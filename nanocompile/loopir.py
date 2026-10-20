"""The loop IR: buffers, loop nests, loads, stores and scalar expressions.

A kernel is what one graph primitive becomes once it has an iteration space:

    Kernel(params=(f32 out, f32 a, f32 b), body=[
        Loop("i0", Sym(0), [
            Loop("i1", 896, [
                Store(0, i0*896 + i1, Load(1, i0*896 + i1) + Load(2, i1)),
            ]),
        ]),
    ])

Parameter 0 is always the output. Every other parameter is an operand, read
through the strides of the view it was lowered from, so a transposed weight or
a broadcast bias is an index expression here and not a copy. ``Sym(i)`` is the
run-time value of the graph's i-th symbol (``seq``, ``past``).

Everything is immutable and compared by value. That is what lets identical
kernels from 24 identical layers be emitted once: two kernels are the same
kernel exactly when their IR is equal.

Schedules (phase 4) rewrite this IR and never leave it. A loop carries a
``kind`` saying how it is to be run (in order, across threads, or as SIMD
lanes), and accumulators that a schedule privatizes become small local arrays.
Every scheduled kernel is still a plain loop nest with a meaning in scalar C,
which is how each schedule is tested: against the same IR run as scalar code.

The constructors fold constants and identities (``x * 1``, ``x + 0``,
``x * 0``) as they build. Index arithmetic is where nearly all of that
happens, and generated C full of ``i1 * 1 + 0`` is harder to read and review
for no benefit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nanocompile import dtypes
from nanocompile.dtypes import DType
from nanocompile.symbolic import Dim, Expr

# -- scalar expressions ------------------------------------------------------


@dataclass(frozen=True)
class Const:
    value: int | float
    dtype: DType


@dataclass(frozen=True)
class Var:
    """A loop index or a kernel-local variable."""

    name: str
    dtype: DType = dtypes.i64


@dataclass(frozen=True)
class Sym:
    """The run-time value of graph symbol ``index``."""

    index: int
    dtype: DType = dtypes.i64


@dataclass(frozen=True)
class Load:
    param: int
    index: "Scalar"
    dtype: DType


@dataclass(frozen=True)
class Binary:
    """``add sub mul div idiv mod`` keep the operand dtype; ``le lt ne or`` give bool.

    ``idiv`` and ``mod`` are integer index arithmetic, used only on
    non-negative operands, where C's truncating division is floor division.
    """

    op: str
    a: "Scalar"
    b: "Scalar"
    dtype: DType


@dataclass(frozen=True)
class Unary:
    """``neg exp sqrt reciprocal sin cos``."""

    op: str
    a: "Scalar"
    dtype: DType


@dataclass(frozen=True)
class Select:
    cond: "Scalar"
    a: "Scalar"
    b: "Scalar"
    dtype: DType


@dataclass(frozen=True)
class Cast:
    a: "Scalar"
    dtype: DType


@dataclass(frozen=True)
class Local:
    """Element ``index`` of the kernel-local array ``name``."""

    name: str
    index: "Scalar"
    dtype: DType


Scalar = Const | Var | Sym | Load | Binary | Unary | Select | Cast | Local

COMPARISONS = frozenset({"le", "lt", "ne", "or"})


def const(value, dtype: DType = dtypes.i64) -> Const:
    return Const(value, dtype)


def _is(x: Scalar, value) -> bool:
    return isinstance(x, Const) and x.value == value


def binary(op: str, a: Scalar, b: Scalar) -> Scalar:
    """Build ``a op b``, folding what can be folded without changing a result.

    Only integer identities are folded. ``x * 1.0`` and ``x + 0.0`` are left
    alone for floats: ``-0.0 + 0.0`` is ``+0.0``, so even that is not an
    identity, and this IR promises to compute what the graph says.
    """
    dtype = dtypes.bool_ if op in COMPARISONS else a.dtype
    if a.dtype.is_int and b.dtype.is_int and op in ("add", "sub", "mul"):
        if isinstance(a, Const) and isinstance(b, Const):
            value = {"add": a.value + b.value, "sub": a.value - b.value, "mul": a.value * b.value}[op]
            return Const(value, dtype)
        if op == "add" and _is(a, 0):
            return b
        if op in ("add", "sub") and _is(b, 0):
            return a
        if op == "mul" and (_is(a, 0) or _is(b, 0)):
            return Const(0, dtype)
        if op == "mul" and _is(a, 1):
            return b
        if op == "mul" and _is(b, 1):
            return a
    return Binary(op, a, b, dtype)


def idiv(a: Scalar, b: Scalar) -> Scalar:
    if _is(b, 1):
        return a
    if isinstance(a, Const) and isinstance(b, Const):
        return Const(a.value // b.value, a.dtype)
    return Binary("idiv", a, b, a.dtype)


def mod(a: Scalar, b: Scalar) -> Scalar:
    if _is(b, 1):
        return Const(0, a.dtype)
    if isinstance(a, Const) and isinstance(b, Const):
        return Const(a.value % b.value, a.dtype)
    return Binary("mod", a, b, a.dtype)


def add(a: Scalar, b: Scalar) -> Scalar:
    return binary("add", a, b)


def mul(a: Scalar, b: Scalar) -> Scalar:
    return binary("mul", a, b)


def from_dim(d: Dim, symbols: tuple[str, ...]) -> Scalar:
    """A shape expression (a polynomial in symbols) as an i64 scalar."""
    if isinstance(d, int):
        return Const(d, dtypes.i64)
    total: Scalar = Const(0, dtypes.i64)
    # Monomials in a fixed order, so the same Expr always renders the same C.
    for monomial, coeff in d.terms():
        term: Scalar = Const(coeff, dtypes.i64)
        for name in monomial:
            term = mul(term, Sym(symbols.index(name)))
        total = add(total, term)
    return total


def linear_index(indices: list[Scalar], strides: tuple[Dim, ...], symbols: tuple[str, ...]) -> Scalar:
    """``sum(index_d * stride_d)``: the element offset of a strided access."""
    total: Scalar = Const(0, dtypes.i64)
    for index, stride in zip(indices, strides):
        total = add(total, mul(index, from_dim(stride, symbols)))
    return total


# -- statements --------------------------------------------------------------


LOOP_KINDS = ("serial", "parallel", "vector")


@dataclass(frozen=True)
class Loop:
    """``for var in 0..extent``.

    ``kind`` is how the iterations run: ``serial`` in order, ``parallel``
    shared across threads, ``vector`` as the lanes of one SIMD instruction. A
    schedule only marks a loop parallel or vector when its iterations are
    independent, so every kind computes what ``serial`` does.
    """

    var: str
    extent: Scalar
    body: tuple["Stmt", ...]
    kind: str = "serial"


@dataclass(frozen=True)
class Store:
    param: int
    index: Scalar
    value: Scalar


@dataclass(frozen=True)
class Let:
    """Declare and initialize a kernel-local variable."""

    var: str
    value: Scalar


@dataclass(frozen=True)
class Assign:
    var: str
    value: Scalar


@dataclass(frozen=True)
class Fail:
    """``if (cond) return code;``: a kernel that detects bad input stops."""

    cond: Scalar
    code: int


@dataclass(frozen=True)
class Array:
    """Declare the kernel-local array ``name``, every element set to ``init``."""

    name: str
    size: int
    init: Const


@dataclass(frozen=True)
class ArrayStore:
    name: str
    index: Scalar
    value: Scalar


@dataclass(frozen=True)
class Block:
    """A nested scope. Unrolling makes one per copy, so their names never clash."""

    body: tuple["Stmt", ...]


Stmt = Loop | Store | Let | Assign | Fail | Array | ArrayStore | Block


@dataclass(frozen=True)
class Param:
    """A kernel argument.

    ``packed`` is set when the kernel reads a ``[rows, cols]`` operand in
    blocks of ``packed`` rows, column-major within a block: element
    ``(r, c)`` at ``(r / packed) * cols * packed + c * packed + r % packed``.
    The runtime hands the kernel that layout, made once from the weight.
    """

    dtype: DType
    writable: bool
    packed: int | None = None


@dataclass(frozen=True)
class Kernel:
    """One loop nest, callable as ``int32 k(void **params, const int64 *syms)``.

    ``bytes_read`` and ``bytes_written`` are the compulsory traffic: every
    distinct element of every operand read once, every output element written
    once. They are a lower bound on what the kernel costs in memory traffic,
    and they are exact, so phase 3 can assert what fusion removed.
    """

    primitive: str
    params: tuple[Param, ...]
    body: tuple[Stmt, ...]
    # Not part of a kernel's identity: two kernels with the same code are the
    # same kernel even when their traffic is described differently.
    bytes_read: Dim = field(default=0, compare=False)
    bytes_written: Dim = field(default=0, compare=False)


def loop_nest(names: list[str], extents: list[Scalar], body: list[Stmt]) -> list[Stmt]:
    """Nest ``body`` in one loop per (name, extent), outermost first.

    A loop of constant extent 1 is not emitted: its variable is the constant 0.
    """
    stmts = body
    for name, extent in reversed(list(zip(names, extents))):
        if _is(extent, 1):
            continue
        stmts = [Loop(name, extent, tuple(stmts))]
    return stmts


def substitute_unit_loops(names: list[str], extents: list[Scalar]) -> list[Scalar]:
    """The index scalar for each loop: ``Var`` normally, ``0`` for unit loops."""
    return [Const(0, dtypes.i64) if _is(e, 1) else Var(n) for n, e in zip(names, extents)]


# -- rewriting ---------------------------------------------------------------
#
# Schedules are rewrites of this IR. These are the traversals they share:
# rebuilding a scalar bottom-up, applying that to every scalar in a block of
# statements, and substituting expressions for loop variables. Rebuilding goes
# through the folding constructors, so substituting a constant for a loop
# variable folds the index arithmetic it lands in.


def _rebuild_binary(op: str, a: Scalar, b: Scalar) -> Scalar:
    if op == "idiv":
        return idiv(a, b)
    if op == "mod":
        return mod(a, b)
    return binary(op, a, b)


def map_scalar(x: Scalar, f) -> Scalar:
    """``x`` rebuilt bottom-up, each node replaced by ``f(node)``."""
    if isinstance(x, Load):
        x = Load(x.param, map_scalar(x.index, f), x.dtype)
    elif isinstance(x, Local):
        x = Local(x.name, map_scalar(x.index, f), x.dtype)
    elif isinstance(x, Binary):
        x = _rebuild_binary(x.op, map_scalar(x.a, f), map_scalar(x.b, f))
    elif isinstance(x, Unary):
        x = Unary(x.op, map_scalar(x.a, f), x.dtype)
    elif isinstance(x, Select):
        x = Select(map_scalar(x.cond, f), map_scalar(x.a, f), map_scalar(x.b, f), x.dtype)
    elif isinstance(x, Cast):
        x = Cast(map_scalar(x.a, f), x.dtype)
    return f(x)


def map_stmts(stmts, f) -> tuple[Stmt, ...]:
    """Every scalar in ``stmts`` (loop extents included) rebuilt by ``map_scalar``."""
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Loop):
            out.append(Loop(s.var, map_scalar(s.extent, f), map_stmts(s.body, f), s.kind))
        elif isinstance(s, Store):
            out.append(Store(s.param, map_scalar(s.index, f), map_scalar(s.value, f)))
        elif isinstance(s, Let):
            out.append(Let(s.var, map_scalar(s.value, f)))
        elif isinstance(s, Assign):
            out.append(Assign(s.var, map_scalar(s.value, f)))
        elif isinstance(s, Fail):
            out.append(Fail(map_scalar(s.cond, f), s.code))
        elif isinstance(s, ArrayStore):
            out.append(ArrayStore(s.name, map_scalar(s.index, f), map_scalar(s.value, f)))
        elif isinstance(s, Block):
            out.append(Block(map_stmts(s.body, f)))
        else:
            out.append(s)
    return tuple(out)


def substitute(stmts, mapping: dict[str, Scalar]) -> tuple[Stmt, ...]:
    """``stmts`` with each ``Var(name)`` in ``mapping`` replaced by its expression."""
    def f(x: Scalar) -> Scalar:
        return mapping.get(x.name, x) if isinstance(x, Var) else x
    return map_stmts(stmts, f)


def walk_scalar(x: Scalar):
    """``x`` and every node below it, depth first."""
    yield x
    if isinstance(x, (Load, Local)):
        yield from walk_scalar(x.index)
    elif isinstance(x, Binary):
        yield from walk_scalar(x.a)
        yield from walk_scalar(x.b)
    elif isinstance(x, (Unary, Cast)):
        yield from walk_scalar(x.a)
    elif isinstance(x, Select):
        yield from walk_scalar(x.cond)
        yield from walk_scalar(x.a)
        yield from walk_scalar(x.b)


def scalars_of(stmts):
    """Every scalar node in ``stmts``, loop extents included, depth first."""
    for s in stmts:
        if isinstance(s, Loop):
            yield from walk_scalar(s.extent)
            yield from scalars_of(s.body)
        elif isinstance(s, Block):
            yield from scalars_of(s.body)
        elif isinstance(s, (Store, ArrayStore)):
            yield from walk_scalar(s.index)
            yield from walk_scalar(s.value)
        elif isinstance(s, (Let, Assign)):
            yield from walk_scalar(s.value)
        elif isinstance(s, Fail):
            yield from walk_scalar(s.cond)


def vars_of(x: Scalar) -> set[str]:
    return {n.name for n in walk_scalar(x) if isinstance(n, Var)}


def stmt_vars(stmts) -> set[str]:
    """Names of every variable read anywhere in ``stmts``."""
    return {n.name for n in scalars_of(stmts) if isinstance(n, Var)}
