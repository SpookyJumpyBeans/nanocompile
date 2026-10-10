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


Scalar = Const | Var | Sym | Load | Binary | Unary | Select | Cast

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


@dataclass(frozen=True)
class Loop:
    var: str
    extent: Scalar
    body: tuple["Stmt", ...]


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


Stmt = Loop | Store | Let | Assign | Fail


@dataclass(frozen=True)
class Param:
    dtype: DType
    writable: bool


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
