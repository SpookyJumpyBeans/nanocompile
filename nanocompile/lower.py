"""Lowering a graph to kernel calls: one loop nest per compute primitive.

Movement primitives (``reshape``, ``permute``, ``broadcast``, ``slice``) do
not become kernels. They become *views*: a buffer, a shape, per-axis strides
and an offset, all symbolic. A kernel that reads a view reads through its
strides, so the transposed weight in ``x @ W.T`` is the weight itself with its
strides swapped, and a broadcast bias is a stride of zero. Without that, every
decode step would copy all 2 GB of weights into transposed buffers first.

The one movement that cannot always be a view is ``reshape``. Merging or
splitting axes is only an index change if the elements are already laid out
contiguously; reshaping a transposed or broadcast view needs the data moved.
Then a ``copy`` kernel materializes the view and the reshape is a view of the
copy. Those copies are real traffic and are counted like any other kernel.

Everything else (arithmetic, reductions, ``concat``, ``matmul``, ``gather``,
``iota``, ``dim``) is one kernel per node, writing a fresh contiguous buffer.
No fusion: that is phase 3, and this phase's numbers are the baseline it is
measured against.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from nanocompile import dtypes
from nanocompile import loopir as L
from nanocompile.dtypes import DType
from nanocompile.ir import Graph, Node
from nanocompile.symbolic import Dim, Expr, Shape, as_expr, canonical, dims_equal, product


class Buffer:
    """Storage for one value: an input, a weight, a constant, or a kernel output.

    ``kind`` is ``input``, ``weight``, ``const``, ``temp`` (planned into the
    arena, dead after its last use) or ``output`` (returned to the caller).
    Compared by identity.
    """

    __slots__ = ("kind", "name", "dtype", "shape", "value")

    def __init__(self, kind: str, dtype: DType, shape: Shape, name: str | None = None,
                 value: np.ndarray | None = None) -> None:
        self.kind = kind
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.value = value

    @property
    def nbytes(self) -> Dim:
        return canonical(as_expr(product(self.shape)) * self.dtype.numpy.itemsize)

    def __repr__(self) -> str:
        return f"<Buffer {self.kind} {self.name or ''} {self.dtype}{list(self.shape)}>"


def contiguous_strides(shape: Shape) -> tuple[Dim, ...]:
    strides: list[Dim] = []
    running: Dim = 1
    for d in reversed(shape):
        strides.append(running)
        running = canonical(as_expr(running) * d)
    return tuple(reversed(strides))


@dataclass(frozen=True)
class View:
    buffer: Buffer
    shape: Shape
    strides: tuple[Dim, ...]
    offset: Dim

    @property
    def dtype(self) -> DType:
        return self.buffer.dtype

    @staticmethod
    def of(buffer: Buffer) -> "View":
        return View(buffer, buffer.shape, contiguous_strides(buffer.shape), 0)

    @property
    def is_contiguous(self) -> bool:
        """Row-major with no gaps, ignoring axes of size 1 (their stride is never used)."""
        expected = contiguous_strides(self.shape)
        return all(
            dims_equal(d, 1) or dims_equal(s, e)
            for d, s, e in zip(self.shape, self.strides, expected)
        )

    @property
    def distinct_elements(self) -> Dim:
        """Elements a full read actually touches: a broadcast axis touches one."""
        return product(d for d, s in zip(self.shape, self.strides) if not dims_equal(s, 0))


@dataclass(frozen=True)
class Call:
    kernel: int
    output: Buffer
    operands: tuple[View, ...]
    primitive: str
    node: int            # graph node number, for error messages


@dataclass
class Program:
    """A graph lowered to an ordered list of kernel calls."""

    graph: Graph
    symbols: tuple[str, ...]
    kernels: list[L.Kernel]
    calls: list[Call]
    buffers: list[Buffer]
    outputs: dict[str, Buffer]

    @property
    def bytes_moved(self) -> Dim:
        """Compulsory bytes read and written by one run, as a polynomial in the symbols."""
        total = Expr.const(0)
        for call in self.calls:
            kernel = self.kernels[call.kernel]
            total = total + kernel.bytes_read + kernel.bytes_written
        return canonical(total)

    def last_uses(self) -> dict[Buffer, int]:
        """For each temp buffer, the index of the last call that reads it."""
        last: dict[Buffer, int] = {}
        for index, call in enumerate(self.calls):
            if call.output.kind == "temp":
                last.setdefault(call.output, index)  # written and never read: dead at once
            for view in call.operands:
                if view.buffer.kind == "temp":
                    last[view.buffer] = index
        return last


# -- kernel generators -------------------------------------------------------

ELEMENTWISE: dict[str, Callable[..., L.Scalar]] = {
    "add": lambda a, b: L.binary("add", a, b),
    "sub": lambda a, b: L.binary("sub", a, b),
    "mul": lambda a, b: L.binary("mul", a, b),
    "div": lambda a, b: L.binary("div", a, b),
    "less_equal": lambda a, b: L.binary("le", a, b),
    "where": lambda c, a, b: L.Select(c, a, b, a.dtype),
    "neg": lambda a: L.Unary("neg", a, a.dtype),
    "exp": lambda a: L.Unary("exp", a, a.dtype),
    "sqrt": lambda a: L.Unary("sqrt", a, a.dtype),
    "reciprocal": lambda a: L.Unary("reciprocal", a, a.dtype),
    "sin": lambda a: L.Unary("sin", a, a.dtype),
    "cos": lambda a: L.Unary("cos", a, a.dtype),
    "copy": lambda a: a,
}


def _cast(a: L.Scalar, dtype: DType) -> L.Scalar:
    if dtype == dtypes.bool_:
        return L.binary("ne", a, L.const(0, a.dtype))
    return L.Cast(a, dtype)


def _itemsize(dtype: DType) -> int:
    return dtype.numpy.itemsize


def _bytes(elements: Dim, dtype: DType) -> Dim:
    return canonical(as_expr(elements) * _itemsize(dtype))


class _KernelBuilder:
    """Shared plumbing: loop variables, extents and strided loads for one kernel."""

    def __init__(self, symbols: tuple[str, ...], out_dtype: DType, operands: tuple[View, ...]) -> None:
        self.symbols = symbols
        self.operands = operands
        self.params = (L.Param(out_dtype, True),) + tuple(L.Param(v.dtype, False) for v in operands)

    def extents(self, shape: Shape) -> list[L.Scalar]:
        return [L.from_dim(d, self.symbols) for d in shape]

    def load(self, operand: int, indices: list[L.Scalar]) -> L.Scalar:
        view = self.operands[operand]
        index = L.linear_index(indices, view.strides, self.symbols)
        return L.Load(operand + 1, index, view.dtype)

    def store(self, shape: Shape, indices: list[L.Scalar], value: L.Scalar) -> L.Store:
        return L.Store(0, L.linear_index(indices, contiguous_strides(shape), self.symbols), value)

    def read_bytes(self) -> Dim:
        total = Expr.const(0)
        for view in self.operands:
            total = total + as_expr(_bytes(view.distinct_elements, view.dtype))
        return canonical(total)

    def kernel(self, primitive: str, body: list[L.Stmt], out_shape: Shape, out_dtype: DType,
               bytes_read: Dim | None = None) -> L.Kernel:
        return L.Kernel(
            primitive,
            self.params,
            tuple(body),
            bytes_read=self.read_bytes() if bytes_read is None else bytes_read,
            bytes_written=_bytes(product(out_shape), out_dtype),
        )


def _names(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(count)]


def _elementwise_kernel(b: _KernelBuilder, primitive: str, shape: Shape, dtype: DType,
                        compute: Callable[[list[L.Scalar], list[L.Scalar]], L.Scalar]) -> L.Kernel:
    names = _names("i", len(shape))
    extents = b.extents(shape)
    idx = L.substitute_unit_loops(names, extents)
    loads = [b.load(p, idx) for p in range(len(b.operands))]
    body = L.loop_nest(names, extents, [b.store(shape, idx, compute(loads, idx))])
    return b.kernel(primitive, body, shape, dtype)


def _reduce_kernel(b: _KernelBuilder, node: Node, axis: int) -> L.Kernel:
    view = b.operands[0]
    names = _names("i", node.rank)
    extents = b.extents(node.shape)
    idx = L.substitute_unit_loops(names, extents)
    in_idx = list(idx)
    in_idx[axis] = L.Var("r")
    value = b.load(0, in_idx)
    acc = L.Var("acc", node.dtype)

    if node.op == "reduce_sum":
        init: L.Scalar = L.const(0.0 if node.dtype.is_float else 0, node.dtype)
        update = L.binary("add", acc, value)
    else:
        init = L.const(-np.inf if node.dtype.is_float else int(np.iinfo(node.dtype.numpy).min), node.dtype)
        bigger = L.binary("lt", acc, value)
        if node.dtype.is_float:
            # NaN propagates, as NumPy's max does: once seen, nothing compares above it.
            bigger = L.binary("or", bigger, L.binary("ne", value, value))
        update = L.Select(bigger, value, acc, node.dtype)

    inner = [
        L.Let("acc", init),
        L.Loop("r", L.from_dim(view.shape[axis], b.symbols), (L.Assign("acc", update),)),
        b.store(node.shape, idx, acc),
    ]
    return b.kernel(node.op, L.loop_nest(names, extents, inner), node.shape, node.dtype)


def _matmul_kernel(b: _KernelBuilder, node: Node) -> L.Kernel:
    """``acc += A[..., i, k] * B[..., k, j]``, sequentially in k. No FMA, no reordering."""
    a, w = b.operands
    names = _names("i", node.rank)
    extents = b.extents(node.shape)
    idx = L.substitute_unit_loops(names, extents)
    k = L.Var("k")
    product_ = L.binary("mul", b.load(0, idx[:-1] + [k]), b.load(1, idx[:-2] + [k, idx[-1]]))
    acc = L.Var("acc", node.dtype)
    inner = [
        L.Let("acc", L.const(0.0, node.dtype)),
        L.Loop("k", L.from_dim(a.shape[-1], b.symbols), (L.Assign("acc", L.binary("add", acc, product_)),)),
        b.store(node.shape, idx, acc),
    ]
    return b.kernel("matmul", L.loop_nest(names, extents, inner), node.shape, node.dtype)


def _gather_kernel(b: _KernelBuilder, node: Node) -> L.Kernel:
    """Rows of the table by index, failing with code 1 on an index out of range."""
    table, ids = b.operands
    outer_names = _names("i", len(ids.shape))
    outer_extents = b.extents(ids.shape)
    outer = L.substitute_unit_loops(outer_names, outer_extents)
    rest_shape = table.shape[1:]
    rest_names = _names("j", len(rest_shape))
    rest_extents = b.extents(rest_shape)
    rest = L.substitute_unit_loops(rest_names, rest_extents)

    row = L.Var("row")
    rows = L.from_dim(table.shape[0], b.symbols)
    out_of_range = L.binary("or", L.binary("lt", row, L.const(0)), L.binary("le", rows, row))
    inner = L.loop_nest(rest_names, rest_extents, [
        b.store(node.shape, outer + rest, b.load(0, [row] + rest)),
    ])
    body = L.loop_nest(outer_names, outer_extents, [
        L.Let("row", L.Cast(b.load(1, outer), dtypes.i64)),
        L.Fail(out_of_range, 1),
        *inner,
    ])
    touched = as_expr(product(ids.shape)) * product(rest_shape)
    bytes_read = canonical(as_expr(_bytes(touched, table.dtype)) + _bytes(ids.distinct_elements, ids.dtype))
    return b.kernel("gather", body, node.shape, node.dtype, bytes_read=bytes_read)


def _concat_kernel(b: _KernelBuilder, node: Node) -> L.Kernel:
    """One loop nest per operand, each writing its slab of the output."""
    axis = node.attrs["axis"] % node.rank
    body: list[L.Stmt] = []
    start: Dim = 0
    for p, view in enumerate(b.operands):
        names = _names("i", node.rank)
        extents = b.extents(view.shape)
        idx = L.substitute_unit_loops(names, extents)
        out_idx = list(idx)
        out_idx[axis] = L.add(idx[axis], L.from_dim(start, b.symbols))
        body.extend(L.loop_nest(names, extents, [b.store(node.shape, out_idx, b.load(p, idx))]))
        start = canonical(as_expr(start) + view.shape[axis])
    return b.kernel("concat", body, node.shape, node.dtype)


def _iota_kernel(b: _KernelBuilder, node: Node) -> L.Kernel:
    axis = node.attrs["axis"] % node.rank
    return _elementwise_kernel(b, "iota", node.shape, node.dtype,
                               lambda loads, idx: L.Cast(idx[axis], node.dtype))


def _dim_kernel(b: _KernelBuilder, node: Node) -> L.Kernel:
    value = L.from_dim(node.attrs["expr"], b.symbols)
    return b.kernel("dim", [L.Store(0, L.const(0), value)], (), node.dtype)


# -- the lowering ------------------------------------------------------------


class _Lowering:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.symbols = graph.symbols
        self.kernels: list[L.Kernel] = []
        self.kernel_ids: dict[L.Kernel, int] = {}
        self.calls: list[Call] = []
        self.buffers: list[Buffer] = []
        self.values: dict[Node, View] = {}

    def buffer(self, kind: str, dtype: DType, shape: Shape, name: str | None = None,
               value: np.ndarray | None = None) -> Buffer:
        buf = Buffer(kind, dtype, shape, name, value)
        self.buffers.append(buf)
        return buf

    def emit(self, kernel: L.Kernel, output: Buffer, operands: tuple[View, ...], node: int) -> None:
        index = self.kernel_ids.get(kernel)
        if index is None:
            index = self.kernel_ids[kernel] = len(self.kernels)
            self.kernels.append(kernel)
        self.calls.append(Call(index, output, operands, kernel.primitive, node))

    def copy(self, view: View, node: int, into: Buffer | None = None) -> View:
        """Materialize a view into a contiguous buffer of the same shape."""
        out = into or self.buffer("temp", view.dtype, view.shape)
        b = _KernelBuilder(self.symbols, view.dtype, (view,))
        kernel = _elementwise_kernel(b, "copy", view.shape, view.dtype, lambda loads, idx: loads[0])
        self.emit(kernel, out, (view,), node)
        return View.of(out)

    def lower_node(self, index: int, node: Node) -> View:
        op = node.op
        if op == "input":
            return View.of(self.buffer("input", node.dtype, node.shape, node.attrs["name"]))
        if op == "weight":
            return View.of(self.buffer("weight", node.dtype, node.shape, node.attrs["name"]))
        if op == "const":
            value = node.attrs["value"]
            return View.of(self.buffer("const", node.dtype, node.shape, f"const{index}", value))

        operands = tuple(self.values[x] for x in node.inputs)

        if op == "reshape":
            (view,) = operands
            if not view.is_contiguous:
                view = self.copy(view, index)
            return View(view.buffer, node.shape, contiguous_strides(node.shape), view.offset)
        if op == "permute":
            (view,) = operands
            perm = node.attrs["perm"]
            return View(view.buffer, node.shape, tuple(view.strides[p] for p in perm), view.offset)
        if op == "broadcast":
            (view,) = operands
            strides = tuple(
                0 if dims_equal(have, 1) and not dims_equal(want, 1) else stride
                for have, want, stride in zip(view.shape, node.shape, view.strides)
            )
            return View(view.buffer, node.shape, strides, view.offset)
        if op == "slice":
            (view,) = operands
            axis = node.attrs["axis"] % node.rank
            offset = as_expr(view.offset) + as_expr(node.attrs["start"]) * view.strides[axis]
            return View(view.buffer, node.shape, view.strides, canonical(offset))

        b = _KernelBuilder(self.symbols, node.dtype, operands)
        if op in ELEMENTWISE:
            compute = ELEMENTWISE[op]
            kernel = _elementwise_kernel(b, op, node.shape, node.dtype, lambda loads, idx: compute(*loads))
        elif op == "cast":
            kernel = _elementwise_kernel(b, op, node.shape, node.dtype,
                                         lambda loads, idx: _cast(loads[0], node.dtype))
        elif op in ("reduce_sum", "reduce_max"):
            kernel = _reduce_kernel(b, node, node.attrs["axis"] % node.rank)
        elif op == "matmul":
            kernel = _matmul_kernel(b, node)
        elif op == "gather":
            kernel = _gather_kernel(b, node)
        elif op == "concat":
            kernel = _concat_kernel(b, node)
        elif op == "iota":
            kernel = _iota_kernel(b, node)
        elif op == "dim":
            kernel = _dim_kernel(b, node)
        else:
            raise NotImplementedError(f"no lowering for {op!r}")

        out = self.buffer("temp", node.dtype, node.shape)
        self.emit(kernel, out, operands, index)
        return View.of(out)

    def run(self) -> Program:
        for index, node in enumerate(self.graph.nodes):
            self.values[node] = self.lower_node(index, node)

        outputs: dict[str, Buffer] = {}
        claimed: set[Buffer] = set()
        for name, node in self.graph.outputs.items():
            view = self.values[node]
            whole = (
                view.buffer.kind == "temp"
                and view.buffer not in claimed
                and view.is_contiguous
                and dims_equal(view.offset, 0)
                and len(view.shape) == len(view.buffer.shape)
                and all(dims_equal(a, b) for a, b in zip(view.shape, view.buffer.shape))
            )
            if whole:
                # The kernel that made it writes straight into the returned array.
                view.buffer.kind, view.buffer.name = "output", name
                buf = view.buffer
            else:
                buf = self.buffer("output", node.dtype, node.shape, name)
                self.copy(view, self.graph.ids[node], into=buf)
            claimed.add(buf)
            outputs[name] = buf

        return Program(self.graph, self.symbols, self.kernels, self.calls, self.buffers, outputs)


def lower(graph: Graph) -> Program:
    return _Lowering(graph).run()
