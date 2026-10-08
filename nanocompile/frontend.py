"""Building graphs with ordinary Python operators.

A :class:`Tensor` wraps a node. Arithmetic on tensors does not compute
anything; it creates nodes, so model code reads like the NumPy it replaces:

    b = GraphBuilder("demo")
    seq = b.symbol("seq")
    x = b.input("x", f32, (seq, 16))
    y = (x * x).sum(-1) / 16.0          # nodes, typed f32[seq, 1]

Two conveniences live here and nowhere else, so the IR stays strict:

* **Broadcasting.** Operands are aligned NumPy-style and the frontend inserts
  explicit ``reshape`` and ``broadcast`` nodes. The IR's elementwise
  primitives only ever see equal shapes.
* **Scalars.** A Python number takes the dtype of the tensor it meets, the way
  NumPy treats Python scalars as weakly typed. ``x * 0.5`` on an ``f32``
  tensor multiplies by an ``f32`` constant, never promotes to float64.
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence

import numpy as np

from nanocompile import dtypes
from nanocompile.dtypes import DType
from nanocompile.ir import Graph, Node, ShapeError, make
from nanocompile.symbolic import (
    Dim,
    Expr,
    Shape,
    canonical_shape,
    dims_equal,
    format_shape,
    shapes_equal,
)


class GraphBuilder:
    """Collects the inputs and weights of one graph while it is written."""

    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self._inputs: dict[str, Node] = {}
        self._weights: dict[str, Node] = {}

    def symbol(self, name: str) -> Expr:
        return Expr.symbol(name)

    def input(self, name: str, dtype: DType, shape: Sequence[Dim]) -> "Tensor":
        if name in self._inputs:
            raise ValueError(f"input {name!r} is already declared")
        node = make("input", name=name, dtype=dtype, shape=tuple(shape))
        self._inputs[name] = node
        return Tensor(node, self)

    def weight(self, name: str, shape: Sequence[int], dtype: DType = dtypes.f32) -> "Tensor":
        """A named parameter, bound when the graph runs.

        Asking for the same name twice returns the same node, which is how the
        tied embedding becomes one weight read in two places rather than two
        weights that happen to share a name.
        """
        shape = canonical_shape(shape)
        existing = self._weights.get(name)
        if existing is not None:
            if existing.dtype != dtype or not shapes_equal(existing.shape, shape):
                raise ShapeError(
                    f"weight {name!r} requested as {dtype}{format_shape(shape)}, "
                    f"already declared as {existing.type_str}"
                )
            return Tensor(existing, self)
        node = make("weight", name=name, dtype=dtype, shape=shape)
        self._weights[name] = node
        return Tensor(node, self)

    def const(self, value, dtype: DType | None = None) -> "Tensor":
        """A constant array. Python numbers need an explicit dtype."""
        if dtype is None:
            if not isinstance(value, np.ndarray):
                raise TypeError("a constant from a Python value needs an explicit dtype")
            array = value
        else:
            array = np.asarray(value, dtype=dtype.numpy)
        array = np.array(array, copy=True)  # owned and immutable from here on
        array.flags.writeable = False
        return Tensor(make("const", value=array), self)

    def iota(self, shape: Sequence[Dim], axis: int, dtype: DType = dtypes.i64) -> "Tensor":
        """``0, 1, 2, ...`` along ``axis``, repeated across the others."""
        return Tensor(make("iota", shape=tuple(shape), axis=axis, dtype=dtype), self)

    def dim(self, expr: Dim) -> "Tensor":
        """The run-time value of a dimension, as an ``i64`` scalar."""
        if isinstance(expr, int):
            return self.const(expr, dtypes.i64)
        return Tensor(make("dim", expr=expr), self)

    def build(self, outputs: Mapping[str, "Tensor"]) -> Graph:
        nodes = {}
        for name, tensor in outputs.items():
            if not isinstance(tensor, Tensor) or tensor.builder is not self:
                raise ValueError(f"output {name!r} is not a tensor from this builder")
            nodes[name] = tensor.node
        return Graph(list(self._inputs.values()), nodes, name=self.name)


def broadcast_shape(a: Shape, b: Shape) -> Shape:
    """NumPy's rule: align from the right, a size-1 axis stretches."""
    rank = max(len(a), len(b))
    a = (1,) * (rank - len(a)) + tuple(a)
    b = (1,) * (rank - len(b)) + tuple(b)
    out: list[Dim] = []
    for x, y in zip(a, b):
        if dims_equal(x, y) or dims_equal(y, 1):
            out.append(x)
        elif dims_equal(x, 1):
            out.append(y)
        else:
            raise ShapeError(f"cannot broadcast {format_shape(a)} with {format_shape(b)}")
    return tuple(out)


class Tensor:
    """A node, plus the operators that build new ones from it."""

    __slots__ = ("node", "builder")

    def __init__(self, node: Node, builder: GraphBuilder) -> None:
        self.node = node
        self.builder = builder

    # -- type --------------------------------------------------------------

    @property
    def shape(self) -> Shape:
        return self.node.shape

    @property
    def dtype(self) -> DType:
        return self.node.dtype

    @property
    def ndim(self) -> int:
        return self.node.rank

    def __repr__(self) -> str:
        return f"Tensor({self.node.op} : {self.node.type_str})"

    # Tensors are nodes, not values: ``==`` building a comparison would make
    # them unhashable, and truthiness would be a question about data that
    # does not exist yet.
    __hash__ = object.__hash__

    def __bool__(self) -> bool:
        raise TypeError("a Tensor has no value while the graph is being built")

    # -- helpers -----------------------------------------------------------

    def _new(self, op: str, *operands: "Tensor", **attrs) -> "Tensor":
        return Tensor(make(op, *(t.node for t in operands), **attrs), self.builder)

    def _lift(self, other) -> "Tensor":
        """Turn a Python or NumPy scalar into a constant of this tensor's dtype."""
        if isinstance(other, Tensor):
            if other.builder is not self.builder:
                raise ValueError("cannot combine tensors from different builders")
            return other
        if isinstance(other, (numbers.Number, np.number)) and not isinstance(other, bool):
            if isinstance(other, (float, np.floating)) and not self.dtype.is_float:
                raise TypeError(f"a float scalar cannot meet a {self.dtype} tensor")
            return self.builder.const(other, self.dtype)
        raise TypeError(f"cannot use a {type(other).__name__} in a graph")

    def broadcast_to(self, shape: Sequence[Dim]) -> "Tensor":
        """Explicit reshape and broadcast nodes, only where they are needed."""
        shape = canonical_shape(shape)
        if shapes_equal(self.shape, shape):
            return self
        if len(shape) < self.ndim:
            raise ShapeError(f"cannot broadcast {self.node.type_str} to {format_shape(shape)}")
        x = self
        if self.ndim < len(shape):
            x = x.reshape((1,) * (len(shape) - self.ndim) + self.shape)
        if not shapes_equal(x.shape, shape):
            x = x._new("broadcast", x, shape=shape)
        return x

    def _binary(self, op: str, other, reverse: bool = False) -> "Tensor":
        other = self._lift(other)
        a, b = (other, self) if reverse else (self, other)
        shape = broadcast_shape(a.shape, b.shape)
        return self._new(op, a.broadcast_to(shape), b.broadcast_to(shape))

    # -- operators ---------------------------------------------------------

    def __add__(self, other): return self._binary("add", other)
    def __radd__(self, other): return self._binary("add", other, reverse=True)
    def __sub__(self, other): return self._binary("sub", other)
    def __rsub__(self, other): return self._binary("sub", other, reverse=True)
    def __mul__(self, other): return self._binary("mul", other)
    def __rmul__(self, other): return self._binary("mul", other, reverse=True)
    def __truediv__(self, other): return self._binary("div", other)
    def __rtruediv__(self, other): return self._binary("div", other, reverse=True)
    def __le__(self, other): return self._binary("less_equal", other)
    def __ge__(self, other): return self._binary("less_equal", other, reverse=True)
    def __neg__(self): return self._new("neg", self)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return self._new("matmul", self, self._lift(other))

    # -- elementwise functions ---------------------------------------------

    def exp(self) -> "Tensor": return self._new("exp", self)
    def sqrt(self) -> "Tensor": return self._new("sqrt", self)
    def reciprocal(self) -> "Tensor": return self._new("reciprocal", self)
    def sin(self) -> "Tensor": return self._new("sin", self)
    def cos(self) -> "Tensor": return self._new("cos", self)

    def cast(self, dtype: DType) -> "Tensor":
        return self if dtype == self.dtype else self._new("cast", self, dtype=dtype)

    # -- reductions (the reduced axis is kept, with size 1) -----------------

    def sum(self, axis: int) -> "Tensor":
        return self._new("reduce_sum", self, axis=axis)

    def max(self, axis: int) -> "Tensor":
        return self._new("reduce_max", self, axis=axis)

    # -- movement ----------------------------------------------------------

    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = canonical_shape(shape)
        if shapes_equal(shape, self.shape):
            return self
        return self._new("reshape", self, shape=shape)

    def permute(self, *perm: int) -> "Tensor":
        if len(perm) == 1 and isinstance(perm[0], (tuple, list)):
            perm = tuple(perm[0])
        return self._new("permute", self, perm=tuple(perm))

    def swapaxes(self, a: int, b: int) -> "Tensor":
        perm = list(range(self.ndim))
        perm[a], perm[b] = perm[b], perm[a]
        return self.permute(*perm)

    def slice(self, axis: int, start: Dim, stop: Dim) -> "Tensor":
        return self._new("slice", self, axis=axis, start=start, stop=stop)


def where(cond: Tensor, a, b) -> Tensor:
    """``a`` where ``cond`` holds, else ``b``; scalars take the other's dtype."""
    anchor = a if isinstance(a, Tensor) else b
    if not isinstance(anchor, Tensor):
        raise TypeError("where needs at least one of a and b to be a Tensor")
    a, b = anchor._lift(a), anchor._lift(b)
    shape = broadcast_shape(broadcast_shape(cond.shape, a.shape), b.shape)
    return cond._new("where", cond.broadcast_to(shape), a.broadcast_to(shape), b.broadcast_to(shape))


def concat(tensors: Sequence[Tensor], axis: int) -> Tensor:
    if not tensors:
        raise ValueError("concat needs at least one tensor")
    first = tensors[0]
    return first._new("concat", *tensors, axis=axis)


def gather(table: Tensor, ids: Tensor) -> Tensor:
    return table._new("gather", table, ids)
