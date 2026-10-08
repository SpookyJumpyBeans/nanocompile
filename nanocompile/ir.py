"""The graph IR: nodes, the primitives they apply, and the graph around them.

Every value is produced by exactly one node and never changes (SSA). A node
knows its dtype and shape the moment it is created, because shape inference
runs in the constructor: a mismatch is an error at the line of model code that
caused it, not a broadcasting failure somewhere inside the interpreter.

The primitive set is small on purpose. ``rms_norm`` is not a primitive and
neither is ``softmax``, ``silu``, RoPE or attention; they are compositions
(``nn.py``), and the graph sees the multiplies and reductions they are made
of. That is the raw material for phase 3: there is nothing to fuse in a graph
whose nodes are already whole layers.

The primitives, by family:

    leaves       input, weight, const, iota, dim
    elementwise  add, sub, mul, div, less_equal, where,
                 neg, exp, sqrt, reciprocal, sin, cos, cast
    reductions   reduce_sum, reduce_max
    movement     reshape, permute, broadcast, slice, concat
    other        matmul, gather

Elementwise primitives require their operands to have identical shapes.
NumPy-style implicit broadcasting is a frontend convenience that inserts
explicit ``reshape`` and ``broadcast`` nodes, so every broadcast in the model is
visible in the IR and a later pass never has to rediscover one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from nanocompile import dtypes
from nanocompile.dtypes import DType
from nanocompile.symbolic import (
    Dim,
    Expr,
    Shape,
    as_expr,
    canonical,
    canonical_shape,
    dims_equal,
    format_shape,
    product,
    shapes_equal,
    symbols_of,
)


class ShapeError(ValueError):
    """A node's operands do not fit the primitive it applies."""


class GraphError(ValueError):
    """A graph is malformed as a whole: undeclared inputs, clashing names."""


class Node:
    """One value in the graph, and the primitive that produces it.

    Identity is the node itself, not its contents: two ``add`` nodes over the
    same operands are two values until phase 3's CSE decides otherwise.
    """

    __slots__ = ("op", "inputs", "attrs", "dtype", "shape")

    def __init__(
        self,
        op: str,
        inputs: tuple["Node", ...],
        attrs: Mapping[str, Any],
        dtype: DType,
        shape: Shape,
    ) -> None:
        self.op = op
        self.inputs = inputs
        self.attrs = MappingProxyType(dict(attrs))
        self.dtype = dtype
        self.shape = shape

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def type_str(self) -> str:
        return f"{self.dtype}{format_shape(self.shape)}"

    def __repr__(self) -> str:
        return f"<Node {self.op} : {self.type_str}>"


# -- shape inference ---------------------------------------------------------

Infer = Callable[[tuple[Node, ...], dict[str, Any]], tuple[DType, Shape]]


@dataclass(frozen=True)
class OpDef:
    name: str
    family: str
    arity: int | None          # None for variadic
    infer: Infer


OPS: dict[str, OpDef] = {}


def _register(name: str, family: str, arity: int | None) -> Callable[[Infer], Infer]:
    def wrap(infer: Infer) -> Infer:
        OPS[name] = OpDef(name, family, arity, infer)
        return infer

    return wrap


def _fail(op: str, message: str) -> ShapeError:
    return ShapeError(f"{op}: {message}")


def _normalize_axis(op: str, axis: int, rank: int) -> int:
    if not -rank <= axis < rank:
        raise _fail(op, f"axis {axis} is out of range for rank {rank}")
    return axis % rank


def _check_shape_attr(op: str, shape: Shape) -> Shape:
    for d in shape:
        if isinstance(d, int) and d < 0:
            raise _fail(op, f"negative dimension in {format_shape(shape)}")
    return canonical_shape(shape)


# leaves


@_register("input", "leaf", 0)
def _infer_input(inputs, attrs):
    return attrs["dtype"], _check_shape_attr("input", attrs["shape"])


@_register("weight", "leaf", 0)
def _infer_weight(inputs, attrs):
    shape = _check_shape_attr("weight", attrs["shape"])
    if symbols_of(shape):
        raise _fail("weight", f"{attrs['name']!r} has a symbolic shape {format_shape(shape)}")
    return attrs["dtype"], shape


@_register("const", "leaf", 0)
def _infer_const(inputs, attrs):
    value: np.ndarray = attrs["value"]
    return dtypes.from_numpy(value.dtype), tuple(int(d) for d in value.shape)


@_register("iota", "leaf", 0)
def _infer_iota(inputs, attrs):
    shape = _check_shape_attr("iota", attrs["shape"])
    _normalize_axis("iota", attrs["axis"], len(shape))
    if not attrs["dtype"].is_numeric:
        raise _fail("iota", f"dtype must be numeric, got {attrs['dtype']}")
    return attrs["dtype"], shape


@_register("dim", "leaf", 0)
def _infer_dim(inputs, attrs):
    """A dimension's value at run time, as an ``i64`` scalar.

    This is how a symbol reaches arithmetic: positions are ``iota + past``,
    and ``past`` is a size, not a tensor anyone passed in.
    """
    return dtypes.i64, ()


# elementwise


def _same_operands(op: str, inputs: tuple[Node, ...]) -> None:
    first = inputs[0]
    for other in inputs[1:]:
        if other.dtype != first.dtype:
            raise _fail(op, f"dtype mismatch: {first.type_str} and {other.type_str}")
        if not shapes_equal(first.shape, other.shape):
            raise _fail(
                op,
                f"shape mismatch: {first.type_str} and {other.type_str} "
                "(elementwise operands must match; the frontend inserts broadcasts)",
            )


def _binary(name: str, float_only: bool = False) -> None:
    @_register(name, "elementwise", 2)
    def infer(inputs, attrs):
        _same_operands(name, inputs)
        dtype = inputs[0].dtype
        if not dtype.is_numeric or (float_only and not dtype.is_float):
            kind = "a float" if float_only else "a numeric"
            raise _fail(name, f"needs {kind} dtype, got {dtype}")
        return dtype, inputs[0].shape


for _name in ("add", "sub", "mul"):
    _binary(_name)
# Integer division has three common definitions and the model uses none of them.
_binary("div", float_only=True)


@_register("less_equal", "elementwise", 2)
def _infer_less_equal(inputs, attrs):
    _same_operands("less_equal", inputs)
    if not inputs[0].dtype.is_numeric:
        raise _fail("less_equal", f"needs a numeric dtype, got {inputs[0].dtype}")
    return dtypes.bool_, inputs[0].shape


@_register("where", "elementwise", 3)
def _infer_where(inputs, attrs):
    cond, a, b = inputs
    if cond.dtype != dtypes.bool_:
        raise _fail("where", f"condition must be bool, got {cond.type_str}")
    _same_operands("where", (a, b))
    if not shapes_equal(cond.shape, a.shape):
        raise _fail("where", f"condition {cond.type_str} does not match {a.type_str}")
    return a.dtype, a.shape


def _unary(name: str, float_only: bool) -> None:
    @_register(name, "elementwise", 1)
    def infer(inputs, attrs):
        dtype = inputs[0].dtype
        if not dtype.is_numeric or (float_only and not dtype.is_float):
            kind = "a float" if float_only else "a numeric"
            raise _fail(name, f"needs {kind} dtype, got {dtype}")
        return dtype, inputs[0].shape


_unary("neg", float_only=False)
for _name in ("exp", "sqrt", "reciprocal", "sin", "cos"):
    _unary(_name, float_only=True)


@_register("cast", "elementwise", 1)
def _infer_cast(inputs, attrs):
    return attrs["dtype"], inputs[0].shape


# reductions


def _reduction(name: str) -> None:
    @_register(name, "reduce", 1)
    def infer(inputs, attrs):
        (x,) = inputs
        if not x.dtype.is_numeric:
            raise _fail(name, f"needs a numeric dtype, got {x.dtype}")
        if x.rank == 0:
            raise _fail(name, "cannot reduce a scalar")
        axis = _normalize_axis(name, attrs["axis"], x.rank)
        if name == "reduce_max" and dims_equal(x.shape[axis], 0):
            raise _fail(name, "max over an empty axis has no value")
        # Reductions keep the reduced axis with size 1. Dropping it is a
        # reshape, and keeping it means the result broadcasts straight back.
        return x.dtype, x.shape[:axis] + (1,) + x.shape[axis + 1 :]


_reduction("reduce_sum")
_reduction("reduce_max")


# movement


@_register("reshape", "movement", 1)
def _infer_reshape(inputs, attrs):
    (x,) = inputs
    shape = _check_shape_attr("reshape", attrs["shape"])
    if not dims_equal(product(x.shape), product(shape)):
        raise _fail(
            "reshape",
            f"cannot reshape {x.type_str} ({product(x.shape)} elements) "
            f"to {format_shape(shape)} ({product(shape)} elements)",
        )
    return x.dtype, shape


@_register("permute", "movement", 1)
def _infer_permute(inputs, attrs):
    (x,) = inputs
    perm = tuple(attrs["perm"])
    if sorted(perm) != list(range(x.rank)):
        raise _fail("permute", f"{perm} is not a permutation of {x.rank} axes")
    return x.dtype, tuple(x.shape[p] for p in perm)


@_register("broadcast", "movement", 1)
def _infer_broadcast(inputs, attrs):
    """Stretch size-1 axes. Same rank in and out: adding axes is a reshape."""
    (x,) = inputs
    shape = _check_shape_attr("broadcast", attrs["shape"])
    if len(shape) != x.rank:
        raise _fail("broadcast", f"rank {x.rank} to rank {len(shape)}; reshape first")
    for have, want in zip(x.shape, shape):
        if not (dims_equal(have, want) or dims_equal(have, 1)):
            raise _fail("broadcast", f"cannot broadcast {x.type_str} to {format_shape(shape)}")
    return x.dtype, shape


@_register("slice", "movement", 1)
def _infer_slice(inputs, attrs):
    """``x[..., start:stop, ...]`` on one axis, with a unit step.

    Bounds are checked here when they are decidable from the expressions, and
    by the interpreter when they depend on what the symbols are bound to.
    """
    (x,) = inputs
    axis = _normalize_axis("slice", attrs["axis"], x.rank)
    start, stop, size = as_expr(attrs["start"]), as_expr(attrs["stop"]), as_expr(x.shape[axis])

    for label, gap in (("start", start), ("stop - start", stop - start), ("size - stop", size - stop)):
        if gap.is_constant and gap.constant < 0:
            raise _fail("slice", f"{label} is negative: [{start}:{stop}] on axis of size {size}")

    shape = list(x.shape)
    shape[axis] = canonical(stop - start)
    return x.dtype, tuple(shape)


@_register("concat", "movement", None)
def _infer_concat(inputs, attrs):
    if not inputs:
        raise _fail("concat", "needs at least one operand")
    first = inputs[0]
    axis = _normalize_axis("concat", attrs["axis"], first.rank)
    total: Dim = 0
    for x in inputs:
        if x.dtype != first.dtype:
            raise _fail("concat", f"dtype mismatch: {first.type_str} and {x.type_str}")
        if x.rank != first.rank:
            raise _fail("concat", f"rank mismatch: {first.type_str} and {x.type_str}")
        for i, (a, b) in enumerate(zip(first.shape, x.shape)):
            if i != axis and not dims_equal(a, b):
                raise _fail("concat", f"{first.type_str} and {x.type_str} differ off axis {axis}")
        total = as_expr(total) + x.shape[axis]
    shape = list(first.shape)
    shape[axis] = canonical(total)
    return first.dtype, tuple(shape)


# the rest


@_register("matmul", "matmul", 2)
def _infer_matmul(inputs, attrs):
    """``[..., M, K] @ [..., K, N]``, with batch dimensions that already match.

    No batch broadcasting. Attention repeats its KV heads explicitly, and the
    repeat being a node is the point (see ``nn.repeat_kv``).
    """
    a, b = inputs
    if a.rank < 2 or b.rank < 2:
        raise _fail("matmul", f"operands need rank >= 2, got {a.type_str} and {b.type_str}")
    if a.rank != b.rank:
        raise _fail("matmul", f"rank mismatch: {a.type_str} and {b.type_str}")
    if a.dtype != b.dtype or not a.dtype.is_float:
        raise _fail("matmul", f"needs matching float operands, got {a.type_str} and {b.type_str}")
    if not shapes_equal(a.shape[:-2], b.shape[:-2]):
        raise _fail("matmul", f"batch dimensions differ: {a.type_str} and {b.type_str}")
    if not dims_equal(a.shape[-1], b.shape[-2]):
        raise _fail("matmul", f"inner dimensions differ: {a.type_str} @ {b.type_str}")
    return a.dtype, a.shape[:-1] + (b.shape[-1],)


@_register("gather", "gather", 2)
def _infer_gather(inputs, attrs):
    """Rows of ``table`` by index: ``table[ids]``, the embedding lookup."""
    table, ids = inputs
    if table.rank < 1:
        raise _fail("gather", f"table must have rank >= 1, got {table.type_str}")
    if not ids.dtype.is_int:
        raise _fail("gather", f"indices must be integers, got {ids.type_str}")
    return table.dtype, ids.shape + table.shape[1:]


def make(op: str, *inputs: Node, **attrs: Any) -> Node:
    """Create a node, inferring its type. The only way nodes are made."""
    definition = OPS.get(op)
    if definition is None:
        raise ValueError(f"unknown primitive {op!r}")
    if definition.arity is not None and len(inputs) != definition.arity:
        raise _fail(op, f"takes {definition.arity} operands, got {len(inputs)}")
    for x in inputs:
        if not isinstance(x, Node):
            raise TypeError(f"{op}: operands must be Nodes, got {type(x).__name__}")
    dtype, shape = definition.infer(inputs, attrs)
    return Node(op, inputs, attrs, dtype, canonical_shape(shape))


# -- the graph ---------------------------------------------------------------


def _attr_symbols(node: Node) -> frozenset[str]:
    names: set[str] = set()
    for value in node.attrs.values():
        if isinstance(value, Expr):
            names |= value.symbols
        elif isinstance(value, tuple):
            names |= symbols_of(v for v in value if isinstance(v, (int, Expr)) and not isinstance(v, bool))
    return frozenset(names)


class Graph:
    """A computation from named inputs and weights to named outputs.

    ``nodes`` is a topological order: every node after all of its operands.
    The order is deterministic for a given construction, which matters from
    phase 2 on, where compiling the same graph twice must emit the same C.
    """

    def __init__(
        self,
        inputs: Sequence[Node],
        outputs: Mapping[str, Node],
        name: str = "graph",
    ) -> None:
        if not outputs:
            raise GraphError("a graph needs at least one output")
        self.name = name
        self.inputs: tuple[Node, ...] = tuple(inputs)
        self.outputs: dict[str, Node] = dict(outputs)
        self.nodes: list[Node] = _toposort([*self.inputs, *self.outputs.values()])
        self.ids: dict[Node, int] = {node: i for i, node in enumerate(self.nodes)}
        self.weights: dict[str, Node] = {
            n.attrs["name"]: n for n in self.nodes if n.op == "weight"
        }
        self.validate()

    @property
    def input_names(self) -> list[str]:
        return [n.attrs["name"] for n in self.inputs]

    @property
    def symbols(self) -> tuple[str, ...]:
        """Symbols the inputs' shapes bind, in the order they first appear."""
        seen: list[str] = []
        for node in self.inputs:
            for d in node.shape:
                if isinstance(d, Expr):
                    name = d.as_symbol
                    if name is not None and name not in seen:
                        seen.append(name)
        return tuple(seen)

    def validate(self) -> None:
        """Check the invariants every pass must preserve.

        Run on construction, and by tests after every pass from phase 3 on: a
        pass that leaves a stale shape or an orphaned input is caught at the
        pass, not three passes later.
        """
        names: set[str] = set()
        for node in self.inputs:
            if node.op != "input":
                raise GraphError(f"declared input is a {node.op} node, not an input")
            name = node.attrs["name"]
            if name in names:
                raise GraphError(f"two inputs named {name!r}")
            names.add(name)

        weight_nodes: dict[str, Node] = {}
        position = self.ids
        for node in self.nodes:
            if node.op == "input" and node not in self.inputs:
                raise GraphError(f"input {node.attrs['name']!r} is used but not declared")
            if node.op == "weight":
                other = weight_nodes.setdefault(node.attrs["name"], node)
                if other is not node:
                    raise GraphError(f"two weight nodes named {node.attrs['name']!r}")
            for operand in node.inputs:
                if position[operand] >= position[node]:
                    raise GraphError(f"node %{position[node]} uses a later node")
            dtype, shape = OPS[node.op].infer(node.inputs, dict(node.attrs))
            if dtype != node.dtype or not shapes_equal(canonical_shape(shape), node.shape):
                raise GraphError(
                    f"node %{position[node]} ({node.op}) is typed {node.type_str} "
                    f"but its operands give {dtype}{format_shape(shape)}"
                )

        bindable = set(self.symbols)
        for node in self.nodes:
            loose = (symbols_of(node.shape) | _attr_symbols(node)) - bindable
            if loose:
                raise GraphError(
                    f"node %{position[node]} ({node.op}) uses {', '.join(sorted(loose))}, "
                    "which no input shape binds"
                )

    def __str__(self) -> str:
        from nanocompile.printer import format_graph

        return format_graph(self)


def _toposort(roots: Sequence[Node]) -> list[Node]:
    """Post-order DFS, iterative: 24 layers make chains deeper than Python's stack.

    Nodes are immutable and can only be built from nodes that already exist,
    so the graph cannot contain a cycle and none is checked for.
    """
    order: list[Node] = []
    done: set[Node] = set()
    for root in roots:
        if root in done:
            continue
        stack: list[tuple[Node, int]] = [(root, 0)]
        while stack:
            node, next_operand = stack[-1]
            if next_operand < len(node.inputs):
                stack[-1] = (node, next_operand + 1)
                child = node.inputs[next_operand]
                if child not in done:
                    stack.append((child, 0))
            else:
                stack.pop()
                if node not in done:
                    done.add(node)
                    order.append(node)
    return order
