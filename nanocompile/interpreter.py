"""The reference interpreter: what a graph means, evaluated with NumPy.

This is the definition every later phase is held to. Phase 2's generated C,
phase 3's fused graph and phase 4's schedules are all tested by running the
same graph here and comparing. So it is written to be obviously right rather
than fast, and it is never optimized: an interpreter that took shortcuts would
be a second implementation to doubt rather than the one to check against.

It is also strict where NumPy is lenient. After every node, the result's dtype
and shape are checked against what shape inference promised. NumPy promotes
dtypes and broadcasts silently, and either would let a wrong graph produce a
right-looking answer; here it is an error naming the node.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from nanocompile.ir import Graph, Node
from nanocompile.symbolic import Expr, evaluate_shape, format_shape


class InterpreterError(RuntimeError):
    """Inputs that do not fit the graph, or a node that broke its own type."""


def bind_symbols(
    graph: Graph,
    inputs: Mapping[str, np.ndarray],
    bindings: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Read each symbol's value off the input arrays, and check they agree.

    ``past`` is bound by the third axis of ``past_keys``; if ``past_values``
    disagrees about it, that is an error here rather than a mismatched concat
    twenty layers in. Compound dimensions (``past + seq``) are checked after
    every plain symbol is bound.
    """
    bound: dict[str, int] = dict(bindings or {})
    for node in graph.inputs:
        array = inputs[node.attrs["name"]]
        for axis, d in enumerate(node.shape):
            name = d.as_symbol if isinstance(d, Expr) else None
            if name is None:
                continue
            size = array.shape[axis]
            if bound.setdefault(name, size) != size:
                raise InterpreterError(
                    f"{node.attrs['name']!r} axis {axis} has size {size}, "
                    f"but {name} is already bound to {bound[name]}"
                )
    for name, value in bound.items():
        if value < 0:
            raise InterpreterError(f"{name} is bound to a negative size {value}")
    return bound


def _check_inputs(graph: Graph, inputs: Mapping[str, np.ndarray], bound: Mapping[str, int]) -> None:
    for node in graph.inputs:
        name = node.attrs["name"]
        array = inputs[name]
        expected = evaluate_shape(node.shape, bound)
        if array.dtype != node.dtype.numpy:
            raise InterpreterError(f"input {name!r} is {array.dtype}, expected {node.dtype}")
        if array.shape != expected:
            raise InterpreterError(
                f"input {name!r} has shape {array.shape}, expected "
                f"{format_shape(node.shape)} = {expected}"
            )


def _check_weights(graph: Graph, weights: Mapping[str, np.ndarray]) -> None:
    missing = sorted(set(graph.weights) - weights.keys())
    if missing:
        raise InterpreterError(f"missing weights: {', '.join(missing[:5])}")
    for name, node in graph.weights.items():
        array = weights[name]
        if array.dtype != node.dtype.numpy or array.shape != node.shape:
            raise InterpreterError(
                f"weight {name!r} is {array.dtype}{list(array.shape)}, "
                f"expected {node.type_str}"
            )


def check_arguments(
    graph: Graph,
    inputs: Mapping[str, np.ndarray],
    weights: Mapping[str, np.ndarray],
    bindings: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Validate a call's inputs and weights, and return the symbol bindings.

    Shared with the generated-code runtime, so both backends accept and
    reject exactly the same arguments with the same messages.
    """
    unknown = sorted(set(inputs) - set(graph.input_names))
    missing = sorted(set(graph.input_names) - set(inputs))
    if unknown or missing:
        raise InterpreterError(f"inputs: missing {missing}, unexpected {unknown}")

    bound = bind_symbols(graph, inputs, bindings)
    _check_inputs(graph, inputs, bound)
    _check_weights(graph, weights)
    return bound


def _value(d, bound: Mapping[str, int]) -> int:
    return d if isinstance(d, int) else d.evaluate(bound)


# -- one function per primitive ---------------------------------------------

Kernel = Callable[[Node, list[np.ndarray], Mapping[str, int]], np.ndarray]


def _iota(node, args, bound):
    shape = evaluate_shape(node.shape, bound)
    axis = node.attrs["axis"] % len(shape)
    line = np.arange(shape[axis], dtype=node.dtype.numpy)
    view = [1] * len(shape)
    view[axis] = shape[axis]
    return np.broadcast_to(line.reshape(view), shape)


def _slice(node, args, bound):
    (x,) = args
    axis = node.attrs["axis"] % x.ndim
    start = _value(node.attrs["start"], bound)
    stop = _value(node.attrs["stop"], bound)
    if not 0 <= start <= stop <= x.shape[axis]:
        raise InterpreterError(
            f"slice [{start}:{stop}] is out of bounds for axis {axis} of size {x.shape[axis]}"
        )
    index = [slice(None)] * x.ndim
    index[axis] = slice(start, stop)
    return x[tuple(index)]


def _gather(node, args, bound):
    table, ids = args
    if ids.size and (ids.min() < 0 or ids.max() >= table.shape[0]):
        bad = ids[(ids < 0) | (ids >= table.shape[0])].flat[0]
        # NumPy would wrap a negative index and quietly return a real row.
        raise InterpreterError(f"gather index {bad} is outside 0..{table.shape[0] - 1}")
    return table[ids]


def _matmul(node, args, bound):
    a, b = args
    return np.matmul(a, b)


KERNELS: dict[str, Kernel] = {
    "const": lambda node, args, bound: node.attrs["value"],
    "iota": _iota,
    "dim": lambda node, args, bound: np.array(_value(node.attrs["expr"], bound), dtype=np.int64),
    "add": lambda node, args, bound: np.add(*args),
    "sub": lambda node, args, bound: np.subtract(*args),
    "mul": lambda node, args, bound: np.multiply(*args),
    "div": lambda node, args, bound: np.divide(*args),
    "less_equal": lambda node, args, bound: np.less_equal(*args),
    "where": lambda node, args, bound: np.where(*args),
    "neg": lambda node, args, bound: np.negative(args[0]),
    "exp": lambda node, args, bound: np.exp(args[0]),
    "sqrt": lambda node, args, bound: np.sqrt(args[0]),
    "reciprocal": lambda node, args, bound: np.reciprocal(args[0]),
    "sin": lambda node, args, bound: np.sin(args[0]),
    "cos": lambda node, args, bound: np.cos(args[0]),
    "cast": lambda node, args, bound: args[0].astype(node.dtype.numpy),
    # dtype pinned: NumPy would sum int8 into a wider type.
    "reduce_sum": lambda node, args, bound: np.sum(
        args[0], axis=node.attrs["axis"], keepdims=True, dtype=node.dtype.numpy
    ),
    "reduce_max": lambda node, args, bound: np.max(args[0], axis=node.attrs["axis"], keepdims=True),
    "reshape": lambda node, args, bound: np.reshape(args[0], evaluate_shape(node.shape, bound)),
    "permute": lambda node, args, bound: np.transpose(args[0], node.attrs["perm"]),
    "broadcast": lambda node, args, bound: np.broadcast_to(args[0], evaluate_shape(node.shape, bound)),
    "slice": _slice,
    "concat": lambda node, args, bound: np.concatenate(args, axis=node.attrs["axis"]),
    "matmul": _matmul,
    "gather": _gather,
}


def run(
    graph: Graph,
    inputs: Mapping[str, np.ndarray],
    weights: Mapping[str, np.ndarray],
    bindings: Mapping[str, int] | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate ``graph``, returning its outputs by name.

    Intermediates are dropped after their last use, so peak memory is the
    weights plus the live activations rather than every value ever computed.
    """
    bound = check_arguments(graph, inputs, weights, bindings)

    last_use: dict[Node, int] = {}
    for index, node in enumerate(graph.nodes):
        for operand in node.inputs:
            last_use[operand] = index
    keep = set(graph.outputs.values())

    values: dict[Node, np.ndarray] = {}
    for index, node in enumerate(graph.nodes):
        if node.op == "input":
            result = inputs[node.attrs["name"]]
        elif node.op == "weight":
            result = weights[node.attrs["name"]]
        else:
            result = KERNELS[node.op](node, [values[x] for x in node.inputs], bound)
            result = np.asarray(result)

        expected = evaluate_shape(node.shape, bound)
        if result.dtype != node.dtype.numpy or result.shape != expected:
            raise InterpreterError(
                f"node %{index} ({node.op}) produced {result.dtype}{list(result.shape)}, "
                f"but is typed {node.type_str} = {node.dtype}{list(expected)}"
            )
        values[node] = result

        for operand in set(node.inputs):
            if last_use[operand] == index and operand not in keep:
                del values[operand]

    # Contiguous, so callers can hand outputs to C. Not ascontiguousarray: it
    # promotes a 0-d scalar to shape (1,), which breaks the typed shape.
    return {name: np.asarray(values[node], order="C") for name, node in graph.outputs.items()}
