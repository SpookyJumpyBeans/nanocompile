"""Graph-to-graph passes: common subexpressions, constant folding, dead code.

Each pass returns a new graph and leaves the old one untouched, and each is
**bitwise**: it never changes what a node computes, only how many nodes
compute it. CSE merges two nodes that apply the same primitive to the same
operands; folding evaluates a node whose operands are all constants with the
interpreter's own kernels, so the constant it produces is the value the node
would have had.

Dead code elimination is not a separate pass. A ``Graph`` is built by walking
back from its outputs, so a rewrite that orphans a node drops it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from nanocompile import interpreter
from nanocompile.ir import Graph, Node, make
from nanocompile.symbolic import symbols_of

# Folding stops here: a constant this large is better computed than embedded.
FOLD_LIMIT = 1 << 16


def rewrite(graph: Graph, fn: Callable[[Node, tuple[Node, ...]], Node]) -> Graph:
    """Rebuild ``graph`` bottom-up: ``fn(old_node, new_operands)`` gives the new node.

    Inputs keep their identity, so the rewritten graph has the same signature.
    """
    mapping: dict[Node, Node] = {}
    for node in graph.nodes:
        if node.op == "input":
            mapping[node] = node
            continue
        mapping[node] = fn(node, tuple(mapping[x] for x in node.inputs))
    return Graph(graph.inputs, {k: mapping[v] for k, v in graph.outputs.items()}, name=graph.name)


def _same(node: Node, inputs: tuple[Node, ...]) -> Node:
    if inputs == node.inputs:
        return node
    return make(node.op, *inputs, **dict(node.attrs))


def _attr_key(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, tuple):
        return tuple(_attr_key(v) for v in value)
    return value


def eliminate_common_subexpressions(graph: Graph) -> Graph:
    """One node per distinct (primitive, operands, attributes).

    Merges what the frontend repeats: every ``x + eps`` builds its own scalar
    constant and its own reshape and broadcast of it.
    """
    seen: dict[tuple, Node] = {}

    def visit(node: Node, inputs: tuple[Node, ...]) -> Node:
        attrs = tuple(sorted((k, _attr_key(v)) for k, v in node.attrs.items()))
        key = (node.op, tuple(id(x) for x in inputs), attrs)
        existing = seen.get(key)
        if existing is not None:
            return existing
        new = _same(node, inputs)
        seen[key] = new
        return new

    return rewrite(graph, visit)


def fold_constants(graph: Graph) -> Graph:
    """Replace a node whose operands are all constants by the constant it computes.

    Only when the result's shape has no symbols in it, and only up to
    ``FOLD_LIMIT`` elements.
    """

    def visit(node: Node, inputs: tuple[Node, ...]) -> Node:
        if node.op in ("input", "weight", "const", "dim") or not inputs:
            return _same(node, inputs)
        if not all(x.op == "const" for x in inputs) or symbols_of(node.shape):
            return _same(node, inputs)
        if int(np.prod(node.shape, dtype=np.int64)) > FOLD_LIMIT:
            return _same(node, inputs)
        kernel = interpreter.KERNELS[node.op]
        value = np.asarray(kernel(node, [x.attrs["value"] for x in inputs], {}))
        value = np.array(value, dtype=node.dtype.numpy, copy=True)
        value.flags.writeable = False
        return make("const", value=value)

    return rewrite(graph, visit)


def optimize(graph: Graph) -> Graph:
    """The pass pipeline: CSE, fold, then CSE again over what folding exposed."""
    return eliminate_common_subexpressions(fold_constants(eliminate_common_subexpressions(graph)))
