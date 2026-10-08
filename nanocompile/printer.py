"""The graph as text: what ``print(graph)`` shows.

One line per node, operands by number, the type on the right:

    %7 = mul %5, %6                     : f32[seq, 16]
    %8 = reduce_sum %7 axis=-1          : f32[seq, 1]

The format is for reading and for golden tests, not for parsing back. It is
deterministic, because the node order is, so a diff between two dumps is a
diff between two graphs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from nanocompile.dtypes import DType
from nanocompile.symbolic import Expr, format_shape

if TYPE_CHECKING:
    from nanocompile.ir import Graph, Node

_TYPE_COLUMN = 40


def _format_attr(value: Any) -> str:
    if isinstance(value, DType):
        return str(value)
    if isinstance(value, Expr):
        return f"({value})" if " " in str(value) else str(value)
    if isinstance(value, tuple):
        if all(isinstance(v, (int, Expr)) and not isinstance(v, bool) for v in value):
            return format_shape(value)
        return "(" + ", ".join(_format_attr(v) for v in value) + ")"
    return str(value)


def _format_const(value: np.ndarray) -> str:
    # str() of a NumPy scalar is its shortest round-trip form in its own
    # precision: an f32 epsilon prints as 1e-06, not as the float64 widening.
    if value.ndim == 0:
        return str(value[()])
    return f"<{value.size} values>"


def format_node(node: "Node", ids: dict["Node", int]) -> str:
    parts = [node.op]
    attrs = dict(node.attrs)

    if node.op in ("input", "weight"):
        parts.append(f'"{attrs["name"]}"')
        attrs = {}
    elif node.op == "const":
        parts.append(_format_const(attrs.pop("value")))
    elif node.op == "dim":
        parts.append(_format_attr(attrs.pop("expr")))

    if node.inputs:
        parts.append(", ".join(f"%{ids[x]}" for x in node.inputs))
    # dtype and shape are already in the type column unless they are the point.
    if node.op not in ("cast", "iota"):
        attrs.pop("dtype", None)
    if node.op == "iota":
        attrs.pop("shape", None)
        attrs.pop("dtype", None)
    for key, value in attrs.items():
        # A permutation is a tuple of ints too, but it is not a shape.
        text = "(" + ", ".join(map(str, value)) + ")" if key == "perm" else _format_attr(value)
        parts.append(f"{key}={text}")

    body = f"%{ids[node]} = " + " ".join(parts)
    return f"{body:<{_TYPE_COLUMN}} : {node.type_str}"


def format_graph(graph: "Graph") -> str:
    signature = ", ".join(f"{n.attrs['name']}: {n.type_str}" for n in graph.inputs)
    lines = [f"graph {graph.name}({signature}) {{"]
    for node in graph.nodes:
        lines.append("  " + format_node(node, graph.ids))
    outputs = ", ".join(f"{name}=%{graph.ids[node]}" for name, node in graph.outputs.items())
    lines.append(f"  return {outputs}")
    lines.append("}")
    return "\n".join(lines)
