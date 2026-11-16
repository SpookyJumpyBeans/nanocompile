"""Phase 6: per-row int8 weights, and the graph pass that puts them in.

nanoinfer quantizes each projection weight to symmetric per-row int8: one
float32 scale per output row, ``max(|row|) / 127``, and the row divided by it
and rounded half away from zero. :func:`quantize_int8` does exactly that,
with the same NumPy operations in the same order, so the integers and scales
here are nanoinfer's, byte for byte. The tests compare them.

:func:`quantize` rewrites a graph to read those weights. Every
``x @ W.T`` whose ``W`` is a 2-D weight read only that way becomes one of the
int8 matmuls (``ir.py``), reading ``W@int8`` and ``W@scales`` in place of
``W``. Two modes, nanoinfer's two kernels:

* ``activations="f32"``: ``matmul_i8``. The activations stay float32 and
  each int8 weight is widened to float in the kernel. nanoinfer's default,
  and what its int8 perplexity (23.3078) and token tests measure.
* ``activations="int8"``: ``matmul_i8_vnni``. The activations are quantized
  too, one scale per ``block`` values, so the dot product is bytes times
  bytes into int32, the operation VPDPBUSD does. nanoinfer's opt-in fast
  path. The activation quantization is not a primitive: it is written here
  in the graph's own operations (a max, a divide, ``round``, a clamp, two
  sums), the same arithmetic as nanoinfer's ``quantize_activations_blocked``.

Weights a ``keep`` list names stay float32, and so does any weight read some
other way. The tied embedding is both a gather table and the LM head, so it
stays float32 without being named; that matches nanoinfer, whose 23.3078 is
with the embeddings unquantized, and whose LM head never quantizes its
activations because their error lands straight on the logits.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from nanocompile import dtypes
from nanocompile.frontend import GraphBuilder, Tensor, where
from nanocompile.ir import Graph, Node, make
from nanocompile.passes import rewrite

INT8_MAX = 127
ACTIVATIONS = ("f32", "int8")
DEFAULT_BLOCK = 32          # nanoinfer's _VNNI_BLOCK, and llama.cpp's Q8_0


def int8_name(name: str) -> str:
    return f"{name}@int8"


def scales_name(name: str) -> str:
    return f"{name}@scales"


def quantize_int8(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``[out, in]`` float32 to per-row int8 values and float32 scales.

    nanoinfer's ``quantization.quantize_int8``, operation for operation: the
    scale is computed in float32, an all-zero row gets scale 1, and rounding
    is ``sign(x) * floor(|x| + 0.5)`` in float32, not NumPy's half-to-even.
    """
    if weights.ndim != 2:
        raise ValueError(f"expected a 2-D weight matrix, got shape {weights.shape}")
    weights = weights.astype(np.float32, copy=False)
    scales = np.max(np.abs(weights), axis=1) / INT8_MAX
    scales[scales == 0] = 1.0
    scaled = weights / scales[:, None]
    rounded = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
    values = np.clip(rounded, -INT8_MAX, INT8_MAX).astype(np.int8)
    return values, scales.astype(np.float32)


def _projection_weight(matmul: Node) -> Node | None:
    """``W`` if ``matmul`` is ``x @ permute(W, (1, 0))`` for a 2-D f32 weight ``W``."""
    if matmul.op != "matmul" or matmul.rank != 2:
        return None
    t = matmul.inputs[1]
    if t.op != "permute" or tuple(t.attrs["perm"]) != (1, 0):
        return None
    w = t.inputs[0]
    if w.op != "weight" or w.rank != 2 or w.dtype != dtypes.f32:
        return None
    return w


def projections(graph: Graph) -> list[str]:
    """Weights read only as ``x @ W.T``: the ones :func:`quantize` can rewrite."""
    users: dict[Node, list[Node]] = {}
    for node in graph.nodes:
        for x in node.inputs:
            users.setdefault(x, []).append(node)
    found = []
    for name, w in graph.weights.items():
        transposes = users.get(w, [])
        if not transposes or any(t.op != "permute" or tuple(t.attrs["perm"]) != (1, 0) for t in transposes):
            continue
        if all(m.op == "matmul" and _projection_weight(m) is w for t in transposes for m in users.get(t, [])):
            found.append(name)
    return found


def quantize_activations(x: Node, block: int) -> tuple[Node, Node, Node]:
    """``x: f32[..., K]`` to ``(xq, xs, offset)``, block by block, as nanoinfer does it.

    ``scale = max(|x| in block) / 127`` (1 for an all-zero block),
    ``xq = clamp(round(x / scale), -127, 127)`` as int8, and
    ``offset = 128 * sum_b(sum(xq in block b) * scale_b)``, the correction
    ``matmul_i8_vnni`` subtracts, summed over blocks in order.
    """
    k = x.shape[-1]
    if not isinstance(k, int) or k % block:
        raise ValueError(f"width {k} is not a whole number of blocks of {block}")
    b = GraphBuilder("quantize")
    lead, blocks = x.shape[:-1], k // block
    groups = Tensor(x, b).reshape(lead + (blocks, block))
    magnitude = where(0.0 <= groups, groups, -groups).max(-1)
    scale = where(magnitude <= 0.0, 1.0, magnitude / float(INT8_MAX))
    rounded = (groups / scale).round()
    clamped = where(rounded <= float(INT8_MAX), rounded, float(INT8_MAX))
    clamped = where(float(-INT8_MAX) <= clamped, clamped, float(-INT8_MAX))
    values = clamped.cast(dtypes.i8)
    block_sums = values.cast(dtypes.i32).sum(-1)
    offset = (block_sums.cast(dtypes.f32) * scale).reshape(lead + (blocks,)).sum(-1) * 128.0
    return values.reshape(lead + (k,)).node, scale.reshape(lead + (blocks,)).node, offset.node


def quantize(graph: Graph, activations: str = "f32", block: int = DEFAULT_BLOCK,
             keep: Iterable[str] = ()) -> Graph:
    """``graph`` with its projections reading per-row int8 weights.

    ``activations`` picks the kernel (see the module docstring); ``block`` is
    the activation block for ``"int8"``. Weights named in ``keep`` stay
    float32.
    """
    if activations not in ACTIVATIONS:
        raise ValueError(f"activations must be one of {ACTIVATIONS}, got {activations!r}")
    chosen = set(projections(graph)) - set(keep)
    weights: dict[str, tuple[Node, Node]] = {}
    quantized: dict[Node, tuple[Node, Node, Node]] = {}

    def int8_weight(w: Node) -> tuple[Node, Node]:
        name = w.attrs["name"]
        if name not in weights:
            n, k = w.shape
            weights[name] = (make("weight", name=int8_name(name), dtype=dtypes.i8, shape=(n, k)),
                             make("weight", name=scales_name(name), dtype=dtypes.f32, shape=(n,)))
        return weights[name]

    def visit(node: Node, inputs: tuple[Node, ...]) -> Node:
        w = _projection_weight(node)
        if w is None or w.attrs["name"] not in chosen:
            if inputs == node.inputs:
                return node
            return make(node.op, *inputs, **dict(node.attrs))
        q, scales = int8_weight(w)
        x = inputs[0]
        if activations == "f32":
            return make("matmul_i8", x, q, scales)
        if x not in quantized:
            quantized[x] = quantize_activations(x, block)
        xq, xs, offset = quantized[x]
        return make("matmul_i8_vnni", xq, xs, offset, q, scales, block=block)

    return rewrite(graph, visit)


def quantized_weights(graph: Graph, weights: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The arrays ``graph`` reads, quantizing from ``weights`` the int8 ones it names."""
    out: dict[str, np.ndarray] = {}
    for name in graph.weights:
        if name in weights:
            out[name] = weights[name]
            continue
        if name.endswith("@int8") or name.endswith("@scales"):
            source = name.rsplit("@", 1)[0]
            if int8_name(source) not in out:
                out[int8_name(source)], out[scales_name(source)] = quantize_int8(weights[source])
            continue
        raise KeyError(f"no weight {name!r}")
    return out
