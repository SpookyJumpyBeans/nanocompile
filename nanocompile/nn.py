"""The transformer's building blocks, as compositions of primitives.

Each function here mirrors one in nanoinfer and does the same float32
arithmetic in the same order, because the phase 1 gate is agreement with
nanoinfer and the cheapest way to agree is not to differ. Where nanoinfer
guards an input the model never produces, the difference is stated rather
than reproduced.

None of these is a primitive. ``rms_norm`` becomes a multiply, a reduction, a
divide, an add, a square root, a reciprocal and two more multiplies, each its
own node and, in phase 2, its own pass over memory. That is the cost phase 3
is built to remove, and it is only visible because it is spelled out here.
"""

from __future__ import annotations

import numpy as np

from nanocompile import dtypes
from nanocompile.frontend import GraphBuilder, Tensor, concat, where
from nanocompile.symbolic import Dim


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """``x @ W.T (+ b)`` with ``W`` stored as ``[out_features, in_features]``.

    The transpose is a ``permute`` node, not a copied weight: the interpreter
    hands NumPy a transposed view, the same single sgemm nanoinfer makes.
    """
    y = x @ weight.swapaxes(-1, -2)
    return y if bias is None else y + bias


def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """``x / sqrt(mean(x^2) + eps) * weight``, epsilon inside the root."""
    n = x.shape[-1]
    mean_square = (x * x).sum(-1) / float(n)
    return x * (mean_square + eps).sqrt().reciprocal() * weight


def sigmoid(x: Tensor) -> Tensor:
    """``1 / (1 + exp(-x))``, computed on the side that cannot overflow.

    nanoinfer's form: with ``e = exp(-|x|)`` it is ``1 / (1 + e)`` for x >= 0
    and ``e / (1 + e)`` below. ``-|x|`` is written as a select, since there is
    no ``abs`` primitive and one ``where`` costs the same pass over memory.
    """
    non_negative = x >= 0.0
    e = where(non_negative, -x, x).exp()
    return where(non_negative, 1.0, e) / (e + 1.0)


def silu(x: Tensor) -> Tensor:
    return x * sigmoid(x)


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Max-subtracted softmax.

    nanoinfer also guards a row that is entirely ``-inf``, returning zeros
    instead of NaN. Under a causal mask that row cannot occur (every position
    attends at least to itself), so the guard would be two dead nodes per
    layer and is left out.
    """
    shifted = (x - x.max(axis)).exp()
    return shifted / shifted.sum(axis)


def inverse_frequencies(head_dim: int, theta: float) -> np.ndarray:
    """RoPE's per-plane rotation rates, in float32 exactly as nanoinfer computes them.

    A build-time constant: it depends on the config alone.
    """
    exponent = np.arange(0, head_dim, 2, dtype=np.int64).astype(np.float32) / np.float32(head_dim)
    return (np.float32(1.0) / (np.float32(theta) ** exponent)).astype(np.float32)


def rope_tables(positions: Tensor, head_dim: int, theta: float) -> tuple[Tensor, Tensor]:
    """``(cos, sin)``, each ``[seq, head_dim]``, for the given positions.

    nanoinfer precomputes a table and indexes it by position; here the angles
    are computed in the graph from the positions. Each table entry is the same
    float32 product and the same ``cos``, so the values agree, and the graph
    needs no table sized for the longest sequence it might ever see.

    The angles are duplicated, not interleaved: entry i and entry i + d/2
    share an angle. That is the half-split convention Qwen2 uses.
    """
    builder = positions.builder
    inv_freq = builder.const(inverse_frequencies(head_dim, theta))
    seq = positions.shape[0]
    angles = positions.cast(dtypes.f32).reshape(seq, 1) * inv_freq.reshape(1, head_dim // 2)
    doubled = concat([angles, angles], axis=-1)
    return doubled.cos(), doubled.sin()


def rotate_half(x: Tensor) -> Tensor:
    """``[a | b] -> [-b | a]`` on the last axis: the half-split pairing."""
    d = x.shape[-1]
    half = d // 2
    return concat([-x.slice(-1, half, d), x.slice(-1, 0, half)], axis=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate ``x`` of shape ``[heads, seq, head_dim]`` by the tables."""
    return x * cos + rotate_half(x) * sin


def repeat_kv(x: Tensor, repeats: int) -> Tensor:
    """``[kv, seq, d] -> [kv * repeats, seq, d]``, each KV head repeated in place.

    Consecutive, not tiled: KV head 0 serves query heads 0..repeats-1. Done as
    reshape, broadcast, reshape, so the repeat is visible in the IR. Phase 3
    can fold it into the attention matmul's indexing instead of materializing
    seven copies of every key.
    """
    if repeats == 1:
        return x
    kv, seq, d = x.shape
    return x.reshape(kv, 1, seq, d).broadcast_to((kv, repeats, seq, d)).reshape(kv * repeats, seq, d)


def causal_mask(builder: GraphBuilder, seq: Dim, past: Dim) -> Tensor:
    """``[seq, past + seq]``: 0 where a new token may attend, ``-inf`` where not.

    New token ``i`` sits at absolute position ``past + i`` and may see every
    key up to and including its own position. With ``seq == 1`` (a decode
    step) nothing is masked. The boundary is ``<=``: ``<`` would forbid a token
    from attending to itself.
    """
    total = past + seq
    rows = builder.iota((seq, total), axis=0)
    cols = builder.iota((seq, total), axis=1)
    allowed = cols <= rows + builder.dim(past)
    return where(allowed, 0.0, builder.const(-np.inf, dtypes.f32))
