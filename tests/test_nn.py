"""The composite building blocks, against nanoinfer's NumPy versions.

These are bitwise comparisons. ``nn.py`` performs the same float32 operations
in the same order as nanoinfer, so the interpreter calls the same NumPy
functions on the same values, and any difference at all means the composition
is not the one nanoinfer computes. A tolerance here would hide exactly the
reordering that this phase promises not to do.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanocompile import interpreter, nn
from nanocompile.dtypes import f32, i64
from nanocompile.frontend import GraphBuilder


@pytest.fixture(scope="module")
def ref(nanoinfer_root):
    import nanoinfer.ops
    import nanoinfer.rope

    return nanoinfer


def run_unary(fn, x: np.ndarray, **weights) -> np.ndarray:
    b = GraphBuilder()
    t = b.input("x", f32, (b.symbol("seq"),) + x.shape[1:])
    ws = {name: b.weight(name, value.shape) for name, value in weights.items()}
    graph = b.build({"y": fn(t, **ws)})
    return interpreter.run(graph, {"x": x}, weights)["y"]


@pytest.fixture
def x() -> np.ndarray:
    # Wide enough range to reach both branches of sigmoid and to overflow a
    # naive exp, with exact zeros to exercise the x >= 0 boundary.
    values = np.random.default_rng(0).standard_normal((5, 32)).astype(np.float32) * 30
    values[0, :4] = 0.0
    return values


def test_rms_norm(ref, x):
    weight = np.random.default_rng(1).standard_normal(32).astype(np.float32)
    out = run_unary(lambda t, w: nn.rms_norm(t, w, 1e-6), x, w=weight)
    np.testing.assert_array_equal(out, ref.ops.rms_norm(x, weight, 1e-6))


def test_sigmoid_and_silu(ref, x):
    with np.errstate(over="raise"):
        np.testing.assert_array_equal(run_unary(nn.sigmoid, x), ref.ops.sigmoid(x))
        np.testing.assert_array_equal(run_unary(nn.silu, x), ref.ops.silu(x))


def test_softmax(ref, x):
    np.testing.assert_array_equal(run_unary(nn.softmax, x), ref.ops.softmax(x, axis=-1))


def test_inverse_frequencies(ref):
    np.testing.assert_array_equal(
        nn.inverse_frequencies(64, 1_000_000.0), ref.rope.inverse_frequencies(64, 1_000_000.0)
    )


@pytest.mark.parametrize("past", [0, 1, 13])
def test_rope(ref, past):
    head_dim, theta, heads, seq = 64, 1_000_000.0, 3, 4
    q = np.random.default_rng(past).standard_normal((heads, seq, head_dim)).astype(np.float32)

    b = GraphBuilder()
    s, p = b.symbol("seq"), b.symbol("past")
    b.input("cache", f32, (p,))
    t = b.input("q", f32, (heads, s, head_dim))
    positions = b.iota((s,), axis=0) + b.dim(p)
    cos, sin = nn.rope_tables(positions, head_dim, theta)
    graph = b.build({"y": nn.apply_rope(t, cos, sin)})
    out = interpreter.run(graph, {"q": q, "cache": np.zeros(past, np.float32)}, {})["y"]

    rope = ref.rope.RotaryEmbedding(head_dim, theta)
    np.testing.assert_array_equal(out, rope.apply(q, np.arange(past, past + seq)))


def test_rope_is_half_split_not_interleaved(ref):
    x = np.arange(8, dtype=np.float32).reshape(1, 8)
    out = run_unary(nn.rotate_half, x)
    np.testing.assert_array_equal(out, ref.rope.rotate_half(x))
    assert not np.array_equal(out, ref.rope.rotate_interleaved(x))


def test_repeat_kv_is_consecutive(ref):
    x = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    b = GraphBuilder()
    t = b.input("x", f32, (2, b.symbol("seq"), 4))
    graph = b.build({"y": nn.repeat_kv(t, 7)})
    out = interpreter.run(graph, {"x": x}, {})["y"]
    np.testing.assert_array_equal(out, ref.ops.repeat_kv(x, 7))
    assert not np.array_equal(out, np.tile(x, (7, 1, 1)))


@pytest.mark.parametrize("seq,past", [(5, 0), (1, 9), (3, 4)])
def test_causal_mask(ref, seq, past):
    b = GraphBuilder()
    s, p = b.symbol("seq"), b.symbol("past")
    b.input("cache", f32, (p,))
    b.input("ids", i64, (s,))
    graph = b.build({"mask": nn.causal_mask(b, s, p)})
    out = interpreter.run(
        graph, {"cache": np.zeros(past, np.float32), "ids": np.zeros(seq, np.int64)}, {}
    )["mask"]
    np.testing.assert_array_equal(out, ref.ops.causal_mask(seq, past))


def test_linear_with_bias(ref):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    w = rng.standard_normal((5, 8)).astype(np.float32)
    bias = rng.standard_normal(5).astype(np.float32)
    out = run_unary(lambda t, w, bias: nn.linear(t, w, bias), x, w=w, bias=bias)
    np.testing.assert_array_equal(out, x @ w.T + bias)
