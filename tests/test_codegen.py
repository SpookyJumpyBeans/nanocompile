"""The phase 2 gate: generated C against the interpreter, primitive by primitive.

Every graph here is run twice, by the interpreter and by compiled C, with the
symbol bound to several sizes including 1. Most primitives must agree
bitwise: IEEE add, multiply, divide, square root and reciprocal are exactly
rounded in both, and the compile flags forbid FMA contraction and
reassociation. Three kinds are compared with a tolerance, and each for a
stated reason:

* ``exp``, ``sin``, ``cos``: NumPy and the C library implement them
  differently, and neither is required to round correctly.
* ``reduce_sum`` and ``matmul``: NumPy sums pairwise, or hands the matmul to
  BLAS; the generated loop adds in index order.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanocompile import codegen_c, interpreter
from nanocompile.dtypes import bool_, f32, i8, i32, i64
from nanocompile.frontend import GraphBuilder, concat, gather, where
from nanocompile.ir import OPS
from nanocompile.lower import View, lower
from nanocompile.runtime import CompiledGraph, KernelError, plan_memory

SIZES = [1, 3, 7]
APPROX = dict(rtol=2e-6, atol=1e-6)


def rand(rng, *shape, dtype=np.float32):
    if np.dtype(dtype).kind == "f":
        return rng.standard_normal(shape).astype(dtype)
    if np.dtype(dtype).kind == "b":
        return rng.random(shape) > 0.5
    return rng.integers(-100, 100, size=shape).astype(dtype)


def both(graph, inputs, weights=None):
    """The interpreter's outputs, and the fused C's, after checking unfused C.

    Fused and unfused code must agree bitwise (phase 3's gate: fusion moves
    operations between kernels and never reorders them). Each is also held to
    the interpreter by the caller.
    """
    weights = weights or {}
    fused = CompiledGraph(graph)(inputs, weights)
    unfused = CompiledGraph(graph, fuse=False)(inputs, weights)
    for name in fused:
        np.testing.assert_array_equal(fused[name], unfused[name], err_msg=f"{name}: fused != unfused")
    return interpreter.run(graph, inputs, weights), fused


def assert_same(expected, actual, exact=True):
    assert expected.keys() == actual.keys()
    for name in expected:
        assert actual[name].dtype == expected[name].dtype, name
        assert actual[name].shape == expected[name].shape, name
        if exact:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
        else:
            np.testing.assert_allclose(actual[name], expected[name], err_msg=name, **APPROX)


# -- every primitive -----------------------------------------------------------

EXACT_UNARY = {
    "neg": lambda x: -x,
    "sqrt": lambda x: x.sqrt(),
    "reciprocal": lambda x: x.reciprocal(),
}
APPROX_UNARY = {
    "exp": lambda x: x.exp(),
    "sin": lambda x: x.sin(),
    "cos": lambda x: x.cos(),
}


@pytest.mark.parametrize("op", sorted(EXACT_UNARY) + sorted(APPROX_UNARY))
@pytest.mark.parametrize("seq", SIZES)
def test_unary(op, seq):
    fn = {**EXACT_UNARY, **APPROX_UNARY}[op]
    b = GraphBuilder()
    x = b.input("x", f32, (b.symbol("seq"), 5))
    graph = b.build({"y": fn(x)})
    xv = rand(np.random.default_rng(seq), seq, 5) * 4
    if op == "sqrt":
        xv = np.abs(xv)
    assert_same(*both(graph, {"x": xv}), exact=op in EXACT_UNARY)


@pytest.mark.parametrize("op", ["add", "sub", "mul", "div", "less_equal"])
@pytest.mark.parametrize("dtype", [f32, i64, i32, i8])
@pytest.mark.parametrize("seq", SIZES)
def test_binary(op, dtype, seq):
    if op == "div" and not dtype.is_float:
        pytest.skip("div is float-only")
    build = {
        "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b, "less_equal": lambda a, b: a <= b,
    }[op]
    b = GraphBuilder()
    s = b.symbol("seq")
    x, y = b.input("x", dtype, (s, 6)), b.input("y", dtype, (s, 6))
    graph = b.build({"z": build(x, y)})
    rng = np.random.default_rng(seq)
    inputs = {"x": rand(rng, seq, 6, dtype=dtype.numpy), "y": rand(rng, seq, 6, dtype=dtype.numpy)}
    assert_same(*both(graph, inputs))


def test_int8_wraps_like_numpy():
    b = GraphBuilder()
    x = b.input("x", i8, (4,))
    graph = b.build({"y": x + x, "z": x * 3, "n": -x})
    assert_same(*both(graph, {"x": np.array([100, -100, -128, 7], np.int8)}))


def test_where_and_cast():
    b = GraphBuilder()
    s = b.symbol("seq")
    c, x = b.input("c", bool_, (s, 4)), b.input("x", f32, (s, 4))
    n = b.input("n", i64, (s,))
    graph = b.build({
        "w": where(c, x, -x),
        "f": n.cast(f32),
        "q": n.cast(i8),
        "back": x.cast(i32),
        "flag": n.cast(bool_),
        "from_bool": c.cast(f32),
    })
    rng = np.random.default_rng(0)
    inputs = {
        "c": rand(rng, 5, 4, dtype=np.bool_),
        "x": rand(rng, 5, 4) * 50,
        "n": np.array([0, 1, 300, -3, 2**24 + 1], np.int64),
    }
    assert_same(*both(graph, inputs))


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("seq", SIZES)
def test_reductions(axis, seq):
    b = GraphBuilder()
    x = b.input("x", f32, (b.symbol("seq"), 33))
    graph = b.build({"s": x.sum(axis), "m": x.max(axis)})
    xv = rand(np.random.default_rng(seq), seq, 33)
    expected, actual = both(graph, {"x": xv})
    np.testing.assert_array_equal(actual["m"], expected["m"])
    np.testing.assert_allclose(actual["s"], expected["s"], **APPROX)


def test_max_propagates_nan_and_handles_negative_infinity():
    b = GraphBuilder()
    x = b.input("x", f32, (3, 4))
    graph = b.build({"m": x.max(-1)})
    xv = np.array([[1, np.nan, 3, 2], [-np.inf] * 4, [-5, -1, -7, -2]], np.float32)
    assert_same(*both(graph, {"x": xv}))


def test_integer_sum_wraps():
    b = GraphBuilder()
    x = b.input("x", i8, (4,))
    graph = b.build({"s": x.sum(0)})
    assert_same(*both(graph, {"x": np.full(4, 100, np.int8)}))


@pytest.mark.parametrize("seq", SIZES)
def test_movement_views(seq):
    """Every movement primitive, alone and stacked, read by a kernel."""
    b = GraphBuilder()
    s = b.symbol("seq")
    x = b.input("x", f32, (s, 4, 6))
    graph = b.build({
        "reshape": x.reshape(s, 24) * 1.0,
        "permute": x.permute(2, 0, 1) * 1.0,
        "broadcast": x.slice(1, 0, 1).broadcast_to((s, 5, 6)) * 1.0,
        "slice": x.slice(-1, 2, 5) * 1.0,
        "last": x.slice(0, s - 1, s) * 1.0,
        "merge_after_permute": x.permute(1, 0, 2).reshape(4, s * 6) * 1.0,
        "permute_then_slice": x.permute(2, 1, 0).slice(0, 1, 4) * 1.0,
        "concat": concat([x, -x, x.slice(1, 0, 1)], axis=1),
        "output_is_a_view": x.permute(1, 2, 0),
        "output_is_an_input": x,
    })
    assert_same(*both(graph, {"x": rand(np.random.default_rng(seq), seq, 4, 6)}))


@pytest.mark.parametrize("seq", SIZES)
def test_matmul(seq):
    b = GraphBuilder()
    s = b.symbol("seq")
    x, w = b.input("x", f32, (s, 40)), b.input("w", f32, (7, 40))
    q, k = b.input("q", f32, (3, s, 16)), b.input("k", f32, (3, s, 16))
    graph = b.build({"linear": x @ w.swapaxes(0, 1), "scores": q @ k.swapaxes(-1, -2)})
    rng = np.random.default_rng(seq)
    inputs = {"x": rand(rng, seq, 40), "w": rand(rng, 7, 40), "q": rand(rng, 3, seq, 16), "k": rand(rng, 3, seq, 16)}
    assert_same(*both(graph, inputs), exact=False)


def test_gather_and_its_bounds():
    b = GraphBuilder()
    table, ids = b.input("t", f32, (10, 3)), b.input("i", i64, (b.symbol("seq"),))
    graph = b.build({"y": gather(table, ids)})
    tv = rand(np.random.default_rng(0), 10, 3)
    assert_same(*both(graph, {"t": tv, "i": np.array([9, 0, 0, 4])}))
    compiled = CompiledGraph(graph)
    for bad in (-1, 10):
        with pytest.raises(KernelError, match=f"index {bad}"):
            compiled({"t": tv, "i": np.array([0, bad])}, {})


def test_iota_dim_and_an_empty_operand():
    b = GraphBuilder()
    seq, past = b.symbol("seq"), b.symbol("past")
    cache = b.input("cache", f32, (2, past, 4))
    x = b.input("x", f32, (2, seq, 4))
    graph = b.build({
        "positions": b.iota((seq,), axis=0) + b.dim(past),
        "cols": b.iota((seq, past + seq), axis=1),
        "total": b.dim(past + seq),
        "joined": concat([cache, x], axis=1),
    })
    rng = np.random.default_rng(0)
    for past_len in (0, 5):
        assert_same(*both(graph, {"cache": rand(rng, 2, past_len, 4), "x": rand(rng, 2, 3, 4)}))


def test_every_primitive_is_tested_here():
    covered = {
        "input", "weight", "const", "iota", "dim", "add", "sub", "mul", "div",
        "less_equal", "where", "cast", "reduce_sum", "reduce_max", "reshape",
        "permute", "broadcast", "slice", "concat", "matmul", "gather",
    } | set(EXACT_UNARY) | set(APPROX_UNARY)
    assert covered == set(OPS)


# -- what lowering promises ----------------------------------------------------


def primitives(graph):
    return [call.primitive for call in lower(graph).calls]


def test_unary_minus_binds_its_whole_operand():
    """-(a + b) must not print as -a + b; fusion is what first inlined a sum there."""
    from nanocompile import loopir as L

    total = L.binary("add", L.Var("a", f32), L.Var("b", f32))
    assert codegen_c.render_scalar(L.Unary("neg", total, f32)) == "(-(a + b))"
    assert codegen_c.render_scalar(L.Unary("reciprocal", total, f32)) == "(1.0f / (a + b))"


def test_transposed_weight_is_read_in_place():
    """``x @ W.T`` must not copy the weight: that would be 2 GB per token."""
    b = GraphBuilder()
    x = b.input("x", f32, (b.symbol("seq"), 8))
    graph = b.build({"y": x @ b.weight("w", (5, 8)).swapaxes(0, 1) + b.weight("b", (5,))})
    assert primitives(graph) == ["matmul", "add"]


def test_reshape_of_a_transposed_view_copies():
    b = GraphBuilder()
    x = b.input("x", f32, (3, b.symbol("seq"), 4))
    graph = b.build({"y": x.permute(1, 0, 2).reshape(b.symbol("seq"), 12) * 2.0})
    assert primitives(graph) == ["copy", "mul"]


def test_identical_kernels_are_emitted_once():
    b = GraphBuilder()
    x = b.input("x", f32, (b.symbol("seq"), 8))
    for _ in range(10):
        x = x * x
    program = lower(b.build({"y": x}))
    assert len(program.calls) == 10
    assert len(program.kernels) == 1


def test_bytes_moved_is_exact():
    b = GraphBuilder()
    seq = b.symbol("seq")
    x = b.input("x", f32, (seq, 8))
    graph = b.build({"y": x + b.weight("bias", (8,))})
    # read x (seq*8 floats) and the bias once (8 floats), write seq*8 floats.
    assert lower(graph).bytes_moved == seq * 8 * 4 * 2 + 8 * 4


# -- determinism and the cache -------------------------------------------------


def build_mlp():
    b = GraphBuilder("mlp")
    x = b.input("x", f32, (b.symbol("seq"), 16))
    h = x @ b.weight("w1", (32, 16)).swapaxes(0, 1)
    h = h * (h >= 0.0).cast(f32)
    return b.build({"y": (h @ b.weight("w2", (16, 32)).swapaxes(0, 1)).sum(-1)})


def test_compiling_twice_emits_identical_c():
    first = codegen_c.render_library(lower(build_mlp()).kernels)
    second = codegen_c.render_library(lower(build_mlp()).kernels)
    assert first == second


def test_second_compile_comes_from_the_cache():
    CompiledGraph(build_mlp())
    again = CompiledGraph(build_mlp())
    assert again.cached


# -- the memory plan -----------------------------------------------------------


def test_live_buffers_never_overlap(tiny_model_dir):
    from nanocompile.models.qwen2 import Qwen2Config, build_qwen2

    program = lower(build_qwen2(Qwen2Config.from_model_dir(tiny_model_dir)))
    last_use = program.last_uses()
    for bound in ({"seq": 1, "past": 9}, {"seq": 7, "past": 0}):
        plan = plan_memory(program, last_use, bound)
        first_use = {}
        for index, call in enumerate(program.calls):
            first_use.setdefault(call.output, index)
        spans = []
        for buf, offset in plan.offsets.items():
            elements = int(np.prod([d if isinstance(d, int) else d.evaluate(bound) for d in buf.shape]))
            size = elements * buf.dtype.numpy.itemsize
            spans.append((first_use[buf], last_use[buf], offset, offset + max(size, 1)))
        for i, (s1, e1, a1, b1) in enumerate(spans):
            for s2, e2, a2, b2 in spans[i + 1:]:
                live_together = s1 <= e2 and s2 <= e1
                assert not (live_together and a1 < b2 and a2 < b1)
        assert plan.arena_bytes <= plan.temp_bytes
