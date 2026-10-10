"""The phase 3 gate: what fusion removes, counted exactly, and nothing it changes.

Fused and unfused code are compared bitwise (in ``test_codegen.both`` for
every primitive, and here for the whole model with its cache). The structure
claims are asserted on the lowered programs: which patterns become one
kernel, how many kernels a transformer block costs, and how many bytes a
decode step moves.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nanocompile import codegen_c, interpreter, nn
from nanocompile.dtypes import f32
from nanocompile.frontend import GraphBuilder
from nanocompile.fusion import fuse
from nanocompile.lower import lower
from nanocompile.models.qwen2 import Qwen2, Qwen2Config, build_qwen2
from nanocompile.passes import optimize
from nanocompile.runtime import CBackend, CompiledGraph, plan_memory
from nanocompile.symbolic import Expr
from tests import oracle


def kinds(graph):
    return [call.primitive for call in fuse(optimize(graph)).calls]


def run_both(graph, inputs, weights):
    expected = interpreter.run(graph, inputs, weights)
    actual = CompiledGraph(graph)(inputs, weights)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=2e-6, atol=1e-6)


# -- patterns that become one kernel -------------------------------------------


def test_rms_norm_is_one_row_kernel():
    b = GraphBuilder()
    x = b.input("x", f32, (b.symbol("seq"), 16))
    graph = b.build({"y": nn.rms_norm(x, b.weight("w", (16,)), 1e-6)})
    assert kinds(graph) == ["row"]


def test_softmax_is_one_row_kernel_with_two_reductions():
    b = GraphBuilder()
    x = b.input("x", f32, (3, b.symbol("seq"), 16))
    graph = b.build({"y": nn.softmax(x)})
    assert kinds(graph) == ["row"]
    source = codegen_c.render_library(fuse(optimize(graph)).kernels)
    assert "acc1" in source and "acc2" not in source


def test_bias_and_residual_fold_into_the_matmul():
    b = GraphBuilder()
    s = b.symbol("seq")
    x, residual = b.input("x", f32, (s, 8)), b.input("r", f32, (s, 4))
    y = residual + nn.linear(x, b.weight("w", (4, 8)), b.weight("b", (4,)))
    assert kinds(b.build({"y": y})) == ["matmul+epilogue"]


def test_gate_and_up_share_one_k_loop():
    b = GraphBuilder()
    x = b.input("x", f32, (b.symbol("seq"), 8))
    h = nn.silu(nn.linear(x, b.weight("g", (6, 8)))) * nn.linear(x, b.weight("u", (6, 8)))
    graph = b.build({"h": h})
    assert kinds(graph) == ["matmul+epilogue"]
    (kernel,) = fuse(optimize(graph)).kernels
    loops = [s for s in kernel.body[0].body[0].body if s.__class__.__name__ == "Loop"]
    assert len(loops) == 1                          # one k loop, two accumulators
    assert len(loops[0].body) == 2


def test_a_reduction_read_off_its_row_is_refused_and_still_correct():
    """Row fusion needs each row's result read by that row. Transposed, it is not."""
    b = GraphBuilder()
    x = b.input("x", f32, (4, 4))
    m = x.max(-1).reshape(1, 4)
    graph = b.build({"y": x + m})
    assert "reduce" in kinds(graph)
    run_both(graph, {"x": np.random.default_rng(0).standard_normal((4, 4)).astype(np.float32)}, {})


# -- the model -------------------------------------------------------------------


@pytest.fixture(scope="module")
def config(tiny_model_dir: Path) -> Qwen2Config:
    return Qwen2Config.from_model_dir(tiny_model_dir)


def program(config, layers=None, last_only=False):
    if layers is not None:
        config = replace(config, num_hidden_layers=layers)
    return fuse(optimize(build_qwen2(config, last_only=last_only)))


def test_a_transformer_block_is_thirteen_kernels(config):
    """rms_norm, q, k, v, rope q, rope k, scores, softmax, context, o + residual,
    rms_norm, gate + up + silu, down + residual."""
    two, three = program(config, 2), program(config, 3)
    assert len(three.calls) - len(two.calls) == 13
    assert len(three.kernels) == len(two.kernels)


def test_the_kv_history_is_never_copied(config):
    calls = program(config).calls
    # The only copies left write the two cache outputs; repeat_kv and the
    # per-layer concat are index arithmetic inside the attention kernels.
    assert [c.primitive for c in calls].count("copy") == 2


def test_each_cached_token_is_read_exactly_once_per_step(config):
    """The ``past`` coefficient of bytes moved is the KV cache's own size per token."""
    kv_bytes_per_token = config.num_hidden_layers * 2 * config.num_key_value_heads * config.head_dim * 4
    bytes_moved = program(config, last_only=True).bytes_moved
    past_only = {m: c for m, c in bytes_moved.terms() if m == ("past",)}
    assert past_only == {("past",): kv_bytes_per_token}
    # Unfused, each step moved every cached token 3 + 2G times over (G query
    # heads per KV head): read and rewrite it in the concat, read it again to
    # make G copies, and read those copies in the matmul. 17x for the real
    # model's G = 7.
    unfused = {m: c for m, c in lower(build_qwen2(config, last_only=True)).bytes_moved.terms()}
    assert unfused[("past",)] == (3 + 2 * config.kv_group_size) * kv_bytes_per_token


def test_every_weight_is_still_read_once(config, tiny_model_dir: Path):
    """The constant term is weights read once plus fixed-size writes (the logits).

    Fusion must not raise it: no weight copied or reread. It goes down a
    little, because a scalar constant becomes a literal instead of a load.
    """
    from nanoinfer.weights import ModelWeights

    weight_bytes = sum(w.nbytes for w in oracle.weights_by_name(ModelWeights.load(tiny_model_dir)).values())
    fused = dict(program(config, last_only=True).bytes_moved.terms())
    unfused = dict(lower(build_qwen2(config, last_only=True)).bytes_moved.terms())
    assert weight_bytes <= fused[()] <= unfused[()]
    for monomial, coefficient in fused.items():
        assert coefficient <= unfused[monomial], monomial


def test_fused_and_unfused_models_agree_bitwise(tiny_model_dir: Path, config):
    from nanoinfer.weights import ModelWeights

    weights = oracle.weights_by_name(ModelWeights.load(tiny_model_dir))
    fused, unfused = Qwen2(config, weights, run=CBackend()), Qwen2(config, weights, run=CBackend(fuse=False))
    ids = [3, 1, 4, 1, 5, 9, 2, 6]
    np.testing.assert_array_equal(fused.forward(ids), unfused.forward(ids))
    a, b = fused.new_cache(capacity=2), unfused.new_cache(capacity=2)
    for chunk in (ids[:5], ids[5:6], ids[6:7], ids[7:]):
        np.testing.assert_array_equal(fused.forward(chunk, cache=a), unfused.forward(chunk, cache=b))
    for x, y in zip(a.past(), b.past()):
        np.testing.assert_array_equal(x, y)


def test_fused_buffers_never_overlap_while_live(config):
    prog = program(config)
    last_use = prog.last_uses()
    first_use: dict = {}
    for index, call in enumerate(prog.calls):
        first_use.setdefault(call.output, index)
    for bound in ({"seq": 1, "past": 9}, {"seq": 7, "past": 0}):
        plan = plan_memory(prog, last_use, bound)
        spans = []
        for buf, offset in plan.offsets.items():
            elements = int(np.prod([d if isinstance(d, int) else d.evaluate(bound) for d in buf.shape]))
            spans.append((first_use[buf], last_use[buf], offset, offset + elements * buf.dtype.numpy.itemsize))
        for i, (s1, e1, a1, b1) in enumerate(spans):
            for s2, e2, a2, b2 in spans[i + 1:]:
                assert not (s1 <= e2 and s2 <= e1 and a1 < b2 and a2 < b1)
