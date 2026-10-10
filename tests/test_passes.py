"""The graph passes: fewer nodes, the same values, bitwise."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nanocompile import interpreter
from nanocompile.dtypes import f32
from nanocompile.frontend import GraphBuilder
from nanocompile.models.qwen2 import Qwen2Config, build_qwen2
from nanocompile.passes import eliminate_common_subexpressions, fold_constants, optimize
from tests import oracle


def test_cse_merges_repeated_constants_and_their_broadcasts():
    b = GraphBuilder()
    x = b.input("x", f32, (b.symbol("seq"), 4))
    graph = b.build({"a": x + 1.0, "b": x + 1.0})
    merged = eliminate_common_subexpressions(graph)
    assert len(merged.nodes) < len(graph.nodes)
    assert merged.outputs["a"] is merged.outputs["b"]


def test_cse_keeps_different_constants_apart():
    b = GraphBuilder()
    x = b.input("x", f32, (4,))
    graph = eliminate_common_subexpressions(b.build({"a": x + 1.0, "b": x + 2.0}))
    assert graph.outputs["a"] is not graph.outputs["b"]


def test_folding_evaluates_constant_subgraphs():
    b = GraphBuilder()
    c = b.const(np.arange(6, dtype=np.float32)).reshape(2, 3).permute(1, 0)
    graph = fold_constants(b.build({"y": c * 2.0}))
    assert [n.op for n in graph.nodes] == ["const"]
    np.testing.assert_array_equal(graph.outputs["y"].attrs["value"], np.arange(6, dtype=np.float32).reshape(2, 3).T * 2)


def test_folding_leaves_symbolic_shapes_alone():
    b = GraphBuilder()
    x = b.input("x", f32, (b.symbol("seq"), 4))
    graph = fold_constants(b.build({"y": x + 1.0}))
    assert "broadcast" in [n.op for n in graph.nodes]


def test_inputs_keep_their_identity():
    b = GraphBuilder()
    x = b.input("x", f32, (4,))
    graph = b.build({"y": x + 1.0})
    assert optimize(graph).inputs == graph.inputs


def test_optimized_qwen2_computes_the_same_bits(tiny_model_dir: Path):
    from nanoinfer.weights import ModelWeights

    config = Qwen2Config.from_model_dir(tiny_model_dir)
    weights = oracle.weights_by_name(ModelWeights.load(tiny_model_dir))
    graph = build_qwen2(config)
    optimized = optimize(graph)
    assert len(optimized.nodes) < len(graph.nodes)

    rng = np.random.default_rng(0)
    inputs = {
        "token_ids": np.array([3, 1, 4, 1, 5]),
        "past_keys": rng.standard_normal((2, 2, 3, 4)).astype(np.float32),
        "past_values": rng.standard_normal((2, 2, 3, 4)).astype(np.float32),
    }
    before = interpreter.run(graph, inputs, weights)
    after = interpreter.run(optimized, inputs, weights)
    for name in before:
        np.testing.assert_array_equal(after[name], before[name])
