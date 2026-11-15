"""Phase 6: int8 weights, checked against nanoinfer's quantizer and its Rust kernels.

The gate is bitwise. ``quantize_int8`` must produce nanoinfer's integers and
scales; both int8 matmuls must produce what nanoinfer's Rust kernels produce
from the same activations, to the bit, unscheduled and under every point of
their schedule space; and the quantized model must generate nanoinfer's int8
engine's tokens. The interpreter's meaning for ``matmul_i8`` is simulated
quantization, which is what nanoinfer's perplexity figures measure, so on the
tiny model it is held bitwise to nanoinfer's simulated model too.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from nanocompile import dtypes, interpreter, quantize, schedules
from nanocompile.frontend import GraphBuilder
from nanocompile.ir import ShapeError, make
from nanocompile.models.qwen2 import Qwen2, Qwen2Config, build_qwen2, greedy
from nanocompile.nn import linear, silu
from nanocompile.runtime import CBackend, CompiledGraph
from tests import oracle


def bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


@pytest.fixture(scope="module")
def rust(nanoinfer_root: Path):
    """nanoinfer's ``kernels`` module, with its Rust library loaded. Skips without it."""
    from nanoinfer import kernels

    if kernels._LIBRARY is None:
        pytest.skip("nanoinfer's Rust kernels are not built")
    yield kernels
    kernels.use_vnni(False)


def projection(k: int, n: int, bias: bool = False):
    b = GraphBuilder("projection")
    x = b.input("x", dtypes.f32, (b.symbol("seq"), k))
    w = b.weight("w", (n, k))
    return b.build({"y": linear(x, w, b.weight("b", (n,)) if bias else None)})


def rust_matmul(kernels, values, scales, x, vnni: bool) -> np.ndarray:
    kernels.use_vnni(vnni)
    if vnni:
        if not kernels.vnni_available():
            pytest.skip("this CPU has no AVX-VNNI")
        kernels.vnni_block(quantize.DEFAULT_BLOCK)
    try:
        return np.ascontiguousarray(kernels.matmul_i8(values, scales, x))
    finally:
        kernels.use_vnni(False)


# -- the quantizer -------------------------------------------------------------


def test_quantize_int8_is_nanoinfers_byte_for_byte(nanoinfer_root):
    from nanoinfer.quantization import quantize_int8 as reference

    rng = np.random.default_rng(0)
    w = (rng.standard_normal((37, 70)) * 0.04).astype(np.float32)
    w[3] = 0.0                                     # an all-zero row: scale 1
    w[5, :4] = [0.5, -0.5, 1.5, -2.5]              # halves, away from zero
    w[5, 4] = 127.0 / 64                           # the row's magnitude: exactly 127 steps
    values, scales = quantize.quantize_int8(w)
    expected = reference(w)
    assert np.array_equal(values, expected.values) and values.dtype == np.int8
    assert np.array_equal(bits(scales), bits(expected.scales))
    assert scales[3] == 1.0 and not values[3].any()


def test_round_is_half_away_from_zero_in_both_backends():
    values = np.array([0.5, -0.5, 1.5, 2.5, -2.5, 0.49999997, -0.49999997, 1e-30, -0.0, 0.0,
                       2.0**23 + 1, -(2.0**24), 3.7, -3.7, 126.5, -126.5], dtype=np.float32)
    b = GraphBuilder("round")
    x = b.input("x", dtypes.f32, (b.symbol("n"),))
    g = b.build({"y": x.round()})
    expected = np.array([math.copysign(math.floor(abs(v) + 0.5), v) if abs(v) < 2**23 else v
                         for v in values.astype(np.float64)], dtype=np.float32)
    got = interpreter.run(g, {"x": values}, {})["y"]
    assert np.array_equal(bits(got), bits(expected))
    for schedule in (None, "default"):
        compiled = CompiledGraph(g, schedule=schedule)({"x": values}, {})["y"]
        assert np.array_equal(bits(compiled), bits(expected))


# -- the graph pass ------------------------------------------------------------


def test_the_pass_rewrites_projections_only(tiny_model_dir):
    config = Qwen2Config.from_model_dir(tiny_model_dir)
    graph = build_qwen2(config)
    rewritten = quantize.quantize(graph, "f32")
    names = set(rewritten.weights)
    # The tied embedding is a gather table too: it stays float32, as the LM head.
    assert "model.embed_tokens.weight" in names
    assert rewritten.weights["model.embed_tokens.weight"].dtype == dtypes.f32
    assert quantize.int8_name("model.layers.0.mlp.down_proj.weight") in names
    assert "model.layers.0.mlp.down_proj.weight" not in names
    assert sum(n.op == "matmul_i8" for n in rewritten.nodes) == 7 * config.num_hidden_layers
    kept = quantize.quantize(graph, "f32", keep=["model.layers.1.self_attn.o_proj.weight"])
    assert "model.layers.1.self_attn.o_proj.weight" in kept.weights


def test_int8_activations_need_whole_blocks(tiny_model_dir):
    # The tiny model is 16 wide; nanoinfer's VNNI kernel would take its
    # scalar path there, and this one has none.
    graph = build_qwen2(Qwen2Config.from_model_dir(tiny_model_dir))
    with pytest.raises(ValueError, match="whole number of blocks"):
        quantize.quantize(graph, "int8")


def test_the_primitives_check_their_operands():
    b = GraphBuilder("bad")
    x = b.input("x", dtypes.f32, (4, 64)).node
    q = b.weight("q", (8, 64), dtypes.i8).node
    s = b.weight("s", (8,)).node
    assert make("matmul_i8", x, q, s).type_str == "f32[4, 8]"
    with pytest.raises(ShapeError, match="one f32 scale per row"):
        make("matmul_i8", x, q, b.weight("s2", (9,)).node)
    with pytest.raises(ShapeError, match="must be i8"):
        make("matmul_i8", x, b.weight("w", (8, 64)).node, s)
    xq = b.input("xq", dtypes.i8, (4, 64)).node
    xs = b.input("xs", dtypes.f32, (4, 2)).node
    off = b.input("off", dtypes.f32, (4, 1)).node
    assert make("matmul_i8_vnni", xq, xs, off, q, s, block=32).type_str == "f32[4, 8]"
    with pytest.raises(ShapeError, match="power of two"):
        make("matmul_i8_vnni", xq, xs, off, q, s, block=48)


# -- the kernels against nanoinfer's Rust ----------------------------------------
#
# Widths chosen for the loops of nanoinfer's kernels: 896 and 4864 are the
# model's; 77 has 32-wide blocks, 8-wide groups and a scalar tail; 40 has no
# 32-wide block; 160 and 96 leave 32-byte VNNI groups past the 128-byte stride.

WIDTHS = [(896, 72), (4864, 24), (77, 40), (40, 16), (160, 40), (96, 24)]
TOKENS = [1, 3, 7]


def inputs(k: int, n: int, tokens: int, seed: int):
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((n, k)) * 0.05).astype(np.float32)
    x = (rng.standard_normal((tokens, k)) * 2).astype(np.float32)
    x[:, ::13] *= 40.0          # outliers, as the model's activations have
    return w, x


@pytest.mark.parametrize("activations", quantize.ACTIVATIONS)
@pytest.mark.parametrize("k,n", WIDTHS)
def test_kernel_is_bitwise_nanoinfers_rust(rust, activations, k, n):
    if activations == "int8" and k % quantize.DEFAULT_BLOCK:
        pytest.skip("int8 activations need whole blocks")
    graph = quantize.quantize(projection(k, n), activations)
    for tokens in TOKENS:
        w, x = inputs(k, n, tokens, seed=k + tokens)
        weights = quantize.quantized_weights(graph, {"w": w})
        values, scales = weights["w@int8"], weights["w@scales"]
        expected = rust_matmul(rust, values, scales, x, vnni=activations == "int8")
        for options in ({"fuse": False}, {"schedule": None}, {}):
            got = CompiledGraph(graph, **options)({"x": x}, weights)["y"]
            assert np.array_equal(bits(got), bits(expected)), (options, tokens)


@pytest.mark.slow
@pytest.mark.parametrize("activations", quantize.ACTIVATIONS)
def test_every_schedule_is_bitwise_nanoinfers_rust(rust, activations):
    """Every point of the int8 space, through the epilogue SwiGLU fuses on."""
    k, n = (96, 72)
    b = GraphBuilder("swiglu")
    x = b.input("x", dtypes.f32, (b.symbol("seq"), k))
    gate, up = (linear(x, b.weight(name, (n, k))) for name in ("gate", "up"))
    graph = quantize.quantize(b.build({"y": silu(gate) * up}), activations)
    rng = np.random.default_rng(5)
    source = {name: (rng.standard_normal((n, k)) * 0.05).astype(np.float32) for name in ("gate", "up")}
    weights = quantize.quantized_weights(graph, source)
    xs = (rng.standard_normal((5, k)) * 2).astype(np.float32)
    unscheduled = CompiledGraph(graph, schedule=None)({"x": xs}, weights)["y"]
    # Unfused, gate and up are separate kernels, each the one tested above
    # against nanoinfer; fused, they share one k loop and must not notice.
    unfused = CompiledGraph(graph, fuse=False)({"x": xs}, weights)["y"]
    assert np.array_equal(bits(unfused), bits(unscheduled))
    seen = 0
    for config in schedules.space_of_class("int8"):
        def schedule(kernel, calls, config=config):
            if schedules.kernel_class(kernel) != "int8":
                return kernel
            return schedules.apply(kernel, calls, config)
        compiled = CompiledGraph(graph, schedule=schedule)
        assert any(k.primitive.startswith("matmul_i8") for k in compiled.program.kernels)
        got = compiled({"x": xs}, weights)["y"]
        assert np.array_equal(bits(got), bits(unscheduled)), config
        seen += 1
    assert seen == len(schedules.space_of_class("int8"))


def test_the_vnni_vector_form_is_used_when_the_compiler_has_it(rust):
    graph = quantize.quantize(projection(256, 32), "int8")
    source = CompiledGraph(graph).source
    assert "nc_v_dp4(" in source and "defined(__AVXVNNI__)" in source


# -- models ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny8(tiny_model_dir: Path):
    from nanoinfer.model import Qwen2 as Reference
    from nanoinfer.quantization import quantize_model

    reference = Reference.from_model_dir(tiny_model_dir)
    config = Qwen2Config.from_model_dir(tiny_model_dir)
    weights = oracle.weights_by_name(reference.weights)
    simulated, _ = quantize_model(reference.weights, bits=8, quantize_embeddings=False)
    stored, _ = quantize_model(reference.weights, bits=8, quantize_embeddings=False, dequantize=False)
    return config, weights, Reference(simulated), Reference(stored)


def test_tiny_interpreter_is_nanoinfers_simulated_int8_bitwise(tiny8):
    config, weights, simulated, _ = tiny8
    ids = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
    ours = Qwen2(config, weights, quantization="int8").forward(ids)
    assert np.array_equal(bits(ours), bits(simulated.forward(ids)))


def test_tiny_int8_greedy_matches_nanoinfers_int8_engine(rust, tiny8):
    from nanoinfer.generate import greedy as reference_greedy

    config, weights, _, stored = tiny8
    model = Qwen2(config, weights, run=CBackend(), quantization="int8")
    for prompt in ([4, 5, 6], [7, 8], [1]):
        expected = reference_greedy(stored, prompt, max_new_tokens=8).generated_ids
        assert greedy(model, prompt, 8) == expected
        assert greedy(model, prompt, 8, use_cache=False) == expected


@pytest.fixture(scope="module")
def real8(real_model_dir: Path, rust):
    from nanoinfer.model import Qwen2 as Reference
    from nanoinfer.quantization import quantize_model
    from nanoinfer.tokenizer import Tokenizer

    reference = Reference.from_model_dir(real_model_dir)
    stored, _ = quantize_model(reference.weights, bits=8, quantize_embeddings=False, dequantize=False)
    return (Qwen2Config.from_model_dir(real_model_dir), oracle.weights_by_name(reference.weights),
            Reference(stored), Tokenizer.from_model_dir(real_model_dir))


@pytest.mark.reference
@pytest.mark.slow
@pytest.mark.parametrize("quantization", ["int8", "int8-vnni"])
def test_real_int8_tokens_are_nanoinfers(real8, rust, quantization):
    from nanoinfer.generate import greedy as reference_greedy

    config, weights, stored, tokenizer = real8
    model = Qwen2(config, weights, run=CBackend(), quantization=quantization)
    vnni = quantization == "int8-vnni"
    for prompt in ("The capital of France is", "def fibonacci(n):"):
        ids = tokenizer.encode(prompt)
        rust.use_vnni(vnni)
        rust.vnni_block(quantize.DEFAULT_BLOCK)
        try:
            expected = reference_greedy(stored, ids, max_new_tokens=24).generated_ids
        finally:
            rust.use_vnni(False)
        assert greedy(model, ids, 24) == expected


@pytest.mark.reference
@pytest.mark.slow
def test_real_projection_is_bitwise_nanoinfers_rust(real8, rust):
    """One real layer's down_proj, on activations of the real model's shape."""
    config, weights, _, _ = real8
    name = "model.layers.3.mlp.down_proj.weight"
    graph = quantize.quantize(projection(config.intermediate_size, config.hidden_size), "f32")
    q = quantize.quantized_weights(graph, {"w": weights[name]})
    _, x = inputs(config.intermediate_size, 1, 9, seed=3)
    for vnni, mode in ((False, "f32"), (True, "int8")):
        g = quantize.quantize(projection(config.intermediate_size, config.hidden_size), mode)
        got = CompiledGraph(g)({"x": x}, q)["y"]
        assert np.array_equal(bits(got), bits(rust_matmul(rust, q["w@int8"], q["w@scales"], x, vnni)))
