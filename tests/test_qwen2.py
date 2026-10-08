"""The phase 1 gate: Qwen2 in the graph IR computes what nanoinfer computes.

On the 2-layer test model the gate is 1e-5; on the real 494M weights it is
1e-3 on the logits and identical greedy tokens. Both are met with room to
spare: the interpreter performs nanoinfer's float32 operations in nanoinfer's
order, and the result is bitwise identical. That is asserted separately on the
tiny model, as a stronger claim than the gate, so that if a later change to
the frontend reorders arithmetic the gate still holds and the bitwise test
says exactly what was given up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nanocompile.interpreter import InterpreterError
from nanocompile.models.qwen2 import KVCache, Qwen2, Qwen2Config, build_qwen2, greedy
from tests import oracle

TINY_TOLERANCE = 1e-5
REAL_TOLERANCE = 1e-3

# Three of nanoinfer's phase 3 prompts: prose, code, and the chat template.
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
]


# -- the tiny model ----------------------------------------------------------


@pytest.fixture(scope="module")
def tiny(tiny_model_dir: Path):
    from nanoinfer.model import Qwen2 as Reference

    reference = Reference.from_model_dir(tiny_model_dir)
    model = Qwen2(Qwen2Config.from_model_dir(tiny_model_dir), oracle.weights_by_name(reference.weights))
    return model, reference


TINY_IDS = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])


def test_tiny_logits_meet_the_gate(tiny):
    model, reference = tiny
    np.testing.assert_allclose(
        model.forward(TINY_IDS), reference.forward(TINY_IDS), rtol=0, atol=TINY_TOLERANCE
    )


def test_tiny_logits_are_bitwise_identical(tiny):
    model, reference = tiny
    np.testing.assert_array_equal(model.forward(TINY_IDS), reference.forward(TINY_IDS))


def test_last_only_is_the_final_row(tiny):
    """Bitwise against nanoinfer's ``last_only``, which slices at the same point.

    Against our own full pass it is only close: a one-row matmul takes a
    different BLAS path than an eleven-row one and rounds differently (6e-8
    here). Asserting bitwise equality there would assert something false.
    """
    model, reference = tiny
    full = model.forward(TINY_IDS)
    last = model.forward(TINY_IDS, last_only=True)
    assert last.shape == (1, full.shape[1])
    np.testing.assert_array_equal(last, reference.forward(TINY_IDS, last_only=True))
    np.testing.assert_allclose(last, full[-1:], rtol=0, atol=1e-6)


def test_cached_steps_match_the_uncached_pass(tiny):
    """Prefill then one token at a time, against one pass over everything.

    Logits agree to float tolerance rather than bitwise, for the reason
    nanoinfer gives: the cache changes the shape of every matmul, and BLAS sums
    a different shape in a different order.
    """
    model, _ = tiny
    cache = model.new_cache(capacity=4)  # small, so the cache has to grow
    rows = [model.forward(TINY_IDS[:5], cache=cache)]
    for token in TINY_IDS[5:]:
        rows.append(model.forward([token], cache=cache))
    stepped = np.concatenate(rows)

    assert cache.length == len(TINY_IDS)
    np.testing.assert_allclose(stepped, model.forward(TINY_IDS), rtol=0, atol=1e-6)


def test_cached_keys_match_nanoinfer_cache(tiny):
    """The keys stored are post-RoPE and land in the right layer and slot."""
    model, reference = tiny
    ours = model.new_cache()
    model.forward(TINY_IDS, cache=ours)
    theirs = reference.new_cache(len(TINY_IDS))
    reference.forward(TINY_IDS, cache=theirs)
    keys, values = ours.past()
    for layer in range(model.config.num_hidden_layers):
        np.testing.assert_array_equal(keys[layer], theirs.keys(layer))
        np.testing.assert_array_equal(values[layer], theirs.values(layer))


def test_greedy_matches_nanoinfer(tiny):
    from nanoinfer.generate import greedy as reference_greedy

    model, reference = tiny
    for prompt in ([4, 5, 6], [7, 8], [1]):
        expected = reference_greedy(reference, prompt, max_new_tokens=8).generated_ids
        assert greedy(model, prompt, 8) == expected
        assert greedy(model, prompt, 8, use_cache=False) == expected


def test_rejects_bad_token_ids(tiny):
    model, _ = tiny
    with pytest.raises(ValueError, match="non-empty"):
        model.forward([])
    with pytest.raises(ValueError, match="1-D"):
        model.forward([[1, 2]])
    with pytest.raises(InterpreterError, match="index 64"):
        model.forward([1, 64])
    with pytest.raises(InterpreterError, match="index -1"):
        model.forward([-1])


def test_graph_signature(tiny):
    model, _ = tiny
    graph = model.graph
    assert graph.input_names == ["token_ids", "past_keys", "past_values"]
    assert graph.symbols == ("seq", "past")
    assert list(graph.outputs) == ["logits", "keys", "values"]
    # Tied embeddings: no lm_head weight, and the embedding is one node.
    assert "lm_head.weight" not in graph.weights
    assert len(graph.weights) == 2 + 12 * model.config.num_hidden_layers


def test_untied_model_has_an_output_projection(tiny):
    model, _ = tiny
    config = Qwen2Config(**{**model.config.__dict__, "tie_word_embeddings": False})
    assert "lm_head.weight" in build_qwen2(config).weights


def test_kv_cache_growth_preserves_contents():
    config = Qwen2Config(16, 8, 16, 2, 2, 1, 1e-6, 10000.0, True)
    cache = KVCache(config, capacity=2)
    chunks = [np.full((2, 1, n, 4), float(i), np.float32) for i, n in enumerate([1, 2, 3])]
    for chunk in chunks:
        cache.append(chunk, -chunk)
    keys, values = cache.past()
    assert cache.length == 6 and cache.capacity >= 6
    np.testing.assert_array_equal(keys, np.concatenate(chunks, axis=2))
    np.testing.assert_array_equal(values, -keys)


# -- the real model ----------------------------------------------------------


@pytest.fixture(scope="module")
def real(real_model_dir: Path):
    from nanoinfer.model import Qwen2 as Reference
    from nanoinfer.tokenizer import Tokenizer

    reference = Reference.from_model_dir(real_model_dir)
    model = Qwen2(Qwen2Config.from_model_dir(real_model_dir), oracle.weights_by_name(reference.weights))
    return model, reference, Tokenizer.from_model_dir(real_model_dir)


@pytest.mark.reference
@pytest.mark.slow
@pytest.mark.parametrize("prompt", PROMPTS)
def test_real_logits_meet_the_gate(real, prompt):
    model, reference, tokenizer = real
    ids = np.array(tokenizer.encode(prompt))
    ours, theirs = model.forward(ids), reference.forward(ids)
    np.testing.assert_allclose(ours, theirs, rtol=0, atol=REAL_TOLERANCE)
    np.testing.assert_array_equal(ours.argmax(-1), theirs.argmax(-1))


@pytest.mark.reference
@pytest.mark.slow
@pytest.mark.parametrize("prompt", PROMPTS)
def test_real_greedy_tokens_are_identical(real, prompt):
    from nanoinfer.generate import greedy as reference_greedy

    model, reference, tokenizer = real
    ids = tokenizer.encode(prompt)
    expected = reference_greedy(reference, ids, max_new_tokens=24).generated_ids
    assert greedy(model, ids, 24) == expected


@pytest.mark.reference
@pytest.mark.slow
def test_real_cached_and_uncached_agree(real):
    model, _, tokenizer = real
    ids = tokenizer.encode(PROMPTS[0])
    assert greedy(model, ids, 12) == greedy(model, ids, 12, use_cache=False)
