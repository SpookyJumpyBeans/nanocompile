"""Qwen2 end to end in generated C: the phase 2 gate on the whole model.

The gate is identical greedy tokens to nanoinfer. Logits are not bitwise
here, unlike phase 1: the generated matmul adds in index order where NumPy
calls BLAS, so the last bits differ, and the check on them is a tolerance.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nanocompile.lower import lower
from nanocompile.models.qwen2 import Qwen2, Qwen2Config, build_qwen2, greedy
from nanocompile.runtime import CBackend, KernelError
from tests import oracle

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
]
TINY_IDS = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])


@pytest.fixture(scope="module")
def tiny(tiny_model_dir: Path):
    from nanoinfer.model import Qwen2 as Reference

    reference = Reference.from_model_dir(tiny_model_dir)
    config = Qwen2Config.from_model_dir(tiny_model_dir)
    weights = oracle.weights_by_name(reference.weights)
    return Qwen2(config, weights, run=CBackend()), Qwen2(config, weights), reference


def test_tiny_logits_match_the_interpreter(tiny):
    compiled, interpreted, _ = tiny
    np.testing.assert_allclose(compiled.forward(TINY_IDS), interpreted.forward(TINY_IDS), rtol=0, atol=1e-6)


def test_tiny_greedy_matches_nanoinfer(tiny):
    from nanoinfer.generate import greedy as reference_greedy

    compiled, _, reference = tiny
    for prompt in ([4, 5, 6], [7, 8], [1]):
        expected = reference_greedy(reference, prompt, max_new_tokens=8).generated_ids
        assert greedy(compiled, prompt, 8) == expected
        assert greedy(compiled, prompt, 8, use_cache=False) == expected


def test_tiny_cached_steps_match_one_pass(tiny):
    compiled, _, _ = tiny
    cache = compiled.new_cache(capacity=4)
    rows = [compiled.forward(TINY_IDS[:5], cache=cache)]
    rows += [compiled.forward([t], cache=cache) for t in TINY_IDS[5:]]
    np.testing.assert_allclose(np.concatenate(rows), compiled.forward(TINY_IDS), rtol=0, atol=1e-6)


def test_bad_token_ids_are_caught_by_the_kernel(tiny):
    compiled, _, _ = tiny
    with pytest.raises(KernelError, match="index 64"):
        compiled.forward([1, 64])
    with pytest.raises(KernelError, match="index -1"):
        compiled.forward([-1])


def test_layers_share_kernels(tiny_model_dir: Path):
    """A third layer adds calls but no new code: the blocks are the same kernels."""
    config = Qwen2Config.from_model_dir(tiny_model_dir)
    two = lower(build_qwen2(config))
    three = lower(build_qwen2(replace(config, num_hidden_layers=3)))
    assert len(three.calls) > len(two.calls)
    assert len(three.kernels) == len(two.kernels)


@pytest.fixture(scope="module")
def real(real_model_dir: Path):
    from nanoinfer.model import Qwen2 as Reference
    from nanoinfer.tokenizer import Tokenizer

    reference = Reference.from_model_dir(real_model_dir)
    model = Qwen2(
        Qwen2Config.from_model_dir(real_model_dir),
        oracle.weights_by_name(reference.weights),
        run=CBackend(),
    )
    return model, reference, Tokenizer.from_model_dir(real_model_dir)


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
def test_real_logits_within_tolerance(real):
    model, reference, tokenizer = real
    ids = np.array(tokenizer.encode(PROMPTS[0]))
    np.testing.assert_allclose(model.forward(ids), reference.forward(ids), rtol=0, atol=1e-3)
