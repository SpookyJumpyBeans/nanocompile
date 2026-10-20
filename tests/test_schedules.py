"""The default schedules on the model: what each kernel gets, and that it changes nothing.

Phase 4's claim is stronger than its gate. The roadmap allowed a tolerance
wherever a schedule reorders a sum; none of these schedules reorders one, so
the scheduled model is held to bitwise equality with phase 3's, prompt and
cached decode steps alike, and its tokens to nanoinfer's.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from nanocompile import schedules
from nanocompile.fusion import fuse
from nanocompile.models.qwen2 import Qwen2, Qwen2Config, build_qwen2, greedy
from nanocompile.passes import optimize
from nanocompile.runtime import CBackend
from nanocompile.schedule import schedule_program
from tests import oracle

IDS = [3, 1, 4, 1, 5, 9, 2, 6]


def assert_same_model(a: Qwen2, b: Qwen2, ids) -> None:
    np.testing.assert_array_equal(a.forward(ids), b.forward(ids))
    ca, cb = a.new_cache(capacity=2), b.new_cache(capacity=2)
    for chunk in (ids[:5], ids[5:6], ids[6:7], ids[7:]):
        np.testing.assert_array_equal(a.forward(chunk, cache=ca), b.forward(chunk, cache=cb))
    for x, y in zip(ca.past(), cb.past()):
        np.testing.assert_array_equal(x, y)


def scheduled_program(config: Qwen2Config):
    return schedule_program(fuse(optimize(build_qwen2(config))), schedules.default)


def test_tiny_scheduled_model_is_bitwise_phase_3(tiny_model_dir: Path):
    from nanoinfer.weights import ModelWeights

    config = Qwen2Config.from_model_dir(tiny_model_dir)
    weights = oracle.weights_by_name(ModelWeights.load(tiny_model_dir))
    scheduled = Qwen2(config, weights, run=CBackend())
    plain = Qwen2(config, weights, run=CBackend(schedule=None))
    assert_same_model(scheduled, plain, IDS)


def test_tiny_projections_get_the_column_recipe(tiny_model_dir: Path):
    config = Qwen2Config.from_model_dir(tiny_model_dir)
    program = scheduled_program(config)
    packed = {i for i, k in enumerate(program.kernels) if any(p.packed for p in k.params)}
    # Per layer q, k, v, o, gate+up and down, then the LM head.
    assert sum(call.kernel in packed for call in program.calls) == 6 * config.num_hidden_layers + 1
    for i in packed:
        assert program.kernels[i].body[0].kind == "parallel"


@pytest.fixture(scope="module")
def real(real_model_dir: Path):
    from nanoinfer.model import Qwen2 as Reference
    from nanoinfer.tokenizer import Tokenizer

    reference = Reference.from_model_dir(real_model_dir)
    config = Qwen2Config.from_model_dir(real_model_dir)
    weights = oracle.weights_by_name(reference.weights)
    return config, weights, reference, Tokenizer.from_model_dir(real_model_dir)


@pytest.mark.reference
@pytest.mark.slow
def test_real_scheduled_model_is_bitwise_phase_3(real):
    config, weights, _, tokenizer = real
    ids = tokenizer.encode("The capital of France is Paris, and the capital of Spain is")
    assert len(ids) >= 8
    assert_same_model(Qwen2(config, weights, run=CBackend()),
                      Qwen2(config, weights, run=CBackend(schedule=None)), ids)


@pytest.mark.reference
@pytest.mark.slow
def test_real_scheduled_tokens_match_nanoinfer(real):
    from nanoinfer.generate import greedy as reference_greedy

    config, weights, reference, tokenizer = real
    model = Qwen2(config, weights, run=CBackend())
    ids = tokenizer.encode("def fibonacci(n):")
    assert greedy(model, ids, 24) == reference_greedy(reference, ids, max_new_tokens=24).generated_ids


@pytest.mark.reference
def test_every_real_projection_is_packed_and_tiled(real_model_dir: Path):
    program = scheduled_program(Qwen2Config.from_model_dir(real_model_dir))
    calls = Counter(program.calls[i].kernel for i in range(len(program.calls)))
    packed = {i for i, k in enumerate(program.kernels) if any(p.packed for p in k.params)}
    # Per layer: q, k, v, o, gate+up, down = 6 calls; and the LM head once.
    assert sum(calls[i] for i in packed) == 6 * 24 + 1
