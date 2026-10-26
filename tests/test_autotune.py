"""The phase 5 gate: tuned kernels compute the same bits, and a run is reproducible.

Three layers. The **space** is checked whole: every configuration of every
kernel in the phase 4 cases, run against the unscheduled kernel at sizes that
leave split tails of every length, bit for bit. A **tuned** program then goes
through phase 4's own ``check`` and, for the tiny model, phase 4's model
comparison. The **search** is tested on cost functions whose answer is known,
so reproducibility, the noise margin and replay are asserted exactly rather
than hoped for on a noisy machine.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from nanocompile import autotune, schedules
from nanocompile.autotune import Autotuner, Batch, Candidate, Operands, TuningCache, TuningError
from nanocompile.dtypes import f32
from nanocompile.frontend import GraphBuilder
from nanocompile.fusion import fuse
from nanocompile.models.qwen2 import Qwen2, Qwen2Config
from nanocompile.passes import optimize
from nanocompile.runtime import CBackend, CompiledGraph
from nanocompile.schedule import ScheduleError
from tests import oracle
from tests.test_schedule import CASES, SIZES, check
from tests.test_schedules import IDS, assert_same_model

# Small, so a tuning run in a test takes seconds; the bench uses the defaults.
FAST = dict(samples=3, steps=1, rounds=3)
WORKLOADS = (("small", {"seq": 3}, 1.0), ("tile", {"seq": 8}, 1.0 / 8))


def kernels_of(graph):
    program = fuse(optimize(graph))
    calls: dict[int, list] = {}
    for call in program.calls:
        calls.setdefault(call.kernel, []).append(call)
    return program, [(k, calls[i]) for i, k in enumerate(program.kernels) if i in calls]


# -- the space -------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CASES))
def test_default_is_one_point_of_the_space(name):
    _, kernels = kernels_of(CASES[name].graph)
    for kernel, calls in kernels:
        config = schedules.default_config(kernel, calls)
        if config is None:
            assert schedules.default(kernel, calls) is kernel
            continue
        assert schedules.canonical(config) in {schedules.canonical(c) for c in schedules.space(kernel)}
        assert schedules.apply(kernel, calls, config) == schedules.default(kernel, calls)


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_configuration_computes_the_unscheduled_bits(name):
    """The whole space, not just the winners: any point the search picks is correct."""
    program, kernels = kernels_of(CASES[name].graph)
    tuner = Autotuner(cache=None)
    checked = 0
    for kernel, calls in kernels:
        candidates = tuner.legal_space(kernel, calls)
        batch = Batch(kernel, candidates)
        for seq in SIZES:
            batch.check(Operands(kernel, calls[0], program.symbols, {"seq": seq},
                                 np.random.default_rng(seq)))
        checked += len(candidates)
    assert checked > 0


def test_a_matmul_space_is_large_and_has_no_duplicates():
    program, kernels = kernels_of(CASES["wide matmul"].graph)
    kernel, calls = next((k, c) for k, c in kernels if k.primitive.startswith("matmul"))
    candidates = Autotuner(cache=None).legal_space(kernel, calls)
    assert len(candidates) > 20
    assert len({c.source for c in candidates}) == len(candidates)
    # The hand-written schedule comes first, so it stands for its own kernel.
    assert candidates[0].config == schedules.default_config(kernel, calls)


def impostor_of_elementwise():
    """A kernel with the elementwise case's signature and shapes, computing
    ``exp(x + y) + x`` instead of ``exp(x * y) + x``. Same operands, so it is
    wrong without writing anywhere it should not."""
    b = GraphBuilder()
    x, y = b.input("x", f32, (b.symbol("seq"), 13)), b.input("y", f32, (13,))
    _, kernels = kernels_of(b.build({"z": (x + y).exp() + x}))
    return schedules.default(*kernels[0])


def test_a_wrong_candidate_is_caught_before_it_is_timed():
    program, kernels = kernels_of(CASES["elementwise"].graph)
    kernel, calls = kernels[0]
    batch = Batch(kernel, [Candidate({"vectorize": True, "parallel": True}, impostor_of_elementwise(), "")])
    operands = Operands(kernel, calls[0], program.symbols, {"seq": 3}, np.random.default_rng(0))
    with pytest.raises(TuningError, match="computes something else"):
        batch.check(operands)


def test_a_write_past_the_output_is_caught_even_when_the_output_is_right():
    """The guard regions: a kernel that writes one row too many."""
    from nanocompile import loopir as L
    program, kernels = kernels_of(CASES["elementwise"].graph)
    kernel, calls = kernels[0]
    loop = kernel.body[0]
    assert isinstance(loop, L.Loop)
    longer = L.Loop(loop.var, L.add(loop.extent, L.const(1)), loop.body, loop.kind)
    overrun = L.Kernel(kernel.primitive, kernel.params, (longer,) + kernel.body[1:])
    batch = Batch(kernel, [Candidate({"vectorize": False, "parallel": False}, overrun, "")])
    # x is read past its end too, so the operands sit in a larger buffer.
    operands = Operands(kernel, calls[0], program.symbols, {"seq": 3}, np.random.default_rng(0))
    operands.bases[0] = np.concatenate([operands.bases[0], np.zeros((1, 13), np.float32)])
    with pytest.raises(TuningError, match="writes after its output"):
        batch.check(operands)


# -- tuned programs ----------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CASES))
def test_tuned_kernels_pass_the_phase_4_check(name, tmp_path):
    tuner = Autotuner(**FAST, workloads=WORKLOADS, cache=tmp_path / "tuning.json")
    if all(schedules.default_config(k, c) is None for k, c in kernels_of(CASES[name].graph)[1]):
        pytest.skip("nothing to schedule")
    check(name, tuner, changes=False)
    assert tuner.records


def test_tiny_tuned_model_is_bitwise_phase_3(tiny_model_dir: Path, tmp_path):
    from nanoinfer.weights import ModelWeights

    config = Qwen2Config.from_model_dir(tiny_model_dir)
    weights = oracle.weights_by_name(ModelWeights.load(tiny_model_dir))
    tuner = Autotuner(**FAST, cache=tmp_path / "tuning.json")
    tuned = Qwen2(config, weights, run=CBackend(schedule=tuner))
    plain = Qwen2(config, weights, run=CBackend(schedule=None))
    assert_same_model(tuned, plain, IDS)
    assert any(r.evaluations for r in tuner.records)


# -- the search, on costs whose answer is known -------------------------------------


SPACE = [c for c in schedules.space_of_class("matmul")]
DEFAULT = dict(schedules.DEFAULTS["matmul"][0])


def target_cost(target: dict, scale: float = 1.0):
    """Cost grows with how many knobs differ from ``target``: one clear optimum."""
    def cost(stage, configs):
        return [scale * (1.0 + sum(c.get(k) != v for k, v in target.items())) for c in configs]
    return cost


def run(seed: int, cost, samples=8, steps=4, margin=0.03):
    rng = random.Random(autotune.seed_material(seed, "kernel"))
    return autotune.search(SPACE, DEFAULT, "matmul", rng, cost, samples, steps, margin)


def tried(log):
    record = autotune.TuneRecord("kernel", "matmul", "matmul", len(SPACE), DEFAULT, None, evaluations=log)
    return record.tried


def test_the_same_seed_tries_the_same_candidates():
    target = {**DEFAULT, "vectors": 2, "rows": 3, "unroll": False}
    a, b = run(7, target_cost(target)), run(7, target_cost(target))
    assert tried(a[1]) == tried(b[1])
    assert a[0] == b[0]


def test_the_sample_depends_on_the_seed_alone_not_on_the_measurements():
    flat = run(3, lambda stage, configs: [1.0] * len(configs))
    steep = run(3, target_cost({**DEFAULT, "vectors": 8}))
    assert flat[1][0].configs == steep[1][0].configs
    assert run(4, lambda s, c: [1.0] * len(c))[1][0].configs != flat[1][0].configs


def test_refinement_walks_to_a_known_optimum():
    target = {**DEFAULT, "vectors": 2, "rows": 3, "unroll": False}
    winner, log, default_cost, winner_cost = run(0, target_cost(target), samples=0, steps=10)
    assert schedules.canonical(winner) == schedules.canonical(target)
    assert winner_cost < default_cost
    assert [e.stage for e in log][0] == "sample" and log[-1].stage == "confirm"


def test_a_win_inside_the_margin_keeps_the_hand_written_schedule():
    """Under noise, a 1% difference is not a decision the machine can make."""
    def cost(stage, configs):
        return [0.99 if c != DEFAULT else 1.0 for c in configs]
    winner, *_ = run(0, cost, margin=0.03)
    assert winner == DEFAULT


def test_a_win_that_does_not_survive_confirmation_is_dropped():
    def cost(stage, configs):
        if stage == "confirm":
            return [1.0, 1.0]             # it was noise
        return [1.0 if c == DEFAULT else 0.5 for c in configs]
    winner, *_ = run(0, cost)
    assert winner == DEFAULT


def test_replay_reproduces_what_a_run_tried():
    target = {**DEFAULT, "vectors": 1, "pack": False}
    winner, log, *_ = run(11, target_cost(target), samples=5, steps=3)
    record = autotune.TuneRecord("kernel", "matmul", "matmul", len(SPACE), DEFAULT, winner, evaluations=log)
    assert autotune.replay(record, SPACE, 11, samples=5, steps=3, margin=0.03) == record.tried
    with pytest.raises(TuningError, match="diverged"):
        autotune.replay(record, SPACE, 12, samples=5, steps=3, margin=0.03)


def test_neighbours_change_one_knob_one_step():
    for n in autotune.neighbours(DEFAULT, "matmul"):
        if n.get("recipe") == "scalar":
            assert n == {"recipe": "scalar", "parallel": True}
            continue
        changed = [k for k in DEFAULT if n[k] != DEFAULT[k]]
        assert len(changed) == 1
        values = schedules.SPACES["matmul"][changed[0]]
        assert abs(values.index(n[changed[0]]) - values.index(DEFAULT[changed[0]])) == 1


# -- the cache ---------------------------------------------------------------------


def test_a_cached_winner_is_applied_without_measuring(tmp_path):
    path = tmp_path / "tuning.json"
    first = Autotuner(**FAST, workloads=WORKLOADS, cache=path)
    a = CompiledGraph(CASES["wide matmul"].graph, schedule=first)
    assert any(r.evaluations for r in first.records)

    second = Autotuner(**FAST, workloads=WORKLOADS, cache=path)
    b = CompiledGraph(CASES["wide matmul"].graph, schedule=second)
    assert a.source == b.source
    assert all(r.cached and not r.evaluations for r in second.records if r.default is not None)


def test_the_cache_is_keyed_by_machine(tmp_path, monkeypatch):
    path = tmp_path / "tuning.json"
    CompiledGraph(CASES["wide matmul"].graph, schedule=Autotuner(**FAST, workloads=WORKLOADS, cache=path))
    monkeypatch.setattr(autotune, "machine_key", lambda: "another machine")
    elsewhere = Autotuner(**FAST, workloads=WORKLOADS, cache=path)
    CompiledGraph(CASES["wide matmul"].graph, schedule=elsewhere)
    assert any(r.evaluations for r in elsewhere.records)
    assert len(json.loads(path.read_text())["machines"]) == 2


def test_a_cache_from_another_version_is_ignored(tmp_path):
    path = tmp_path / "tuning.json"
    path.write_text(json.dumps({"version": -1, "machines": {"x": {}}}))
    assert TuningCache(path).get("anything") is None
    path.write_text("not json")
    assert TuningCache(path).get("anything") is None


def test_a_stale_cached_configuration_is_tuned_again(tmp_path):
    path = tmp_path / "tuning.json"
    tuner = Autotuner(**FAST, workloads=WORKLOADS, cache=path)
    CompiledGraph(CASES["wide matmul"].graph, schedule=tuner)
    data = json.loads(path.read_text())
    for entries in data["machines"].values():
        for entry in entries.values():
            if entry["winner"] is not None and entry["winner"].get("recipe") is not None:
                entry["winner"] = {**entry["winner"], "vectors": 3}     # not a legal tile
    path.write_text(json.dumps(data))
    again = Autotuner(**FAST, workloads=WORKLOADS, cache=path)
    CompiledGraph(CASES["wide matmul"].graph, schedule=again)
    assert any(r.evaluations for r in again.records if r.kernel_class == "matmul")


# -- the real model ------------------------------------------------------------------


@pytest.mark.reference
@pytest.mark.slow
def test_real_tuned_model_is_bitwise_phase_3(real_model_dir: Path, tmp_path):
    """The winners bench/phase5.py chose on this machine, on the real weights."""
    from nanoinfer.model import Qwen2 as Reference
    from nanoinfer.tokenizer import Tokenizer

    cached = Path(__file__).resolve().parent.parent / "bench" / "phase5_tuning.json"
    if not cached.exists() or autotune.machine_key() not in json.loads(cached.read_text())["machines"]:
        pytest.skip("no tuning recorded for this machine; run python -m bench.phase5")
    copy = tmp_path / "tuning.json"
    copy.write_text(cached.read_text())

    reference = Reference.from_model_dir(real_model_dir)
    config = Qwen2Config.from_model_dir(real_model_dir)
    weights = oracle.weights_by_name(reference.weights)
    tuner = Autotuner(cache=copy)
    tuned = Qwen2(config, weights, run=CBackend(schedule=tuner))
    plain = Qwen2(config, weights, run=CBackend(schedule=None))
    ids = Tokenizer.from_model_dir(real_model_dir).encode("The capital of France is Paris, and the capital of Spain is")
    assert_same_model(tuned, plain, ids)
    assert not any(r.evaluations for r in tuner.records), "everything came from the cache"
    assert any(r.winner != r.default for r in tuner.records), "and some of it differs from phase 4"
