"""Phase 5 measurements: autotuning the real model, run several times.

    python -m bench.phase5 [--runs 3] [--seed 0]         # tune, then time the model
    python -m bench.phase5 --model-only [--rounds 6]     # time the model from the cache

Each part appends one JSON line to ``bench/results.jsonl``.

**Tuning** (``part: tuning``):

* **Per kernel**, from the first run: the size of its space, how many
  configurations the search tried, how long it took, the winner, and the
  winner against phase 4's hand-written schedule, measured in one alternating
  batch (the search's final comparison).
* **Search time** of each whole-model tuning run.
* **Stability.** Every run uses the same seed, so the random sample is the
  same in every run and any difference comes from measurement alone. For
  each kernel: how many distinct winners the runs chose, and whether the runs
  tried the same candidates. Every run's records are also replayed on their
  own costs, which must reproduce exactly the candidates the run tried.

The first run's winners are written to ``--cache`` (``bench/phase5_tuning.json``).

**The model** (``part: model``): phase 4's schedules against the cached tuned
ones, alternating in one process, the engine that goes first swapping every
round, tokens checked identical. Run it in a fresh process (``--model-only``,
which the full run does by itself): a process that has just spent half an
hour allocating and freeing gigabytes of candidate buffers is not the machine
a model runs on.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from bench.phase1 import BLAS_ENV, PROMPTS, RESULTS, decode_steps
from nanocompile import autotune, runtime, schedules
from nanocompile.autotune import Autotuner
from nanocompile.fusion import fuse
from nanocompile.models.qwen2 import Qwen2, Qwen2Config, build_qwen2
from nanocompile.passes import optimize
from nanocompile.runtime import CBackend
from tests import oracle

CACHE = Path(__file__).resolve().parent / "phase5_tuning.json"


def fmt(config: dict | None) -> str:
    if config is None:
        return "unscheduled"
    return " ".join(f"{k}={'y' if v is True else 'n' if v is False else v}" for k, v in config.items())


def header(args) -> dict:
    return {
        "phase": 5,
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": platform.processor() or platform.machine(),
        "cpu_threads": os.cpu_count(),
        "pool_threads": runtime.threads(),
        "blas_env": {k: os.environ.get(k) for k in BLAS_ENV},
        "seed": args.seed,
    }


def append(record: dict) -> None:
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"appended to {RESULTS}")


# -- tuning ------------------------------------------------------------------------


def tuning(args) -> None:
    config = Qwen2Config.from_model_dir(oracle.model_dir())
    program = fuse(optimize(build_qwen2(config)))
    uses = Counter(c.kernel for c in program.calls)
    if args.cache.exists():
        args.cache.unlink()               # the first run measures; it does not read a stale file

    runs = []
    for r in range(args.runs):
        tuner = Autotuner(seed=args.seed, cache=args.cache if r == 0 else None)
        started = time.perf_counter()
        tuner.schedule_program(program)
        seconds = time.perf_counter() - started
        runs.append((tuner, seconds))
        print(f"run {r + 1}: tuned {len(program.kernels)} kernels in {seconds:.0f} s")

    # Replay: each run's search, fed its own measurements, tries what it tried.
    first = runs[0][0]
    replayed = 0
    for tuner, _ in runs:
        for i, record in enumerate(tuner.records):
            if not record.evaluations:
                continue
            calls = [c for c in program.calls if c.kernel == i]
            space = [c.config for c in first.legal_space(program.kernels[i], calls)]
            assert autotune.replay(record, space, args.seed, first.samples, first.steps, first.margin) == record.tried
            replayed += 1
    print(f"replayed {replayed} kernel searches on their own costs: every one tried exactly what it tried")

    kernels = []
    print()
    print(f"{'#':>3} {'kernel':<16} {'calls':>5} {'space':>5} {'tried':>5} {'secs':>6} {'vs hand':>8} "
          f"{'winners':>7}  winner")
    for i, record in enumerate(first.records):
        winners = [schedules.canonical(t.records[i].winner) if t.records[i].winner else None for t, _ in runs]
        tried_same = len({json.dumps([schedules.canonical(c) for c in t.records[i].tried]) for t, _ in runs}) == 1
        row = {
            "index": i, "primitive": record.primitive, "class": record.kernel_class, "calls": uses[i],
            "space": record.space_size, "tried": len(record.tried), "seconds": round(record.seconds, 1),
            "default": record.default, "winner": record.winner,
            "speedup": None if record.speedup is None else round(record.speedup, 3),
            "default_cost_us": None if record.default_cost is None else round(record.default_cost * 1e6, 2),
            "winner_cost_us": None if record.winner_cost is None else round(record.winner_cost * 1e6, 2),
            "distinct_winners": len(set(winners)), "same_candidates_every_run": tried_same,
            "winners_by_run": [t.records[i].winner for t, _ in runs],
        }
        kernels.append(row)
        speedup = "-" if row["speedup"] is None else f"{row['speedup']:.2f}x"
        print(f"{i:>3} {record.primitive:<16} {uses[i]:>5} {record.space_size:>5} {len(record.tried):>5} "
              f"{record.seconds:>6.1f} {speedup:>8} {row['distinct_winners']:>4}/{args.runs}  {fmt(record.winner)}")

    searched = [k for k in kernels if k["tried"] > 1]
    changed = [k for k in searched if k["distinct_winners"] > 1]
    diverged = [k for k in searched if not k["same_candidates_every_run"]]
    beats = [k for k in searched if k["winner"] != k["default"]]
    print()
    print(f"searched {len(searched)} kernels; the winner beat the hand-written schedule on {len(beats)}")
    print(f"winner changed between runs on {len(changed)} of {len(searched)}; "
          f"candidates tried differed on {len(diverged)} of {len(searched)}")
    print("search time per run: " + ", ".join(f"{s:.0f} s" for _, s in runs))

    append({**header(args), "part": "tuning", "runs": args.runs,
            "search": {"samples": first.samples, "steps": first.steps, "rounds": first.rounds,
                       "margin": first.margin, "workloads": [list(w) for w in first.workloads]},
            "search_seconds": [round(s, 1) for _, s in runs], "kernels": kernels,
            "winner_changed": len(changed), "candidates_differed": len(diverged),
            "searched": len(searched), "beat_hand_written": len(beats), "replayed": replayed})


# -- the model ---------------------------------------------------------------------


def model(args) -> None:
    if not args.cache.exists():
        raise SystemExit(f"no tuning cache at {args.cache}; run without --model-only first")
    model_dir = oracle.model_dir()
    oracle.import_nanoinfer()
    from nanoinfer.model import Qwen2 as Reference
    from nanoinfer.tokenizer import Tokenizer

    reference = Reference.from_model_dir(model_dir)
    tokenizer = Tokenizer.from_model_dir(model_dir)
    config = Qwen2Config.from_model_dir(model_dir)
    weights = oracle.weights_by_name(reference.weights)
    del reference
    tuner = Autotuner(seed=args.seed, cache=args.cache)
    models = {"hand-written": Qwen2(config, weights, run=CBackend()),
              "tuned": Qwen2(config, weights, run=CBackend(schedule=tuner))}

    prompt_ids = tokenizer.encode(PROMPTS[0])
    capacity = len(prompt_ids) + args.tokens

    def run(m):
        cache = m.new_cache(capacity)
        return decode_steps(lambda ids, first: m.forward(ids, cache=cache, last_only=True)[0],
                            prompt_ids, args.tokens)

    # One untimed generation each: compile, load, and pack every weight once.
    outputs = {name: run(m)[2] for name, m in models.items()}
    measured = [r for r in tuner.records if not r.cached and r.evaluations]
    if measured:
        raise SystemExit(f"{len(measured)} kernels were not in the cache; it does not match this program")

    steps = {name: [] for name in models}
    prefill = {name: [] for name in models}
    names = list(models)
    for round_index in range(args.rounds):
        order = names if round_index % 2 == 0 else names[::-1]
        for name in order:
            first, times, generated = run(models[name])
            assert generated == outputs[name]
            steps[name].extend(times)
            prefill[name].append(first)
            print(f"round {round_index + 1} {name:<12} median step {statistics.median(times) * 1e3:7.1f} ms"
                  f"  prefill {first * 1e3:7.1f} ms")
    identical = outputs["hand-written"] == outputs["tuned"]

    speed = {}
    print()
    for name in names:
        median = statistics.median(steps[name])
        speed[name] = {"median_step_ms": round(median * 1e3, 1), "min_step_ms": round(min(steps[name]) * 1e3, 1),
                       "decode_tok_s": round(1 / median, 3),
                       "median_prefill_ms": round(statistics.median(prefill[name]) * 1e3, 1),
                       "prefill_ms": [round(p * 1e3, 1) for p in prefill[name]]}
        print(f"{name:<12} {speed[name]['median_step_ms']:>7.1f} ms (min {speed[name]['min_step_ms']})"
              f"  {speed[name]['decode_tok_s']:.2f} tok/s   prefill {speed[name]['median_prefill_ms']} ms")
    print(f"tokens identical: {identical}")
    append({**header(args), "part": "model", "rounds": args.rounds, "tokens": args.tokens,
            "speed": speed, "tokens_identical": identical})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--model-only", action="store_true")
    args = parser.parse_args()
    if args.model_only:
        model(args)
        return 0
    tuning(args)
    # The model in a fresh process, for the reason the docstring gives.
    return subprocess.call([sys.executable, "-m", "bench.phase5", "--model-only", "--seed", str(args.seed),
                            "--rounds", str(args.rounds), "--tokens", str(args.tokens),
                            "--cache", str(args.cache)])


if __name__ == "__main__":
    raise SystemExit(main())
