"""Phase 2 measurements: naive generated C against nanoinfer.

    python -m bench.phase2 [--rounds 3] [--tokens 24]

Appended as one JSON line to ``bench/results.jsonl``:

* **Compile time**, cold (an empty cache directory) and cached, for the
  decode graph.
* **Kernel calls per token** and distinct kernels, from the lowered program.
* **Bytes moved per token**, the compulsory traffic counted from the loop IR,
  as a polynomial in ``past`` and evaluated at the benchmark's cache length.
* **Arena** size against the sum of every intermediate with no reuse.
* **Speed**, nanoinfer against the generated code, alternating in one
  process, and a per-primitive breakdown of one decode step.

This is the baseline the later phases are measured against: one kernel per
primitive, scalar loops, one thread.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bench.phase1 import BLAS_ENV, PROMPTS, RESULTS, decode_steps
from nanocompile.lower import lower
from nanocompile.models.qwen2 import Qwen2, Qwen2Config
from nanocompile.runtime import CBackend, CompiledGraph
from tests import oracle


def compile_times(graph) -> dict:
    saved = os.environ.get("NANOCOMPILE_CACHE")
    # Windows cannot delete a DLL that is still loaded, and ctypes never
    # unloads one, so the scratch cache is left for the OS to clean up.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as empty:
        os.environ["NANOCOMPILE_CACHE"] = empty
        try:
            cold = CompiledGraph(graph)
            warm = CompiledGraph(graph)
        finally:
            if saved is None:
                os.environ.pop("NANOCOMPILE_CACHE")
            else:
                os.environ["NANOCOMPILE_CACHE"] = saved
        assert not cold.cached and warm.cached
        result = {"cold_s": round(cold.compile_seconds, 2), "cached_s": round(warm.compile_seconds, 3)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=24)
    args = parser.parse_args()

    model_dir = oracle.model_dir()
    oracle.import_nanoinfer()
    from nanoinfer.model import Qwen2 as Reference
    from nanoinfer.tokenizer import Tokenizer

    reference = Reference.from_model_dir(model_dir)
    tokenizer = Tokenizer.from_model_dir(model_dir)
    config = Qwen2Config.from_model_dir(model_dir)
    backend = CBackend()
    model = Qwen2(config, oracle.weights_by_name(reference.weights), run=backend)

    decode_graph = model.graph_last
    compiled = compile_times(decode_graph)
    print(f"compile: {compiled['cold_s']} s cold, {compiled['cached_s']} s cached")

    program = lower(decode_graph)
    prompt_ids = tokenizer.encode(PROMPTS[0])
    past = len(prompt_ids) + args.tokens // 2           # a mid-generation step
    bytes_formula = program.bytes_moved
    step_bound = {"seq": 1, "past": past}
    step_bytes = bytes_formula if isinstance(bytes_formula, int) else bytes_formula.evaluate(step_bound)
    print(f"calls per token: {len(program.calls)}, distinct kernels: {len(program.kernels)}")
    print(f"bytes per run: {bytes_formula}  ->  {step_bytes / 1e9:.3f} GB at seq=1, past={past}")

    capacity = len(prompt_ids) + args.tokens

    def nanoinfer_engine():
        cache = reference.new_cache(capacity)
        return lambda ids, first: reference.next_token_logits(np.array(ids), cache=cache)

    def compiled_engine():
        cache = model.new_cache(capacity)
        return lambda ids, first: model.forward(ids, cache=cache, last_only=True)[0]

    engines = {"nanoinfer": nanoinfer_engine, "generated C": compiled_engine}
    steps = {name: [] for name in engines}
    outputs = {}
    for round_index in range(args.rounds):
        for name, make in engines.items():
            _, times, generated = decode_steps(make(), prompt_ids, args.tokens)
            steps[name].extend(times)
            outputs.setdefault(name, generated)
            print(f"round {round_index + 1} {name:<12} median step {statistics.median(times) * 1e3:8.1f} ms")
    identical = outputs["nanoinfer"] == outputs["generated C"]

    # One profiled decode step at the same cache length.
    runner = backend.compiled(decode_graph)
    cache = model.new_cache(capacity)
    model.forward(list(range(100, 100 + past)), cache=cache)
    keys, values = cache.past()
    profile: dict[str, float] = {}
    started = time.perf_counter()
    runner({"token_ids": np.array([5]), "past_keys": keys, "past_values": values},
           model.weights, profile=profile)
    profiled_total = time.perf_counter() - started
    plan = runner.last_plan

    speed = {}
    for name, samples in steps.items():
        median = statistics.median(samples)
        speed[name] = {
            "median_step_ms": round(median * 1e3, 1),
            "min_step_ms": round(min(samples) * 1e3, 1),
            "max_step_ms": round(max(samples) * 1e3, 1),
            "decode_tok_s": round(1 / median, 3),
        }
    ratio = speed["generated C"]["median_step_ms"] / speed["nanoinfer"]["median_step_ms"]
    bandwidth = step_bytes / (speed["generated C"]["median_step_ms"] / 1e3) / 1e9

    print()
    for name, s in speed.items():
        print(f"{name:<12} {s['median_step_ms']:>8.1f} ms  (min {s['min_step_ms']}, max {s['max_step_ms']})"
              f"  {s['decode_tok_s']:.3f} tok/s")
    print(f"generated C / nanoinfer: {ratio:.2f}x; tokens identical: {identical}")
    print(f"effective bandwidth: {bandwidth:.2f} GB/s")
    print(f"arena {plan.arena_bytes / 1e6:.2f} MB vs {plan.temp_bytes / 1e6:.2f} MB without reuse")
    print(f"profiled step {profiled_total * 1e3:.0f} ms:")
    kernel_total = sum(profile.values())
    for op, seconds in sorted(profile.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {op:<12} {seconds * 1e3:8.1f} ms")
    print(f"  {'(outside kernels)':<12} {(profiled_total - kernel_total) * 1e3:8.1f} ms")

    record = {
        "phase": 2,
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": platform.processor() or platform.machine(),
        "cpu_threads": os.cpu_count(),
        "blas_env": {k: os.environ.get(k) for k in BLAS_ENV},
        "rounds": args.rounds,
        "tokens": args.tokens,
        "compile": compiled,
        "calls_per_token": len(program.calls),
        "distinct_kernels": len(program.kernels),
        "bytes_per_run": str(bytes_formula),
        "bytes_at_step": step_bytes,
        "step_bindings": step_bound,
        "arena_bytes": plan.arena_bytes,
        "temp_bytes_without_reuse": plan.temp_bytes,
        "speed": speed,
        "generated_over_nanoinfer": round(ratio, 2),
        "effective_gb_s": round(bandwidth, 2),
        "tokens_identical": identical,
        "profile_ms": {k: round(v * 1e3, 2) for k, v in sorted(profile.items(), key=lambda kv: -kv[1])},
        "profiled_step_ms": round(profiled_total * 1e3, 1),
    }
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"appended to {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
