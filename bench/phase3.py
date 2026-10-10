"""Phase 3 measurements: what fusion removed, and what it bought.

    python -m bench.phase3 [--rounds 3] [--tokens 24]

Appended as one JSON line to ``bench/results.jsonl``:

* **Kernels**: calls per token and per transformer block, unfused and fused.
* **Bytes moved per run**, both polynomials, and both evaluated at a
  mid-generation decode step.
* **Speed**: nanoinfer, unfused C and fused C alternating in one process, and
  a per-kind breakdown of one fused step.
* **The fused norm and activations alone**, against nanoinfer's NumPy
  versions on the same arrays: rms_norm, softmax and SwiGLU's
  ``silu(gate) * up``, at decode and prefill sizes.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np

from bench.phase1 import BLAS_ENV, PROMPTS, RESULTS, decode_steps
from nanocompile import nn
from nanocompile.dtypes import f32
from nanocompile.frontend import GraphBuilder
from nanocompile.fusion import fuse
from nanocompile.lower import lower
from nanocompile.models.qwen2 import Qwen2, Qwen2Config, build_qwen2
from nanocompile.passes import optimize
from nanocompile.runtime import CBackend, CompiledGraph
from tests import oracle


def _bytes_at(formula, bound) -> int:
    return formula if isinstance(formula, int) else formula.evaluate(bound)


def _alternate(fns: dict, repeats: int) -> dict[str, float]:
    """Median seconds per call, the candidates interleaved call by call."""
    samples: dict[str, list[float]] = {name: [] for name in fns}
    for name, fn in fns.items():
        fn()                                      # warm: compile, page in
    for _ in range(repeats):
        for name, fn in fns.items():
            started = time.perf_counter()
            fn()
            samples[name].append(time.perf_counter() - started)
    return {name: statistics.median(s) for name, s in samples.items()}


def microbenchmarks(ops) -> list[dict]:
    """Each fused pattern as its own compiled graph, against nanoinfer's NumPy."""
    rng = np.random.default_rng(0)
    results = []

    def compiled(build, inputs):
        b = GraphBuilder()
        graph = b.build(build(b))
        runner = CompiledGraph(graph)
        weights = {}
        return lambda: runner(inputs, weights)

    for seq in (1, 64):
        x = rng.standard_normal((seq, 896)).astype(np.float32)
        w = rng.standard_normal(896).astype(np.float32)

        def build_norm(b, seq=seq):
            t = b.input("x", f32, (b.symbol("seq"), 896))
            return {"y": nn.rms_norm(t, b.input("w", f32, (896,)), 1e-6)}

        times = _alternate({
            "nanoinfer": lambda: ops.rms_norm(x, w, 1e-6),
            "fused C": compiled(build_norm, {"x": x, "w": w}),
        }, 200)
        results.append({"pattern": "rms_norm", "shape": [seq, 896], **_us(times)})

        gate = rng.standard_normal((seq, 4864)).astype(np.float32)
        up = rng.standard_normal((seq, 4864)).astype(np.float32)

        def build_swiglu(b):
            s = b.symbol("seq")
            return {"y": nn.silu(b.input("g", f32, (s, 4864))) * b.input("u", f32, (s, 4864))}

        times = _alternate({
            "nanoinfer": lambda: ops.silu(gate) * up,
            "fused C": compiled(build_swiglu, {"g": gate, "u": up}),
        }, 200)
        results.append({"pattern": "silu(gate) * up", "shape": [seq, 4864], **_us(times)})

    for seq, keys in ((1, 128), (64, 64)):
        scores = rng.standard_normal((14, seq, keys)).astype(np.float32)

        def build_softmax(b, seq=seq, keys=keys):
            return {"y": nn.softmax(b.input("s", f32, (14, b.symbol("seq"), keys)))}

        times = _alternate({
            "nanoinfer": lambda: ops.softmax(scores, axis=-1),
            "fused C": compiled(build_softmax, {"s": scores}),
        }, 200)
        results.append({"pattern": "softmax", "shape": [14, seq, keys], **_us(times)})
    return results


def _us(times: dict[str, float]) -> dict:
    return {f"{name}_us": round(t * 1e6, 1) for name, t in times.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=24)
    args = parser.parse_args()

    model_dir = oracle.model_dir()
    oracle.import_nanoinfer()
    from nanoinfer import ops
    from nanoinfer.model import Qwen2 as Reference
    from nanoinfer.tokenizer import Tokenizer

    reference = Reference.from_model_dir(model_dir)
    tokenizer = Tokenizer.from_model_dir(model_dir)
    config = Qwen2Config.from_model_dir(model_dir)
    weights = oracle.weights_by_name(reference.weights)
    fused_backend, unfused_backend = CBackend(), CBackend(fuse=False)
    fused_model = Qwen2(config, weights, run=fused_backend)
    unfused_model = Qwen2(config, weights, run=unfused_backend)

    decode_graph = fused_model.graph_last
    fused_program = fuse(optimize(decode_graph))
    unfused_program = lower(decode_graph)
    per_block = {}
    for name, make in (("unfused", lower), ("fused", lambda g: fuse(optimize(g)))):
        one = make(build_qwen2(replace(config, num_hidden_layers=1), last_only=True))
        two = make(build_qwen2(replace(config, num_hidden_layers=2), last_only=True))
        per_block[name] = len(two.calls) - len(one.calls)

    prompt_ids = tokenizer.encode(PROMPTS[0])
    past = len(prompt_ids) + args.tokens // 2
    bound = {"seq": 1, "past": past}
    traffic = {
        "unfused": (str(unfused_program.bytes_moved), _bytes_at(unfused_program.bytes_moved, bound)),
        "fused": (str(fused_program.bytes_moved), _bytes_at(fused_program.bytes_moved, bound)),
    }
    print(f"calls per token: {len(unfused_program.calls)} -> {len(fused_program.calls)}; "
          f"per block {per_block['unfused']} -> {per_block['fused']}; "
          f"distinct kernels {len(unfused_program.kernels)} -> {len(fused_program.kernels)}")
    for name, (formula, at_step) in traffic.items():
        print(f"bytes {name:<8} {formula}  ->  {at_step:,} at seq=1, past={past}")

    capacity = len(prompt_ids) + args.tokens

    def engine(model):
        def make():
            cache = model.new_cache(capacity)
            return lambda ids, first: model.forward(ids, cache=cache, last_only=True)[0]
        return make

    def nanoinfer_engine():
        cache = reference.new_cache(capacity)
        return lambda ids, first: reference.next_token_logits(np.array(ids), cache=cache)

    engines = {"nanoinfer": nanoinfer_engine, "unfused C": engine(unfused_model), "fused C": engine(fused_model)}
    steps = {name: [] for name in engines}
    outputs = {}
    for round_index in range(args.rounds):
        for name, make in engines.items():
            _, times, generated = decode_steps(make(), prompt_ids, args.tokens)
            steps[name].extend(times)
            outputs.setdefault(name, generated)
            print(f"round {round_index + 1} {name:<10} median step {statistics.median(times) * 1e3:8.1f} ms")
    identical = len({tuple(v) for v in outputs.values()}) == 1

    runner = fused_backend.compiled(decode_graph)
    cache = fused_model.new_cache(capacity)
    fused_model.forward(list(range(100, 100 + past)), cache=cache)
    keys, values = cache.past()
    profile: dict[str, float] = {}
    started = time.perf_counter()
    runner({"token_ids": np.array([5]), "past_keys": keys, "past_values": values}, weights, profile=profile)
    profiled = time.perf_counter() - started

    speed = {}
    for name, samples in steps.items():
        median = statistics.median(samples)
        speed[name] = {
            "median_step_ms": round(median * 1e3, 1),
            "min_step_ms": round(min(samples) * 1e3, 1),
            "max_step_ms": round(max(samples) * 1e3, 1),
            "decode_tok_s": round(1 / median, 3),
        }
    print()
    for name, s in speed.items():
        print(f"{name:<10} {s['median_step_ms']:>8.1f} ms (min {s['min_step_ms']}, max {s['max_step_ms']})"
              f"  {s['decode_tok_s']:.3f} tok/s")
    print(f"tokens identical across all three: {identical}")
    print(f"profiled fused step {profiled * 1e3:.0f} ms:")
    for kind, seconds in sorted(profile.items(), key=lambda kv: -kv[1]):
        print(f"  {kind:<16} {seconds * 1e3:8.1f} ms")
    print(f"  {'(outside kernels)':<16} {(profiled - sum(profile.values())) * 1e3:8.1f} ms")

    micro = microbenchmarks(ops)
    print("\nfused patterns alone (median of 200, alternating):")
    for row in micro:
        print(f"  {row['pattern']:<16} {str(row['shape']):<16} nanoinfer {row['nanoinfer_us']:>9.1f} us"
              f"   fused C {row['fused C_us']:>9.1f} us")

    record = {
        "phase": 3,
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": platform.processor() or platform.machine(),
        "cpu_threads": os.cpu_count(),
        "blas_env": {k: os.environ.get(k) for k in BLAS_ENV},
        "rounds": args.rounds,
        "tokens": args.tokens,
        "calls_per_token": {"unfused": len(unfused_program.calls), "fused": len(fused_program.calls)},
        "calls_per_block": per_block,
        "distinct_kernels": {"unfused": len(unfused_program.kernels), "fused": len(fused_program.kernels)},
        "bytes_per_run": {k: v[0] for k, v in traffic.items()},
        "bytes_at_step": {k: v[1] for k, v in traffic.items()},
        "step_bindings": bound,
        "speed": speed,
        "tokens_identical": identical,
        "profile_ms": {k: round(v * 1e3, 2) for k, v in sorted(profile.items(), key=lambda kv: -kv[1])},
        "profiled_step_ms": round(profiled * 1e3, 1),
        "microbenchmarks": micro,
    }
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"appended to {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
