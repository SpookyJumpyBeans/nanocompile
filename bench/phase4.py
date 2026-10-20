"""Phase 4 measurements: schedules on every matmul shape, and on the model.

    python -m bench.phase4 [--rounds 3] [--tokens 24]

Appended as one JSON line to ``bench/results.jsonl``:

* **The machine**: read bandwidth and peak arithmetic rate, measured by
  ``bench/probe.c`` on nanocompile's thread pool. They are the roofline.
* **Every matmul shape in the model**, at decode (1 row) and prefill (64
  rows): scheduled C, phase 3's unscheduled C and NumPy's OpenBLAS,
  alternating. Each shape runs as 24 copies, one per layer, read in turn,
  so a weight is streamed from memory as it is in a decode step rather than
  sitting in cache from the previous call.
* **Per-layer projection time** at decode size, against nanoinfer's Rust
  int8 kernel on the same shapes, the comparison nanoinfer's own kernel
  table makes.
* **Thread count**: one decode step at each thread count.
* **The model**: nanoinfer, phase 3 C and scheduled C alternating in one
  process, tokens checked identical, and a per-kind profile of one step.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bench.phase1 import BLAS_ENV, PROMPTS, RESULTS, decode_steps
from nanocompile import codegen_c, nn, runtime
from nanocompile.dtypes import f32
from nanocompile.frontend import GraphBuilder
from nanocompile.models.qwen2 import Qwen2, Qwen2Config
from nanocompile.runtime import CBackend, CompiledGraph
from tests import oracle

LAYERS = 24
# (name, out features, in features, copies): every projection per layer, and the LM head once.
SHAPES = [
    ("q_proj", 896, 896, LAYERS),
    ("k_proj", 128, 896, LAYERS),
    ("o_proj", 896, 896, LAYERS),
    ("gate", 4864, 896, LAYERS),
    ("down", 896, 4864, LAYERS),
    ("lm_head", 151936, 896, 1),
]


def alternate(fns: dict, repeats: int) -> dict[str, list[float]]:
    """Seconds per call, the candidates interleaved call by call."""
    samples: dict[str, list[float]] = {name: [] for name in fns}
    for fn in fns.values():
        fn()
    for _ in range(repeats):
        for name, fn in fns.items():
            started = time.perf_counter()
            fn()
            samples[name].append(time.perf_counter() - started)
    return samples


# -- the machine -----------------------------------------------------------------


def machine() -> dict:
    source = (Path(__file__).with_name("probe.c")).read_text(encoding="utf-8")
    path, _ = codegen_c.build_library(source)
    probe = ctypes.CDLL(str(path))
    probe.nc_bind_pool.argtypes = [ctypes.c_void_p]
    probe.nc_bind_pool(ctypes.cast(runtime._pool().nc_pool_run, ctypes.c_void_p))
    probe.nc_read.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    probe.nc_compute.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]

    buf = np.ones(256 * 2**20, dtype=np.float32)            # 1 GiB, far past any cache
    per = 2**18
    chunks = buf.size // per
    sums = np.empty(chunks, dtype=np.float32)
    reads = []
    for _ in range(10):
        started = time.perf_counter()
        probe.nc_read(buf.ctypes.data, chunks, per, sums.ctypes.data)
        reads.append(time.perf_counter() - started)
    assert np.all(sums == per)

    tasks, iterations = 4 * runtime.threads(), 2_000_000
    out = np.empty(tasks, dtype=np.float32)
    computes = []
    for _ in range(5):
        started = time.perf_counter()
        probe.nc_compute(out.ctypes.data, tasks, iterations)
        computes.append(time.perf_counter() - started)
    flops = tasks * iterations * 16 * 8 * 2
    return {"read_gb_s": round(buf.nbytes / min(reads) / 1e9, 1),
            "peak_gflop_s": round(flops / min(computes) / 1e9, 1)}


# -- matmul shapes ---------------------------------------------------------------


def stacked(n_out: int, n_in: int, copies: int):
    """``copies`` independent projections of one input, as a decode step runs them."""
    b = GraphBuilder()
    x = b.input("x", f32, (b.symbol("seq"), n_in))
    return b.build({f"y{c}": nn.linear(x, b.weight(f"w{c}", (n_out, n_in))) for c in range(copies)})


def roofline_s(rows: int, n_out: int, n_in: int, bw: float, peak: float) -> float:
    traffic = 4 * (n_out * n_in + rows * n_in + rows * n_out)
    return max(traffic / bw, 2 * rows * n_out * n_in / peak)


def matmul_shapes(repeats: int, bw: float, peak: float, reference_kernels) -> list[dict]:
    rng = np.random.default_rng(0)
    rows_out = []
    for name, n_out, n_in, copies in SHAPES:
        graph = stacked(n_out, n_in, copies)
        weights = {f"w{c}": rng.standard_normal((n_out, n_in)).astype(np.float32) for c in range(copies)}
        scheduled, plain = CompiledGraph(graph), CompiledGraph(graph, schedule=None)
        for rows in (1, 64):
            x = rng.standard_normal((rows, n_in)).astype(np.float32)
            ws = list(weights.values())
            fns = {
                "scheduled C": lambda: scheduled({"x": x}, weights),
                "phase 3 C": lambda: plain({"x": x}, weights),
                "numpy": lambda: [x @ w.T for w in ws],
            }
            if rows == 1 and reference_kernels is not None:
                kernels, quantized = reference_kernels
                qs = [quantized(w) for w in ws]
                v = x[0]
                fns["nanoinfer rust int8"] = lambda: [kernels.matvec_i8(q.values, q.scales, v) for q in qs]
            samples = alternate(fns, repeats if rows == 1 else max(3, repeats // 4))
            flops = 2 * rows * n_out * n_in
            row = {"shape": name, "out": n_out, "in": n_in, "rows": rows, "copies": copies,
                   "roofline_us": round(roofline_s(rows, n_out, n_in, bw, peak) * 1e6, 1)}
            for engine, s in samples.items():
                per_copy = statistics.median(s) / copies
                row[f"{engine}_us"] = round(per_copy * 1e6, 1)
                row[f"{engine}_gflop_s"] = round(flops / per_copy / 1e9, 1)
            rows_out.append(row)
            print(f"{name:<8} rows={rows:<3} " + "  ".join(
                f"{e} {row[e + '_us']:>8.1f} us ({row[e + '_gflop_s']:>6.1f} GF/s)" for e in samples)
                + f"  roofline {row['roofline_us']:.1f} us")
        del scheduled, plain, weights
    return rows_out


# -- the model -------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--skip-shapes", action="store_true")
    args = parser.parse_args()

    model_dir = oracle.model_dir()
    oracle.import_nanoinfer()
    from nanoinfer import kernels as rust
    from nanoinfer.model import Qwen2 as Reference
    from nanoinfer.quantization import quantize_int8
    from nanoinfer.tokenizer import Tokenizer

    hw = machine()
    print(f"threads {runtime.threads()}: read {hw['read_gb_s']} GB/s, peak {hw['peak_gflop_s']} GFLOP/s (mul + add)")
    bw, peak = hw["read_gb_s"] * 1e9, hw["peak_gflop_s"] * 1e9

    reference_kernels = (rust, quantize_int8) if rust.available() else None
    shapes = [] if args.skip_shapes else matmul_shapes(args.repeats, bw, peak, reference_kernels)

    reference = Reference.from_model_dir(model_dir)
    tokenizer = Tokenizer.from_model_dir(model_dir)
    config = Qwen2Config.from_model_dir(model_dir)
    weights = oracle.weights_by_name(reference.weights)
    scheduled_backend = CBackend()
    scheduled_model = Qwen2(config, weights, run=scheduled_backend)
    plain_model = Qwen2(config, weights, run=CBackend(schedule=None))

    prompt_ids = tokenizer.encode(PROMPTS[0])
    capacity = len(prompt_ids) + args.tokens

    def engine(model):
        def make():
            cache = model.new_cache(capacity)
            return lambda ids, first: model.forward(ids, cache=cache, last_only=True)[0]
        return make

    def nanoinfer_engine():
        cache = reference.new_cache(capacity)
        return lambda ids, first: reference.next_token_logits(np.array(ids), cache=cache)

    # Threads: the scheduled engine alone, one short generation per count.
    default_threads = runtime.threads()
    sweep = {}
    for n in (1, 2, 4, 6, 8, 12, 16, 20):
        runtime.set_threads(n)
        _, times, _ = decode_steps(engine(scheduled_model)(), prompt_ids, 12)
        sweep[n] = round(statistics.median(times) * 1e3, 1)
        print(f"threads {n:>2}: median decode step {sweep[n]} ms")
    runtime.set_threads(default_threads)

    engines = {"nanoinfer": nanoinfer_engine, "phase 3 C": engine(plain_model),
               "scheduled C": engine(scheduled_model)}
    steps = {name: [] for name in engines}
    prefill = {name: [] for name in engines}
    outputs = {}
    for round_index in range(args.rounds):
        for name, make in engines.items():
            first, times, generated = decode_steps(make(), prompt_ids, args.tokens)
            steps[name].extend(times)
            prefill[name].append(first)
            outputs.setdefault(name, generated)
            print(f"round {round_index + 1} {name:<12} median step {statistics.median(times) * 1e3:8.1f} ms")
    identical = len({tuple(v) for v in outputs.values()}) == 1

    decode_graph = scheduled_model.graph_last
    runner = scheduled_backend.compiled(decode_graph)
    past = len(prompt_ids) + args.tokens // 2
    cache = scheduled_model.new_cache(capacity)
    scheduled_model.forward(list(range(100, 100 + past)), cache=cache)
    keys, values = cache.past()
    profile: dict[str, float] = {}
    started = time.perf_counter()
    runner({"token_ids": np.array([5]), "past_keys": keys, "past_values": values}, weights, profile=profile)
    profiled = time.perf_counter() - started

    speed = {}
    for name, samples in steps.items():
        median = statistics.median(samples)
        speed[name] = {"median_step_ms": round(median * 1e3, 1), "min_step_ms": round(min(samples) * 1e3, 1),
                       "max_step_ms": round(max(samples) * 1e3, 1), "decode_tok_s": round(1 / median, 3),
                       "median_prefill_ms": round(statistics.median(prefill[name]) * 1e3, 1)}
    print()
    for name, s in speed.items():
        print(f"{name:<12} {s['median_step_ms']:>8.1f} ms (min {s['min_step_ms']}, max {s['max_step_ms']})"
              f"  {s['decode_tok_s']:.2f} tok/s   prefill {s['median_prefill_ms']} ms")
    print(f"tokens identical across all three: {identical}")
    print(f"profiled scheduled step {profiled * 1e3:.1f} ms:")
    for kind, seconds in sorted(profile.items(), key=lambda kv: -kv[1]):
        print(f"  {kind:<16} {seconds * 1e3:8.2f} ms")
    print(f"  {'(outside kernels)':<16} {(profiled - sum(profile.values())) * 1e3:8.2f} ms")

    record = {
        "phase": 4,
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": platform.processor() or platform.machine(),
        "cpu_threads": os.cpu_count(),
        "pool_threads": default_threads,
        "blas_env": {k: os.environ.get(k) for k in BLAS_ENV},
        "rust_kernels": reference_kernels is not None,
        "roofline": hw,
        "matmul_shapes": shapes,
        "thread_sweep_ms": sweep,
        "rounds": args.rounds,
        "tokens": args.tokens,
        "speed": speed,
        "tokens_identical": identical,
        "profile_ms": {k: round(v * 1e3, 3) for k, v in sorted(profile.items(), key=lambda kv: -kv[1])},
        "profiled_step_ms": round(profiled * 1e3, 2),
    }
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"appended to {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
