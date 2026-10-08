"""Phase 1 measurements: how big the graph is, and what interpreting it costs.

    python -m bench.phase1 [--rounds 5] [--tokens 24]

Three things, appended as one JSON line to ``bench/results.jsonl``:

* **Graph size**, counted exactly. Nodes per transformer block is the
  difference between a 2-layer and a 1-layer build at the real model's
  widths, so the embedding, mask, RoPE tables and output projection are
  excluded rather than amortized.
* **Agreement** with nanoinfer on the real weights: the largest absolute logit
  difference over the test prompts.
* **Speed**, nanoinfer against the interpreter, alternating in one process
  with the same prompt and the same number of tokens. Medians and the spread.
  The interpreter is an oracle, not a backend, so this is a baseline: the gap
  is the cost of dispatching ~3,600 nodes from Python instead of ~25 NumPy
  calls per layer.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from nanocompile.models.qwen2 import Qwen2, Qwen2Config, build_qwen2
from tests import oracle

RESULTS = Path(__file__).resolve().parent / "results.jsonl"
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
]
BLAS_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def graph_size(config: Qwen2Config) -> dict:
    one = build_qwen2(replace(config, num_hidden_layers=1))
    two = build_qwen2(replace(config, num_hidden_layers=2))
    full = build_qwen2(config)

    block = Counter(n.op for n in two.nodes) - Counter(n.op for n in one.nodes)
    return {
        "nodes_total": len(full.nodes),
        "nodes_per_block": sum(block.values()),
        "nodes_outside_blocks": len(one.nodes) - sum(block.values()),
        "block_histogram": dict(block.most_common()),
    }


def decode_steps(step, prompt_ids: list[int], tokens: int) -> tuple[float, list[float], list[int]]:
    """Prefill time, per-step decode times, and the tokens, for one engine.

    ``step(ids, first)`` runs one forward and returns the next-token logits.
    """
    started = time.perf_counter()
    logits = step(prompt_ids, True)
    prefill = time.perf_counter() - started

    generated, times = [int(np.argmax(logits))], []
    for _ in range(tokens - 1):
        started = time.perf_counter()
        logits = step([generated[-1]], False)
        times.append(time.perf_counter() - started)
        generated.append(int(np.argmax(logits)))
    return prefill, times, generated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=24)
    args = parser.parse_args()

    model_dir = oracle.model_dir()
    if model_dir is None:
        print("real weights not found; see tools/check_env.py", file=sys.stderr)
        return 1
    oracle.import_nanoinfer()
    from nanoinfer.model import Qwen2 as Reference
    from nanoinfer.tokenizer import Tokenizer

    reference = Reference.from_model_dir(model_dir)
    tokenizer = Tokenizer.from_model_dir(model_dir)
    config = Qwen2Config.from_model_dir(model_dir)
    model = Qwen2(config, oracle.weights_by_name(reference.weights))

    size = graph_size(config)
    print(f"graph: {size['nodes_total']} nodes, {size['nodes_per_block']} per block")
    print("block:", ", ".join(f"{op} {n}" for op, n in size["block_histogram"].items()))

    max_diff = 0.0
    for prompt in PROMPTS:
        ids = np.array(tokenizer.encode(prompt))
        max_diff = max(max_diff, float(np.abs(model.forward(ids) - reference.forward(ids)).max()))
    print(f"max |logit difference| vs nanoinfer over {len(PROMPTS)} prompts: {max_diff:.3g}")

    prompt_ids = tokenizer.encode(PROMPTS[0])
    capacity = len(prompt_ids) + args.tokens

    def nanoinfer_engine():
        cache = reference.new_cache(capacity)
        return lambda ids, first: reference.next_token_logits(np.array(ids), cache=cache)

    def interpreter_engine():
        cache = model.new_cache(capacity)
        return lambda ids, first: model.forward(ids, cache=cache, last_only=True)[0]

    engines = {"nanoinfer": nanoinfer_engine, "interpreter": interpreter_engine}
    samples = {name: {"prefill": [], "step": []} for name in engines}
    outputs = {}
    for round_index in range(args.rounds):
        for name, make in engines.items():
            prefill, steps, generated = decode_steps(make(), prompt_ids, args.tokens)
            samples[name]["prefill"].append(prefill)
            samples[name]["step"].extend(steps)
            outputs.setdefault(name, generated)
            print(f"round {round_index + 1} {name:<12} median step {statistics.median(steps) * 1e3:7.1f} ms")

    identical = outputs["nanoinfer"] == outputs["interpreter"]
    speed = {}
    for name, s in samples.items():
        step = statistics.median(s["step"])
        speed[name] = {
            "median_step_ms": round(step * 1e3, 2),
            "min_step_ms": round(min(s["step"]) * 1e3, 2),
            "max_step_ms": round(max(s["step"]) * 1e3, 2),
            "decode_tok_s": round(1 / step, 2),
            "median_prefill_ms": round(statistics.median(s["prefill"]) * 1e3, 1),
        }
    ratio = speed["interpreter"]["median_step_ms"] / speed["nanoinfer"]["median_step_ms"]

    print()
    print(f"{'':<12} {'step (median)':>14} {'min':>9} {'max':>9} {'tok/s':>7} {'prefill':>9}")
    for name, s in speed.items():
        print(
            f"{name:<12} {s['median_step_ms']:>11.1f} ms {s['min_step_ms']:>6.1f} ms "
            f"{s['max_step_ms']:>6.1f} ms {s['decode_tok_s']:>7.2f} {s['median_prefill_ms']:>6.0f} ms"
        )
    print(f"interpreter / nanoinfer step time: {ratio:.2f}x; tokens identical: {identical}")

    record = {
        "phase": 1,
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": platform.processor() or platform.machine(),
        "cpu_threads": os.cpu_count(),
        "blas_env": {k: os.environ.get(k) for k in BLAS_ENV},
        "numpy": np.__version__,
        "rounds": args.rounds,
        "tokens": args.tokens,
        "prompt_tokens": len(prompt_ids),
        "graph": size,
        "max_logit_diff_vs_nanoinfer": max_diff,
        "tokens_identical": identical,
        "speed": speed,
        "interpreter_over_nanoinfer": round(ratio, 3),
    }
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"appended to {RESULTS.relative_to(Path.cwd()) if RESULTS.is_relative_to(Path.cwd()) else RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
