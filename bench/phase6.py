"""Phase 6 measurements: int8 weights, against float32 and against nanoinfer's int8.

    python -m bench.phase6 --part speed [--rounds 6] [--tokens 24]
    python -m bench.phase6 --part perplexity

Each part appends one JSON line to ``bench/results.jsonl``.

**Speed** (``part: speed``): five engines in one fresh process, a generation
each per round, the order rotating every round so no engine always runs
first or last: nanocompile at float32, int8 and int8-vnni (phase 4's hand-
written schedules for all three, so the difference is the weights), and
nanoinfer's int8 engine with float activations and with VNNI. nanoinfer runs
at the 4 rayon threads its own README measured fastest for these kernels;
nanocompile at its pool's 6. Each engine's tokens must match its own first
generation, and nanocompile's int8 tokens must match nanoinfer's for the same
mode.

**Perplexity** (``part: perplexity``): nanoinfer's held-out text, its first
512 tokens in chunks of 128 as nanoinfer's ``tools/measure_quantization.py``
scores it, so the numbers are comparable with its README: nanoinfer's
simulated int8 (the 23.3078), its two int8 kernels, nanocompile's
interpreter (simulated, like nanoinfer's), and nanocompile's C at float32,
int8 and int8-vnni.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone

from bench.phase1 import BLAS_ENV, PROMPTS, RESULTS, decode_steps
from nanocompile import quantize, runtime
from nanocompile.models.qwen2 import Qwen2, Qwen2Config, build_qwen2
from nanocompile.runtime import CBackend
from tests import oracle

NANOINFER_THREADS = 4
HELD_OUT_TOKENS = 512
CHUNK = 128


def header() -> dict:
    return {
        "phase": 6,
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": platform.processor() or platform.machine(),
        "cpu_threads": os.cpu_count(),
        "pool_threads": runtime.threads(),
        "blas_env": {k: os.environ.get(k) for k in BLAS_ENV},
    }


def append(record: dict) -> None:
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"appended to {RESULTS}")


def load():
    model_dir = oracle.model_dir()
    oracle.import_nanoinfer()
    from nanoinfer import kernels
    from nanoinfer.model import Qwen2 as Reference
    from nanoinfer.quantization import quantize_model
    from nanoinfer.tokenizer import Tokenizer

    reference = Reference.from_model_dir(model_dir)
    config = Qwen2Config.from_model_dir(model_dir)
    weights = oracle.weights_by_name(reference.weights)
    return reference, config, weights, kernels, quantize_model, Reference, Tokenizer.from_model_dir(model_dir)


# -- speed -------------------------------------------------------------------------


def speed(args) -> None:
    reference, config, weights, kernels, quantize_model, Reference, tokenizer = load()
    threads = kernels._LIBRARY.nanoinfer_set_threads(NANOINFER_THREADS)
    stored, _ = quantize_model(reference.weights, bits=8, quantize_embeddings=False, dequantize=False)
    nanoinfer_int8 = Reference(stored)
    del reference

    # One int8 copy for both int8 engines: this is a 16 GB machine, and a
    # benchmark that pages measures the disk.
    weights = {**weights, **quantize.quantized_weights(quantize.quantize(build_qwen2(config), "f32"), weights)}
    ours = {name: Qwen2(config, weights, run=CBackend(), quantization=q)
            for name, q in (("nanocompile fp32", None), ("nanocompile int8", "int8"),
                            ("nanocompile int8-vnni", "int8-vnni"))}
    prompt_ids = tokenizer.encode(PROMPTS[0])
    capacity = len(prompt_ids) + args.tokens

    def ours_run(model):
        cache = model.new_cache(capacity)
        return decode_steps(lambda ids, first: model.forward(ids, cache=cache, last_only=True)[0],
                            prompt_ids, args.tokens)

    def theirs_run(vnni):
        import numpy as np

        kernels.use_vnni(vnni)
        kernels.vnni_block(quantize.DEFAULT_BLOCK)
        cache = nanoinfer_int8.new_cache(capacity)
        try:
            return decode_steps(lambda ids, first: nanoinfer_int8.next_token_logits(np.asarray(ids), cache=cache),
                                prompt_ids, args.tokens)
        finally:
            kernels.use_vnni(False)

    engines = {**{name: (lambda m=m: ours_run(m)) for name, m in ours.items()},
               "nanoinfer int8": lambda: theirs_run(False),
               "nanoinfer int8-vnni": lambda: theirs_run(True)}
    names = list(engines)
    outputs = {name: run()[2] for name, run in engines.items()}       # compile, load, warm
    steps = {name: [] for name in names}
    prefill = {name: [] for name in names}
    for r in range(args.rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        for name in order:
            first, times, generated = engines[name]()
            assert generated == outputs[name], name
            steps[name].extend(times)
            prefill[name].append(first)
        print(f"round {r + 1}: " + ", ".join(f"{n.split()[-1] if n.startswith('nanocompile') else n} "
                                              f"{statistics.median(steps[n][-(args.tokens - 1):]) * 1e3:.1f}"
                                              for n in names))
    agree = {"int8": outputs["nanocompile int8"] == outputs["nanoinfer int8"],
             "int8-vnni": outputs["nanocompile int8-vnni"] == outputs["nanoinfer int8-vnni"]}

    result = {}
    print()
    for name in names:
        median = statistics.median(steps[name])
        result[name] = {"median_step_ms": round(median * 1e3, 2), "decode_tok_s": round(1 / median, 2),
                        "min_step_ms": round(min(steps[name]) * 1e3, 2),
                        "median_prefill_ms": round(statistics.median(prefill[name]) * 1e3, 1)}
        print(f"{name:<24} {result[name]['median_step_ms']:7.2f} ms/token  {result[name]['decode_tok_s']:6.2f} tok/s"
              f"   prefill {result[name]['median_prefill_ms']:7.1f} ms")
    print(f"tokens identical to nanoinfer: {agree}")
    append({**header(), "part": "speed", "rounds": args.rounds, "tokens": args.tokens,
            "prompt_tokens": len(prompt_ids), "nanoinfer_threads": threads, "speed": result,
            "tokens_identical_to_nanoinfer": agree})


# -- perplexity --------------------------------------------------------------------


def perplexity(args) -> None:
    reference, config, weights, kernels, quantize_model, Reference, tokenizer = load()
    from nanoinfer import perplexity as ppl

    ids = ppl.held_out_tokens(tokenizer, limit=HELD_OUT_TOKENS)
    scores = {}

    def score(name, model):
        started = time.perf_counter()
        result = ppl.evaluate(model, ids, CHUNK)
        scores[name] = {"perplexity": result.perplexity, "tokens_scored": result.tokens_scored}
        print(f"{name:<32} {result.perplexity:.6f}  ({time.perf_counter() - started:.0f} s)", flush=True)

    simulated, _ = quantize_model(reference.weights, bits=8, quantize_embeddings=False)
    score("nanoinfer int8 simulated", Reference(simulated))
    del simulated
    stored, _ = quantize_model(reference.weights, bits=8, quantize_embeddings=False, dequantize=False)
    engine = Reference(stored)
    kernels.use_vnni(False)
    score("nanoinfer int8 kernel", engine)
    kernels.use_vnni(True)
    kernels.vnni_block(quantize.DEFAULT_BLOCK)
    score("nanoinfer int8-vnni kernel", engine)
    kernels.use_vnni(False)
    del engine, stored
    score("nanocompile int8 interpreter", Qwen2(config, weights, quantization="int8"))
    score("nanocompile fp32", Qwen2(config, weights, run=CBackend()))
    score("nanocompile int8", Qwen2(config, weights, run=CBackend(), quantization="int8"))
    score("nanocompile int8-vnni", Qwen2(config, weights, run=CBackend(), quantization="int8-vnni"))
    append({**header(), "part": "perplexity", "tokens": HELD_OUT_TOKENS, "chunk": CHUNK, "scores": scores})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=("speed", "perplexity"), required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--tokens", type=int, default=24)
    args = parser.parse_args()
    {"speed": speed, "perplexity": perplexity}[args.part](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
