# nanocompile

A tensor compiler for the transformer that [nanoinfer](https://github.com/SpookyJumpyBeans/nanoinfer)
runs by hand.

nanoinfer computes Qwen2.5-0.5B-Instruct with code I wrote and tuned myself:
NumPy for most of the forward pass, AVX2 and AVX-VNNI kernels in Rust for the
int8 projections, every phase verified against a reference and the result
benchmarked against llama.cpp. nanocompile takes the same model, written as a
graph of about fifteen primitive operations, and generates the kernels instead.
It fuses them, tiles, vectorizes and parallelizes the loops, emits C, compiles
it with gcc and loads the result over ctypes.

The question it exists to answer: **how close does a small compiler get to
kernels written and tuned by hand, and where does it beat them?** There are
places it should. nanoinfer's `rms_norm` is five NumPy calls, each a full pass
over memory, on a decode step phase 4 measured as bandwidth-bound. A compiler
that fuses them makes one.

The compiler's runtime dependencies are `numpy` and a C compiler. nanoinfer is
the correctness oracle and the source of weights and the tokenizer; tests and
tools use it, and `nanocompile/` never imports it.

## Status

| Phase | What | State |
|------:|------|-------|
| 1 | Graph IR with symbolic shapes, Qwen2 in the frontend, reference interpreter; logits within 1e-3 of nanoinfer | not started |
| 2 | Loop IR, one kernel per primitive, C codegen through gcc; tokens identical to nanoinfer | not started |
| 3 | Fusion: elementwise, reductions, matmul epilogues; bytes moved per token, counted exactly | not started |
| 4 | Schedules: tiling, AVX2 vectorization, a persistent thread pool; matmul against OpenBLAS and nanoinfer's Rust kernel | not started |
| 5 | Autotuning over the schedule space, on a machine known to be noisy | not started |
| 6 | INT8 lowering with AVX-VNNI; bitwise agreement with nanoinfer's int8 kernel | not started |
| 7 | Whole-model compilation: one library, one call per token, a static memory plan | not started |
| 8 | Benchmark against nanoinfer and llama.cpp on identical hardware | not started |

Each phase is verified before the next one starts, and each has a written gate
in [`docs/ROADMAP.md`](docs/ROADMAP.md). A phase that misses its gate says so
here rather than being quietly redefined.

## Pipeline

```
Qwen2 in the frontend            a Python graph builder; rms_norm, softmax, RoPE
        │                        and attention are compositions of primitives
        ▼
Graph IR                         SSA over ~15 primitives, dtypes, symbolic dims
        │                        (seq, past) so one graph serves every step
        │   passes: fusion (3), quantization (6)
        ▼
Loop IR                          loop nests over buffers, plus a schedule:
        │                        split, reorder, vectorize, parallel (4, 5)
        ▼
C  ──►  gcc -O3  ──►  shared library  ──►  ctypes
```

## How correctness is checked

Every layer of the compiler is checked against the layer above it, and the top
of the chain is code that has already been checked against `transformers`:

```
transformers  ──(nanoinfer phase 3: 6.2e-05)──►  nanoinfer
nanoinfer     ──(phase 1: 1e-3, identical tokens)──►  graph interpreter
interpreter   ──(phase 2 on: every primitive, random and symbolic shapes)──►  generated C
```

The interpreter is never deleted or optimized. It is the definition of what a
graph means, and every pass after phase 1 is tested by running the graph
before and after it.

Where results are claimed equal, the claim is exactly as strong as the
arithmetic allows. A pass that reorders a float32 sum changes the bits, and
saying "identical" there would be false; integer accumulation in phase 6 does
not, and there the claim is bitwise.

## Measuring

Same rules as nanoinfer, for the same reason: this laptop has stalled for
seconds at a time under sustained load.

- Medians and minima, never means, with the spread printed beside them.
- A/B comparisons alternate inside one process.
- Head-to-head results are judged with an exact sign test over paired runs.
- Thread counts and the machine are recorded with every result.
- Work (bytes moved, kernels launched, FLOPs) is counted exactly from the IR
  and asserted in tests. Wall time lives in `bench/`.

## Hardware

| | |
|---|---|
| CPU | Intel i7-12700H, 14 cores / 20 threads, AVX2, AVX-VNNI |
| RAM | 16 GB |
| Compiler | gcc 15.2 (MSYS2 UCRT64) |

## Running

nanocompile expects nanoinfer checked out next to it, with its model
downloaded. Set `NANOINFER_PATH` or `NANOCOMPILE_MODEL` to use other locations.

```
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt
python tools/check_env.py
pytest
```

Tests that need the real 494M-parameter weights are marked `reference` and
skip without them.

## Layout

```
nanocompile/   the compiler; imports numpy and nothing else
tests/         differential tests; nanoinfer is the oracle
tools/         environment checks, glue that loads weights through nanoinfer
bench/         benchmark scripts and their appended results
docs/          the roadmap and its gates
```
