# nanocompile

A tensor compiler for the transformer that [nanoinfer](https://github.com/SpookyJumpyBeans/nanoinfer)
runs by hand.

nanoinfer computes Qwen2.5-0.5B-Instruct with code I wrote and tuned myself:
NumPy for most of the forward pass, AVX2 and AVX-VNNI kernels in Rust for the
int8 projections, every phase verified against a reference and the result
benchmarked against llama.cpp. nanocompile takes the same model, written as a
graph of 26 primitive operations, and generates the kernels instead.
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
| 1 | Graph IR with symbolic shapes, Qwen2 in the frontend, reference interpreter; logits within 1e-3 of nanoinfer | **done**: bitwise identical to nanoinfer, tokens identical |
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
Graph IR                         SSA over 26 primitives, dtypes, symbolic dims
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

## Results

```
python -m bench.phase1
```

### Phase 1: graph IR and reference interpreter

Qwen2 is written once in the frontend (`nanocompile/models/qwen2.py`) and
traced into a graph of primitives. One graph serves the prompt and every
decode step: its shapes are in terms of `seq` and `past`, and an uncached pass
is just `past = 0`. A NumPy interpreter defines what the graph means.

Correctness, against nanoinfer on the real 494M weights:

| Check | Result |
|---|---|
| Max absolute logit difference, 3 prompts | **0** (gate: 1e-3) |
| Greedy tokens, 24 per prompt | identical on all 3 prompts |
| Tiny 2-layer model logits | bitwise identical (gate: 1e-5) |
| Cached KV entries vs nanoinfer's cache | bitwise identical, every layer |
| Cached decode vs uncached | identical tokens |

**The gate was 1e-3 and the difference is zero.** That is not tolerance
being generous; the two engines do the same arithmetic. Every composite in
`nn.py` performs nanoinfer's float32 operations in nanoinfer's order (the
mean square is a sum then a divide, `-|x|` in the sigmoid is a select, RoPE's
angles are the same `f32` products), and the interpreter hands NumPy the same
arrays, including a transposed view for every `x @ W.T`, so BLAS takes the
same path. The tiny-model test asserts bitwise equality on its own, separate
from the gate, so that if a later change reorders arithmetic the gate still
passes and that test says exactly what was given up.

The one place equality is not claimed: the last-row logits from a `last_only`
pass differ from row `-1` of a full pass by 6e-8. A one-row matmul takes a
different BLAS path from an eleven-row one. nanoinfer slices at the same
point, and against nanoinfer's `last_only` the result is bitwise identical.

The graph, counted exactly:

| | |
|---|---:|
| Nodes, whole model | 3,656 |
| Nodes per transformer block | 150 |
| Nodes outside the blocks | 56 |

Per block: reshape 30, broadcast 24, mul 13, weight 12, permute 12, add 11,
matmul 9, const 8, slice 6, div 4, concat 4, reduce_sum 3, neg 3, sqrt 2,
reciprocal 2, where 2, exp 2, less_equal 1, reduce_max 1, sub 1.

**54 of the 150 nodes are `reshape` and `broadcast`.** That is the frontend
making every NumPy broadcast explicit: a bias add, a norm's gain vector and
the epsilon each become a reshape and a broadcast before the add. They cost
nothing in the interpreter, where they are views, and they are what phase 3
has to look through when it fuses an elementwise chain into one loop. Only 9
nodes are matmuls (seven projections and the two attention products), and
those are where the time goes.

Speed, nanoinfer and the interpreter alternating in one process, 5 rounds of
24 tokens:

| | Median step | Min | Max | Decode |
|---|---:|---:|---:|---:|
| nanoinfer | 185.4 ms | 125.0 ms | 258.2 ms | 5.39 tok/s |
| interpreter | 293.4 ms | 169.7 ms | 423.3 ms | 3.41 tok/s |

The interpreter is **1.58x slower**, and it is an oracle, not a backend, so
that is the expected kind of number. A per-primitive breakdown of one decode
step (226.6 ms on that run) shows where the gap comes from:

| | |
|---:|---|
| 124.9 ms | matmul: the same BLAS calls nanoinfer makes |
| 26.6 ms | broadcast and reshape views (`np.broadcast_to` costs ~29 us a call) |
| ~20 ms | every other primitive |
| ~55 ms | the interpreter itself: dispatch and the per-node dtype and shape check, 3,656 times |

The shape check stays. It is how the interpreter catches a node that does not
produce what shape inference promised, which already found one bug while
writing the tests: `np.ascontiguousarray` turns a 0-d array into shape `(1,)`.

Both engines are slower here than nanoinfer's own published 82.9 ms step,
because this machine was busy during the run. The ratio is the measurement;
alternating the engines in one process is what keeps it meaningful.

**26 primitives, not the ~15 the roadmap planned.** `iota`, `dim`, `sin` and
`cos` were added so that positions, the causal mask and the RoPE tables are
computed inside the graph from `past` and `seq`, instead of being handed in as
inputs the driver has to keep consistent with the cache.

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
skip without them. `pytest -m "not slow"` runs everything else in seconds.

To print the IR for one block at the real model's widths:

```
python tools/dump_graph.py --layers 1
```

## Layout

```
nanocompile/   the compiler; imports numpy and nothing else
tests/         differential tests; nanoinfer is the oracle
tools/         environment checks, glue that loads weights through nanoinfer
bench/         benchmark scripts and their appended results
docs/          the roadmap and its gates
```
