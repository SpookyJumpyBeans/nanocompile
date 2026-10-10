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
| 2 | Loop IR, one kernel per primitive, C codegen through gcc; tokens identical to nanoinfer | **done**: tokens identical, 0.97 tok/s naive baseline (6.0x slower) |
| 3 | Fusion: elementwise, reductions, matmul epilogues; bytes moved per token, counted exactly | **done**: bitwise identical to unfused, 62 to 13 kernels per block, 1.18x faster |
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
python -m bench.phase2
python -m bench.phase3
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

### Phase 2: loop IR and naive C codegen

Every compute primitive is lowered to one loop nest in a small loop IR,
emitted as C, compiled by gcc into one shared library, and called over ctypes
(`nanocompile/lower.py`, `codegen_c.py`, `runtime.py`). Movement primitives
cost nothing: `reshape`, `permute`, `broadcast` and `slice` become strided
views, so the `W.T` in every projection is the weight read with its strides
swapped and a broadcast bias is a stride of zero. A reshape that a view cannot
express gets an explicit `copy` kernel, counted like any other. All
intermediates live in one arena, reused by liveness.

To see what it generates: `python tools/dump_c.py --layers 1`.

Correctness:

| Check | Result |
|---|---|
| Every primitive vs the interpreter, `seq` in {1, 3, 7} | bitwise, except `exp`/`sin`/`cos` and sums (within 2e-6) |
| Real model, greedy tokens, 24 per prompt | **identical to nanoinfer** on all 3 prompts |
| Real model, logits | within 1e-3 (3.6e-4 observed) |
| Compiling the same graph twice | byte-identical C |
| Every pair of live buffers in the arena | never overlapping |

Phase 1 was bitwise and this is not, for one reason: the generated matmul adds
its products in index order, and NumPy hands the same matmul to BLAS, which
does not. The compile flags rule out everything else that would change a
result (`-ffp-contract=off` forbids fusing a multiply and an add into an FMA,
and there is no `-ffast-math`, so no sum is reordered), which is why every
elementwise kernel still matches bit for bit.

One decode step, as compiled:

| | |
|---|---:|
| Kernel calls per token | 1,514 |
| Distinct kernels | 63 |
| Compile, cold / cached | 4.98 s / 0.70 s |
| Arena for all intermediates | 0.13 MB (10.22 MB without reuse) |

63 distinct kernels for 1,514 calls is the 24 layers sharing code: a kernel is
its loop IR, and two layers' kernels are the same IR. A 25th layer would add
calls and no C.

**Bytes moved per run**, counted exactly from the loop IR as a polynomial in
the two symbols:

```
18966*past*seq + 18966*seq*seq + 417792*past + 16357208*seq + 1976743344
```

Each term says something:

- **1,976,743,344**: every float32 weight, read exactly once. This is the
  check that views work. If a single projection copied its transposed
  weight, this term would grow by that weight's size twice over.
- **417,792 per cached token**: the KV cache holds 24,576 bytes per token,
  and each step moves **17x** that. The concat that appends the new key reads
  and rewrites the whole history, and `repeat_kv` materializes seven copies of
  every key and value for the attention matmuls to read back. Phase 3 can fold
  the repeat into the matmul's indexing instead.
- **16,357,208 per new token** and the `seq` products: activations and the
  attention scores, which grow with the prompt.

Speed, nanoinfer and the generated code alternating in one process, 3 rounds
of 24 tokens:

| | Median step | Min | Max | Decode |
|---|---:|---:|---:|---:|
| nanoinfer | 171.0 ms | 140.7 ms | 862.8 ms | 5.85 tok/s |
| generated C | 1,031.9 ms | 970.1 ms | 1,276.2 ms | **0.97 tok/s** |

**6.0x slower than nanoinfer**, with identical tokens. A profiled step shows
where the time goes:

| | |
|---:|---|
| 920.0 ms | matmul |
| 9.6 ms | the other 1,297 kernel calls together |
| 72.8 ms | outside the kernels: 1,514 ctypes calls from Python, the plan, input checks |

**It is not the memory.** 2.0 GB in 1.03 s is 1.94 GB/s, a twelfth of the
~24 GB/s nanoinfer's phase 4 measured this machine's decode step reaching. The
matmul is bound by latency instead. Each output is `acc = acc + a * b` down a
row of 896 or 4,864, and every add waits for the one before it, about 494
million times per token. gcc could break that chain with several accumulators
or vector lanes, but only by reordering the sum, and the flags forbid exactly
that. Phase 4 reorders it deliberately, with the tolerance that costs.

The 72.8 ms outside the kernels is Python calling 1,514 functions one at a
time. It is the cost phase 7 removes by compiling the whole step into one
call.

**One change from the roadmap:** the compile cache works per library, not per
kernel. The 63 distinct kernels compile together in under 5 seconds, and
loading 63 separate libraries would cost more than recompiling the rare one
that changed.

### Phase 3: fusion

Two graph passes first, both bitwise: common subexpression elimination
merges what the frontend repeats (every `+ eps` built its own constant,
reshape and broadcast), and constant folding evaluates anything computed from
constants alone. Then fusion (`nanocompile/fusion.py`) decides which values
become buffers at all. Matmuls, gathers and outputs do. An elementwise value
is inlined into the one kernel that consumes it and never stored; reductions
fold into the kernel that reads them row by row; a matmul's bias, residual
add or activation is applied as its result is stored. Index-only primitives
never become buffers: a kernel reads through them with strides, a divide and
modulo, or a select.

Correctness:

| Check | Result |
|---|---|
| Fused vs unfused C, every primitive test from phase 2 | **bitwise identical** |
| Fused vs unfused C, whole model with cached steps and the cache itself | **bitwise identical** |
| Real model, greedy tokens, 24 per prompt | identical to nanoinfer on all 3 prompts |
| Optimized vs original graph in the interpreter | bitwise identical |

Bitwise, not within a tolerance, because fusion moves operations between
kernels and never reorders the ones that produce a value. A row kernel sums
in the same order the separate reduce kernel did; a matmul epilogue adds the
bias to the same finished accumulator. That makes "fused equals unfused" an
exact test, and it caught a real bug while it was being written: the C
printer rendered `-(a + b)` as `-a + b`, which phase 2 never exercised
because a negation's operand had always been a single load.

What a transformer block became:

| | unfused | fused |
|---|---:|---:|
| Kernel calls per block | 62 | **13** |
| Kernel calls per token | 1,514 | 319 |
| Distinct kernels | 63 | 20 |

The 13 per block are rms_norm, the q, k and v projections, RoPE on q and on
k, attention scores, softmax, the context matmul, the output projection with
its residual add, rms_norm, `gate` and `up` with SwiGLU, and the down
projection with its residual add.

Bytes moved per run, both counted from the loop IR:

```
unfused  18966*past*seq + 18966*seq*seq + 417792*past + 16357208*seq + 1976743344
fused     5376*past*seq +  5376*seq*seq +  24576*past +  2644488*seq + 1976742656
```

- **`past`: 417,792 to 24,576 bytes per cached token**, and 24,576 is the
  KV cache's own size per token. Each cached key and value is now read
  exactly once per step, and a test asserts that equality. `repeat_kv` and
  the per-layer `concat` became index arithmetic inside the attention
  kernels: a key head is `h / 7`, the history and the new token are a select.
- **`seq`: 6.2x less** per new token. Activations that used to be written by
  one kernel and read by the next now live in registers.
- **The constant term barely moved**, and that is the point it makes: it is
  the weights, read once each, and fusion cannot shrink them. At a decode
  step (`seq` = 1, `past` = 17) the total goes from 2.0005 GB to 1.9799 GB,
  1% less. Decode was never limited by activation traffic.

Speed, nanoinfer, unfused and fused C alternating in one process, 3 rounds of
24 tokens:

| | Median step | Min | Max | Decode |
|---|---:|---:|---:|---:|
| nanoinfer | 96.7 ms | 84.9 ms | 115.1 ms | 10.34 tok/s |
| unfused C | 294.2 ms | 273.0 ms | 337.3 ms | 3.40 tok/s |
| fused C | 249.3 ms | 223.3 ms | 294.5 ms | **4.01 tok/s** |

**Fusion is 1.18x faster, and almost none of that is the bytes.** With 1%
less traffic, the gain had to come from somewhere else, so I measured it
directly at decode size:

| `[1, 896]` against `[4864, 896]` weights | time |
|---|---:|
| one matmul, one accumulator | 2.79 ms |
| `gate` and `up` fused, two accumulators | 3.32 ms |

Twice the arithmetic in 1.19x the time. Phase 2 found the matmul bound by a
chain of dependent adds; the fused kernel runs two independent chains in one
loop, and the CPU overlaps them. Unfused, `gate` and `up` cost 2 x 2.79 ms
per layer; fused they cost 3.32, which saves about 54 ms over 24 layers,
roughly the whole measured gain of 45 ms. Phase 4 applies the same idea on
purpose, with more accumulators and vector lanes.

**These numbers are not comparable with phase 2's.** Both C paths and
nanoinfer ran far faster in this session than in phase 2's (unfused C: 294 ms
here, 1,032 ms there), and not by the same factor: unfused C over nanoinfer
was 6.0x then and 3.0x now. The machine was loaded during phase 2's run. Only
comparisons within one run mean anything, which is why every benchmark here
alternates its engines.

The fused patterns alone, against nanoinfer's NumPy on the same arrays
(median of 200 calls, alternating):

| Pattern | Shape | nanoinfer | fused C |
|---|---|---:|---:|
| rms_norm | [1, 896] | 11.2 us | 21.1 us |
| rms_norm | [64, 896] | 52.8 us | 57.4 us |
| silu(gate) * up | [1, 4864] | 18.4 us | 41.9 us |
| silu(gate) * up | [64, 4864] | 4,418.7 us | **3,487.7 us** |
| softmax | [14, 1, 128] | 20.8 us | 31.7 us |
| softmax | [14, 64, 64] | 149.0 us | 414.3 us |

Mostly a loss, and honestly so. At decode sizes a standalone compiled call
pays about 20 us of Python argument checking and planning before its kernel
runs, which these patterns do not have to spare; inside the model that cost
is paid once per step, not per pattern. Only SwiGLU at prefill size wins,
where one pass replaces NumPy's eight. Softmax loses badly: the row kernel
computes `exp` twice per element (once for the sum, once for the output),
and its max and sum loops are scalar, because the NaN-propagating max and
the in-order sum cannot be vectorized without changing a result. The
profile says none of this matters yet: of a 246 ms fused step, every
non-matmul kernel together takes 0.3 ms.

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

To print the IR, and the C it compiles to, for one block at the real model's widths:

```
python tools/dump_graph.py --layers 1
python tools/dump_c.py --layers 1
```

## Layout

```
nanocompile/   the compiler; imports numpy and nothing else
tests/         differential tests; nanoinfer is the oracle
tools/         environment checks, glue that loads weights through nanoinfer
bench/         benchmark scripts and their appended results
docs/          the roadmap and its gates
```
