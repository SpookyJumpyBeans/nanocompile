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
| 4 | Schedules: tiling, AVX2 vectorization, a persistent thread pool; matmul against OpenBLAS and nanoinfer's Rust kernel | **done**: bitwise identical to phase 3, 3.1x faster, matmul at the roofline (199 of 200.5 GFLOP/s) |
| 5 | Autotuning over the schedule space, on a machine known to be noisy | **done**: bitwise identical, 4% faster decode, winner unstable on 7 of 20 kernels and reported |
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
OPENBLAS_NUM_THREADS=6 python -m bench.phase4
OPENBLAS_NUM_THREADS=6 python -m bench.phase5
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

### Phase 4: schedules

Schedules are rewrites of the loop IR, kept apart from what a kernel computes
(`nanocompile/schedule.py`): `split`, `reorder`, `unroll`, `privatize`,
`vectorize`, `parallel` and `pack`. Each acts on loops by name, checks that it
is legal where it applies, and raises otherwise. Each kernel class gets a
hand-written recipe built from them (`nanocompile/schedules.py`), with
fallbacks down to the kernel as fusion wrote it. Vector loops become AVX2
(`codegen_avx2.py`), parallel loops run on a persistent thread pool
(`pool.c`), and both are on by default.

The roadmap expected this phase to vectorize the matmul by reordering its
k sum, and to pay for that with a tolerance. It does not reorder anything.
The parallelism comes from *across* outputs instead. A projection is tiled
2 rows by 32 columns, and `privatize` moves the tile's 64 accumulators inside
the k loop, where they are independent sums and sit in eight AVX2
registers. Each output's sum still runs over k in order:

```
for jo in N / 32, in parallel:             32 output columns per task
    for io in seq / 2:                      2 rows per tile
        acc[2][4] = 0                       8 vector registers
        for k in K:
            a0, a1 = broadcast x[2io, k], x[2io+1, k]
            for v in 0..3:                  w = packed W[jo*4 + v, k, 0..7]
                acc[0][v] += a0 * w         one multiply, one add, no FMA
                acc[1][v] += a1 * w
        epilogue on the tile, 8 columns at a time
```

The weight is *packed* so the eight columns one vector covers are adjacent:
row-major `W[j, k]` puts them 896 floats apart, and each vector would be
eight scalar loads. The runtime packs each weight once, on first use. An
index simplifier that knows loop ranges is what turns the packed index back
into `base + lane`, so the vectorizer sees a contiguous load. A decode step
has one row, so it runs the tile's one-row tail, four vectors wide.

Correctness:

| Check | Result |
|---|---|
| A grid of schedules over every kernel shape fusion emits, at sizes leaving every tail length | **bitwise identical** to the unscheduled kernel |
| Every primitive test from phase 2, fused and scheduled vs unfused | **bitwise identical** |
| Scheduled vs phase 3 model, real weights: prompt, cached decode steps and the cache | **bitwise identical** |
| Parallel schedules at 1, 2, 3 and 8 threads | bitwise identical |
| Real model, greedy tokens | identical to nanoinfer |

Bitwise because no schedule reorders the operations behind a value, and AVX2
renders each lane as the scalar code would: IEEE add, multiply, divide and
square root are exactly rounded either way, FMA contraction stays off, and
`exp`, `sin` and `cos` call the same libm function per lane. What has no
exact vector form is refused: integer arithmetic that differs between lanes
(RoPE's half-split, the causal mask's compare), a value carried between
lanes, a reduction. Those kernels run scalar, in parallel where they can.

The real model found a bug the tests had not. The attention context product
reads V through the KV cache's concat, `t < past ? cache[t] : new[t - past]`.
Scalar C loads only the branch it takes; the first AVX2 select was a blend,
which loads both, and for `t < past` the new token's index is negative. The
tiny test model never vectorized that kernel (its heads are 4 wide), and the
real one crashed reading outside a buffer. A select whose condition every
lane shares is now a C `?:` on vectors, and one whose condition differs by
lane may only blend values that read no operand.

**The machine**, measured on six threads by `bench/probe.c` with the
instruction mix the kernels use: **34.5 GB/s** read bandwidth and
**200.5 GFLOP/s** of multiply plus separate add. A decode step reads 1.98 GB
of float32 weights, so 57 ms is the floor on a step, whatever the kernels do.

Every matmul shape in the model, 24 copies of each weight read in turn so a
decode-size call streams from memory as it does in a decode step (median per
copy, alternating):

| Shape | Rows | scheduled C | phase 3 C | NumPy | Rust int8 | roofline |
|---|---:|---:|---:|---:|---:|---:|
| q, o: 896 x 896 | 1 | 109 us | 746 us | 171 us | 77 us | 93 us |
| k, v: 128 x 896 | 1 | 28 us | 107 us | 22 us | 70 us | 13 us |
| gate, up: 4864 x 896 | 1 | 562 us | 4,088 us | 643 us | 454 us | 506 us |
| down: 896 x 4864 | 1 | 714 us | 4,408 us | 730 us | 548 us | 506 us |
| LM head: 151936 x 896 | 1 | 17.3 ms | 137.1 ms | 18.7 ms | 4.8 ms | 15.8 ms |
| q, o: 896 x 896 | 64 | **546 us** | 38.3 ms | 816 us | | 513 us |
| k, v: 128 x 896 | 64 | **125 us** | 5.8 ms | 305 us | | 73 us |
| gate, up: 4864 x 896 | 64 | **3.02 ms** | 237.6 ms | 3.39 ms | | 2.78 ms |
| down: 896 x 4864 | 64 | 3.64 ms | 272.5 ms | 3.50 ms | | 2.78 ms |
| LM head: 151936 x 896 | 64 | **87.6 ms** | 7.71 s | 102.1 ms | | 86.9 ms |

- **At 64 rows the tile is compute-bound and at the roofline.** The LM head
  runs at **199.0 GFLOP/s against a measured peak of 200.5**, and the scheduled
  kernels beat OpenBLAS on four of the five shapes, `down` being the one they
  lose, by 4%. Against phase 3's scalar loops, the same arithmetic in the
  same order is **46 to 88x faster**.
- **At one row it is bound by memory, like everything else.** The large
  shapes reach 91% of the read bandwidth (the LM head's 545 MB in 17.3 ms),
  level with NumPy. The small ones (`q`, `k`) sit above the roofline: 28 us
  of work per call is not much more than what the pool takes to start and
  finish a task.
- **nanoinfer's Rust kernel is faster at one row because it reads a quarter
  of the bytes.** It is int8; this is float32. The roadmap's comparison is
  not like for like until phase 6 makes the same trade.

The model, nanoinfer, phase 3 and scheduled C alternating in one process,
3 rounds of 24 tokens:

| | Median step | Min | Max | Decode | Prefill |
|---|---:|---:|---:|---:|---:|
| nanoinfer | 103.6 ms | 77.5 ms | 289.3 ms | 9.65 tok/s | 270.0 ms |
| phase 3 C | 343.5 ms | 323.4 ms | 392.7 ms | 2.91 tok/s | 1,664.5 ms |
| scheduled C | 112.4 ms | 80.0 ms | 346.2 ms | **8.89 tok/s** | **92.2 ms** |

**3.1x faster than phase 3, and level with nanoinfer within this machine's
noise** (8% slower at the median, a similar minimum). Time to first token is
**2.9x faster than nanoinfer's**, because prefill is where the tile's
compute-bound rate counts. One profiled step:

| | |
|---:|---|
| 94.7 ms | matmuls, at the memory bandwidth |
| 0.8 ms | every other kernel together |
| 13.2 ms | outside the kernels: 319 ctypes calls from Python |

The step is now the weights' read time plus Python. What is left in the
kernels is bytes, which phase 6 quarters. What is left outside them is the
13 ms of Python, which phase 7 removes by compiling the step into one call.

**Six threads.** The pool's default comes from a thread sweep of the decode
step. 4 to 8 threads are one plateau within noise (two runs put the best at
6 and at 8), since six already reach the memory bandwidth. Past 8 it gets
steadily worse, to 3.3x slower at 20, as workers land on E-cores and on the
hyperthread siblings of the thread that runs Python between kernels.

**OpenBLAS was measured at six threads too**, and the results record it.
Its default on this laptop is one thread per logical CPU, and at twenty it
was erratic: the same `[64, 4864] @ [4864, 896]` took 2.6 ms on six threads
and anywhere from 0.2 to 2.4 s on twenty, before nanocompile's pool existed
in the process. A first run of this benchmark under that default showed
NumPy almost 400x slower on `down` and a 72-second prefill for nanoinfer; it was
discarded rather than published as a win.

### Phase 5: autotuning

Phase 4's recipes are now functions of a few knobs (`nanocompile/schedules.py`).
A projection's are column vectors per tile, rows per tile, whether to pack the
weight, whether to unroll, and whether to run on the pool; a row or
elementwise kernel's are whether to vectorize and whether to go parallel.
Phase 4's hand-written schedule is one point of each space, and the refactor
reproduces it exactly: the default-scheduled C of the real model is the same
text, byte for byte, as before.

`nanocompile/autotune.py` searches each kernel's space on this machine:

- **The space** is every configuration that applies, counted once per distinct
  generated kernel: 130 for a projection, 2 to 26 for everything else.
- **Cost is measured.** Candidates are compiled into one library and timed by
  a C loop, in alternation: every round times each candidate once, starting
  one later each round, and the cost is the median over rounds. Weights rotate
  through enough copies to stream from memory, as a decode step's do. The
  workload is a decode step plus a 32-token prefill per token, so a knob only
  prefill uses (rows per tile) is chosen by measurement.
- **The search is seeded**: a random sample of 8 with the hand-written
  schedule, then greedy refinement, one knob one step at a time. A final run
  times the winner against the hand-written schedule, and it is kept only if
  it is more than 3% faster in that run.
- **The cache** (`tuning.json`) is keyed by machine (compiler, CPU, threads)
  and by kernel (its C, its operands, the workloads); a hit applies the
  recorded winner without measuring.

Correctness:

| Check | Result |
|---|---|
| Every configuration of every kernel in phase 4's cases, at sizes leaving every tail length | **bitwise identical** to the unscheduled kernel, and no write outside the output |
| Tuned programs through phase 4's own `check` | **bitwise identical** |
| Tuned tiny model, and tuned real model from the recorded winners | **bitwise identical** to phase 3 |
| Every candidate the tuner compiles, before it is timed | checked bitwise against the unscheduled kernel, with guard regions either side of the output |

The guard regions came from a bug. The full suite crashed once in four runs
with heap corruption, and the cause was a test that handed one kernel
another's operands: it wrote 60 floats into a 39-float output, and the
bitwise check passed, because the 39 it compared were right. A write past an
output corrupts its neighbours without changing the output, so the tuner now
checks for that separately, and a test proves it catches one.

**Per kernel**, the real model, the winner against the hand-written schedule
in the final comparison:

| Kernel | Calls/step | Space | Tried | Tuned vs hand | Winners in 3 runs |
|---|---:|---:|---:|---:|---:|
| RoPE on k (bias add, rotate) | 24 | 2 | 2 | **15.8x** | **1**: serial |
| RoPE on q | 24 | 2 | 2 | **5.9x** | **1**: serial |
| KV cache concats | 1 + 1 | 4 | 4 | **3.0x, 3.2x** | **1**: serial |
| `k_proj`, 128 x 896 | 24 | 130 | 20 | 1.43x | 3 |
| `v_proj` + bias | 24 | 130 | 25 | 1.15x | 3 |
| `down` + residual | 23 | 130 | 28 | 1.06x | 2 |
| LM head | 1 | 130 | 22 | 1.04x | 2 |
| `q_proj`, `o_proj`, `gate`/`up`, the last `down`, both attention matmuls | 1 to 24 | 2 to 130 | 2 to 32 | hand-written kept | 1 or 2 |
| norms, softmax, embedding gather, RoPE tables | 1 to 48 | 2 to 4 | 2 to 4 | hand-written kept | 1 or 2 |

Everything the search found is small work. A decode step's RoPE kernels
touch 128 and 896 floats, and waking six threads costs more than doing them
on one; the hand-written recipe ran them on the pool because every kernel
with more than one row did. The `k_proj` and `v_proj` gains are phase 4's own
open question answered: a 128-wide output in 32-column blocks is 4 tasks for
6 threads, and 8-column blocks are 16.

**How often the winner changes**, three runs with the same seed (so the same
sample, and any difference is the machine): on **7 of 20** kernels. Every
kernel whose win was 3x or more chose the same winner every time. The ones
that changed are all matmuls or a row kernel with differences of 0 to 6%,
except `k_proj`, whose 1.43x appeared in one run of three: the other two kept
the hand-written tile or changed only its rows. On a 20 us decode matmul,
this machine's noise is the size of the differences between schedules, and
a search can only report that, not resolve it. The refinement path, which
follows measurements, differed on 9 of 20.

**Reproducibility**, the gate: every run's 20 searches were replayed on
their own recorded costs, 60 in all, and each tried exactly the candidates
it had tried. The sample is the same in every run by construction.

**Search time**: 674, 531 and 535 s for the whole model. The LM head is
most of it, 382 s, because each of its samples streams 545 MB.

**The model**, phase 4's schedules against the first run's winners, in a fresh
process, the engine going first swapping every round, 6 rounds of 24 tokens:

| | Median step | Min | Decode | Prefill |
|---|---:|---:|---:|---:|
| hand-written (phase 4) | 75.9 ms | 69.7 ms | 13.17 tok/s | 78.6 ms |
| tuned | 72.8 ms | 65.0 ms | **13.74 tok/s** | 82.5 ms |

**4% faster per decode step, tuned ahead in 6 of 6 rounds** (sign test,
one-sided p = 0.016), tokens identical. Prefill is level within noise.
Which kernels the 4% comes from is not separable here: the stable RoPE and
concat wins, or the first run's matmul winners, which the other runs did not
all repeat.

**A first measurement was discarded.** Timed in the process that had just
spent half an hour tuning, with the hand-written model always first, it
showed the tuned model **1.46x** faster and a 2.2 s hand-written prefill
against phase 4's 92 ms. The per-kernel gains could not add up to that, so
the benchmark now times the model in a fresh process and swaps the order;
the tuning statistics from that run stand, and its model timing is marked
discarded in `bench/results.jsonl`.

```
python -m bench.phase5                 # tune 3 times, then time the model
python -m bench.phase5 --model-only    # time the model from bench/phase5_tuning.json
```

In code, `CompiledGraph(graph, schedule=Autotuner())` tunes on first compile
and reads the cache after that.

## Measuring

Same rules as nanoinfer, for the same reason: this laptop has stalled for
seconds at a time under sustained load.

- Medians and minima, never means, with the spread printed beside them.
- A/B comparisons alternate inside one process.
- Head-to-head results are judged with an exact sign test over paired runs.
- Thread counts and the machine are recorded with every result, BLAS's included.
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

Parallel loops use six threads unless `NANOCOMPILE_THREADS` says otherwise
(or `nanocompile.runtime.set_threads(n)` at run time).

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
