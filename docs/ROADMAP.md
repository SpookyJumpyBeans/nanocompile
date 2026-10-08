# Roadmap

Each phase has four parts. **Build** is the scope. **Gate** is what has to pass
before the next phase starts; it is written before the code, and it is not
edited afterwards to fit what the code does. **Measure** is what gets published
in the README. **Not here** is scope deliberately left out, so a later phase
can pick it up knowingly.

Phases build strictly on each other. Phase 2 lowers phase 1's graph, phase 3
rewrites it, and so on; nothing is thrown away.

---

## Phase 1: graph IR and reference interpreter

**Build**

- A graph IR in SSA form: every value is produced once, by one node, and has a
  dtype and a shape known when the node is built.
- Dtypes: `f32`, `i32`, `i64`, `i8`, `bool`.
- Symbolic dimensions. A shape entry is an integer or an affine expression over
  named symbols (`seq`, `past`, `past + seq`), so the same graph serves the
  prompt and every decode step without being rebuilt. Shape inference works on
  these expressions and rejects a mismatch when the node is built, not when it
  runs.
- About fifteen primitives, chosen so that everything above them is a
  composition:
  - elementwise: `add`, `sub`, `mul`, `div`, `max`, `neg`, `exp`, `sqrt`,
    `reciprocal`, `where`, `cast`, comparisons
  - reductions: `sum`, `max` over one axis
  - movement: `reshape`, `permute`, `broadcast`, `slice`, `concat`
  - `matmul`, kept primitive because phases 4 and 6 schedule it specially
  - `gather`, for the embedding lookup
  - graph inputs, named weights, constants
- A frontend: tensor-like Python objects whose operators build nodes.
  `rms_norm`, `silu`, `softmax`, RoPE, grouped-query attention and the causal
  mask are frontend functions that decompose into primitives. That is what
  gives phase 3 something to fuse.
- Qwen2 written in the frontend, taking its config and weights as plain dicts
  and arrays.
- A KV cache expressed in the graph: past keys and values are inputs of shape
  `[layers, kv_heads, past, head_dim]`, new ones are outputs.
- A textual IR printer, a validator, and a NumPy interpreter.

**Gate**

- Every primitive against plain NumPy on seeded random inputs, including
  symbolic shapes bound to several sizes and the size-1 edge cases.
- Shape inference rejects each kind of mismatch, with a test per error.
- The tiny 2-layer test model: logits within 1e-5 of nanoinfer.
- The real model: logits within **1e-3** of nanoinfer, greedy tokens identical
  on three prompts.
- Cached decode through the graph gives the same tokens as uncached.

**Measure**

- Nodes per transformer block, and a histogram of primitives.
- Interpreter speed, as a baseline only. It is an oracle, not a backend.

**Not here.** Code generation, any optimization, int8.

---

## Phase 2: loop IR and naive C codegen

**Build**

- A loop IR: buffers, loop nests, loads, stores and scalar expressions.
- A lowering for each primitive into one loop nest, one kernel per primitive.
  No fusion.
- A C emitter, a gcc driver, and a content-hash cache of compiled libraries so
  an unchanged kernel is never recompiled.
- A ctypes runtime and a buffer planner that reuses intermediate buffers by
  liveness.
- Symbolic dimensions become runtime arguments of the kernels.

**Gate**

- Differential tests: every primitive's generated kernel against the
  interpreter, over seeded random shapes, symbolic dimensions bound to several
  sizes, and size-1 axes.
- Compiling the same graph twice produces byte-identical C.
- The real model end to end: greedy tokens identical to nanoinfer.

**Measure**

- Decode tok/s. Expected to be far below nanoinfer: naive loops against
  OpenBLAS. It is the baseline, reported as one.
- Compile time, cold and from the cache.
- Kernels per token and bytes moved per token, counted from the loop IR.

**Not here.** Fusion, vectorization, threads.

---

## Phase 3: fusion

**Build**

- Elementwise fusion into producers and consumers.
- Reduction fusion, so `rms_norm` and `softmax` are one kernel each.
- Epilogue fusion into `matmul`: bias, residual add, `silu(gate) * up`.
- Common subexpression elimination, constant folding, dead code elimination.

**Gate**

- Fused and unfused graphs agree on every test in phases 1 and 2. Bitwise
  where the pass does not reorder arithmetic, within float tolerance where it
  does, and each pass says which it is.
- Real-model tokens identical to nanoinfer.

**Measure**

- Kernels per block and bytes moved per token, before and after, counted
  exactly and asserted in tests.
- Wall time for the fused norms and activations against nanoinfer's NumPy.

**Not here.** Changing loop structure; that is scheduling.

---

## Phase 4: schedules

**Build**

- Schedules kept separate from the algorithm, in the style of Halide: `split`,
  `reorder`, `vectorize`, `unroll`, `parallel`, plus packing and cache tiling
  for `matmul`.
- AVX2 code generation for vectorized loops.
- A persistent thread pool in the runtime. nanoinfer phase 7 measured a
  spawn-per-call pool as 12x slower than one thread, and that lesson is
  carried over rather than relearned.
- Hand-written default schedules for each kernel class.

**Gate**

- Every schedule transformation preserves results: a test grid of schedules,
  each differential-tested against the interpreter.

**Measure**

- GFLOP/s on every matmul shape in the model, against NumPy's OpenBLAS and the
  machine's roofline.
- Per-layer projection time against nanoinfer's Rust kernel table, same
  shapes, same alternating method.

**Not here.** Searching for schedules; these are written by hand.

---

## Phase 5: autotuning

**Build**

- A search space per kernel, built from the phase 4 schedule primitives.
- Cost by measurement: medians of alternating runs, never one sample.
- A search strategy (random sampling, then greedy refinement), seeded.
- An on-disk tuning cache keyed by kernel hash and machine.

**Gate**

- Every tuned kernel passes the same differential tests.
- A tuning run is reproducible from its seed in the candidates it tries. The
  winner may differ under noise, and how often it does is reported rather than
  hidden.

**Measure**

- Tuned against hand-written schedules, per kernel.
- Search time.
- How often the winner changes between runs on this machine.

**Not here.** Learned cost models.

---

## Phase 6: INT8

**Build**

- A quantization pass that rewrites `matmul` against weights to per-row int8,
  the same format nanoinfer uses.
- An int8 `matmul` lowering with int32 accumulation, and an AVX-VNNI
  (`vpdpbusd`) path.
- The LM head stays float32, following nanoinfer's measurement that its
  activation error lands directly on the logits.

**Gate**

- Integer accumulation is exact, so given identical inputs the int8 kernel's
  output is **bitwise identical** to nanoinfer's Rust int8 kernel.
- Perplexity on nanoinfer's held-out text matches its int8 figure (23.3078).
- Real-model tokens identical to nanoinfer's int8 engine.

**Measure**

- Decode tok/s, int8 against fp32, and against nanoinfer's int8.

**Not here.** INT4. nanoinfer measured naive INT4 at +32% perplexity, and a
compiler cannot fix a quantization scheme.

---

## Phase 7: whole-model compilation

**Build**

- The whole decode step compiled into one library with one entry point.
  nanoinfer measured an 8 us floor per ctypes call; at a few hundred kernels
  per token, crossing the boundary per kernel is milliseconds of nothing.
- A static memory plan: every intermediate at a fixed offset in one arena.
- The KV cache as a buffer the library owns.
- Separate prefill and decode specializations.

**Gate**

- Tokens identical to nanoinfer on the generation test set, fp32 and int8.
- No Python inside a decode step except sampling.

**Measure**

- Per-token overhead outside kernels, before and after.
- Time to first token and decode tok/s.

---

## Phase 8: against nanoinfer and llama.cpp

**Build**

- The benchmark from nanoinfer phase 8, pointed at three engines: nanocompile,
  nanoinfer and llama.cpp, fp32 and int8, same prompts, same machine.

**Gate**

- Perplexity parity with nanoinfer at each precision.
- Paired runs judged with an exact sign test.

**Measure**

- Decode tok/s and time to first token for all three, with spreads.
- The result is published whichever way it lands.

---

## After phase 8

A second backend for the Cyclone V FPGA that
[toy-cpu](https://github.com/SpookyJumpyBeans/toy-cpu) was synthesized for: an
int8 systolic array in VHDL, driven by phase 6's kernels and checked bitwise
against them. Out of scope here, and the reason the loop IR keeps the backend
behind one interface.
