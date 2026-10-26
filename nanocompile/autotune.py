"""Autotuning: each kernel's schedule chosen by measurement, on a noisy machine.

Phase 4 wrote one schedule per kernel class by hand. Here each kernel's
schedule is searched for, over the space ``schedules.SPACES`` defines, and
chosen by timing candidates on this machine. Every decision is recorded, so
a tuning run can be read afterwards and replayed.

**The space** of a kernel is every configuration of its class that applies to
it (``schedules.space``), with configurations that generate the same C counted
once. Phase 4's hand-written schedule is always a candidate.

**Cost** is measured, never modelled, and never one sample. Candidates are
compiled into one library, and a small C loop (``_HARNESS``) times each, so a
sample is not dominated by Python's call overhead. The candidates run in
alternation: each round times every candidate once, starting from a different
one each round, so drift in the machine (a clock change, another process)
falls on all of them alike, and the cost is the median over rounds. A weight
operand is rotated through enough copies to stream from memory, as the
weights of a decode step do, rather than sitting in a cache warmed by the
previous call.

A schedule is scored on a **workload**: values for the program's symbols. The
default is a decode step and a 32-token prefill, each per token, so a knob
only a prefill exercises (rows per tile) is chosen by measurement and not by
whichever value the decode step happened to ignore.

**Search** is seeded. A random sample of the space, drawn from the seed and
the kernel alone, is measured with the hand-written schedule. Then greedy
refinement: the measured best and its neighbours (one knob one step away)
are measured together, and the best becomes the incumbent if it beats it by
more than ``margin``. A final run times the incumbent against the
hand-written schedule; it replaces it only if it is faster by more than
``margin``. So under noise a tuned kernel is never chosen on a difference the
machine cannot resolve.

**Reproducibility.** The sample depends only on the seed and the kernel, so
it is the same in every run. Refinement follows measurements, so it can go
differently when the machine does; every comparison it made is in the
record, and ``replay`` re-runs the search on recorded costs, which reproduces
exactly the candidates a run tried. How often the winner changes between runs
is measured, not assumed (``bench/phase5.py``).

**Correctness.** Each candidate is run once on random inputs before it is
timed, and its output must equal the unscheduled kernel's bit for bit. The
primitives guarantee that, and this checks it on every kernel this machine
ever tunes.

**The cache** is a JSON file keyed by machine (compiler, CPU, threads) and
by kernel (its C, what it is called on, the workloads): a hit applies the
recorded configuration without measuring anything.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from nanocompile import codegen_c, runtime, schedules
from nanocompile import loopir as L
from nanocompile.schedule import ScheduleError
from nanocompile.symbolic import evaluate_shape

# A workload: (name, symbol values, weight). The weight turns seconds per call
# into seconds per token, so a decode step and a prefill add up sensibly.
Workload = tuple[str, Mapping[str, int], float]
DECODE = ("decode", {"seq": 1, "past": 64}, 1.0)
PREFILL = ("prefill", {"seq": 32, "past": 0}, 1.0 / 32)
WORKLOADS: tuple[Workload, ...] = (DECODE, PREFILL)

STREAM_BYTES = 64 * 2**20      # weight bytes a sample streams: past this CPU's 24 MiB L3
MAX_COPIES = 32
SAMPLE_SECONDS = 2e-3          # a sample is at least this long, repeating calls if need be
CACHE_VERSION = 1              # bump when the space or the cost changes meaning
GUARD_BYTES = 4096             # either side of a checked output: past any vector or tile
GUARD_BYTE = 0xA5

_HARNESS = r"""
#include <stdint.h>
#ifdef _WIN32
#include <windows.h>
static double nc_now(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
#define NC_EXPORT __declspec(dllexport)
#else
#include <time.h>
static double nc_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
#define NC_EXPORT
#endif
typedef int32_t (*nc_kernel)(void **, const int64_t *);

/* Seconds for reps passes over n_sets argument sets, -1 if a call failed. */
NC_EXPORT double nc_time(void *kernel, void **sets, int64_t n_args, int64_t n_sets,
                         const int64_t *syms, int64_t reps) {
    nc_kernel k = (nc_kernel)kernel;
    double started = nc_now();
    for (int64_t r = 0; r < reps; r++)
        for (int64_t s = 0; s < n_sets; s++)
            if (k(sets + s * n_args, syms) != 0)
                return -1.0;
    return nc_now() - started;
}
"""


class TuningError(RuntimeError):
    """A candidate failed to run, or computed something its kernel does not."""


def _harness() -> ctypes.CDLL:
    global _harness_lib
    if _harness_lib is None:
        path, _ = codegen_c.build_library(_HARNESS)
        lib = ctypes.CDLL(str(path))
        lib.nc_time.restype = ctypes.c_double
        lib.nc_time.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64]
        _harness_lib = lib
    return _harness_lib


_harness_lib: ctypes.CDLL | None = None


# -- what a kernel runs on -------------------------------------------------------


class Operands:
    """Buffers for one call of one kernel at one workload, made once.

    Weights come in ``copies`` independent copies, in both the plain and the
    packed layout, so every candidate streams them however it reads them.
    Everything else is shared by the copies: inputs are small next to weights.
    """

    def __init__(self, kernel: L.Kernel, call, symbols: Sequence[str],
                 bindings: Mapping[str, int], rng: np.random.Generator) -> None:
        missing = [s for s in symbols if s not in bindings]
        if missing:
            raise TuningError(f"the workload gives no value for {missing}")
        self.syms = (ctypes.c_int64 * max(1, len(symbols)))(*[int(bindings[s]) for s in symbols])
        self.bindings = dict(bindings)
        self.output_shape = evaluate_shape(call.output.shape, bindings)
        self.output_dtype = call.output.dtype.numpy
        self._fill = rng
        # One base array per operand, its offset in bytes, and whether it is a weight.
        self.bases: list[np.ndarray] = []
        self.offsets: list[int] = []
        self.weights: list[bool] = []
        for view in call.operands:
            buf = view.buffer
            shape = evaluate_shape(buf.shape, bindings)
            if buf.kind == "const":
                base = np.ascontiguousarray(buf.value)
            else:
                base = self._random(shape, buf.dtype.numpy)
            self.bases.append(base)
            itemsize = buf.dtype.numpy.itemsize
            offset = view.offset if isinstance(view.offset, int) else view.offset.evaluate(bindings)
            self.offsets.append(int(offset) * itemsize)
            self.weights.append(buf.kind == "weight")
        weight_bytes = sum(b.nbytes for b, w in zip(self.bases, self.weights) if w)
        self.copies = 1 if weight_bytes == 0 else max(1, min(MAX_COPIES, math.ceil(STREAM_BYTES / weight_bytes)))
        self._copies: dict[tuple[int, int | None], list[np.ndarray]] = {}
        self._keep: list[object] = []

    def _random(self, shape, dtype) -> np.ndarray:
        if np.issubdtype(dtype, np.floating):
            return self._fill.standard_normal(shape).astype(dtype)
        return np.zeros(shape, dtype=dtype)      # indices: zero is always in range

    def weight(self, i: int, packed: int | None, copy: int) -> np.ndarray:
        key = (i, packed)
        if key not in self._copies:
            source = self.bases[i]
            layout = source if packed is None else runtime.pack_weight(source, packed)
            self._copies[key] = [layout] + [layout.copy() for _ in range(self.copies - 1)]
        return self._copies[key][copy]

    def argument_sets(self, kernel: L.Kernel, copies: int | None = None) -> tuple[object, int, int, np.ndarray]:
        """Pointer arrays for ``copies`` calls (output first), and their output."""
        copies = self.copies if copies is None else copies
        n_args = 1 + len(self.bases)
        output = np.empty(self.output_shape, dtype=self.output_dtype)
        pointers: list[int] = []
        for c in range(copies):
            pointers.append(output.ctypes.data)
            for i, base in enumerate(self.bases):
                if self.weights[i]:
                    array = self.weight(i, kernel.params[i + 1].packed, c)
                    pointers.append(array.ctypes.data)
                else:
                    pointers.append(base.ctypes.data + self.offsets[i])
        sets = (ctypes.c_void_p * len(pointers))(*pointers)
        self._keep.append(sets)
        return sets, n_args, copies, output


# -- measuring -------------------------------------------------------------------


@dataclass
class Candidate:
    config: dict
    kernel: L.Kernel
    source: str


class Batch:
    """Candidates compiled into one library, with the unscheduled kernel as kernel 0."""

    def __init__(self, reference: L.Kernel, candidates: list[Candidate]) -> None:
        self.candidates = candidates
        kernels = [reference] + [c.kernel for c in candidates]
        path, _ = codegen_c.build_library(codegen_c.render_library(kernels))
        self._lib = ctypes.CDLL(str(path))
        self._lib.nc_bind_pool.argtypes = [ctypes.c_void_p]
        self._lib.nc_bind_pool(ctypes.cast(runtime._pool().nc_pool_run, ctypes.c_void_p))
        self.kernels = kernels
        self.addresses = [ctypes.cast(getattr(self._lib, codegen_c.kernel_name(i)), ctypes.c_void_p).value
                          for i in range(len(kernels))]

    def check(self, operands: Operands) -> None:
        """Every candidate's output equals the unscheduled kernel's, bit for bit,
        and no candidate writes outside its output."""
        expected = self._run_once(0, operands)
        for i, candidate in enumerate(self.candidates, start=1):
            got = self._run_once(i, operands)
            if expected.tobytes() != got.tobytes():
                raise TuningError(f"configuration {candidate.config} of a {self.kernels[0].primitive} "
                                  f"kernel computes something else at {operands.bindings}")

    def _run_once(self, index: int, operands: Operands) -> np.ndarray:
        kernel = self.kernels[index]
        # The output sits between two guard regions holding a byte pattern no
        # kernel writes. A write past either end leaves the output bitwise
        # right and corrupts whatever was next to it, so it is checked apart.
        size = int(np.prod(operands.output_shape)) * np.dtype(operands.output_dtype).itemsize
        padded = np.full(size + 2 * GUARD_BYTES, GUARD_BYTE, dtype=np.uint8)
        output = padded[GUARD_BYTES:GUARD_BYTES + size].view(operands.output_dtype).reshape(operands.output_shape)
        pointers = [padded.ctypes.data + GUARD_BYTES]
        keep = [padded]
        for i, base in enumerate(operands.bases):
            param = kernel.params[i + 1]
            if operands.weights[i]:
                pointers.append(operands.weight(i, param.packed, 0).ctypes.data)
                continue
            # An operand the kernel may write starts from the same values for
            # every candidate; one it only reads is shared.
            array = base.copy() if param.writable else base
            keep.append(array)
            pointers.append(array.ctypes.data + operands.offsets[i])
        args = (ctypes.c_void_p * len(pointers))(*pointers)
        seconds = _harness().nc_time(self.addresses[index], args, len(pointers), 1, operands.syms, 1)
        if seconds < 0:
            raise TuningError(f"a {kernel.primitive} kernel failed at {operands.bindings}")
        guards = np.concatenate([padded[:GUARD_BYTES], padded[GUARD_BYTES + size:]])
        if np.any(guards != GUARD_BYTE):
            where = "before" if np.any(padded[:GUARD_BYTES] != GUARD_BYTE) else "after"
            name = "the unscheduled kernel" if index == 0 else f"configuration {self.candidates[index - 1].config}"
            raise TuningError(f"{name} of a {kernel.primitive} kernel writes {where} its output "
                              f"at {operands.bindings}")
        return output.copy()

    def time(self, index: int, operands: Operands, reps: int) -> float:
        """Seconds per call of candidate ``index`` (1-based), streaming the weight copies."""
        sets, n_args, copies, _ = operands.argument_sets(self.kernels[index])
        seconds = _harness().nc_time(self.addresses[index], sets, n_args, copies, operands.syms, reps)
        if seconds < 0:
            raise TuningError(f"a {self.kernels[index].primitive} kernel failed at {operands.bindings}")
        return seconds / (reps * copies)


def measure(batch: Batch, workloads: Sequence[tuple[Operands, float]], rounds: int) -> list[float]:
    """Cost of each candidate: sum over workloads of weight x median seconds per call.

    Candidates alternate within a round, and each round starts one candidate
    later than the last, so no candidate is always first after a pause.
    """
    n = len(batch.candidates)
    costs = [0.0] * n
    for operands, weight in workloads:
        # Calibrate on the first candidate: how many passes make a sample long enough.
        one_pass = batch.time(1, operands, 1) * operands.copies
        reps = max(1, min(1000, math.ceil(SAMPLE_SECONDS / max(one_pass, 1e-9))))
        samples: list[list[float]] = [[] for _ in range(n)]
        for i in range(n):                        # warm up every candidate once
            batch.time(i + 1, operands, 1)
        for r in range(rounds):
            for step in range(n):
                i = (r + step) % n
                samples[i].append(batch.time(i + 1, operands, reps))
        for i in range(n):
            costs[i] += weight * statistics.median(samples[i])
    return costs


# -- searching -------------------------------------------------------------------


@dataclass
class Evaluation:
    stage: str                  # "sample", "refine" or "confirm"
    configs: list[dict]
    costs: list[float]


@dataclass
class TuneRecord:
    """What tuning one kernel did, in order, and what it chose."""

    kernel_key: str
    primitive: str
    kernel_class: str
    space_size: int
    default: dict | None
    winner: dict | None
    default_cost: float | None = None
    winner_cost: float | None = None
    evaluations: list[Evaluation] = field(default_factory=list)
    seconds: float = 0.0
    cached: bool = False

    @property
    def tried(self) -> list[dict]:
        """Every distinct configuration measured, in the order first measured."""
        seen, out = set(), []
        for e in self.evaluations:
            for c in e.configs:
                key = schedules.canonical(c)
                if key not in seen:
                    seen.add(key)
                    out.append(c)
        return out

    @property
    def speedup(self) -> float | None:
        if not self.default_cost or not self.winner_cost:
            return None
        return self.default_cost / self.winner_cost

    def as_json(self) -> dict:
        return {"kernel": self.kernel_key, "primitive": self.primitive, "class": self.kernel_class,
                "space": self.space_size, "default": self.default, "winner": self.winner,
                "default_cost": self.default_cost, "winner_cost": self.winner_cost,
                "speedup": self.speedup, "seconds": round(self.seconds, 3), "cached": self.cached,
                "evaluations": [{"stage": e.stage, "configs": e.configs, "costs": e.costs}
                                for e in self.evaluations]}


def neighbours(config: Mapping, kernel_class: str) -> list[dict]:
    """Configurations one knob one step away from ``config``, in a fixed order."""
    knobs = schedules.SPACES[kernel_class]
    out: list[dict] = []
    for knob, values in knobs.items():
        if knob not in config:
            continue
        i = values.index(config[knob])
        for j in (i - 1, i + 1):
            if 0 <= j < len(values):
                changed = dict(config)
                changed[knob] = values[j]
                if knob == "recipe" and changed["recipe"] == "columns":
                    # Leaving the scalar recipe: start from the hand-written tile.
                    changed = {**schedules.DEFAULTS["matmul"][0], "parallel": config["parallel"]}
                elif changed.get("recipe") == "scalar":
                    changed = {"recipe": "scalar", "parallel": changed["parallel"]}
                out.append(changed)
    return out


def search(space: list[dict], default: dict, kernel_class: str, rng: random.Random,
           cost: Callable[[str, list[dict]], list[float]], samples: int, steps: int,
           margin: float) -> tuple[dict, list[Evaluation], float, float]:
    """The seeded search, on any cost function; returns winner, log, costs.

    ``space`` is the legal configurations, ``cost(stage, configs)`` measures a
    group of them together. Separated from measurement so a recorded run can
    be replayed (``replay``) and the search tested on a cost that is known.
    """
    log: list[Evaluation] = []
    legal = {schedules.canonical(c) for c in space}

    def evaluate(stage: str, configs: list[dict]) -> list[float]:
        costs = cost(stage, configs)
        log.append(Evaluation(stage, configs, costs))
        return costs

    others = [c for c in space if schedules.canonical(c) != schedules.canonical(default)]
    sample = rng.sample(others, min(samples, len(others)))
    group = [default] + sample
    costs = evaluate("sample", group)
    best = min(range(len(group)), key=lambda i: costs[i])
    incumbent, incumbent_cost = group[best], costs[best]
    measured = {schedules.canonical(c) for c in group}

    for _ in range(steps):
        frontier = [n for n in neighbours(incumbent, kernel_class)
                    if schedules.canonical(n) in legal and schedules.canonical(n) not in measured]
        if not frontier:
            break
        group = [incumbent] + frontier
        costs = evaluate("refine", group)
        measured |= {schedules.canonical(c) for c in frontier}
        best = min(range(len(group)), key=lambda i: costs[i])
        if best != 0 and costs[best] < costs[0] * (1 - margin):
            incumbent, incumbent_cost = group[best], costs[best]
        else:
            incumbent_cost = costs[0]
            break

    if schedules.canonical(incumbent) == schedules.canonical(default):
        return default, log, incumbent_cost, incumbent_cost
    costs = evaluate("confirm", [default, incumbent])
    if costs[1] < costs[0] * (1 - margin):
        return incumbent, log, costs[0], costs[1]
    return default, log, costs[0], costs[0]


def replay(record: TuneRecord, space: list[dict], seed: int, samples: int, steps: int,
           margin: float) -> list[dict]:
    """The configurations a search tries when every measurement is ``record``'s.

    Proves a run reproducible from its seed: rerun on the costs it measured,
    the search must try exactly what it tried.
    """
    queue = list(record.evaluations)

    def cost(stage: str, configs: list[dict]) -> list[float]:
        expected = queue.pop(0)
        if expected.stage != stage or [schedules.canonical(c) for c in expected.configs] != \
                [schedules.canonical(c) for c in configs]:
            raise TuningError("the search diverged from the recorded run")
        return list(expected.costs)

    rng = random.Random(seed_material(seed, record.kernel_key))
    winner, log, _, _ = search(space, record.default, record.kernel_class, rng, cost, samples, steps, margin)
    trial = TuneRecord(record.kernel_key, record.primitive, record.kernel_class, len(space),
                       record.default, winner, evaluations=log)
    return trial.tried


def seed_material(seed: int, kernel_key: str) -> str:
    """What seeds one kernel's sample: the run's seed and the kernel, nothing else."""
    return f"{seed}:{kernel_key}"


# -- the cache -------------------------------------------------------------------


def machine_key() -> str:
    """Compiler, CPU and thread count: a schedule tuned on one need not suit another."""
    identity = codegen_c._compiler_identity(codegen_c.compiler())
    return hashlib.sha256(f"{identity}|{os.cpu_count()}|{runtime.threads()}".encode()).hexdigest()[:16]


class TuningCache:
    """Winning configurations on disk: ``{machine: {kernel: entry}}``, in JSON."""

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            path = os.environ.get("NANOCOMPILE_TUNING") or codegen_c.cache_dir() / "tuning.json"
        self.path = Path(path)

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {"version": CACHE_VERSION, "machines": {}}
        if data.get("version") != CACHE_VERSION:
            return {"version": CACHE_VERSION, "machines": {}}
        return data

    def get(self, kernel_key: str) -> dict | None:
        return self._load()["machines"].get(machine_key(), {}).get(kernel_key)

    def put(self, kernel_key: str, entry: dict) -> None:
        data = self._load()
        data["machines"].setdefault(machine_key(), {})[kernel_key] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Written under a temporary name and renamed, so a crash never leaves
        # half a cache behind.
        fd, scratch = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, sort_keys=True)
        os.replace(scratch, self.path)


# -- the tuner -------------------------------------------------------------------


class Autotuner:
    """A schedule for ``CompiledGraph``: searches each kernel's space on this machine.

    ``cache`` is a ``TuningCache``, a path, ``True`` for the default file, or
    ``None`` to measure every time. ``records`` holds what each kernel's
    tuning did, in the order the program's kernels came.
    """

    def __init__(self, seed: int = 0, samples: int = 8, steps: int = 4, rounds: int = 7,
                 margin: float = 0.03, workloads: Sequence[Workload] = WORKLOADS,
                 cache: TuningCache | Path | str | bool | None = True, verbose: bool = False) -> None:
        self.seed = seed
        self.samples = samples
        self.steps = steps
        self.rounds = rounds
        self.margin = margin
        self.workloads = tuple(workloads)
        if cache is True:
            cache = TuningCache()
        elif cache is not None and cache is not False and not isinstance(cache, TuningCache):
            cache = TuningCache(cache)
        self.cache: TuningCache | None = cache or None
        self.verbose = verbose
        self.records: list[TuneRecord] = []

    # The cache key covers everything the choice depends on.
    def kernel_key(self, kernel: L.Kernel, calls, symbols: Sequence[str]) -> str:
        operands = [(v.buffer.kind, str(v.buffer.dtype), [str(d) for d in v.buffer.shape], str(v.offset))
                    for v in calls[0].operands] if calls else []
        material = json.dumps({
            "kernel": codegen_c.render_kernel(kernel, "k"),
            "operands": operands,
            "symbols": list(symbols),
            "workloads": [[n, sorted(b.items()), w] for n, b, w in self.workloads],
            "space": {k: [list(map(str, v)) for v in d.values()] for k, d in schedules.SPACES.items()},
        }, sort_keys=True)
        return hashlib.sha256(material.encode()).hexdigest()[:20]

    def schedule_program(self, program):
        calls_of: dict[int, list] = {}
        for call in program.calls:
            calls_of.setdefault(call.kernel, []).append(call)
        kernels = [self.tune(kernel, calls_of.get(i, []), program.symbols)
                   for i, kernel in enumerate(program.kernels)]
        from dataclasses import replace
        return replace(program, kernels=kernels)

    def legal_space(self, kernel: L.Kernel, calls) -> list[Candidate]:
        """The configurations that apply, one per distinct generated kernel."""
        out: list[Candidate] = []
        seen: set[str] = set()
        default = schedules.default_config(kernel, calls)
        configs = schedules.space(kernel)
        if default is not None:
            # The hand-written configuration first, so it represents its kernel.
            configs = [default] + [c for c in configs if schedules.canonical(c) != schedules.canonical(default)]
        for config in configs:
            try:
                scheduled = schedules.apply(kernel, calls, config)
            except ScheduleError:
                continue
            source = codegen_c.render_kernel(scheduled, "k")
            if source in seen:
                continue
            seen.add(source)
            out.append(Candidate(config, scheduled, source))
        return out

    def tune(self, kernel: L.Kernel, calls, symbols: Sequence[str]) -> L.Kernel:
        started = time.perf_counter()
        key = self.kernel_key(kernel, calls, symbols)
        default = schedules.default_config(kernel, calls)
        record = TuneRecord(key, kernel.primitive, schedules.kernel_class(kernel), 0, default, default)

        if self.cache is not None:
            entry = self.cache.get(key)
            if entry is not None:
                try:
                    tuned = kernel if entry["winner"] is None else schedules.apply(kernel, calls, entry["winner"])
                    record.winner, record.cached = entry["winner"], True
                    record.default_cost, record.winner_cost = entry.get("default_cost"), entry.get("winner_cost")
                    record.seconds = time.perf_counter() - started
                    self.records.append(record)
                    return tuned
                except (ScheduleError, KeyError, TypeError):
                    pass                      # stale or malformed: tune again

        candidates = self.legal_space(kernel, calls)
        record.space_size = len(candidates)
        if default is None or not calls or len(candidates) <= 1:
            # Nothing to choose between: the hand-written schedule (or none).
            record.seconds = time.perf_counter() - started
            self.records.append(record)
            return schedules.default(kernel, calls)

        by_key = {schedules.canonical(c.config): c for c in candidates}
        rng_data = np.random.default_rng(0)
        workloads = [(Operands(kernel, calls[0], symbols, b, rng_data), w) for _, b, w in self.workloads]

        def cost(stage: str, configs: list[dict]) -> list[float]:
            batch = Batch(kernel, [by_key[schedules.canonical(c)] for c in configs])
            for operands, _ in workloads:
                batch.check(operands)
            rounds = self.rounds * 2 if stage == "confirm" else self.rounds
            return measure(batch, workloads, rounds)

        rng = random.Random(seed_material(self.seed, key))
        space = [c.config for c in candidates]
        winner, log, default_cost, winner_cost = search(
            space, space[0], record.kernel_class, rng, cost, self.samples, self.steps, self.margin)
        record.winner, record.evaluations = winner, log
        record.default_cost, record.winner_cost = default_cost, winner_cost
        record.seconds = time.perf_counter() - started
        self.records.append(record)
        if self.verbose:
            print(f"tuned {kernel.primitive:<14} {len(record.tried):>3} of {len(space):>3} configs "
                  f"in {record.seconds:5.1f} s: {record.speedup:.2f}x over hand-written  {winner}")
        if self.cache is not None:
            self.cache.put(key, {"winner": winner, "default_cost": default_cost, "winner_cost": winner_cost,
                                 "seed": self.seed, "tuned_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        return by_key[schedules.canonical(winner)].kernel
