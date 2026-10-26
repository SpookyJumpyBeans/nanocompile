"""Schedules as points in a space: one recipe per kernel class, with knobs.

Fusion emits a few kernel shapes; each gets a recipe here, written for this
machine's AVX2 and its 16 vector registers. Phase 4 wrote one schedule per
class by hand. Phase 5 searches, so each recipe is now a function of a
``Config``, a handful of named knobs, and phase 4's hand-written schedule is
one point in that space: ``DEFAULTS``. The search space of a kernel is every
combination of its class's knobs that applies to it.

A configuration that does not apply to a particular kernel (a column block
that does not divide the width, a softmax whose mask compares lane indices)
raises ``ScheduleError``. ``default`` keeps phase 4's behaviour exactly: it
tries the hand-written configurations of the class in order, then the kernel
as fusion wrote it. Every recipe is built from the schedule primitives, so
every point of every space is bitwise identical to the unscheduled kernel.

**Projections**, ``[seq, K] @ W.T`` with ``W`` a ``[N, K]`` weight, are where
the time goes. The recipe vectorizes across output columns, never down the
k sum:

    for jo in N / (vectors * 8), in parallel:  # one column block per task
        for io in seq / rows:                  # rows per tile
            acc[rows][vectors] = 0             # in registers
            for k in K:
                for each row: a = broadcast x[row, k]
                    for each vector: acc += a * Wpacked[block, k, 0..7]
            epilogue on the tile, 8 columns at a time

Each output's sum still runs over k in order, so nothing is reassociated:
the parallelism is across the independent sums of a tile. Its knobs:

* ``vectors``: 8-wide column vectors per tile, shared among the
  accumulators (SwiGLU's gate and up are two), halved until it divides the
  width. Phase 4: 4.
* ``rows``: rows per tile, in a projection with one accumulator. Phase 4: 2.
* ``pack``: read the weight packed, so the eight columns one vector covers
  are adjacent in memory. Phase 4: yes.
* ``unroll``: unroll the vectors and rows of a tile. Phase 4: yes.
* ``parallel``: run column blocks (or heads) on the thread pool. Phase 4: yes.

``recipe="scalar"`` is the fallback, the kernel's outer loops in parallel.

**Attention matmuls** run one task per head. ``softmax @ V`` reads ``V``
contiguously along its output columns and gets the same recipe without
packing. ``q @ K^T`` reads ``K`` with a stride, so it stays scalar per head.

**Row kernels** (``rms_norm``, ``softmax``) keep their reductions scalar and
in order; ``vectorize`` vectorizes the loop that writes the row. **Elementwise**
and **gather** kernels vectorize their innermost loop. ``parallel`` runs a
kernel's outer loops on the pool, never a gather's, whose bounds check may
stop the kernel early.
"""

from __future__ import annotations

import itertools
from typing import Mapping

from nanocompile import loopir as L
from nanocompile.codegen_avx2 import LANES
from nanocompile.schedule import Schedule, ScheduleError, loop_vars
from nanocompile.simplify import Simplifier

Config = Mapping[str, object]

# Each class's knobs and their values, in the order the search treats as
# neighbours (one step along a list is one step of the greedy refinement).
SPACES: dict[str, dict[str, tuple]] = {
    "matmul": {
        "recipe": ("columns", "scalar"),
        "vectors": (1, 2, 4, 8),
        "rows": (1, 2, 3, 4),
        "pack": (True, False),
        "unroll": (True, False),
        "parallel": (True, False),
    },
    "row": {"vectorize": (True, False), "parallel": (True, False)},
    "gather": {"vectorize": (True, False)},
    "elementwise": {"vectorize": (True, False), "parallel": (True, False)},
}

# Phase 4's hand-written schedules: per class, the configurations ``default``
# tries in order before leaving the kernel as fusion wrote it.
DEFAULTS: dict[str, tuple[dict, ...]] = {
    "matmul": (
        {"recipe": "columns", "vectors": 4, "rows": 2, "pack": True, "unroll": True, "parallel": True},
        {"recipe": "scalar", "parallel": True},
    ),
    "row": ({"vectorize": True, "parallel": True}, {"vectorize": False, "parallel": True}),
    "gather": ({"vectorize": True},),
    "elementwise": (
        {"vectorize": True, "parallel": True},
        {"vectorize": True, "parallel": False},
        {"vectorize": False, "parallel": True},
    ),
}


def kernel_class(kernel: L.Kernel) -> str:
    if kernel.primitive.startswith("matmul"):
        return "matmul"
    if kernel.primitive in ("row", "gather"):
        return kernel.primitive
    return "elementwise"


def default(kernel: L.Kernel, calls) -> L.Kernel:
    """``kernel`` with phase 4's hand-written schedule: the first that applies."""
    for config in DEFAULTS[kernel_class(kernel)]:
        try:
            return apply(kernel, calls, config)
        except ScheduleError:
            continue
    return kernel


def default_config(kernel: L.Kernel, calls) -> dict | None:
    """The configuration ``default`` uses for ``kernel``; None if it leaves it unscheduled."""
    for config in DEFAULTS[kernel_class(kernel)]:
        try:
            apply(kernel, calls, config)
            return dict(config)
        except ScheduleError:
            continue
    return None


def space(kernel: L.Kernel) -> list[dict]:
    """Every configuration of ``kernel``'s class, before asking which apply."""
    return space_of_class(kernel_class(kernel))


def space_of_class(cls: str) -> list[dict]:
    """Every configuration of a kernel class.

    A scalar matmul ignores the tile knobs, so it appears once per value of
    ``parallel`` rather than once per tile.
    """
    knobs = SPACES[cls]
    configs: list[dict] = []
    seen: set[tuple] = set()
    for values in itertools.product(*knobs.values()):
        config = dict(zip(knobs, values))
        if config.get("recipe") == "scalar":
            config = {"recipe": "scalar", "parallel": config["parallel"]}
        key = canonical(config)
        if key not in seen:
            seen.add(key)
            configs.append(config)
    return configs


def canonical(config: Config) -> tuple:
    """A hashable, order-independent form of ``config``."""
    return tuple(sorted(config.items()))


_POINTS ={cls: {canonical(c) for c in space_of_class(cls)} for cls in SPACES}


def apply(kernel: L.Kernel, calls, config: Config) -> L.Kernel:
    """``kernel`` scheduled by ``config``; ``ScheduleError`` if it does not apply.

    A configuration outside its class's space is an error too, never rounded
    to a nearby point: a stale tuning cache must not be reinterpreted.
    """
    cls = kernel_class(kernel)
    if canonical(config) not in _POINTS[cls]:
        raise ScheduleError(f"{dict(config)} is not a point of the {cls} space")
    if cls == "matmul":
        if config["recipe"] == "columns":
            return _matmul_columns(kernel, calls, config)
        return _outer_parallel(kernel, calls, config)
    if cls == "row":
        return _row(kernel, calls, config)
    if cls == "gather":
        if not config["vectorize"]:
            return kernel
        return _innermost_vectorized(kernel, calls)
    return _elementwise(kernel, calls, config)


# -- matmul ----------------------------------------------------------------------


def _accumulators(kernel: L.Kernel) -> int:
    """How many products the matmul's k loop accumulates (two for SwiGLU)."""
    def find(stmts):
        for s in stmts:
            if isinstance(s, L.Loop) and s.var == "k":
                return sum(isinstance(x, L.Assign) for x in s.body)
            if isinstance(s, L.Loop):
                found = find(s.body)
                if found:
                    return found
        return 0
    return find(kernel.body) or 1


def _extent(kernel: L.Kernel, var: str) -> L.Scalar | None:
    def find(stmts):
        for s in stmts:
            if isinstance(s, L.Loop):
                if s.var == var:
                    return s.extent
                found = find(s.body)
                if found is not None:
                    return found
        return None
    return find(kernel.body)


def _packable_weights(kernel: L.Kernel, calls, column: str, columns: int) -> list[tuple[int, int, int]]:
    """Operands read as ``W[column, k]`` of a ``[columns, K]`` weight: (param, rows, cols)."""
    found = []
    simplifier = Simplifier({})
    for p in range(1, len(kernel.params)):
        views = [call.operands[p - 1] for call in calls]
        if not views or any(v.buffer.kind != "weight" or len(v.buffer.shape) != 2 or v.offset != 0 for v in views):
            continue
        rows, cols = views[0].buffer.shape
        if not (isinstance(rows, int) and isinstance(cols, int) and rows == columns):
            continue
        loads = [x for x in L.scalars_of(kernel.body) if isinstance(x, L.Load) and x.param == p]
        if loads and all(simplifier.coefficient(x.index, column) == cols and
                         simplifier.coefficient(x.index, "k") == 1 for x in loads):
            found.append((p, rows, cols))
    return found


def _matmul_columns(kernel: L.Kernel, calls, config: Config) -> L.Kernel:
    """Column-vectorized register tiles; parallel over column blocks or heads."""
    outputs = [v for v in loop_vars(kernel.body) if v != "k"]
    column, rows_var = outputs[-1], (outputs[-2] if len(outputs) >= 2 else None)
    extent = _extent(kernel, column)
    if not isinstance(extent, L.Const) or extent.value % LANES:
        raise ScheduleError("the output width is not a multiple of the vector width")
    columns = int(extent.value)
    accumulators = _accumulators(kernel)
    budget = int(config["vectors"])
    vectors = budget // accumulators if accumulators <= budget else 1
    while (columns // LANES) % vectors:
        vectors //= 2
    projection = len(outputs) == 2
    # Rows share a tile only in a projection, where the column block is
    # outside the row loop; per head, the rows are the loop around the tile.
    rows = int(config["rows"]) if accumulators == 1 and projection else 1
    weights = _packable_weights(kernel, calls, column, columns) if config["pack"] else []

    s = Schedule(kernel)
    block = vectors * LANES
    s.split(column, block, outer="jo", inner="jr")
    if projection:
        # [seq, N]: column blocks outermost, so one row still spreads over threads.
        s.reorder(rows_var, "jo")
    s.split("jr", LANES, outer="jv", inner="ji")
    s.privatize("ji").privatize("jv")
    if rows_var is not None and rows > 1:
        s.split(rows_var, rows, outer="io", inner="ii", tail="it").privatize("ii")
        if config["unroll"]:
            s.unroll("ii")
    for param, n, k in weights:
        s.pack(param, rows=n, cols=k)
    if config["unroll"]:
        s.unroll("jv")
    s.vectorize("ji")
    if config["parallel"]:
        s.parallel("jo" if projection else outputs[0])
    return s.done()


# -- everything else -------------------------------------------------------------


def _innermost(kernel: L.Kernel) -> str:
    """The innermost loop of the main nest (the last loop that holds no loop)."""
    def last(stmts):
        found = None
        for s in stmts:
            if isinstance(s, L.Loop):
                found = last(s.body) or s.var
        return found
    found = last(kernel.body)
    if found is None:
        raise ScheduleError("the kernel has no loops")
    return found


def _outer_loops(kernel: L.Kernel) -> list[str]:
    return list(dict.fromkeys(s.var for s in kernel.body if isinstance(s, L.Loop)))


def _innermost_vectorized(kernel: L.Kernel, calls) -> L.Kernel:
    inner = _innermost(kernel)
    return Schedule(kernel).split(inner, LANES).vectorize(f"{inner}i").done()


def _outer_parallel(kernel: L.Kernel, calls, config: Config) -> L.Kernel:
    outer = _outer_loops(kernel)
    if not outer:
        raise ScheduleError("the kernel has no top-level loop")
    if not config["parallel"]:
        return kernel
    s = Schedule(kernel)
    for var in outer:
        s.parallel(var)
    return s.done()


def _elementwise(kernel: L.Kernel, calls, config: Config) -> L.Kernel:
    if not config["vectorize"]:
        return _outer_parallel(kernel, calls, config)
    if not config["parallel"]:
        return _innermost_vectorized(kernel, calls)
    outer = _outer_loops(kernel)
    inner = _innermost(kernel)
    if not outer or outer == [inner]:
        raise ScheduleError("a single loop is vectorized, not split across threads")
    s = Schedule(kernel).split(inner, LANES).vectorize(f"{inner}i")
    for var in outer:
        s.parallel(var)
    return s.done()


def _row(kernel: L.Kernel, calls, config: Config) -> L.Kernel:
    if not config["vectorize"]:
        return _outer_parallel(kernel, calls, config)
    s = Schedule(kernel).split("j0", LANES).vectorize("j0i")
    if config["parallel"]:
        for var in _outer_loops(kernel):
            s.parallel(var)
    return s.done()
