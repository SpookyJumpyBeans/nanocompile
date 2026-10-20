"""Hand-written default schedules, one recipe per kernel class.

Fusion emits a few kernel shapes; each gets a recipe here, written for this
machine's AVX2 and its 16 vector registers. A recipe that does not apply to
a particular kernel (a width that is not a multiple of eight, a softmax whose
mask compares lane indices) falls back to a simpler one, and in the end to
the kernel as fusion wrote it. Every recipe is built from the schedule
primitives, so every one is bitwise identical to the unscheduled kernel.

**Projections**, ``[seq, K] @ W.T`` with ``W`` a ``[N, K]`` weight, are where
the time goes. The recipe vectorizes across output columns, never down the
k sum:

    for jo in N / 32, in parallel:            # 32 columns per task
        for io in seq / MB:                   # MB rows per tile
            acc[MB][4] = 0                    # MB x 4 vectors, in registers
            for k in K:
                for each of the MB rows: a = broadcast x[row, k]
                    for each of 4 vectors: acc += a * Wpacked[block, k, 0..7]
            epilogue on the tile, 8 columns at a time

Each output's sum still runs over k in order, so nothing is reassociated:
the parallelism is across the 32 x MB independent sums of a tile. The weight
is packed so the eight columns one vector covers are adjacent in memory. A
decode step has one row, so it runs the ``seq % MB`` tail with one row of
four vectors.

**Attention matmuls** run one task per head. The context product
``softmax @ V`` reads ``V`` contiguously along its output columns, so it gets
the same column vectorization without packing. The scores ``q @ K^T`` read
``K`` with a stride along the output, so they stay scalar per head.

**Row kernels** (``rms_norm``, ``softmax``) keep their reductions scalar and
in order, and vectorize the loop that writes the row. **Elementwise** and
**gather** kernels vectorize their innermost loop. Anything with more than
one row runs its rows in parallel, except a gather, whose bounds check may
stop the kernel early.
"""

from __future__ import annotations

from nanocompile import loopir as L
from nanocompile.codegen_avx2 import LANES
from nanocompile.schedule import Schedule, ScheduleError, loop_vars
from nanocompile.simplify import Simplifier

VECTORS = 4          # column vectors per tile
ROWS = 2             # rows per tile, for a single accumulator


def default(kernel: L.Kernel, calls) -> L.Kernel:
    """``kernel`` with the first recipe that applies to it."""
    for recipe in _recipes(kernel):
        try:
            return recipe(kernel, calls)
        except ScheduleError:
            continue
    return kernel


def _recipes(kernel: L.Kernel):
    if kernel.primitive.startswith("matmul"):
        return [_matmul_columns, _matmul_scalar_parallel]
    if kernel.primitive == "row":
        return [_row_vectorized, _outer_parallel]
    if kernel.primitive == "gather":
        return [_innermost_vectorized]
    return [_vectorized_and_parallel, _innermost_vectorized, _outer_parallel]


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


def _matmul_columns(kernel: L.Kernel, calls) -> L.Kernel:
    """Column-vectorized register tiles; parallel over column blocks or heads."""
    outputs = [v for v in loop_vars(kernel.body) if v != "k"]
    column, rows_var = outputs[-1], (outputs[-2] if len(outputs) >= 2 else None)
    extent = _extent(kernel, column)
    if not isinstance(extent, L.Const) or extent.value % LANES:
        raise ScheduleError("the output width is not a multiple of the vector width")
    columns = int(extent.value)
    accumulators = _accumulators(kernel)
    vectors = VECTORS // accumulators if accumulators <= VECTORS else 1
    while (columns // LANES) % vectors:
        vectors //= 2
    projection = len(outputs) == 2
    # Rows share a tile only in a projection, where the column block is
    # outside the row loop; per head, the rows are the loop around the tile.
    rows = ROWS if accumulators == 1 and projection else 1
    weights = _packable_weights(kernel, calls, column, columns)

    s = Schedule(kernel)
    block = vectors * LANES
    s.split(column, block, outer="jo", inner="jr")
    if projection:
        # [seq, N]: column blocks outermost, so one row still spreads over threads.
        s.reorder(rows_var, "jo")
    s.split("jr", LANES, outer="jv", inner="ji")
    s.privatize("ji").privatize("jv")
    if rows_var is not None and rows > 1:
        s.split(rows_var, rows, outer="io", inner="ii", tail="it").privatize("ii").unroll("ii")
    for param, n, k in weights:
        s.pack(param, rows=n, cols=k)
    s.unroll("jv").vectorize("ji")
    s.parallel("jo" if projection else outputs[0])
    return s.done()


def _matmul_scalar_parallel(kernel: L.Kernel, calls) -> L.Kernel:
    return _outer_parallel(kernel, calls)


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


def _innermost_vectorized(kernel: L.Kernel, calls) -> L.Kernel:
    inner = _innermost(kernel)
    return Schedule(kernel).split(inner, LANES).vectorize(f"{inner}i").done()


def _outer_parallel(kernel: L.Kernel, calls) -> L.Kernel:
    outer = [s.var for s in kernel.body if isinstance(s, L.Loop)]
    if not outer:
        raise ScheduleError("the kernel has no top-level loop")
    s = Schedule(kernel)
    for var in dict.fromkeys(outer):
        s.parallel(var)
    return s.done()


def _vectorized_and_parallel(kernel: L.Kernel, calls) -> L.Kernel:
    outer = [s.var for s in kernel.body if isinstance(s, L.Loop)]
    inner = _innermost(kernel)
    if not outer or outer == [inner]:
        raise ScheduleError("a single loop is vectorized, not split across threads")
    s = Schedule(kernel).split(inner, LANES).vectorize(f"{inner}i")
    for var in dict.fromkeys(outer):
        s.parallel(var)
    return s.done()


def _row_vectorized(kernel: L.Kernel, calls) -> L.Kernel:
    s = Schedule(kernel).split("j0", LANES).vectorize("j0i")
    for var in dict.fromkeys(v.var for v in kernel.body if isinstance(v, L.Loop)):
        s.parallel(var)
    return s.done()
