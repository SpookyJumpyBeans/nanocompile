"""The phase 4 gate: every schedule computes exactly what the kernel did.

Each test compiles a graph twice, plainly and with a schedule applied to
every kernel, and requires the outputs to be bitwise identical. The symbol is
bound to sizes that leave split tails of every length, including none.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanocompile import loopir as L
from nanocompile import nn
from nanocompile.dtypes import f32, i64
from nanocompile.frontend import GraphBuilder, gather
from nanocompile.fusion import fuse
from nanocompile.passes import optimize
from nanocompile.runtime import CompiledGraph
from nanocompile.schedule import Schedule, ScheduleError

SIZES = [1, 3, 7, 8]


class Case:
    """A graph, and how to make its inputs for a given ``seq``."""

    def __init__(self, graph, make) -> None:
        self.graph = graph
        self.make = make


def _cases() -> dict[str, Case]:
    cases = {}

    b = GraphBuilder()
    s = b.symbol("seq")
    x, r = b.input("x", f32, (s, 12)), b.input("r", f32, (s, 10))
    y = r + nn.linear(x, b.weight("w", (10, 12)), b.weight("b", (10,)))
    cases["matmul+epilogue"] = Case(b.build({"y": y}), lambda rng, n: (
        {"x": rand(rng, n, 12), "r": rand(rng, n, 10)}, {"w": rand(rng, 10, 12), "b": rand(rng, 10)}))

    b = GraphBuilder()
    x = b.input("x", f32, (b.symbol("seq"), 12))
    h = nn.silu(nn.linear(x, b.weight("g", (9, 12)))) * nn.linear(x, b.weight("u", (9, 12)))
    cases["swiglu"] = Case(b.build({"h": h}), lambda rng, n: (
        {"x": rand(rng, n, 12)}, {"g": rand(rng, 9, 12), "u": rand(rng, 9, 12)}))

    b = GraphBuilder()
    x = b.input("x", f32, (b.symbol("seq"), 20))
    cases["rms_norm"] = Case(b.build({"y": nn.rms_norm(x, b.weight("w", (20,)), 1e-6)}), lambda rng, n: (
        {"x": rand(rng, n, 20)}, {"w": rand(rng, 20)}))

    b = GraphBuilder()
    x = b.input("x", f32, (3, b.symbol("seq"), 11))
    cases["softmax"] = Case(b.build({"y": nn.softmax(x)}), lambda rng, n: ({"x": rand(rng, 3, n, 11)}, {}))

    b = GraphBuilder()
    x, y = b.input("x", f32, (b.symbol("seq"), 13)), b.input("y", f32, (13,))
    cases["elementwise"] = Case(b.build({"z": (x * y).exp() + x}), lambda rng, n: (
        {"x": rand(rng, n, 13), "y": rand(rng, 13)}, {}))

    b = GraphBuilder()
    ids = b.input("ids", i64, (b.symbol("seq"),))
    cases["gather"] = Case(b.build({"e": gather(b.weight("t", (6, 5)), ids)}), lambda rng, n: (
        {"ids": rng.integers(0, 6, size=n).astype(np.int64)}, {"t": rand(rng, 6, 5)}))
    return cases


def rand(rng, *shape):
    return rng.standard_normal(shape).astype(np.float32)


CASES = _cases()


def check(name: str, schedule) -> CompiledGraph:
    """Run ``name`` plainly and scheduled at every size; require identical bits."""
    case = CASES[name]
    plain = CompiledGraph(case.graph)
    scheduled = CompiledGraph(case.graph, schedule=schedule)
    assert scheduled.source != plain.source, "the schedule changed nothing"
    rng = np.random.default_rng(len(name))
    for seq in SIZES:
        inputs, weights = case.make(rng, seq)
        expected = plain(inputs, weights)
        actual = scheduled(inputs, weights)
        for out in expected:
            np.testing.assert_array_equal(actual[out], expected[out], err_msg=f"{name} seq={seq} {out}")
    return scheduled


# -- the grid ------------------------------------------------------------------


def split_all(factor):
    def schedule(kernel, calls):
        s = Schedule(kernel)
        for var in s.loops:
            s.split(var, factor)
        return s.done()
    return schedule


def split_and_unroll_innermost(factor):
    def schedule(kernel, calls):
        s = Schedule(kernel)
        inner = s.loops[-1]
        s.split(inner, factor).unroll(f"{inner}i")
        return s.done()
    return schedule


def reorder_every_legal_pair(kernel, calls):
    s = Schedule(kernel)
    loops = s.loops
    for outer, inner in zip(loops, loops[1:]):
        try:
            s.reorder(outer, inner)
        except ScheduleError:
            pass
    return s.done()


@pytest.mark.parametrize("factor", [1, 3, 8])
@pytest.mark.parametrize("name", sorted(CASES))
def test_split_every_loop(name, factor):
    check(name, split_all(factor))


@pytest.mark.parametrize("factor", [2, 4])
@pytest.mark.parametrize("name", sorted(CASES))
def test_split_then_unroll_the_innermost_loop(name, factor):
    check(name, split_and_unroll_innermost(factor))


# A row kernel's loops each hold more than one statement, and a gather's row
# loop holds its bounds check, so neither has a pair of loops to swap.
NOTHING_TO_REORDER = {"gather", "rms_norm"}


@pytest.mark.parametrize("name", sorted(set(CASES) - NOTHING_TO_REORDER))
def test_reorder_every_legal_pair(name):
    check(name, reorder_every_legal_pair)


@pytest.mark.parametrize("name", sorted(NOTHING_TO_REORDER))
def test_kernels_without_a_perfect_nest_refuse_every_reorder(name):
    (kernel,) = fuse(optimize(CASES[name].graph)).kernels
    loops = Schedule(kernel).loops
    for outer, inner in zip(loops, loops[1:]):
        with pytest.raises(ScheduleError):
            Schedule(kernel).reorder(outer, inner)


COLUMNS = {"matmul+epilogue": 10, "swiglu": 9}


def register_block(columns_per_block, rows_per_block):
    """The matmul recipe without SIMD: column blocks outermost, rows blocked,
    the block's columns and rows moved inside the k loop and unrolled."""
    def schedule(kernel, calls):
        s = Schedule(kernel)
        s.split("i1", columns_per_block).reorder("i0", "i1o")
        s.privatize("i1i")
        s.split("i0", rows_per_block).privatize("i0i")
        s.unroll("i0i").unroll("i1i")
        return s.done()
    return schedule


@pytest.mark.parametrize("rows", [1, 2, 3])
@pytest.mark.parametrize("name", sorted(COLUMNS))
def test_privatized_and_unrolled_register_blocks(name, rows):
    # A column block that divides the width, so the column loops nest perfectly.
    block = {10: 5, 9: 3}[COLUMNS[name]]
    check(name, register_block(block, rows))


@pytest.mark.parametrize("name", sorted(COLUMNS))
def test_privatize_a_split_tail_too(name):
    def schedule(kernel, calls):
        s = Schedule(kernel).split("i1", 4)
        s.privatize("i1i").privatize("i1t")
        return s.done()
    check(name, schedule)


def test_privatize_moves_the_columns_inside_the_k_loop():
    s = Schedule(matmul_kernel()).split("i1", 5).privatize("i1i")
    (rows,) = s.kernel.body
    (block,) = rows.body
    array, reduction, epilogue = block.body
    assert array == L.Array("acc0", 5, L.const(0.0, f32))
    assert reduction.var == "k" and reduction.body[0].var == "i1i"
    assert epilogue.var == "i1i"


def test_privatize_needs_accumulators_and_a_constant_extent():
    with pytest.raises(ScheduleError, match="not a constant"):
        Schedule(matmul_kernel()).privatize("i0")
    (row_kernel,) = fuse(optimize(CASES["rms_norm"].graph)).kernels
    with pytest.raises(ScheduleError):
        Schedule(row_kernel).split("j0", 4).privatize("j0i")


# -- legality --------------------------------------------------------------------


def matmul_kernel():
    (kernel,) = fuse(optimize(CASES["matmul+epilogue"].graph)).kernels
    return kernel


def test_a_reduction_loop_cannot_leave_its_accumulator():
    with pytest.raises(ScheduleError, match="not the only statement"):
        Schedule(matmul_kernel()).reorder("i1", "k")


def test_reorder_swaps_the_rows_and_columns_of_a_matmul():
    assert Schedule(matmul_kernel()).reorder("i0", "i1").loops[:2] == ["i1", "i0"]


def test_unknown_loops_and_bad_arguments_are_errors():
    s = Schedule(matmul_kernel())
    with pytest.raises(ScheduleError, match="no loop over 'z'"):
        s.split("z", 4)
    with pytest.raises(ScheduleError, match="positive"):
        s.split("k", 0)
    with pytest.raises(ScheduleError, match="not a constant"):
        s.unroll("i0")            # the seq loop


def test_a_split_leaves_no_tail_when_the_factor_divides():
    assert "kt" not in Schedule(matmul_kernel()).split("k", 4).loops
    assert "kt" in Schedule(matmul_kernel()).split("k", 5).loops
