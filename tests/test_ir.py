"""Shape inference and graph validation: every mismatch rejected when built.

One test per kind of error. The point of inferring shapes in the node
constructor is that a wrong model fails at the line that made it wrong; each
of these is a bug that would otherwise surface as a NumPy broadcasting error,
or worse, as a broadcast that succeeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanocompile import dtypes
from nanocompile.dtypes import f32, i64
from nanocompile.frontend import GraphBuilder, concat, gather, where
from nanocompile.ir import Graph, GraphError, ShapeError, make


@pytest.fixture
def b() -> GraphBuilder:
    return GraphBuilder("test")


def leaf(shape, dtype=f32, name="x"):
    return make("input", name=name, dtype=dtype, shape=shape)


# -- shape inference ---------------------------------------------------------


def test_elementwise_operands_must_match_exactly():
    with pytest.raises(ShapeError, match="add: shape mismatch"):
        make("add", leaf((4, 8)), leaf((1, 8)))


def test_elementwise_dtypes_must_match():
    with pytest.raises(ShapeError, match="dtype mismatch"):
        make("mul", leaf((4,)), leaf((4,), i64))


def test_symbolic_dims_must_be_the_same_symbol(b):
    seq, past = b.symbol("seq"), b.symbol("past")
    with pytest.raises(ShapeError, match="shape mismatch"):
        make("add", leaf((seq, 8)), leaf((past, 8)))


def test_equal_expressions_written_differently_match(b):
    seq, past = b.symbol("seq"), b.symbol("past")
    node = make("add", leaf((seq + past, 8)), leaf((past + seq, 8)))
    assert node.shape[0] == past + seq


def test_div_is_float_only():
    with pytest.raises(ShapeError, match="float dtype"):
        make("div", leaf((4,), i64), leaf((4,), i64))


@pytest.mark.parametrize("op", ["exp", "sqrt", "reciprocal", "sin", "cos"])
def test_transcendentals_are_float_only(op):
    with pytest.raises(ShapeError, match="float dtype"):
        make(op, leaf((4,), i64))


def test_where_condition_must_be_bool():
    with pytest.raises(ShapeError, match="condition must be bool"):
        make("where", leaf((4,)), leaf((4,)), leaf((4,)))


def test_reduce_axis_out_of_range():
    with pytest.raises(ShapeError, match="axis 2 is out of range"):
        make("reduce_sum", leaf((4, 8)), axis=2)


def test_reduce_keeps_the_axis():
    assert make("reduce_sum", leaf((4, 8)), axis=-1).shape == (4, 1)


def test_reshape_must_preserve_element_count(b):
    seq = b.symbol("seq")
    with pytest.raises(ShapeError, match="cannot reshape"):
        make("reshape", leaf((seq, 896)), shape=(seq, 14, 63))
    assert make("reshape", leaf((seq, 896)), shape=(seq, 14, 64)).shape == (seq, 14, 64)


def test_reshape_cannot_move_a_symbol_into_a_constant(b):
    seq = b.symbol("seq")
    with pytest.raises(ShapeError, match="cannot reshape"):
        make("reshape", leaf((seq, 4)), shape=(4, 4))


def test_permute_must_be_a_permutation():
    with pytest.raises(ShapeError, match="not a permutation"):
        make("permute", leaf((2, 3, 4)), perm=(0, 0, 1))


def test_broadcast_only_stretches_size_one():
    with pytest.raises(ShapeError, match="cannot broadcast"):
        make("broadcast", leaf((2, 3)), shape=(4, 3))


def test_broadcast_does_not_add_axes():
    with pytest.raises(ShapeError, match="reshape first"):
        make("broadcast", leaf((3,)), shape=(4, 3))


def test_slice_out_of_bounds_when_decidable(b):
    with pytest.raises(ShapeError, match="size - stop is negative"):
        make("slice", leaf((4, 8)), axis=1, start=0, stop=9)
    with pytest.raises(ShapeError, match="stop - start is negative"):
        make("slice", leaf((4, 8)), axis=1, start=5, stop=3)
    seq = b.symbol("seq")
    with pytest.raises(ShapeError, match="size - stop is negative"):
        make("slice", leaf((seq,)), axis=0, start=0, stop=seq + 1)


def test_symbolic_slice(b):
    seq = b.symbol("seq")
    assert make("slice", leaf((seq, 8)), axis=0, start=seq - 1, stop=seq).shape == (1, 8)


def test_concat_must_agree_off_axis():
    with pytest.raises(ShapeError, match="differ off axis"):
        make("concat", leaf((2, 3)), leaf((2, 4)), axis=0)


def test_concat_adds_symbolic_lengths(b):
    seq, past = b.symbol("seq"), b.symbol("past")
    node = make("concat", leaf((2, past, 4)), leaf((2, seq, 4)), axis=1)
    assert node.shape == (2, past + seq, 4)


def test_matmul_inner_dimension():
    with pytest.raises(ShapeError, match="inner dimensions differ"):
        make("matmul", leaf((4, 8)), leaf((7, 2)))


def test_matmul_does_not_broadcast_batches():
    with pytest.raises(ShapeError, match="batch dimensions differ"):
        make("matmul", leaf((14, 4, 8)), leaf((2, 8, 4)))


def test_matmul_ranks_must_match():
    with pytest.raises(ShapeError, match="rank mismatch"):
        make("matmul", leaf((14, 4, 8)), leaf((8, 4)))


def test_gather_needs_integer_indices():
    with pytest.raises(ShapeError, match="indices must be integers"):
        make("gather", leaf((10, 4)), leaf((3,)))


def test_weights_cannot_be_symbolic(b):
    with pytest.raises(ShapeError, match="symbolic shape"):
        make("weight", name="w", dtype=f32, shape=(b.symbol("seq"), 4))


def test_arity_is_checked():
    with pytest.raises(ShapeError, match="takes 2 operands"):
        make("add", leaf((4,)))


def test_unknown_primitive():
    with pytest.raises(ValueError, match="unknown primitive"):
        make("softmax", leaf((4,)))


# -- graphs ------------------------------------------------------------------


def test_undeclared_input_is_rejected():
    stray = leaf((4,), name="stray")
    with pytest.raises(GraphError, match="used but not declared"):
        Graph([], {"y": make("neg", stray)})


def test_duplicate_input_names_are_rejected():
    a, b_ = leaf((4,), name="x"), leaf((4,), name="x")
    with pytest.raises(GraphError, match="two inputs named"):
        Graph([a, b_], {"y": make("add", a, b_)})


def test_two_weights_with_one_name_are_rejected():
    w1 = make("weight", name="w", dtype=f32, shape=(4,))
    w2 = make("weight", name="w", dtype=f32, shape=(4,))
    with pytest.raises(GraphError, match="two weight nodes"):
        Graph([], {"y": make("add", w1, w2)})


def test_a_symbol_no_input_binds_is_rejected(b):
    seq = b.symbol("seq")
    x = leaf((seq, 4))
    with pytest.raises(GraphError, match="which no input shape binds"):
        Graph([x], {"y": make("dim", expr=b.symbol("past"))})


def test_unused_inputs_stay_in_the_signature():
    x, unused = leaf((4,), name="x"), leaf((4,), name="unused")
    graph = Graph([x, unused], {"y": make("neg", x)})
    assert graph.input_names == ["x", "unused"]


def test_topological_order_and_sharing(b):
    x = b.input("x", f32, (4,))
    y = x * x
    graph = b.build({"a": y + 1.0, "b": y})
    order = graph.nodes
    assert order.index(y.node) < order.index(graph.outputs["a"])
    assert sum(1 for n in order if n is y.node) == 1


def test_deep_graphs_do_not_hit_the_recursion_limit(b):
    x = b.input("x", f32, (4,))
    for _ in range(5000):
        x = -x
    assert len(b.build({"y": x}).nodes) == 5001


def test_validate_catches_a_stale_type(b):
    x = b.input("x", f32, (4,))
    graph = b.build({"y": -x})
    graph.outputs["y"].shape = (5,)
    with pytest.raises(GraphError, match="is typed"):
        graph.validate()


# -- printing ----------------------------------------------------------------


def test_printer_golden(b):
    seq = b.symbol("seq")
    x = b.input("x", f32, (seq, 4))
    w = b.weight("w", (4,))
    y = (x * x).sum(-1) / 4.0 + 1e-6
    graph = b.build({"y": y.sqrt() * w})
    assert str(graph) == "\n".join(
        [
            "graph test(x: f32[seq, 4]) {",
            '  %0 = input "x"                           : f32[seq, 4]',
            "  %1 = mul %0, %0                          : f32[seq, 4]",
            "  %2 = reduce_sum %1 axis=-1               : f32[seq, 1]",
            "  %3 = const 4.0                           : f32[]",
            "  %4 = reshape %3 shape=[1, 1]             : f32[1, 1]",
            "  %5 = broadcast %4 shape=[seq, 1]         : f32[seq, 1]",
            "  %6 = div %2, %5                          : f32[seq, 1]",
            "  %7 = const 1e-06                         : f32[]",
            "  %8 = reshape %7 shape=[1, 1]             : f32[1, 1]",
            "  %9 = broadcast %8 shape=[seq, 1]         : f32[seq, 1]",
            "  %10 = add %6, %9                         : f32[seq, 1]",
            "  %11 = sqrt %10                           : f32[seq, 1]",
            "  %12 = broadcast %11 shape=[seq, 4]       : f32[seq, 4]",
            '  %13 = weight "w"                         : f32[4]',
            "  %14 = reshape %13 shape=[1, 4]           : f32[1, 4]",
            "  %15 = broadcast %14 shape=[seq, 4]       : f32[seq, 4]",
            "  %16 = mul %12, %15                       : f32[seq, 4]",
            "  return y=%16",
            "}",
        ]
    )


def test_printer_is_deterministic(b):
    def build():
        g = GraphBuilder("d")
        x = g.input("x", f32, (g.symbol("seq"), 4))
        return str(g.build({"y": concat([x, -x], axis=0), "z": where(x >= 0.0, x, 0.0)}))

    assert build() == build()
