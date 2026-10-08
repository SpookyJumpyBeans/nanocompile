from __future__ import annotations

import numpy as np
import pytest

from nanocompile import interpreter
from nanocompile.dtypes import f32, i64
from nanocompile.frontend import GraphBuilder, broadcast_shape, where
from nanocompile.ir import ShapeError


def ops_of(tensor):
    """The primitives on the path from a tensor back to the leaves."""
    seen, stack, ops = set(), [tensor.node], []
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        ops.append(node.op)
        stack.extend(node.inputs)
    return ops


def test_broadcast_shape_follows_numpy():
    assert broadcast_shape((4, 1), (3,)) == (4, 3)
    assert broadcast_shape((1,), (2, 5)) == (2, 5)
    with pytest.raises(ShapeError):
        broadcast_shape((4, 2), (3,))


def test_broadcasting_inserts_explicit_nodes():
    b = GraphBuilder()
    x = b.input("x", f32, (b.symbol("seq"), 8))
    w = b.weight("w", (8,))
    y = x * w
    assert y.shape == x.shape
    assert ops_of(y).count("reshape") == 1
    assert ops_of(y).count("broadcast") == 1


def test_equal_shapes_insert_nothing():
    b = GraphBuilder()
    x = b.input("x", f32, (4, 8))
    assert sorted(ops_of(x + x)) == ["add", "input"]


def test_python_scalars_take_the_tensor_dtype():
    """``x * 0.5`` on f32 is an f32 multiply, never a float64 promotion."""
    b = GraphBuilder()
    x = b.input("x", f32, (3,))
    graph = b.build({"y": x * 0.5, "z": 1.0 - x})
    out = interpreter.run(graph, {"x": np.array([1, 2, 3], np.float32)}, {})
    assert out["y"].dtype == np.float32
    np.testing.assert_array_equal(out["z"], [0, -1, -2])


def test_float_scalar_cannot_meet_an_integer_tensor():
    b = GraphBuilder()
    x = b.input("x", i64, (3,))
    with pytest.raises(TypeError, match="float scalar"):
        x * 0.5


def test_integer_scalar_on_integer_tensor():
    b = GraphBuilder()
    x = b.input("x", i64, (3,))
    assert (x + 1).dtype == i64


def test_where_with_scalar_branches():
    b = GraphBuilder()
    x = b.input("x", f32, (4,))
    graph = b.build({"y": where(x >= 0.0, x, 0.0)})
    out = interpreter.run(graph, {"x": np.array([-1, 2, -3, 4], np.float32)}, {})
    np.testing.assert_array_equal(out["y"], [0, 2, 0, 4])


def test_tensors_have_no_truth_value():
    b = GraphBuilder()
    x = b.input("x", f32, (4,))
    with pytest.raises(TypeError, match="no value"):
        if x:
            pass


def test_tensors_from_two_builders_do_not_mix():
    x = GraphBuilder().input("x", f32, (4,))
    y = GraphBuilder().input("y", f32, (4,))
    with pytest.raises(ValueError, match="different builders"):
        x + y


def test_weight_requested_twice_is_one_node():
    b = GraphBuilder()
    assert b.weight("w", (4, 2)).node is b.weight("w", (4, 2)).node


def test_weight_requested_with_another_shape():
    b = GraphBuilder()
    b.weight("w", (4, 2))
    with pytest.raises(ShapeError, match="already declared"):
        b.weight("w", (2, 4))


def test_reshape_and_cast_to_the_same_type_are_free():
    b = GraphBuilder()
    x = b.input("x", f32, (4,))
    assert x.reshape(4) is x
    assert x.cast(f32) is x


def test_constants_are_frozen():
    b = GraphBuilder()
    value = np.arange(3, dtype=np.float32)
    c = b.const(value)
    value[0] = 99
    assert c.node.attrs["value"][0] == 0
    with pytest.raises(ValueError):
        c.node.attrs["value"][0] = 1
