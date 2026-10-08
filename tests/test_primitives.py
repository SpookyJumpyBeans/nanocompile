"""Every primitive in the interpreter, against plain NumPy.

Each graph is built once with a symbolic ``seq`` and run with ``seq`` bound to
several sizes, including 1, the decode step, and the size where off-by-one
errors in broadcasting tend to hide. Comparisons are bitwise: the interpreter
calls the same NumPy function on the same arrays, so anything short of
identical is a bug in the interpreter, not rounding.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanocompile import interpreter
from nanocompile.dtypes import bool_, f32, i8, i32, i64
from nanocompile.frontend import GraphBuilder, concat, gather, where
from nanocompile.interpreter import InterpreterError
from nanocompile.ir import OPS

SIZES = [1, 3, 7]


def rand(rng, *shape, dtype=np.float32):
    if np.dtype(dtype).kind == "f":
        return rng.standard_normal(shape).astype(dtype)
    return rng.integers(-5, 6, size=shape).astype(dtype)


def run1(graph, **inputs):
    return interpreter.run(graph, inputs, {})


def build_unary(fn, dtype=f32):
    b = GraphBuilder()
    x = b.input("x", dtype, (b.symbol("seq"), 5))
    return b.build({"y": fn(x)})


UNARY = {
    "neg": (lambda x: -x, np.negative),
    "exp": (lambda x: x.exp(), np.exp),
    "sqrt": (lambda x: x.sqrt(), lambda a: np.sqrt(np.abs(a))),
    "reciprocal": (lambda x: x.reciprocal(), np.reciprocal),
    "sin": (lambda x: x.sin(), np.sin),
    "cos": (lambda x: x.cos(), np.cos),
}


@pytest.mark.parametrize("op", sorted(UNARY))
@pytest.mark.parametrize("seq", SIZES)
def test_unary(op, seq):
    build, reference = UNARY[op]
    rng = np.random.default_rng(seq)
    x = rand(rng, seq, 5)
    if op == "sqrt":
        x = np.abs(x)
    out = run1(build_unary(build), x=x)["y"]
    np.testing.assert_array_equal(out, reference(x))
    assert out.dtype == np.float32


BINARY = {
    "add": (lambda a, b: a + b, np.add),
    "sub": (lambda a, b: a - b, np.subtract),
    "mul": (lambda a, b: a * b, np.multiply),
    "div": (lambda a, b: a / b, np.divide),
    "less_equal": (lambda a, b: a <= b, np.less_equal),
}


@pytest.mark.parametrize("op", sorted(BINARY))
@pytest.mark.parametrize("seq", SIZES)
def test_binary(op, seq):
    build, reference = BINARY[op]
    b = GraphBuilder()
    seq_ = b.symbol("seq")
    x, y = b.input("x", f32, (seq_, 5)), b.input("y", f32, (seq_, 5))
    graph = b.build({"z": build(x, y)})
    rng = np.random.default_rng(seq)
    xv, yv = rand(rng, seq, 5), rand(rng, seq, 5)
    np.testing.assert_array_equal(run1(graph, x=xv, y=yv)["z"], reference(xv, yv))


@pytest.mark.parametrize("dtype", [i64, i32, i8])
def test_integer_arithmetic_stays_in_its_dtype(dtype):
    """``i8 + i8`` is ``i8``, wrapping, as it will be in generated C."""
    b = GraphBuilder()
    x = b.input("x", dtype, (4,))
    graph = b.build({"y": x + x, "z": x * 3})
    xv = np.array([100, -100, 7, 0], dtype=dtype.numpy)
    out = run1(graph, x=xv)
    assert out["y"].dtype == dtype.numpy
    np.testing.assert_array_equal(out["y"], (xv + xv).astype(dtype.numpy))
    np.testing.assert_array_equal(out["z"], (xv * dtype.numpy.type(3)).astype(dtype.numpy))


def test_where():
    b = GraphBuilder()
    seq = b.symbol("seq")
    c, x, y = b.input("c", bool_, (seq, 3)), b.input("x", f32, (seq, 3)), b.input("y", f32, (seq, 3))
    graph = b.build({"z": where(c, x, y)})
    rng = np.random.default_rng(0)
    cv = rng.random((4, 3)) > 0.5
    xv, yv = rand(rng, 4, 3), rand(rng, 4, 3)
    np.testing.assert_array_equal(run1(graph, c=cv, x=xv, y=yv)["z"], np.where(cv, xv, yv))


def test_cast():
    b = GraphBuilder()
    x = b.input("x", i64, (5,))
    graph = b.build({"y": x.cast(f32), "z": x.cast(i8)})
    xv = np.array([0, 1, 2**24 + 1, -3, 300], dtype=np.int64)
    out = run1(graph, x=xv)
    np.testing.assert_array_equal(out["y"], xv.astype(np.float32))
    np.testing.assert_array_equal(out["z"], xv.astype(np.int8))


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("seq", SIZES)
def test_reductions(axis, seq):
    b = GraphBuilder()
    x = b.input("x", f32, (b.symbol("seq"), 6))
    graph = b.build({"s": x.sum(axis), "m": x.max(axis)})
    xv = rand(np.random.default_rng(seq), seq, 6)
    out = run1(graph, x=xv)
    np.testing.assert_array_equal(out["s"], np.sum(xv, axis=axis, keepdims=True))
    np.testing.assert_array_equal(out["m"], np.max(xv, axis=axis, keepdims=True))


def test_integer_sum_does_not_widen():
    b = GraphBuilder()
    x = b.input("x", i8, (4,))
    out = run1(b.build({"s": x.sum(0)}), x=np.full(4, 100, dtype=np.int8))["s"]
    assert out.dtype == np.int8
    assert out[0] == np.int8(400 - 512)


@pytest.mark.parametrize("seq", SIZES)
def test_movement(seq):
    b = GraphBuilder()
    s = b.symbol("seq")
    x = b.input("x", f32, (s, 4, 6))
    graph = b.build(
        {
            "reshape": x.reshape(s, 24),
            "split": x.reshape(s, 4, 2, 3),
            "permute": x.permute(2, 0, 1),
            "broadcast": x.slice(1, 0, 1).broadcast_to((s, 5, 6)),
            "slice": x.slice(-1, 2, 5),
            "last": x.slice(0, s - 1, s),
            "concat": concat([x, -x, x], axis=1),
        }
    )
    xv = rand(np.random.default_rng(seq), seq, 4, 6)
    out = run1(graph, x=xv)
    np.testing.assert_array_equal(out["reshape"], xv.reshape(seq, 24))
    np.testing.assert_array_equal(out["split"], xv.reshape(seq, 4, 2, 3))
    np.testing.assert_array_equal(out["permute"], xv.transpose(2, 0, 1))
    np.testing.assert_array_equal(out["broadcast"], np.broadcast_to(xv[:, :1], (seq, 5, 6)))
    np.testing.assert_array_equal(out["slice"], xv[..., 2:5])
    np.testing.assert_array_equal(out["last"], xv[-1:])
    np.testing.assert_array_equal(out["concat"], np.concatenate([xv, -xv, xv], axis=1))


def test_concat_with_an_empty_operand():
    """The uncached step: ``past`` is 0 and the cache operand has no rows."""
    b = GraphBuilder()
    past, seq = b.symbol("past"), b.symbol("seq")
    p, n = b.input("p", f32, (2, past, 4)), b.input("n", f32, (2, seq, 4))
    graph = b.build({"y": concat([p, n], axis=1)})
    nv = rand(np.random.default_rng(0), 2, 3, 4)
    out = run1(graph, p=np.zeros((2, 0, 4), np.float32), n=nv)["y"]
    np.testing.assert_array_equal(out, nv)


@pytest.mark.parametrize("seq", SIZES)
def test_matmul(seq):
    b = GraphBuilder()
    s = b.symbol("seq")
    x, w = b.input("x", f32, (s, 8)), b.input("w", f32, (5, 8))
    q, k = b.input("q", f32, (3, s, 4)), b.input("k", f32, (3, s, 4))
    graph = b.build({"y": x @ w.swapaxes(0, 1), "scores": q @ k.swapaxes(-1, -2)})
    rng = np.random.default_rng(seq)
    xv, wv, qv, kv = rand(rng, seq, 8), rand(rng, 5, 8), rand(rng, 3, seq, 4), rand(rng, 3, seq, 4)
    out = run1(graph, x=xv, w=wv, q=qv, k=kv)
    np.testing.assert_array_equal(out["y"], xv @ wv.T)
    np.testing.assert_array_equal(out["scores"], qv @ kv.transpose(0, 2, 1))


def test_gather():
    b = GraphBuilder()
    table, ids = b.input("t", f32, (10, 3)), b.input("i", i64, (b.symbol("seq"),))
    graph = b.build({"y": gather(table, ids)})
    tv = rand(np.random.default_rng(0), 10, 3)
    iv = np.array([9, 0, 0, 4])
    np.testing.assert_array_equal(run1(graph, t=tv, i=iv)["y"], tv[iv])


@pytest.mark.parametrize("bad", [-1, 10])
def test_gather_rejects_out_of_range_indices(bad):
    """NumPy would wrap -1 to the last row and return plausible numbers."""
    b = GraphBuilder()
    table, ids = b.input("t", f32, (10, 3)), b.input("i", i64, (2,))
    graph = b.build({"y": gather(table, ids)})
    with pytest.raises(InterpreterError, match=f"index {bad}"):
        run1(graph, t=np.zeros((10, 3), np.float32), i=np.array([0, bad]))


def test_iota_and_dim():
    b = GraphBuilder()
    seq, past = b.symbol("seq"), b.symbol("past")
    b.input("cache", f32, (past,))
    x = b.input("x", f32, (seq,))
    graph = b.build(
        {
            "positions": b.iota((seq,), axis=0) + b.dim(past),
            "cols": b.iota((seq, past + seq), axis=1),
            "total": b.dim(past + seq),
            "x": x,
        }
    )
    out = run1(graph, cache=np.zeros(5, np.float32), x=np.zeros(3, np.float32))
    np.testing.assert_array_equal(out["positions"], [5, 6, 7])
    np.testing.assert_array_equal(out["cols"], np.broadcast_to(np.arange(8), (3, 8)))
    assert out["total"].shape == () and out["total"] == 8


def test_every_primitive_is_covered():
    """A primitive added to the IR without a test here fails this test."""
    tested = set(UNARY) | set(BINARY) | {
        "input", "weight", "const", "iota", "dim", "where", "cast",
        "reduce_sum", "reduce_max", "reshape", "permute", "broadcast",
        "slice", "concat", "matmul", "gather",
    }
    assert set(OPS) == tested
    assert set(OPS) - {"input", "weight"} == set(interpreter.KERNELS)


# -- the interpreter's own checks ---------------------------------------------


def two_input_graph():
    b = GraphBuilder()
    seq = b.symbol("seq")
    x, y = b.input("x", f32, (seq, 4)), b.input("y", f32, (seq, 4))
    return b.build({"z": x + y})


def test_symbol_bound_inconsistently():
    with pytest.raises(InterpreterError, match="already bound"):
        run1(two_input_graph(), x=np.zeros((3, 4), np.float32), y=np.zeros((2, 4), np.float32))


def test_wrong_input_dtype():
    with pytest.raises(InterpreterError, match="float64"):
        run1(two_input_graph(), x=np.zeros((3, 4)), y=np.zeros((3, 4), np.float32))


def test_wrong_concrete_dimension():
    with pytest.raises(InterpreterError, match="expected"):
        run1(two_input_graph(), x=np.zeros((3, 5), np.float32), y=np.zeros((3, 5), np.float32))


def test_missing_input():
    with pytest.raises(InterpreterError, match="missing"):
        run1(two_input_graph(), x=np.zeros((3, 4), np.float32))


def test_missing_and_misshapen_weights():
    b = GraphBuilder()
    graph = b.build({"y": -b.weight("w", (4,))})
    with pytest.raises(InterpreterError, match="missing weights: w"):
        interpreter.run(graph, {}, {})
    with pytest.raises(InterpreterError, match="expected f32\\[4\\]"):
        interpreter.run(graph, {}, {"w": np.zeros(5, np.float32)})


def test_symbolic_slice_bounds_are_checked_at_run_time():
    b = GraphBuilder()
    seq, past = b.symbol("seq"), b.symbol("past")
    b.input("p", f32, (past,))
    x = b.input("x", f32, (seq,))
    graph = b.build({"y": x.slice(0, 0, past)})
    with pytest.raises(InterpreterError, match="out of bounds"):
        run1(graph, p=np.zeros(5, np.float32), x=np.zeros(3, np.float32))
