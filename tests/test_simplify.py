"""Index simplification: only divisions the loop ranges prove away.

Each rewrite is checked two ways: that it produces the expected simple form,
and that it computes the same value as the original for every combination
of loop variables in range. A simplification that is wrong for one index is
a kernel that reads the wrong element, so the second check is exhaustive.
"""

from __future__ import annotations

import itertools

import pytest

from nanocompile import loopir as L
from nanocompile.simplify import Simplifier, simplify_stmts

v = L.Var


def evaluate(x: L.Scalar, env: dict[str, int], syms=(5, 9)) -> int:
    if isinstance(x, L.Const):
        return int(x.value)
    if isinstance(x, L.Var):
        return env[x.name]
    if isinstance(x, L.Sym):
        return syms[x.index]
    a, b = evaluate(x.a, env, syms), evaluate(x.b, env, syms)
    ops = {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b,
           "idiv": lambda: a // b, "mod": lambda: a % b}
    return ops[x.op]()


def exhaustively_equal(before, after, ranges):
    names = sorted(ranges)
    for values in itertools.product(*(range(int(hi) + 1) for _, hi in (ranges[n] for n in names))):
        env = dict(zip(names, values))
        assert evaluate(before, env) == evaluate(after, env), env


def split_j():
    # j = jo * 32 + jv * 8 + ji, the shape a matmul schedule gives j.
    return L.add(L.add(L.mul(v("jo"), L.const(32)), L.mul(v("jv"), L.const(8))), v("ji"))


RANGES = {"jo": (0, 3), "jv": (0, 3), "ji": (0, 7), "k": (0, 5)}


def test_division_by_the_lane_count_drops_the_lane():
    s = Simplifier(RANGES)
    x = L.idiv(split_j(), L.const(8))
    out = s.simplify(x)
    assert out == L.add(L.mul(v("jo"), L.const(4)), v("jv"))
    exhaustively_equal(x, out, RANGES)


def test_modulo_keeps_only_the_lane():
    s = Simplifier(RANGES)
    x = L.mod(split_j(), L.const(8))
    assert s.simplify(x) == v("ji")
    exhaustively_equal(x, s.simplify(x), RANGES)


def test_packed_weight_index_becomes_contiguous_in_the_lane():
    # Row-major W[j, k] at f = j * K + k, read packed in blocks of 8 rows.
    K = 6
    f = L.add(L.mul(split_j(), L.const(K)), v("k"))
    row, col = L.idiv(f, L.const(K)), L.mod(f, L.const(K))
    packed = L.add(L.add(L.mul(L.idiv(row, L.const(8)), L.const(K * 8)), L.mul(col, L.const(8))), L.mod(row, L.const(8)))
    s = Simplifier(RANGES)
    out = s.simplify(packed)
    assert s.coefficient(out, "ji") == 1
    exhaustively_equal(packed, out, RANGES)
    assert "idiv" not in repr(out) and "mod" not in repr(out)


def test_division_is_kept_when_the_remainder_could_overflow():
    s = Simplifier({"i": (0, 9)})
    x = L.idiv(v("i"), L.const(8))   # i may be 8 or 9
    assert s.simplify(x) == x


def test_unknown_ranges_are_assumed_nothing():
    s = Simplifier({})
    x = L.idiv(L.add(L.mul(v("row"), L.const(8)), v("t")), L.const(8))
    assert s.simplify(x) == x
    y = L.idiv(L.add(L.mul(v("i"), L.const(8)), L.Sym(0)), L.const(8))
    assert Simplifier({"i": (0, 3)}).simplify(y) == y


@pytest.mark.parametrize("expr, var, expected", [
    (L.add(L.mul(v("i"), L.const(896)), v("k")), "k", 1),
    (L.add(L.mul(v("i"), L.const(896)), v("k")), "i", 896),
    (L.mul(v("i"), L.Sym(0)), "i", None),     # symbolic coefficient
    (L.mul(v("i"), v("i")), "i", None),        # not linear
    (L.add(v("k"), L.const(3)), "i", 0),
])
def test_coefficient(expr, var, expected):
    assert Simplifier({}).coefficient(expr, var) == expected


def test_statements_take_ranges_from_their_loops():
    body = (L.Store(0, L.idiv(split_j(), L.const(8)), L.const(1.0, L.dtypes.f32)),)
    nest = L.Loop("jo", L.const(4), (L.Loop("jv", L.const(4), (L.Loop("ji", L.const(8), body),)),))
    (out,) = simplify_stmts((nest,))
    store = out.body[0].body[0].body[0]
    assert store.index == L.add(L.mul(v("jo"), L.const(4)), v("jv"))
    # Outside the ji loop nothing is known about ji, so the same store is kept.
    (alone,) = simplify_stmts(body)
    assert alone.index == body[0].index
