from __future__ import annotations

import numpy as np
import pytest

from nanocompile.symbolic import (
    Expr,
    as_expr,
    canonical,
    dims_equal,
    evaluate_shape,
    format_shape,
    product,
)

seq = Expr.symbol("seq")
past = Expr.symbol("past")


def test_arithmetic_is_canonical():
    assert past + seq == seq + past
    assert (seq + 3) - 3 == seq
    assert seq * 14 * 64 == seq * 896
    assert 2 * (seq + past) == seq + seq + past + past


def test_symbols_cancel_to_an_int():
    assert canonical((past + seq) - seq - past) == 0
    assert canonical((seq - 1) - seq + 5) == 4
    assert isinstance(canonical(seq - seq + 7), int)


def test_products_of_symbols():
    """Reshape compares element counts, and those multiply symbols together."""
    assert seq * past == past * seq
    assert (seq + 1) * (seq + 1) == seq * seq + 2 * seq + 1
    assert product([seq, 4, past]) == 4 * seq * past


def test_evaluate():
    assert (past + seq - 1).evaluate({"past": 10, "seq": 3}) == 12
    assert evaluate_shape((seq, 4, past + seq), {"seq": 2, "past": 5}) == (2, 4, 7)


def test_evaluate_names_the_missing_symbol():
    with pytest.raises(KeyError, match="past"):
        (past + seq).evaluate({"seq": 1})


def test_equality_with_ints():
    assert as_expr(4) == 4
    assert seq - seq == 0
    assert seq != 4
    assert dims_equal(seq - seq + 4, 4)


def test_as_symbol():
    assert seq.as_symbol == "seq"
    assert (seq + 0).as_symbol == "seq"
    assert (2 * seq).as_symbol is None
    assert (seq + past).as_symbol is None


def test_formatting_is_stable():
    assert str(past + seq - 1) == "past + seq - 1"
    assert str(seq * 14) == "14*seq"
    assert str(1 - seq) == "-seq + 1"
    assert format_shape((seq, 896)) == "[seq, 896]"


def test_numpy_integers_are_dimensions():
    assert as_expr(np.int64(5)) == 5


@pytest.mark.parametrize("bad", [True, 2.0, "seq", None])
def test_non_integers_are_not_dimensions(bad):
    with pytest.raises(TypeError):
        as_expr(bad)


def test_hash_matches_equality():
    assert len({seq + past, past + seq, seq * 1 + past}) == 1
