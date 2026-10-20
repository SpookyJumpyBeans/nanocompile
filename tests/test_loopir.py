"""The loop IR's phase 4 additions, run as scalar C.

Local arrays, blocks and loop kinds are what schedules produce. Before any
schedule uses them, each has to mean what it says when rendered plainly.
"""

from __future__ import annotations

import numpy as np

from nanocompile import dtypes
from nanocompile import loopir as L
from tests.kernels import run_kernel

F = dtypes.f32
I = dtypes.i64


def kernel(body, nparams=1):
    params = (L.Param(F, True),) + tuple(L.Param(F, False) for _ in range(nparams))
    return L.Kernel("test", params, tuple(body))


def test_local_array_accumulates_and_reads_back():
    # acc[j] = sum over k of x[k*4 + j]; out[j] = acc[j]
    j, k = L.Var("j"), L.Var("k")
    body = [
        L.Array("acc", 4, L.const(0.0, F)),
        L.Loop("k", L.const(3), (
            L.Loop("j", L.const(4), (
                L.ArrayStore("acc", j, L.binary("add", L.Local("acc", j, F), L.Load(1, L.add(L.mul(k, L.const(4)), j), F))),
            )),
        )),
        L.Loop("j", L.const(4), (L.Store(0, j, L.Local("acc", j, F)),)),
    ]
    x = np.arange(12, dtype=np.float32)
    out = np.zeros(4, dtype=np.float32)
    assert run_kernel(kernel(body), out, [x]) == 0
    np.testing.assert_array_equal(out, x.reshape(3, 4).sum(axis=0))


def test_array_init_fills_every_element():
    i = L.Var("i")
    body = [
        L.Array("m", 5, L.const(-np.inf, F)),
        L.Loop("i", L.const(5), (L.Store(0, i, L.Local("m", i, F)),)),
    ]
    out = np.zeros(5, dtype=np.float32)
    run_kernel(kernel(body, 0), out, [])
    assert np.all(out == -np.inf)


def test_blocks_scope_their_names():
    # Two copies of the same body, each declaring t, as unrolling makes them.
    copies = [
        L.Block((L.Let("t", L.Load(1, L.const(c), F)), L.Store(0, L.const(c), L.binary("mul", L.Var("t", F), L.Var("t", F)))))
        for c in range(3)
    ]
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    run_kernel(kernel(copies), out, [x])
    np.testing.assert_array_equal(out, x * x)


def test_substitute_folds_constant_indices():
    i, j = L.Var("i"), L.Var("j")
    store = L.Store(0, L.add(L.mul(i, L.const(8)), j), L.Load(1, j, F))
    (folded,) = L.substitute((store,), {"i": L.const(2), "j": L.const(3)})
    assert folded.index == L.const(19)
    assert folded.value == L.Load(1, L.const(3), F)


def test_loop_kind_is_part_of_identity():
    body = (L.Store(0, L.Var("i"), L.const(1.0, F)),)
    assert L.Loop("i", L.const(4), body) != L.Loop("i", L.const(4), body, "parallel")
    assert L.Loop("i", L.const(4), body) == L.Loop("i", L.const(4), body, "serial")
