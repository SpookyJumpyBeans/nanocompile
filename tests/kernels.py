"""Compile and call a single loop-IR kernel, for tests of schedules.

Graph-level tests go through ``CompiledGraph``; a schedule acts on one kernel,
so its tests build or take a kernel and run it here, on NumPy arrays, against
the same kernel unscheduled.
"""

from __future__ import annotations

import ctypes

import numpy as np

from nanocompile import codegen_c
from nanocompile import loopir as L

_KERNEL = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int64))


def run_kernel(kernel: L.Kernel, out: np.ndarray, operands: list[np.ndarray], syms: list[int] = ()) -> int:
    """Run ``kernel`` writing into ``out``; returns the kernel's status."""
    source = codegen_c.render_library([kernel])
    path, _ = codegen_c.build_library(source)
    library = ctypes.CDLL(str(path))
    function = _KERNEL((codegen_c.kernel_name(0), library))
    arrays = [out] + [np.ascontiguousarray(a) for a in operands]
    pointers = (ctypes.c_void_p * len(arrays))(*[a.ctypes.data for a in arrays])
    values = (ctypes.c_int64 * max(1, len(syms)))(*syms)
    return function(pointers, values)
