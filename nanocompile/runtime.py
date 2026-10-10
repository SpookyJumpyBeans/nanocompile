"""Running a compiled graph: memory planning and the kernel-call loop.

Intermediates live in one arena. The planner walks the calls in order,
gives each kernel's output a block, and returns a block to the free list after
the last call that reads it. Sizes depend on ``seq`` and ``past``, so the plan
is made per run, which is cheap next to the run itself.

One rule keeps it correct: a call's output is allocated *before* the blocks it
frees are returned, so no kernel ever writes over one of its own operands.
That is also what makes ``restrict`` on the kernel pointers true.

Inputs and weights are passed in place. Inputs are made contiguous first;
the KV cache view the driver hands in is not, and copying it is one of the
costs phase 7 removes by owning the cache.
"""

from __future__ import annotations

import ctypes
import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from nanocompile import codegen_c
from nanocompile.interpreter import check_arguments
from nanocompile.ir import Graph
from nanocompile.fusion import fuse as fuse_graph
from nanocompile.lower import Buffer, Program, lower
from nanocompile.passes import optimize
from nanocompile.symbolic import Dim, evaluate_shape

ALIGN = 64


class KernelError(RuntimeError):
    """A kernel rejected its input (an out-of-range gather index)."""


@dataclass(frozen=True)
class Plan:
    offsets: dict[Buffer, int]
    arena_bytes: int
    temp_bytes: int  # what the temps would need with no reuse at all


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def plan_memory(program: Program, last_use: Mapping[Buffer, int], bound: Mapping[str, int]) -> Plan:
    """First-fit-by-best-size reuse of freed blocks, in call order."""
    free: list[tuple[int, int]] = []          # (offset, size), sorted by offset
    top = 0
    offsets: dict[Buffer, int] = {}
    total = 0
    dying: dict[int, list[Buffer]] = {}
    for buf, index in last_use.items():
        dying.setdefault(index, []).append(buf)

    def allocate(size: int) -> int:
        nonlocal top
        best = None
        for i, (offset, block) in enumerate(free):
            if block >= size and (best is None or block < free[best][1]):
                best = i
        if best is None:
            offset, top = top, top + size
            return offset
        offset, block = free.pop(best)
        if block > size:
            free.append((offset + size, block - size))
            free.sort()
        return offset

    def release(offset: int, size: int) -> None:
        free.append((offset, size))
        free.sort()
        merged: list[tuple[int, int]] = []
        for start, block in free:
            if merged and merged[-1][0] + merged[-1][1] == start:
                merged[-1] = (merged[-1][0], merged[-1][1] + block)
            else:
                merged.append((start, block))
        free[:] = merged

    sizes: dict[Buffer, int] = {}
    for index, call in enumerate(program.calls):
        out = call.output
        if out.kind == "temp" and out not in offsets:
            size = _align(max(1, _nbytes(out, bound)))
            sizes[out] = size
            offsets[out] = allocate(size)
            total += size
        for buf in dying.get(index, ()):
            release(offsets[buf], sizes[buf])
    return Plan(offsets, top, total)


def _nbytes(buf: Buffer, bound: Mapping[str, int]) -> int:
    return int(np.prod(evaluate_shape(buf.shape, bound), dtype=np.int64)) * buf.dtype.numpy.itemsize


def _offset_bytes(offset: Dim, itemsize: int, bound: Mapping[str, int]) -> int:
    return (offset if isinstance(offset, int) else offset.evaluate(bound)) * itemsize


_KERNEL_TYPE = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int64))


class CompiledGraph:
    """A graph lowered, compiled and loaded: call it like ``interpreter.run``.

    ``fuse=True`` (the default) optimizes the graph and fuses it (phase 3);
    ``fuse=False`` is phase 2's one kernel per primitive, kept as the baseline
    fusion is measured against and tested bitwise against.
    """

    def __init__(self, graph: Graph, fuse: bool = True) -> None:
        started = time.perf_counter()
        self.graph = graph
        self.fused = fuse
        self.program = fuse_graph(optimize(graph)) if fuse else lower(graph)
        self.source = codegen_c.render_library(self.program.kernels)
        self.library_path, self.cached = codegen_c.build_library(self.source)
        self.compile_seconds = time.perf_counter() - started

        self._library = ctypes.CDLL(str(self.library_path))
        self._kernels = [
            _KERNEL_TYPE((codegen_c.kernel_name(i), self._library))
            for i in range(len(self.program.kernels))
        ]
        self._last_use = self.program.last_uses()
        self._arena = np.empty(0, dtype=np.uint8)
        self.last_plan: Plan | None = None

    def _arena_base(self, size: int) -> int:
        if self._arena.nbytes < size + ALIGN:
            self._arena = np.empty(size + ALIGN, dtype=np.uint8)
        base = self._arena.ctypes.data
        return base + (-base) % ALIGN

    def __call__(
        self,
        inputs: Mapping[str, np.ndarray],
        weights: Mapping[str, np.ndarray],
        bindings: Mapping[str, int] | None = None,
        profile: dict[str, float] | None = None,
    ) -> dict[str, np.ndarray]:
        """Run the graph. ``profile``, if given, accumulates seconds per primitive."""
        program = self.program
        bound = check_arguments(self.graph, inputs, weights, bindings)
        syms = (ctypes.c_int64 * max(1, len(program.symbols)))(*[bound[s] for s in program.symbols])

        plan = plan_memory(program, self._last_use, bound)
        self.last_plan = plan
        arena = self._arena_base(plan.arena_bytes)

        keep: list[np.ndarray] = []           # owners of every address below
        addresses: dict[Buffer, int] = {}
        outputs: dict[str, np.ndarray] = {}
        for buf in program.buffers:
            if buf.kind == "input":
                array = np.ascontiguousarray(inputs[buf.name])
            elif buf.kind == "weight":
                array = np.ascontiguousarray(weights[buf.name])
            elif buf.kind == "const":
                array = buf.value
            elif buf.kind == "output":
                array = np.empty(evaluate_shape(buf.shape, bound), dtype=buf.dtype.numpy)
                outputs[buf.name] = array
            else:
                continue
            keep.append(array)
            addresses[buf] = array.ctypes.data
        for buf, offset in plan.offsets.items():
            addresses[buf] = arena + offset

        for call in program.calls:
            pointers = [addresses[call.output]]
            for view in call.operands:
                itemsize = view.dtype.numpy.itemsize
                pointers.append(addresses[view.buffer] + _offset_bytes(view.offset, itemsize, bound))
            args = (ctypes.c_void_p * len(pointers))(*pointers)
            if profile is None:
                status = self._kernels[call.kernel](args, syms)
            else:
                started = time.perf_counter()
                status = self._kernels[call.kernel](args, syms)
                elapsed = time.perf_counter() - started
                profile[call.primitive] = profile.get(call.primitive, 0.0) + elapsed
            if status != 0:
                raise self._error(call, status, inputs, bound)

        return {name: outputs[name] for name in program.outputs}

    def _error(self, call, status: int, inputs, bound) -> KernelError:
        """Explain a failed kernel from the graph node it came from.

        The node, not the call's operand order: a fused gather may list its
        operands in any order.
        """
        node = self.program.graph.nodes[call.node]
        if node.op == "gather":
            table, ids = node.inputs
            rows = table.shape[0] if isinstance(table.shape[0], int) else table.shape[0].evaluate(bound)
            if ids.op == "input":
                values = np.asarray(inputs[ids.attrs["name"]])
                bad = values[(values < 0) | (values >= rows)].flat[0]
                return KernelError(f"gather index {bad} is outside 0..{rows - 1}")
            return KernelError(f"gather index outside 0..{rows - 1} (node %{call.node})")
        return KernelError(f"kernel for node %{call.node} ({call.primitive}) failed with {status}")


class CBackend:
    """A ``Runner`` for ``Qwen2``: compiles each graph once, on first use."""

    def __init__(self, fuse: bool = True) -> None:
        self.fuse = fuse
        self._compiled: dict[int, tuple[Graph, CompiledGraph]] = {}

    def compiled(self, graph: Graph) -> CompiledGraph:
        entry = self._compiled.get(id(graph))
        if entry is None or entry[0] is not graph:
            entry = (graph, CompiledGraph(graph, fuse=self.fuse))
            self._compiled[id(graph)] = entry
        return entry[1]

    def __call__(self, graph: Graph, inputs, weights) -> dict[str, np.ndarray]:
        return self.compiled(graph)(inputs, weights)
