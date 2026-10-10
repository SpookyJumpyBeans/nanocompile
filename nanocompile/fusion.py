"""Fusion: deciding which values become buffers, and generating the kernels.

Phase 2 gave every compute node its own kernel and its own buffer. Here most
nodes get neither. A node is *realized* (written to a buffer by a kernel of
its own) only when something requires it:

* matmuls and gathers are realized, unless a matmul folds into an epilogue;
* graph outputs are realized;
* reductions are realized, unless they fold into a row kernel;
* an elementwise node is realized when its value is needed by more than one
  kernel, or feeds a matmul operand. Otherwise it is inlined into the one
  kernel that consumes it and computed there, per element, never stored.

Index-only primitives (``reshape``, ``permute``, ``broadcast``, ``slice``,
``concat``) are never realized. A kernel reading through them computes the
index instead: a strided load where the access is still a strided view, a
divide and modulo where a reshape merges or splits axes the strides cannot
express, and a select where a ``concat`` joins operands. That is what keeps
``repeat_kv`` from materializing seven copies of every key, and the cache
``concat`` from rewriting the history every step.

Three kernel shapes come out of it:

* **elementwise**: one loop nest over the output, the inlined expression in
  the innermost body.
* **row**: for each row, reductions over the last axis into accumulators,
  then one loop writing the row. ``rms_norm`` and ``softmax`` are one each.
* **matmul**: a k loop accumulating one or more products, then an epilogue
  over the accumulators. A bias, a residual add or ``silu(gate) * up`` is
  computed as the result is stored; ``gate`` and ``up`` share one k loop.

Fusion moves where each operation happens and never the order of the
operations that produce one value, so with the compile flags of phase 2 the
fused code is bitwise identical to the unfused code. The tests hold it to
exactly that.

When a planned fusion turns out not to be expressible (a reduction's result
read at an index that is not its row, say), generating the kernel refuses
it, and planning runs again without that fusion. The plan is a heuristic; the
generator is the authority on what is legal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nanocompile import dtypes
from nanocompile import loopir as L
from nanocompile.ir import Graph, Node
from nanocompile.lower import ELEMENTWISE, Buffer, Call, Program, View, _cast, contiguous_strides
from nanocompile.symbolic import Dim, Expr, as_expr, canonical, dims_equal, product, shapes_equal

INDEX_OPS = frozenset({"reshape", "permute", "broadcast", "slice", "concat"})
LEAF_OPS = frozenset({"input", "weight", "const"})
REDUCE_OPS = frozenset({"reduce_sum", "reduce_max"})
TRANSCENDENTAL = frozenset({"exp", "sin", "cos"})
ELEMENTWISE_OPS = frozenset(ELEMENTWISE) | {"cast", "iota", "dim"}


class _Refuse(Exception):
    def __init__(self, node: Node, kind: str) -> None:
        super().__init__(f"cannot fuse {node.op} as {kind}")
        self.node = node
        self.kind = kind


# -- planning ----------------------------------------------------------------


@dataclass
class Plan:
    realized: set[Node]
    root: dict[Node, Node | None]          # inlined node -> its kernel; None = recomputed anywhere
    row: dict[Node, Node] = field(default_factory=dict)        # reduction -> row kernel root
    epilogue: dict[Node, Node] = field(default_factory=dict)   # matmul -> epilogue kernel root

    def reductions_of(self, root: Node) -> list[Node]:
        return [r for r, t in self.row.items() if t is root]

    def matmuls_of(self, root: Node) -> list[Node]:
        return [m for m, t in self.epilogue.items() if t is root]


class _Planner:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.order = {n: i for i, n in enumerate(graph.nodes)}
        self.outputs = set(graph.outputs.values())
        self.users: dict[Node, list[Node]] = {n: [] for n in graph.nodes}
        for n in graph.nodes:
            for x in dict.fromkeys(n.inputs):
                self.users[x].append(n)
        self._free: dict[Node, bool] = {}
        self._consumers: dict[Node, list[tuple[Node, bool]]] = {}
        self.refused: set[tuple[Node, str]] = set()

    def free(self, n: Node) -> bool:
        """Computable from indices and constants alone, cheaply: recompute it anywhere.

        Positions, the causal mask and RoPE's angles are free; the tables of
        cosines are not (transcendental), and nothing that reads an input or a
        weight is.
        """
        if n not in self._free:
            if n.op in ("iota", "dim", "const"):
                self._free[n] = True
            elif n.op in ("input", "weight") or n.op in TRANSCENDENTAL or n.op not in (ELEMENTWISE_OPS | INDEX_OPS):
                self._free[n] = False
            else:
                self._free[n] = all(self.free(x) for x in n.inputs)
        return self._free[n]

    def consumers(self, n: Node) -> list[tuple[Node, bool]]:
        """Compute nodes that read ``n``, looking through index-only nodes.

        Each comes with whether it reads ``n`` as a matmul operand. An
        index-only node that is a graph output counts as a consumer: it
        becomes a copy kernel of its own.
        """
        if n not in self._consumers:
            found: dict[Node, bool] = {}
            stack = list(self.users[n])
            seen: set[Node] = set()
            while stack:
                u = stack.pop()
                if u in seen:
                    continue
                seen.add(u)
                if u.op in INDEX_OPS:
                    if u in self.outputs:
                        found[u] = found.get(u, False)
                    stack.extend(self.users[u])
                else:
                    found[u] = found.get(u, False) or u.op == "matmul"
            self._consumers[n] = sorted(found.items(), key=lambda kv: self.order[kv[0]])
        return self._consumers[n]

    def assign(self, row: dict[Node, Node], epilogue: dict[Node, Node]) -> Plan:
        realized: set[Node] = set()
        root: dict[Node, Node | None] = {}

        def realize(n: Node) -> None:
            realized.add(n)
            root[n] = n

        for n in reversed(self.graph.nodes):
            if n in self.outputs:
                realize(n)
            elif n.op in LEAF_OPS or n.op in INDEX_OPS:
                continue
            elif n.op == "gather":
                realize(n)
            elif n.op == "matmul":
                if n in epilogue:
                    root[n] = epilogue[n]
                else:
                    realize(n)
            elif n.op in REDUCE_OPS:
                if n in row:
                    root[n] = row[n]
                else:
                    realize(n)
            elif self.free(n):
                root[n] = None
            else:
                consumers = self.consumers(n)
                kernels = {root[c] for c, _ in consumers if root.get(c) is not None}
                if any(via_matmul for _, via_matmul in consumers) or len(kernels) != 1:
                    realize(n)
                else:
                    root[n] = kernels.pop()
        return Plan(realized, root, dict(row), dict(epilogue))

    def _unique_root(self, plan: Plan, n: Node) -> Node | None:
        consumers = self.consumers(n)
        if not consumers or any(via for _, via in consumers):
            return None
        kernels = {plan.root.get(c) for c, _ in consumers}
        if len(kernels) != 1:
            return None
        (t,) = kernels
        if t is None or t not in plan.realized or t.op not in ELEMENTWISE_OPS:
            return None
        return t

    def plan(self) -> Plan:
        row: dict[Node, Node] = {}
        epilogue: dict[Node, Node] = {}
        while True:
            plan = self.assign(row, epilogue)
            # A fusion decided earlier can target a node that a later fusion
            # inlined (softmax's max can fuse into the exp kernel before the
            # sum fuses and inlines exp). Drop those and decide them again.
            stale = [r for r, t in row.items() if t not in plan.realized]
            stale += [m for m, t in epilogue.items() if t not in plan.realized]
            if stale:
                for n in stale:
                    row.pop(n, None)
                    epilogue.pop(n, None)
                continue
            changed = False
            claimed: dict[Node, str] = {t: "row" for t in row.values()}
            claimed.update({t: "epilogue" for t in epilogue.values()})

            for n in reversed(self.graph.nodes):
                if n in self.outputs or n in row or n in epilogue:
                    continue
                if n.op in REDUCE_OPS and (n, "row") not in self.refused:
                    t = self._unique_root(plan, n)
                    source = n.inputs[0]
                    if (
                        t is not None
                        and claimed.get(t, "row") == "row"
                        and n.attrs["axis"] % n.rank == n.rank - 1
                        and t.rank == n.rank
                        and shapes_equal(t.shape[:-1], n.shape[:-1])
                        and dims_equal(source.shape[-1], t.shape[-1])
                    ):
                        row[n] = t
                        claimed[t] = "row"
                        changed = True
                elif n.op == "matmul" and (n, "epilogue") not in self.refused:
                    direct = self.users[n]
                    if not direct or any(u.op not in ELEMENTWISE_OPS for u in direct):
                        continue
                    t = self._unique_root(plan, n)
                    if t is None or claimed.get(t, "epilogue") != "epilogue":
                        continue
                    if not shapes_equal(t.shape, n.shape):
                        continue
                    k = n.inputs[0].shape[-1]
                    if any(not dims_equal(m.inputs[0].shape[-1], k) for m, r in epilogue.items() if r is t):
                        continue
                    epilogue[n] = t
                    claimed[t] = "epilogue"
                    changed = True
            if not changed:
                return plan


# -- kernel generation -------------------------------------------------------


class _Fuser:
    """Lowers a planned graph to a ``Program`` of fused kernels."""

    def __init__(self, graph: Graph, plan: Plan) -> None:
        self.graph = graph
        self.plan = plan
        self.symbols = graph.symbols
        self.buffers: list[Buffer] = []
        self.buffer_of: dict[Node, Buffer] = {}
        self._views: dict[Node, View | None] = {}
        self.kernels: list[L.Kernel] = []
        self.kernel_ids: dict[L.Kernel, int] = {}
        self.calls: list[Call] = []
        self.outputs: dict[str, Buffer] = {}

    # buffers and views

    def _buffer(self, kind: str, node: Node, name: str | None = None) -> Buffer:
        value = node.attrs.get("value") if node.op == "const" else None
        buf = Buffer(kind, node.dtype, node.shape, name, value)
        self.buffers.append(buf)
        return buf

    def leaf_buffer(self, node: Node) -> Buffer:
        if node not in self.buffer_of:
            kind = {"input": "input", "weight": "weight", "const": "const"}[node.op]
            name = node.attrs["name"] if kind != "const" else f"const{self.graph.ids[node]}"
            self.buffer_of[node] = self._buffer(kind, node, name)
        return self.buffer_of[node]

    def view_of(self, n: Node) -> View | None:
        """``n`` as a strided view of a buffer, if it is one."""
        if n in self._views:
            return self._views[n]
        view: View | None = None
        if n in self.buffer_of and n.op not in LEAF_OPS:
            view = View.of(self.buffer_of[n])
        elif n.op in LEAF_OPS:
            view = View.of(self.leaf_buffer(n))
        elif n.op in ("reshape", "permute", "broadcast", "slice"):
            base = self.view_of(n.inputs[0])
            if base is not None:
                view = _view_through(n, base)
        self._views[n] = view
        return view

    # programs

    def run(self) -> Program:
        roots = [n for n in self.graph.nodes if n in self.plan.realized]
        output_names: dict[Node, list[str]] = {}
        for name, node in self.graph.outputs.items():
            output_names.setdefault(node, []).append(name)

        for node in roots:
            names = output_names.get(node)
            if names:
                buf = self._buffer("output", node, names[0])
                self.outputs[names[0]] = buf
            else:
                buf = self._buffer("temp", node)
            if node.op not in LEAF_OPS:
                self.buffer_of[node] = buf
            self._root_buffers = getattr(self, "_root_buffers", {})
            self._root_buffers[node] = buf

        for node in roots:
            kernel, params = _KernelGen(self, node).generate()
            self._emit(kernel, self._root_buffers[node], params, node)
            for extra in output_names.get(node, [])[1:]:
                buf = self._buffer("output", node, extra)
                self.outputs[extra] = buf
                kernel, params = _KernelGen(self, node, copy=True).generate()
                self._emit(kernel, buf, params, node)

        ordered = {name: self.outputs[name] for name in self.graph.outputs}
        return Program(self.graph, self.symbols, self.kernels, self.calls, self.buffers, ordered)

    def _emit(self, kernel: L.Kernel, out: Buffer, params: list[View], node: Node) -> None:
        index = self.kernel_ids.get(kernel)
        if index is None:
            index = self.kernel_ids[kernel] = len(self.kernels)
            self.kernels.append(kernel)
        self.calls.append(Call(index, out, tuple(params), kernel.primitive, self.graph.ids[node]))


def _view_through(n: Node, base: View) -> View | None:
    """Phase 2's movement rules: a view of a view, where strides can express it."""
    if n.op == "reshape":
        if not base.is_contiguous:
            return None
        return View(base.buffer, n.shape, contiguous_strides(n.shape), base.offset)
    if n.op == "permute":
        perm = n.attrs["perm"]
        return View(base.buffer, n.shape, tuple(base.strides[p] for p in perm), base.offset)
    if n.op == "broadcast":
        strides = tuple(
            0 if dims_equal(have, 1) and not dims_equal(want, 1) else stride
            for have, want, stride in zip(base.shape, n.shape, base.strides)
        )
        return View(base.buffer, n.shape, strides, base.offset)
    axis = n.attrs["axis"] % n.rank
    offset = as_expr(base.offset) + as_expr(n.attrs["start"]) * base.strides[axis]
    return View(base.buffer, n.shape, base.strides, canonical(offset))


_PROBE = 1_000_003


def _probe(d: Dim) -> int:
    return d if isinstance(d, int) else d.evaluate({s: _PROBE + i for i, s in enumerate(sorted(d.symbols))})


def _reshape_index(in_shape, out_shape, idx: list[L.Scalar], symbols) -> list[L.Scalar]:
    """The input index that a reshape's output index reads.

    Axes are matched up in groups whose sizes multiply to the same thing
    (``[seq, 896]`` to ``[seq, 14, 64]`` is ``seq`` alone and ``896`` against
    ``14 x 64``). Within a group the output index is flattened and split
    across the input axes; a group of one axis on each side is the identity,
    so most reshapes cost no arithmetic at all.
    """
    result: list[L.Scalar] = []
    i = j = 0
    zero = L.const(0)
    while i < len(in_shape) or j < len(out_shape):
        if i < len(in_shape) and dims_equal(in_shape[i], 1):
            result.append(zero)
            i += 1
            continue
        if j < len(out_shape) and dims_equal(out_shape[j], 1):
            j += 1
            continue
        if i >= len(in_shape) or j >= len(out_shape):
            raise AssertionError("reshape element counts disagree")
        gin, gout = [i], [j]
        pin, pout = as_expr(in_shape[i]), as_expr(out_shape[j])
        i, j = i + 1, j + 1
        while pin != pout:
            if _probe(canonical(pin)) < _probe(canonical(pout)):
                pin = pin * in_shape[i]
                gin.append(i)
                i += 1
            else:
                pout = pout * out_shape[j]
                gout.append(j)
                j += 1
        # Flatten the output side of the group...
        flat: L.Scalar = L.const(0)
        stride: Dim = 1
        for axis in reversed(gout):
            flat = L.add(L.mul(idx[axis], L.from_dim(stride, symbols)), flat)
            stride = canonical(as_expr(stride) * out_shape[axis])
        # ...and split it across the input side.
        strides_in = contiguous_strides(tuple(in_shape[a] for a in gin))
        for position, axis in enumerate(gin):
            part = L.idiv(flat, L.from_dim(strides_in[position], symbols))
            if position > 0:
                part = L.mod(part, L.from_dim(in_shape[axis], symbols))
            result.append(part)
    return result


class _KernelGen:
    """One fused kernel, rooted at a realized node."""

    def __init__(self, fuser: _Fuser, root: Node, copy: bool = False) -> None:
        self.fuser = fuser
        self.plan = fuser.plan
        self.symbols = fuser.symbols
        self.root = None if copy else root
        self.node = root
        self.params: list[View] = []
        self.param_ids: dict[tuple[Buffer, Dim], int] = {}
        self.views_read: dict[Buffer, list[View]] = {}
        self.accs: dict[Node, tuple[L.Var, tuple]] = {}
        self.stmts: list[L.Stmt] = []
        self.memo: dict = {}
        self.pure = 0
        self.counter = 0
        # A gather reads only the rows it is asked for, not its whole table.
        self.gathered: tuple[Buffer, Dim] | None = None

    # values

    def fresh(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}{self.counter}"

    def block(self) -> list[L.Stmt]:
        """Start a new statement block; values computed in the last one are out of scope."""
        self.stmts = []
        self.memo = {}
        return self.stmts

    def load(self, view: View, idx: list[L.Scalar]) -> L.Scalar:
        value = view.buffer.value
        if view.buffer.kind == "const" and value is not None and value.size == 1:
            # A scalar constant, broadcast or not, is a literal in the code.
            return L.const(value.reshape(()).item(), view.dtype)
        key = (view.buffer, view.offset)
        param = self.param_ids.get(key)
        if param is None:
            param = self.param_ids[key] = len(self.params) + 1
            self.params.append(View(view.buffer, view.buffer.shape, contiguous_strides(view.buffer.shape), view.offset))
        self.views_read.setdefault(view.buffer, [])
        if view not in self.views_read[view.buffer]:
            self.views_read[view.buffer].append(view)
        return L.Load(param, L.linear_index(idx, view.strides, self.symbols), view.dtype)

    def value(self, n: Node, idx: list[L.Scalar]) -> L.Scalar:
        key = (n, tuple(idx), self.pure > 0)
        if key in self.memo:
            return self.memo[key]
        result = self._value(n, idx)
        self.memo[key] = result
        return result

    def _value(self, n: Node, idx: list[L.Scalar]) -> L.Scalar:
        if n in self.accs:
            var, expected = self.accs[n]
            if tuple(idx) != expected:
                raise _Refuse(n, "row" if n.op in REDUCE_OPS else "epilogue")
            return var

        if n.op == "const" and n.attrs["value"].size == 1:
            return L.const(n.attrs["value"].reshape(()).item(), n.dtype)
        if n is not self.root or n.op in LEAF_OPS:
            view = self.fuser.view_of(n)
            if view is not None:
                return self.load(view, idx)

        op = n.op
        if op in INDEX_OPS:
            return self._index(n, idx)
        if op == "iota":
            return L.Cast(idx[n.attrs["axis"] % n.rank], n.dtype)
        if op == "dim":
            return L.from_dim(n.attrs["expr"], self.symbols)
        if op in ELEMENTWISE or op == "cast":
            args = [self.value(x, idx) for x in n.inputs]
            expr = _cast(args[0], n.dtype) if op == "cast" else ELEMENTWISE[op](*args)
            if self.pure:
                return expr
            name = self.fresh("t")
            self.stmts.append(L.Let(name, expr))
            return L.Var(name, n.dtype)
        raise AssertionError(f"{op} node %{self.fuser.graph.ids[n]} is neither realized nor fused here")

    def _index(self, n: Node, idx: list[L.Scalar]) -> L.Scalar:
        x = n.inputs[0]
        if n.op == "permute":
            mapped: list[L.Scalar] = [L.const(0)] * x.rank
            for k, p in enumerate(n.attrs["perm"]):
                mapped[p] = idx[k]
            return self.value(x, mapped)
        if n.op == "broadcast":
            mapped = [
                L.const(0) if dims_equal(have, 1) and not dims_equal(want, 1) else i
                for i, have, want in zip(idx, x.shape, n.shape)
            ]
            return self.value(x, mapped)
        if n.op == "slice":
            axis = n.attrs["axis"] % n.rank
            mapped = list(idx)
            mapped[axis] = L.add(idx[axis], L.from_dim(n.attrs["start"], self.symbols))
            return self.value(x, mapped)
        if n.op == "reshape":
            return self.value(x, _reshape_index(x.shape, n.shape, idx, self.symbols))
        # concat: a select per operand. Every branch is built as a pure
        # expression, so only the chosen operand is ever loaded.
        axis = n.attrs["axis"] % n.rank
        starts: list[Dim] = []
        start: Dim = 0
        for operand in n.inputs:
            starts.append(start)
            start = canonical(as_expr(start) + operand.shape[axis])
        self.pure += 1
        try:
            branches = []
            for operand, begin in zip(n.inputs, starts):
                mapped = list(idx)
                mapped[axis] = L.binary("sub", idx[axis], L.from_dim(begin, self.symbols))
                branches.append(self.value(operand, mapped))
        finally:
            self.pure -= 1
        result = branches[-1]
        for operand, begin, branch in reversed(list(zip(n.inputs, starts, branches))[:-1]):
            end = L.from_dim(canonical(as_expr(begin) + operand.shape[axis]), self.symbols)
            result = L.Select(L.binary("lt", idx[axis], end), branch, result, n.dtype)
        return result

    # templates

    def _loops(self, shape, prefix: str = "i"):
        names = [f"{prefix}{d}" for d in range(len(shape))]
        extents = [L.from_dim(d, self.symbols) for d in shape]
        return names, extents, L.substitute_unit_loops(names, extents)

    def _store(self, shape, idx, value) -> L.Store:
        return L.Store(0, L.linear_index(idx, contiguous_strides(shape), self.symbols), value)

    def generate(self) -> tuple[L.Kernel, list[View]]:
        n = self.node
        if self.root is None:
            body, kind = self._elementwise(n), "copy"
        elif self.plan.matmuls_of(n) or n.op == "matmul":
            body, kind = self._matmul(n), "matmul+epilogue" if self.plan.matmuls_of(n) else "matmul"
        elif self.plan.reductions_of(n):
            body, kind = self._row(n), "row"
        elif n.op in REDUCE_OPS:
            body, kind = self._reduce(n), "reduce"
        elif n.op == "gather":
            body, kind = self._gather(n), "gather"
        else:
            body, kind = self._elementwise(n), "copy" if n.op in INDEX_OPS | LEAF_OPS else "elementwise"

        params = (L.Param(n.dtype, True),) + tuple(L.Param(v.dtype, False) for v in self.params)
        kernel = L.Kernel(kind, params, tuple(body), self._bytes_read(),
                          canonical(as_expr(product(n.shape)) * n.dtype.numpy.itemsize))
        return kernel, self.params

    def _bytes_read(self) -> Dim:
        """Each buffer once if any read covers all of it, else the views read."""
        total = Expr.const(0)
        for buf, views in self.views_read.items():
            if self.gathered is not None and buf is self.gathered[0]:
                total = total + self.gathered[1]
                continue
            whole = product(buf.shape)
            if any(dims_equal(v.distinct_elements, whole) for v in views):
                total = total + as_expr(whole) * buf.dtype.numpy.itemsize
            else:
                for v in views:
                    total = total + as_expr(v.distinct_elements) * buf.dtype.numpy.itemsize
        return canonical(total)

    def _elementwise(self, n: Node) -> list[L.Stmt]:
        names, extents, idx = self._loops(n.shape)
        body = self.block()
        if self.root is None:
            value = self.load(self.fuser.view_of(n), idx)
        else:
            value = self.value(n, idx)
        body.append(self._store(n.shape, idx, value))
        return L.loop_nest(names, extents, body)

    def _reduce_update(self, r: Node, acc: L.Var, value: L.Scalar) -> L.Scalar:
        if r.op == "reduce_sum":
            return L.binary("add", acc, value)
        bigger = L.binary("lt", acc, value)
        if r.dtype.is_float:
            bigger = L.binary("or", bigger, L.binary("ne", value, value))
        return L.Select(bigger, value, acc, r.dtype)

    def _reduce_init(self, r: Node) -> L.Const:
        if r.op == "reduce_sum":
            return L.const(0.0 if r.dtype.is_float else 0, r.dtype)
        return L.const(-np.inf if r.dtype.is_float else int(np.iinfo(r.dtype.numpy).min), r.dtype)

    def _reduce(self, r: Node) -> list[L.Stmt]:
        axis = r.attrs["axis"] % r.rank
        names, extents, idx = self._loops(r.shape)
        acc = L.Var("acc", r.dtype)
        rbody = self.block()
        in_idx = list(idx)
        in_idx[axis] = L.Var("r")
        rbody.append(L.Assign("acc", self._reduce_update(r, acc, self.value(r.inputs[0], in_idx))))
        inner = [
            L.Let("acc", self._reduce_init(r)),
            L.Loop("r", L.from_dim(r.inputs[0].shape[axis], self.symbols), tuple(rbody)),
            self._store(r.shape, idx, acc),
        ]
        return L.loop_nest(names, extents, inner)

    def _row(self, t: Node) -> list[L.Stmt]:
        """Reductions over each row into scalars, then one pass writing the row."""
        names, extents, outer = self._loops(t.shape[:-1])
        body: list[L.Stmt] = []
        for i, r in enumerate(sorted(self.plan.reductions_of(t), key=lambda x: self.fuser.graph.ids[x])):
            name = f"acc{i}"
            acc = L.Var(name, r.dtype)
            rbody = self.block()
            value = self.value(r.inputs[0], outer + [L.Var("r")])
            rbody.append(L.Assign(name, self._reduce_update(r, acc, value)))
            body += [
                L.Let(name, self._reduce_init(r)),
                L.Loop("r", L.from_dim(r.inputs[0].shape[-1], self.symbols), tuple(rbody)),
            ]
            self.accs[r] = (acc, tuple(outer + [L.const(0)]))
        jnames, jextents, jidx = self._loops(t.shape[-1:], prefix="j")
        jbody = self.block()
        jbody.append(self._store(t.shape, outer + jidx, self.value(t, outer + jidx)))
        body += L.loop_nest(jnames, jextents, jbody)
        return L.loop_nest(names, extents, body)

    def _matmul(self, t: Node) -> list[L.Stmt]:
        """A k loop over every fused matmul at once, then the epilogue."""
        matmuls = sorted(self.plan.matmuls_of(t), key=lambda x: self.fuser.graph.ids[x]) or [t]
        names, extents, idx = self._loops(t.shape)
        k = L.Var("k")
        kbody = self.block()
        accs = []
        for i, m in enumerate(matmuls):
            name = f"acc{i}"
            acc = L.Var(name, m.dtype)
            a = self.value(m.inputs[0], idx[:-1] + [k])
            b = self.value(m.inputs[1], idx[:-2] + [k, idx[-1]])
            kbody.append(L.Assign(name, L.binary("add", acc, L.binary("mul", a, b))))
            accs.append((name, m))
        inner: list[L.Stmt] = [L.Let(name, L.const(0.0, m.dtype)) for name, m in accs]
        inner.append(L.Loop("k", L.from_dim(matmuls[0].inputs[0].shape[-1], self.symbols), tuple(kbody)))
        for name, m in accs:
            self.accs[m] = (L.Var(name, m.dtype), tuple(idx))
        epilogue = self.block()
        epilogue.append(self._store(t.shape, idx, self.value(t, idx)))
        inner += epilogue
        return L.loop_nest(names, extents, inner)

    def _gather(self, g: Node) -> list[L.Stmt]:
        table, ids = g.inputs
        names, extents, outer = self._loops(ids.shape)
        rest_names, rest_extents, rest = self._loops(table.shape[1:], prefix="j")
        body = self.block()
        row = self.value(ids, outer)
        body.append(L.Let("row", L.Cast(row, dtypes.i64)))
        row_var = L.Var("row")
        rows = L.from_dim(table.shape[0], self.symbols)
        body.append(L.Fail(L.binary("or", L.binary("lt", row_var, L.const(0)), L.binary("le", rows, row_var)), 1))
        outer_stmts = body
        inner = self.block()
        inner.append(self._store(g.shape, outer + rest, self.value(table, [row_var] + rest)))
        table_view = self.fuser.view_of(table)
        if table_view is not None:
            rows_read = as_expr(product(ids.shape)) * product(table.shape[1:]) * table.dtype.numpy.itemsize
            self.gathered = (table_view.buffer, canonical(rows_read))
        outer_stmts += L.loop_nest(rest_names, rest_extents, inner)
        return L.loop_nest(names, extents, outer_stmts)


def fuse(graph: Graph) -> Program:
    """Plan, generate, and replan without any fusion the generator refuses."""
    planner = _Planner(graph)
    while True:
        plan = planner.plan()
        try:
            return _Fuser(graph, plan).run()
        except _Refuse as refusal:
            planner.refused.add((refusal.node, refusal.kind))
