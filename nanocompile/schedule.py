"""Schedules: how a kernel's loops run, kept apart from what they compute.

A kernel from fusion says what to compute, as the plainest loop nest that
computes it. A schedule rewrites that nest without changing what it computes,
in the style of Halide:

* ``split(i, f)``: one loop becomes two, ``i = io * f + ii``, with a tail
  loop for the remainder when ``f`` does not divide the extent.
* ``reorder(a, b)``: two perfectly nested loops swap places.
* ``unroll(i)``: a loop of constant extent becomes that many copies of its
  body, each in its own scope.

Every transformation acts on each loop with the given name (a split's tail
copies the body, so a later transformation reaches both copies) and checks
that it is legal where it applies. An illegal schedule is an error, never a
silently different kernel.

None of these reorder the operations that produce a value. Each accumulator
sees its updates in the same order, every element is computed by the same
expression, so a scheduled kernel is bitwise identical to the kernel it was
scheduled from, and the tests hold every schedule to exactly that.
"""

from __future__ import annotations

from dataclasses import replace

from nanocompile import loopir as L
from nanocompile.simplify import simplify_stmts


class ScheduleError(ValueError):
    """A transformation that does not apply, or would change what is computed."""


def _declared(stmts) -> set[str]:
    """Names declared anywhere inside ``stmts``: loop variables, lets, arrays."""
    names: set[str] = set()
    for s in stmts:
        if isinstance(s, L.Loop):
            names.add(s.var)
            names |= _declared(s.body)
        elif isinstance(s, L.Block):
            names |= _declared(s.body)
        elif isinstance(s, L.Let):
            names.add(s.var)
        elif isinstance(s, L.Array):
            names.add(s.name)
    return names


def _written(stmts) -> set[str]:
    """Names of scalars assigned and arrays stored to, anywhere inside ``stmts``."""
    names: set[str] = set()
    for s in stmts:
        if isinstance(s, (L.Loop, L.Block)):
            names |= _written(s.body)
        elif isinstance(s, L.Assign):
            names.add(s.var)
        elif isinstance(s, L.ArrayStore):
            names.add(s.name)
    return names


def independent(loop: L.Loop) -> bool:
    """Whether ``loop``'s iterations could run in any order with the same result.

    An iteration that updates a scalar or an array declared outside the loop
    carries a value into the next one. Stores to the output do not: a kernel
    writes each output element once, at an index its loops determine.
    """
    return not (_written(loop.body) - _declared(loop.body))


def loop_vars(stmts) -> list[str]:
    """Every loop variable in ``stmts``, outermost first, each name once."""
    seen: list[str] = []
    for s in stmts:
        if isinstance(s, L.Loop):
            if s.var not in seen:
                seen.append(s.var)
            for v in loop_vars(s.body):
                if v not in seen:
                    seen.append(v)
        elif isinstance(s, L.Block):
            for v in loop_vars(s.body):
                if v not in seen:
                    seen.append(v)
    return seen


def _rewrite(stmts, var: str, fn, found: list[int]) -> tuple[L.Stmt, ...]:
    """Replace each loop over ``var`` by the statements ``fn(loop)`` returns."""
    out: list[L.Stmt] = []
    for s in stmts:
        if isinstance(s, L.Loop) and s.var == var:
            found[0] += 1
            out.extend(fn(s))
        elif isinstance(s, L.Loop):
            out.append(replace(s, body=_rewrite(s.body, var, fn, found)))
        elif isinstance(s, L.Block):
            out.append(L.Block(_rewrite(s.body, var, fn, found)))
        else:
            out.append(s)
    return tuple(out)


def _to_arrays(stmts, scalars: dict[str, L.Const], sizes: dict[str, int], j: L.Var) -> tuple[L.Stmt, ...]:
    """Accumulator ``acc`` read and written as ``acc[j]``; array ``a`` of size n as ``a[j * n + i]``."""
    def scalar(x: L.Scalar) -> L.Scalar:
        if isinstance(x, L.Var) and x.name in scalars:
            return L.Local(x.name, j, scalars[x.name].dtype)
        if isinstance(x, L.Local) and x.name in sizes:
            return L.Local(x.name, L.add(L.mul(j, L.const(sizes[x.name])), x.index), x.dtype)
        return x

    out: list[L.Stmt] = []
    for s in stmts:
        if isinstance(s, L.Loop):
            out.append(replace(s, extent=L.map_scalar(s.extent, scalar), body=_to_arrays(s.body, scalars, sizes, j)))
            continue
        if isinstance(s, L.Block):
            out.append(L.Block(_to_arrays(s.body, scalars, sizes, j)))
            continue
        (s,) = L.map_stmts((s,), scalar)
        if isinstance(s, L.Assign) and s.var in scalars:
            out.append(L.ArrayStore(s.var, j, s.value))
        elif isinstance(s, L.ArrayStore) and s.name in sizes:
            out.append(L.ArrayStore(s.name, L.add(L.mul(j, L.const(sizes[s.name])), s.index), s.value))
        else:
            out.append(s)
    return tuple(out)


class Schedule:
    """A kernel being scheduled. Each method applies one transformation."""

    def __init__(self, kernel: L.Kernel) -> None:
        self.kernel = kernel
        self.steps: list[str] = []

    @property
    def loops(self) -> list[str]:
        return loop_vars(self.kernel.body)

    def _apply(self, step: str, var: str, fn) -> "Schedule":
        found = [0]
        body = _rewrite(self.kernel.body, var, fn, found)
        if not found[0]:
            raise ScheduleError(f"{step}: no loop over {var!r} (loops: {', '.join(self.loops)})")
        self.kernel = replace(self.kernel, body=body)
        self.steps.append(step)
        return self

    # -- transformations -------------------------------------------------------

    def split(self, var: str, factor: int, outer: str | None = None, inner: str | None = None,
              tail: str | None = None) -> "Schedule":
        """``var = outer * factor + inner``, then the leftover iterations in ``tail``."""
        if factor < 1:
            raise ScheduleError(f"split {var} by {factor}: the factor must be positive")
        outer, inner, tail = outer or f"{var}o", inner or f"{var}i", tail or f"{var}t"
        taken = set(self.loops) | _declared(self.kernel.body)
        for name in (outer, inner, tail):
            if name in taken:
                raise ScheduleError(f"split {var}: {name!r} is already a name in this kernel")

        def fn(loop: L.Loop) -> list[L.Stmt]:
            if loop.kind != "serial":
                raise ScheduleError(f"split {var}: the loop is already {loop.kind}")
            f = L.const(factor)
            if isinstance(loop.extent, L.Const):
                full, rest = divmod(int(loop.extent.value), factor)
                full_extent, rest_extent = L.const(full), L.const(rest)
            else:
                full_extent = L.idiv(loop.extent, f)
                rest_extent = L.mod(loop.extent, f)
                full, rest = None, None
            out: list[L.Stmt] = []
            if full != 0:
                index = L.add(L.mul(L.Var(outer), f), L.Var(inner))
                body = L.substitute(loop.body, {var: index})
                out.append(L.Loop(outer, full_extent, (L.Loop(inner, f, body),)))
            if rest != 0:
                start = L.mul(full_extent, f)
                body = L.substitute(loop.body, {var: L.add(start, L.Var(tail))})
                out.append(L.Loop(tail, rest_extent, body))
            return out

        return self._apply(f"split {var} {factor}", var, fn)

    def reorder(self, outer: str, inner: str) -> "Schedule":
        """Swap ``outer`` and the loop over ``inner`` directly inside it."""
        def fn(loop: L.Loop) -> list[L.Stmt]:
            if len(loop.body) != 1 or not isinstance(loop.body[0], L.Loop) or loop.body[0].var != inner:
                raise ScheduleError(f"reorder {outer} {inner}: {inner} is not the only statement in {outer}")
            nested = loop.body[0]
            if outer in L.vars_of(nested.extent):
                raise ScheduleError(f"reorder {outer} {inner}: {inner}'s extent depends on {outer}")
            if not (independent(loop) and independent(nested)):
                raise ScheduleError(f"reorder {outer} {inner}: an iteration carries a value into the next")
            return [replace(nested, body=(replace(loop, body=nested.body),))]

        return self._apply(f"reorder {outer} {inner}", outer, fn)

    def unroll(self, var: str) -> "Schedule":
        """One copy of the body per iteration, ``var`` replaced by its value."""
        def fn(loop: L.Loop) -> list[L.Stmt]:
            if not isinstance(loop.extent, L.Const):
                raise ScheduleError(f"unroll {var}: the extent is not a constant")
            if loop.kind != "serial":
                raise ScheduleError(f"unroll {var}: the loop is {loop.kind}")
            return [L.Block(L.substitute(loop.body, {var: L.const(c)})) for c in range(int(loop.extent.value))]

        return self._apply(f"unroll {var}", var, fn)

    def privatize(self, var: str) -> "Schedule":
        """Move ``var`` inside the reduction loop it encloses, one accumulator per iteration.

        ``var``'s body must be accumulators (scalars set to a constant, or
        arrays), one reduction loop that updates only them, and an epilogue
        that reads them::

            for j in 0..8:                        acc = [0] * 8
                acc = 0                           for k in 0..K:
                for k in 0..K:            ->          for j in 0..8:
                    acc = acc + a[k] * b[k, j]            acc[j] = acc[j] + a[k] * b[k, j]
                out[j] = acc                      for j in 0..8:
                                                      out[j] = acc[j]

        Each accumulator is updated by the same operations in the same order;
        only the loop that visits the accumulators has moved. Accumulators that
        are already arrays (from privatizing an inner loop) grow by ``var``'s
        extent, so the transformation stacks.
        """
        def fn(loop: L.Loop) -> list[L.Stmt]:
            where = f"privatize {var}"
            if not isinstance(loop.extent, L.Const):
                raise ScheduleError(f"{where}: the extent is not a constant")
            if loop.kind != "serial":
                raise ScheduleError(f"{where}: the loop is {loop.kind}")
            extent = int(loop.extent.value)
            body = list(loop.body)
            scalars: dict[str, L.Const] = {}
            arrays: dict[str, L.Array] = {}
            while body and isinstance(body[0], (L.Let, L.Array)):
                decl = body.pop(0)
                if isinstance(decl, L.Array):
                    arrays[decl.name] = decl
                elif isinstance(decl.value, L.Const):
                    scalars[decl.var] = decl.value
                else:
                    raise ScheduleError(f"{where}: {decl.var} does not start from a constant")
            if not scalars and not arrays:
                raise ScheduleError(f"{where}: {var}'s body declares no accumulators")
            if not body or not isinstance(body[0], L.Loop):
                raise ScheduleError(f"{where}: no reduction loop follows the accumulators")
            reduction, rest = body[0], tuple(body[1:])
            if var in L.vars_of(reduction.extent):
                raise ScheduleError(f"{where}: {reduction.var}'s extent depends on {var}")
            accumulators = set(scalars) | set(arrays)
            if _written(reduction.body) - _declared(reduction.body) - accumulators:
                raise ScheduleError(f"{where}: the reduction updates something besides its accumulators")
            if _written(rest) - _declared(rest):
                raise ScheduleError(f"{where}: the epilogue carries a value between iterations")

            j = L.Var(var)
            decls: list[L.Stmt] = [L.Array(name, extent, init) for name, init in scalars.items()]
            decls += [L.Array(a.name, extent * a.size, a.init) for a in arrays.values()]
            sizes = {name: a.size for name, a in arrays.items()}
            moved = _to_arrays(reduction.body, scalars, sizes, j)
            epilogue = _to_arrays(rest, scalars, sizes, j)
            out = decls + [replace(reduction, body=(L.Loop(var, loop.extent, moved),))]
            if epilogue:
                out.append(L.Loop(var, loop.extent, epilogue))
            return out

        return self._apply(f"privatize {var}", var, fn)

    # -- result ----------------------------------------------------------------

    def done(self) -> L.Kernel:
        """The scheduled kernel, its indices simplified with the new loop ranges."""
        return replace(self.kernel, body=simplify_stmts(self.kernel.body))


def schedule_program(program, fn):
    """``program`` with each distinct kernel replaced by ``fn(kernel, calls)``.

    ``calls`` are the calls that run the kernel, so a schedule can look at what
    it will be handed (whether an operand is always a weight, say). A kernel
    shared by 24 layers is scheduled once.
    """
    calls_of: dict[int, list] = {}
    for call in program.calls:
        calls_of.setdefault(call.kernel, []).append(call)
    kernels = [fn(kernel, calls_of.get(i, [])) for i, kernel in enumerate(program.kernels)]
    return replace(program, kernels=kernels)
