"""Index arithmetic as polynomials, simplified with what loop ranges imply.

Schedules build index expressions that are correct and opaque. Splitting
``j`` into ``jo * 32 + ji`` and then reading a packed weight produces

    ((jo * 32 + ji) / 8) * 7168 + k * 8 + (jo * 32 + ji) % 8

where ``ji`` runs over 0..31. Nothing in that says the access is contiguous in
``ji`` for runs of 8. This module turns integer expressions into polynomials
over atoms (loop variables, symbols, and anything it cannot see into), and
uses the range of each loop variable to remove a division where the
remainder provably stays inside one divisor's worth.

The rule is the only one there is: for ``n = d * q + r`` with ``0 <= r < d``,
``n / d`` is ``q`` and ``n % d`` is ``r``. The loop IR uses ``/`` and ``%``
only on non-negative values, so C's truncation agrees with floor division.
Ranges come from loop extents; a variable whose range is not known (a
``Let``, an extent that is a symbol) is assumed nothing about, and a division
the rule cannot be proven for is left exactly as it was.

The same polynomials answer what vectorization asks: the coefficient of a
lane variable in an index, which is 1 for a contiguous access.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from nanocompile import loopir as L

INF = math.inf

Poly = dict[tuple[str, ...], int]   # monomial (sorted atom keys) -> coefficient
Range = tuple[float, float]


class _Atoms:
    """Atom keys for a polynomial, and the expressions they stand for."""

    def __init__(self) -> None:
        self.exprs: dict[str, L.Scalar] = {}

    def key(self, x: L.Scalar) -> str:
        if isinstance(x, L.Var):
            k = f"v:{x.name}"
        elif isinstance(x, L.Sym):
            k = f"s:{x.index}"
        else:
            k = f"x:{x!r}"
        self.exprs.setdefault(k, x)
        return k


def _padd(p: Poly, q: Poly, sign: int = 1) -> Poly:
    out = dict(p)
    for m, c in q.items():
        out[m] = out.get(m, 0) + sign * c
        if out[m] == 0:
            del out[m]
    return out


def _pmul(p: Poly, q: Poly) -> Poly:
    out: Poly = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            m = tuple(sorted(m1 + m2))
            out[m] = out.get(m, 0) + c1 * c2
            if out[m] == 0:
                del out[m]
    return out


def _imul(a: Range, b: Range) -> Range:
    def times(x: float, y: float) -> float:
        return 0.0 if x == 0 or y == 0 else x * y
    products = [times(x, y) for x in a for y in b]
    return min(products), max(products)


class Simplifier:
    """Polynomials over one kernel's atoms, given each loop variable's range."""

    def __init__(self, ranges: Mapping[str, Range]) -> None:
        self.ranges = dict(ranges)
        self.atoms = _Atoms()

    # -- polynomials -----------------------------------------------------------

    def poly(self, x: L.Scalar) -> Poly | None:
        """``x`` as a polynomial, or ``None`` if it is not an integer expression."""
        if not x.dtype.is_int:
            return None
        if isinstance(x, L.Const):
            return {(): int(x.value)} if x.value else {}
        if isinstance(x, (L.Var, L.Sym)):
            return {(self.atoms.key(x),): 1}
        if isinstance(x, L.Binary) and x.op in ("add", "sub", "mul"):
            a, b = self.poly(x.a), self.poly(x.b)
            if a is not None and b is not None:
                if x.op == "mul":
                    return _pmul(a, b)
                return _padd(a, b, 1 if x.op == "add" else -1)
        if isinstance(x, L.Binary) and x.op in ("idiv", "mod"):
            reduced = self._divide(x)
            if reduced is not None:
                return reduced
        return {(self.atoms.key(x),): 1}

    def _atom_range(self, key: str) -> Range:
        if key.startswith("v:"):
            return self.ranges.get(key[2:], (-INF, INF))
        if key.startswith("s:"):
            return (0.0, INF)
        return (-INF, INF)

    def range_of(self, p: Poly) -> Range:
        lo = hi = 0.0
        for m, c in p.items():
            r: Range = (float(c), float(c))
            for atom in m:
                r = _imul(r, self._atom_range(atom))
            lo, hi = lo + r[0], hi + r[1]
        return lo, hi

    def _divide(self, x: L.Binary) -> Poly | None:
        if not (isinstance(x.b, L.Const) and x.b.value > 0):
            return None
        p = self.poly(x.a)
        if p is None:
            return None
        d = int(x.b.value)
        quotient = {m: c // d for m, c in p.items() if c % d == 0}
        remainder = {m: c for m, c in p.items() if c % d != 0}
        lo, hi = self.range_of(remainder)
        if not (lo >= 0 and hi <= d - 1):
            return None
        return quotient if x.op == "idiv" else remainder

    # -- back to expressions ---------------------------------------------------

    def expr(self, p: Poly) -> L.Scalar:
        """``p`` as an i64 expression: highest degree first, the constant last."""
        total: L.Scalar = L.const(0)
        ordered = sorted(p.items(), key=lambda mc: (len(mc[0]) == 0, -len(mc[0]), mc[0]))
        for monomial, coeff in ordered:
            term: L.Scalar | None = None
            for atom in monomial:
                a = self.atoms.exprs[atom]
                term = a if term is None else L.mul(term, a)
            magnitude = L.const(abs(coeff))
            term = magnitude if term is None else L.mul(term, magnitude)
            if coeff < 0:
                total = L.binary("sub", total, term)
            else:
                total = L.add(total, term)
        return total

    def node(self, x: L.Scalar) -> L.Scalar:
        """One node, with its division removed if the ranges allow it."""
        if isinstance(x, L.Binary) and x.op in ("idiv", "mod"):
            reduced = self._divide(x)
            if reduced is not None:
                return self.expr(reduced)
        return x

    def simplify(self, x: L.Scalar) -> L.Scalar:
        """``x`` with every division the ranges allow removed, and nothing else changed."""
        return L.map_scalar(x, self.node)

    def coefficient(self, x: L.Scalar, var: str) -> int | None:
        """The constant ``c`` with ``x == c * var + (terms free of var)``, else ``None``."""
        p = self.poly(x)
        if p is None:
            return None
        key = f"v:{var}"
        c = 0
        for m, coeff in p.items():
            if key in m:
                if m != (key,):
                    return None
                c = coeff
            elif any(self._mentions(a, var) for a in m):
                return None
        return c

    def _mentions(self, atom: str, var: str) -> bool:
        if not atom.startswith("x:"):
            return False
        return var in L.vars_of(self.atoms.exprs[atom])


def loop_ranges(stmts, outer: Mapping[str, Range] | None = None) -> dict[str, Range]:
    """The range of every loop variable declared in ``stmts``."""
    ranges: dict[str, Range] = dict(outer or {})
    for s in stmts:
        if isinstance(s, L.Loop):
            extent = s.extent
            hi = float(extent.value - 1) if isinstance(extent, L.Const) else INF
            ranges[s.var] = (0.0, hi)
            ranges.update(loop_ranges(s.body))
        elif isinstance(s, L.Block):
            ranges.update(loop_ranges(s.body))
    return ranges


def simplify_stmts(stmts, outer: Mapping[str, Range] | None = None) -> tuple[L.Stmt, ...]:
    """Every index in ``stmts`` simplified with the ranges of the loops around it."""
    out: list[L.Stmt] = []
    ranges = dict(outer or {})
    here = Simplifier(ranges)
    for s in stmts:
        if isinstance(s, L.Loop):
            extent = s.extent
            hi = float(extent.value - 1) if isinstance(extent, L.Const) else INF
            inner = dict(ranges)
            inner[s.var] = (0.0, hi)
            out.append(L.Loop(s.var, here.simplify(extent), simplify_stmts(s.body, inner), s.kind))
        elif isinstance(s, L.Block):
            out.append(L.Block(simplify_stmts(s.body, ranges)))
        else:
            out.extend(L.map_stmts((s,), here.node))
    return tuple(out)


def simplify_kernel(kernel: L.Kernel) -> L.Kernel:
    return L.Kernel(kernel.primitive, kernel.params, simplify_stmts(kernel.body),
                    kernel.bytes_read, kernel.bytes_written)
