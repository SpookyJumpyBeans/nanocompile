"""AVX2 for vector loops: eight float32 lanes, each computing what scalar C would.

A loop a schedule marked ``vector`` has eight iterations and no dependence
between them, so it can run as one instruction per operation, lane ``l``
doing iteration ``l``. This module renders such a loop with AVX2 intrinsics,
and the rule it follows is that every lane must produce the bits the scalar
loop produces:

* ``+ - * /`` and ``sqrt`` are single IEEE operations, correctly rounded in
  both forms, and the compile flags still forbid contracting a multiply and an
  add into an FMA.
* Comparisons use the predicates that match C on NaN: ``<`` and ``<=`` are
  false for a NaN operand, ``!=`` is true.
* ``exp``, ``sin`` and ``cos`` call the same C library function once per
  lane, since a vector approximation would round differently.
* A select is a blend, which moves bits and computes nothing.

Loads and stores are classified by the lane variable's coefficient in their
index: 1 is a contiguous ``loadu``/``storeu``, 0 is one scalar broadcast to
every lane, anything else is eight scalar accesses. Integer values that differ
between lanes, a scalar carried across iterations, a nested loop or a bounds
check have no exact vector form here, and a loop containing one is refused
rather than approximated.

Accumulator arrays that are only ever touched eight lanes at a time, at
constant offsets, are kept in registers: ``acc[16 + l]`` is lane ``l`` of
``acc_2``. That is what makes a register tile a register tile.
"""

from __future__ import annotations

from nanocompile import dtypes
from nanocompile import loopir as L
from nanocompile.simplify import Simplifier

LANES = 8

PRELUDE = """\
#if defined(__AVX2__)
#include <immintrin.h>
#define NC_LANEWISE(fn) \\
    static inline __m256 nc_v_##fn(__m256 v) { \\
        float t[8]; _mm256_storeu_ps(t, v); \\
        for (int l = 0; l < 8; ++l) t[l] = fn(t[l]); \\
        return _mm256_loadu_ps(t); \\
    }
NC_LANEWISE(expf)
NC_LANEWISE(sinf)
NC_LANEWISE(cosf)
static inline __m256 nc_v_mask(int c) { return _mm256_castsi256_ps(_mm256_set1_epi32(c ? -1 : 0)); }
#endif
"""

_ARITH = {"add": "_mm256_add_ps", "sub": "_mm256_sub_ps", "mul": "_mm256_mul_ps", "div": "_mm256_div_ps"}
_CMP = {"lt": "_CMP_LT_OQ", "le": "_CMP_LE_OQ", "ne": "_CMP_NEQ_UQ"}
_LANEWISE = {"exp": "nc_v_expf", "sin": "nc_v_sinf", "cos": "nc_v_cosf"}


class VectorError(ValueError):
    """A loop body with no exact eight-lane form."""


def registers_for(decl: L.Array, scope) -> int | None:
    """How many registers ``decl`` becomes, or ``None`` if it stays in memory.

    ``scope`` is the statements after the declaration in its block, which is
    everywhere the array can be named. It is kept in registers when it holds
    floats, a whole number of vectors, and every access to it is inside a
    vector loop at ``8 * c + lane`` for a constant ``c``.
    """
    if decl.init.dtype != dtypes.f32 or decl.size % LANES:
        return None
    poly = Simplifier({}).poly
    good = True

    def access(index: L.Scalar, lane: str | None) -> None:
        nonlocal good
        if lane is None:
            good = False
            return
        p = poly(index) or {}
        if set(p) - {(), (f"v:{lane}",)} or p.get((f"v:{lane}",)) != 1 or p.get((), 0) % LANES:
            good = False

    def walk(stmts, lane: str | None) -> None:
        for s in stmts:
            if isinstance(s, L.Loop):
                walk(s.body, s.var if s.kind == "vector" else lane)
                continue
            if isinstance(s, L.Block):
                walk(s.body, lane)
                continue
            if isinstance(s, L.ArrayStore) and s.name == decl.name:
                access(s.index, lane)
            for x in L.scalars_of((s,)):
                if isinstance(x, L.Local) and x.name == decl.name:
                    access(x.index, lane)

    walk(scope, None)
    return decl.size // LANES if good else None


class VectorLoop:
    """Renders one vector loop.

    ``scalar`` renders a scalar expression as C and ``ctypes`` maps a dtype to
    its C type; both come from the scalar code generator.
    """

    def __init__(self, loop: L.Loop, promoted: dict[str, int], scalar, ctypes) -> None:
        if not (isinstance(loop.extent, L.Const) and loop.extent.value == LANES):
            raise VectorError(f"a vector loop has {LANES} iterations, not {loop.extent}")
        self.loop = loop
        self.lane = loop.var
        self.promoted = promoted
        self.scalar = scalar
        self.ctypes = ctypes
        self.varying: set[str] = set()
        self.simplifier = Simplifier({loop.var: (0, LANES - 1)})
        self.temps = 0

    # -- what differs between lanes ----------------------------------------------

    def varies(self, x: L.Scalar) -> bool:
        for n in L.walk_scalar(x):
            if isinstance(n, L.Var) and (n.name == self.lane or n.name in self.varying):
                return True
            if isinstance(n, L.Local) and n.name in self.promoted:
                return True
        return False

    def _coefficient(self, index: L.Scalar) -> int | None:
        if any(isinstance(n, L.Var) and n.name in self.varying for n in L.walk_scalar(index)):
            raise VectorError("an index depends on a value computed per lane")
        return self.simplifier.coefficient(index, self.lane)

    def _at(self, index: L.Scalar, lane: int) -> str:
        return self.scalar(L.map_scalar(index, lambda n: L.const(lane) if isinstance(n, L.Var) and n.name == self.lane else n))

    def _register(self, name: str, index: L.Scalar) -> str:
        p = self.simplifier.poly(index) or {}
        return f"{name}_{p.get((), 0) // LANES}"

    # -- expressions -------------------------------------------------------------

    def vec(self, x: L.Scalar) -> str:
        """``x`` as a ``__m256`` (a float value) or a ``__m256`` mask (a bool)."""
        if not self.varies(x):
            if x.dtype == dtypes.bool_:
                return f"nc_v_mask({self.scalar(x)})"
            if x.dtype != dtypes.f32:
                raise VectorError(f"a {x.dtype} value cannot be a lane of a float vector")
            return f"_mm256_set1_ps({self.scalar(x)})"
        if isinstance(x, L.Var):
            if x.name == self.lane:
                raise VectorError(f"the lane index {self.lane} is used as a value")
            return x.name
        if isinstance(x, L.Load):
            return self._load(f"x{x.param}", x.index, x.dtype)
        if isinstance(x, L.Local):
            if x.name in self.promoted:
                return self._register(x.name, x.index)
            return self._load(x.name, x.index, x.dtype)
        if isinstance(x, L.Binary):
            if x.op in _ARITH and x.dtype == dtypes.f32:
                return f"{_ARITH[x.op]}({self.vec(x.a)}, {self.vec(x.b)})"
            if x.op in _CMP and x.a.dtype == dtypes.f32:
                return f"_mm256_cmp_ps({self.vec(x.a)}, {self.vec(x.b)}, {_CMP[x.op]})"
            if x.op == "or":
                return f"_mm256_or_ps({self.vec(x.a)}, {self.vec(x.b)})"
            raise VectorError(f"{x.op} on {x.a.dtype} values that differ between lanes")
        if isinstance(x, L.Unary) and x.dtype == dtypes.f32:
            a = self.vec(x.a)
            if x.op == "neg":
                return f"_mm256_xor_ps({a}, _mm256_set1_ps(-0.0f))"
            if x.op == "sqrt":
                return f"_mm256_sqrt_ps({a})"
            if x.op == "reciprocal":
                return f"_mm256_div_ps(_mm256_set1_ps(1.0f), {a})"
            return f"{_LANEWISE[x.op]}({a})"
        if isinstance(x, L.Select) and x.dtype == dtypes.f32:
            return f"_mm256_blendv_ps({self.vec(x.b)}, {self.vec(x.a)}, {self.vec(x.cond)})"
        if isinstance(x, L.Cast) and x.dtype == dtypes.f32 and x.a.dtype == dtypes.f32:
            return self.vec(x.a)
        raise VectorError(f"no exact vector form for {type(x).__name__} {getattr(x, 'op', '')}".strip())

    def _load(self, base: str, index: L.Scalar, dtype) -> str:
        if dtype != dtypes.f32:
            raise VectorError(f"a {dtype} load that differs between lanes")
        c = self._coefficient(index)
        if c == 1:
            return f"_mm256_loadu_ps(&{base}[{self._at(index, 0)}])"
        lanes = ", ".join(f"{base}[{self._at(index, l)}]" for l in range(LANES))
        return f"_mm256_setr_ps({lanes})"

    # -- statements --------------------------------------------------------------

    def render(self, depth: int) -> list[str]:
        pad = "    " * depth
        lines = [f"{pad}{{ /* {self.lane}: {LANES} lanes */"]
        lines += self._stmts(self.loop.body, depth + 1)
        lines.append(f"{pad}}}")
        return lines

    def _stmts(self, stmts, depth: int) -> list[str]:
        pad = "    " * depth
        lines: list[str] = []
        for s in stmts:
            if isinstance(s, L.Let):
                if self.varies(s.value):
                    if s.value.dtype not in (dtypes.f32, dtypes.bool_):
                        raise VectorError(f"{s.var} is a {s.value.dtype} that differs between lanes")
                    lines.append(f"{pad}__m256 {s.var} = {self.vec(s.value)};")
                    self.varying.add(s.var)
                else:
                    lines.append(f"{pad}{self.ctypes[s.value.dtype]} {s.var} = {self.scalar(s.value)};")
            elif isinstance(s, L.Store):
                lines += self._store(pad, "x0", s.index, s.value)
            elif isinstance(s, L.ArrayStore):
                if s.name in self.promoted:
                    lines.append(f"{pad}{self._register(s.name, s.index)} = {self.vec(s.value)};")
                else:
                    lines += self._store(pad, s.name, s.index, s.value)
            elif isinstance(s, L.Block):
                lines.append(f"{pad}{{")
                lines += self._stmts(s.body, depth + 1)
                lines.append(f"{pad}}}")
            elif isinstance(s, L.Assign):
                raise VectorError(f"{s.var} carries a value from one lane to the next")
            else:
                raise VectorError(f"a vector loop cannot contain a {type(s).__name__}")
        return lines

    def _store(self, pad: str, base: str, index: L.Scalar, value: L.Scalar) -> list[str]:
        c = self._coefficient(index)
        if c == 1:
            return [f"{pad}_mm256_storeu_ps(&{base}[{self._at(index, 0)}], {self.vec(value)});"]
        if c == 0:
            raise VectorError("every lane stores to the same element")
        self.temps += 1
        t = f"nc_t{self.temps}"
        lines = [f"{pad}float {t}[{LANES}];", f"{pad}_mm256_storeu_ps({t}, {self.vec(value)});"]
        lines += [f"{pad}{base}[{self._at(index, l)}] = {t}[{l}];" for l in range(LANES)]
        return lines


def check(loop: L.Loop, scalar, ctypes) -> None:
    """Raise ``VectorError`` unless ``loop`` has an exact AVX2 rendering."""
    VectorLoop(loop, {}, scalar, ctypes).render(0)
