"""Symbolic dimensions: shapes that are known up to a few named sizes.

A decoder runs the same computation on a 12-token prompt and then on one token
at a time against a growing cache. If shapes had to be concrete, every step
would be a new graph, and from phase 2 on a new compile. So a dimension here is
either an integer or an expression over named symbols, ``seq`` and ``past``,
and one graph serves every step.

Expressions are polynomials with integer coefficients. Affine expressions
(``past + seq``, ``seq - 1``) cover every dimension the model has, but reshape
has to compare element counts, and ``seq * 896`` against ``seq * 14 * 64`` is a
product. Polynomials are closed under both, and two of them are equal exactly
when their canonical terms are, so equality never needs a solver.

What this cannot do is prove an inequality: whether ``past + seq - 1`` is a
valid index depends on the values bound at run time. Shape inference checks
what is decidable from the expressions and the interpreter checks the rest.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping

# A monomial is a sorted tuple of symbol names, with repeats for powers;
# () is the constant term.
Monomial = tuple[str, ...]


class Expr:
    """A polynomial over dimension symbols, immutable and hashable."""

    __slots__ = ("_terms",)

    def __init__(self, terms: Mapping[Monomial, int]) -> None:
        self._terms: dict[Monomial, int] = {m: c for m, c in terms.items() if c != 0}

    @staticmethod
    def symbol(name: str) -> "Expr":
        if not name.isidentifier():
            raise ValueError(f"symbol names must be identifiers, got {name!r}")
        return Expr({(name,): 1})

    @staticmethod
    def const(value: int) -> "Expr":
        return Expr({(): int(value)})

    # -- arithmetic --------------------------------------------------------

    def __add__(self, other: "Dim") -> "Expr":
        other = as_expr(other)
        terms = dict(self._terms)
        for monomial, coeff in other._terms.items():
            terms[monomial] = terms.get(monomial, 0) + coeff
        return Expr(terms)

    __radd__ = __add__

    def __neg__(self) -> "Expr":
        return Expr({m: -c for m, c in self._terms.items()})

    def __sub__(self, other: "Dim") -> "Expr":
        return self + -as_expr(other)

    def __rsub__(self, other: "Dim") -> "Expr":
        return as_expr(other) - self

    def __mul__(self, other: "Dim") -> "Expr":
        other = as_expr(other)
        terms: dict[Monomial, int] = {}
        for m1, c1 in self._terms.items():
            for m2, c2 in other._terms.items():
                monomial = tuple(sorted(m1 + m2))
                terms[monomial] = terms.get(monomial, 0) + c1 * c2
        return Expr(terms)

    __rmul__ = __mul__

    # -- comparison --------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            other = Expr.const(other)
        if not isinstance(other, Expr):
            return NotImplemented
        return self._terms == other._terms

    def __hash__(self) -> int:
        return hash(frozenset(self._terms.items()))

    # -- inspection --------------------------------------------------------

    @property
    def symbols(self) -> frozenset[str]:
        return frozenset(name for monomial in self._terms for name in monomial)

    @property
    def is_constant(self) -> bool:
        return all(monomial == () for monomial in self._terms)

    @property
    def constant(self) -> int:
        """The constant term."""
        return self._terms.get((), 0)

    @property
    def as_symbol(self) -> str | None:
        """The name, if this is exactly one symbol with coefficient 1."""
        if len(self._terms) == 1:
            ((monomial, coeff),) = self._terms.items()
            if coeff == 1 and len(monomial) == 1:
                return monomial[0]
        return None

    def evaluate(self, bindings: Mapping[str, int]) -> int:
        missing = self.symbols - bindings.keys()
        if missing:
            raise KeyError(f"no value bound for {', '.join(sorted(missing))} in {self}")
        total = 0
        for monomial, coeff in self._terms.items():
            term = coeff
            for name in monomial:
                term *= bindings[name]
            total += term
        return total

    def __str__(self) -> str:
        if not self._terms:
            return "0"

        # Highest degree first, then alphabetical, constant last:
        # "past + seq - 1", "14*seq".
        order = sorted(self._terms, key=lambda m: (-len(m), m))
        parts: list[str] = []
        for monomial in order:
            coeff = self._terms[monomial]
            body = "*".join(monomial)
            magnitude = abs(coeff)
            if not body:
                text = str(magnitude)
            elif magnitude == 1:
                text = body
            else:
                text = f"{magnitude}*{body}"
            if not parts:
                parts.append(text if coeff > 0 else f"-{text}")
            else:
                parts.append(f"+ {text}" if coeff > 0 else f"- {text}")
        return " ".join(parts)

    def __repr__(self) -> str:
        return f"Expr({self})"


Dim = int | Expr
Shape = tuple[Dim, ...]


def as_expr(value: Dim) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, bool):
        raise TypeError("a dimension must be an int or Expr, got bool")
    try:
        return Expr.const(operator.index(value))
    except TypeError:
        raise TypeError(
            f"a dimension must be an int or Expr, got {type(value).__name__}"
        ) from None


def canonical(value: Dim) -> Dim:
    """An ``int`` if the dimension has no symbols in it, else the ``Expr``.

    Every dimension stored in a node is canonical, so ``shape == (4, 16)``
    behaves the way it reads for concrete shapes.
    """
    expr = as_expr(value)
    return expr.constant if expr.is_constant else expr


def canonical_shape(shape: Iterable[Dim]) -> Shape:
    return tuple(canonical(d) for d in shape)


def dims_equal(a: Dim, b: Dim) -> bool:
    return as_expr(a) == as_expr(b)


def shapes_equal(a: Shape, b: Shape) -> bool:
    return len(a) == len(b) and all(dims_equal(x, y) for x, y in zip(a, b))


def product(dims: Iterable[Dim]) -> Dim:
    total = Expr.const(1)
    for d in dims:
        total = total * d
    return canonical(total)


def symbols_of(shape: Iterable[Dim]) -> frozenset[str]:
    names: set[str] = set()
    for d in shape:
        if isinstance(d, Expr):
            names |= d.symbols
    return frozenset(names)


def evaluate_shape(shape: Iterable[Dim], bindings: Mapping[str, int]) -> tuple[int, ...]:
    return tuple(d if isinstance(d, int) else d.evaluate(bindings) for d in shape)


def format_dim(d: Dim) -> str:
    return str(d)


def format_shape(shape: Iterable[Dim]) -> str:
    return "[" + ", ".join(format_dim(d) for d in shape) + "]"
