"""Deterministic checks used before accepting model-based STEM grading."""

from __future__ import annotations

import re
from collections import Counter

from sympy import simplify
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_SAFE_EXPRESSION = re.compile(r"^[0-9A-Za-z+\-*/^().,\s=]+$")
_SAFE_FUNCTIONS = {"sin", "cos", "tan", "sqrt", "log", "exp", "abs"}
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def math_equivalent(candidate: str, expected: str) -> bool | None:
    """Return equivalence, or ``None`` when an expression is unsafe/unparseable."""
    if not candidate or not expected:
        return None
    if not _SAFE_EXPRESSION.fullmatch(candidate + expected):
        return None
    identifiers = re.findall(r"[A-Za-z]+", candidate + " " + expected)
    if any(len(name) > 2 and name not in _SAFE_FUNCTIONS for name in identifiers):
        return None
    try:
        left = parse_expr(candidate.split("=", 1)[-1], transformations=_TRANSFORMS)
        right = parse_expr(expected.split("=", 1)[-1], transformations=_TRANSFORMS)
        return bool(simplify(left - right) == 0)
    except Exception:
        return None


def units_compatible(candidate: str, expected: str) -> bool | None:
    """Check dimensional compatibility without judging the numeric value."""
    try:
        from pint import UnitRegistry

        registry = UnitRegistry()
        return registry(candidate).dimensionality == registry(expected).dimensionality
    except Exception:
        return None


_ELEMENT = re.compile(r"([A-Z][a-z]?)(\d*)")


def _formula_atoms(formula: str) -> Counter[str] | None:
    formula = formula.strip()
    if not formula or re.search(r"[^A-Za-z0-9]", formula):
        return None
    atoms: Counter[str] = Counter()
    position = 0
    for match in _ELEMENT.finditer(formula):
        if match.start() != position:
            return None
        atoms[match.group(1)] += int(match.group(2) or "1")
        position = match.end()
    return atoms if position == len(formula) else None


def chemical_equation_balanced(equation: str) -> bool | None:
    """Validate common non-parenthesized equations such as ``2H2 + O2 -> 2H2O``."""
    separator = "->" if "->" in equation else "=" if "=" in equation else None
    if not separator:
        return None

    def side_atoms(side: str) -> Counter[str] | None:
        total: Counter[str] = Counter()
        for term in side.split("+"):
            match = re.fullmatch(r"\s*(\d*)\s*([A-Za-z0-9]+)\s*", term)
            if not match:
                return None
            coefficient = int(match.group(1) or "1")
            atoms = _formula_atoms(match.group(2))
            if atoms is None:
                return None
            for element, count in atoms.items():
                total[element] += coefficient * count
        return total

    left, right = equation.split(separator, 1)
    left_atoms, right_atoms = side_atoms(left), side_atoms(right)
    if left_atoms is None or right_atoms is None:
        return None
    return left_atoms == right_atoms
