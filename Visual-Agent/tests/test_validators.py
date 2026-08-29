from education.validators import (
    chemical_equation_balanced,
    math_equivalent,
    units_compatible,
)


def test_math_equivalence():
    assert math_equivalent("(x+1)^2", "x^2+2*x+1") is True
    assert math_equivalent("x+2", "x+3") is False
    assert math_equivalent("__import__(os)", "0") is None


def test_physics_units():
    assert units_compatible("9.8 meter / second ** 2", "kilometer / hour ** 2") is True
    assert units_compatible("meter", "second") is False


def test_chemical_balance():
    assert chemical_equation_balanced("2H2 + O2 -> 2H2O") is True
    assert chemical_equation_balanced("H2 + O2 -> H2O") is False
    assert chemical_equation_balanced("not an equation") is None

