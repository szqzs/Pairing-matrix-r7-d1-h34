import sympy as sp

from rank7_jk import residue_oracle as oracle
from rank7_jk import slow_evaluator
from rank7_jk.rank5_regression import RANK5_PRIMARY_PRIME


def test_oracle_iterated_residue_order_is_visible():
    y1, y2 = sp.symbols("Y1 Y2")
    expr = 1 / (y1 * (y1 + y2))

    assert oracle.iterated_residue(expr, (y1, y2), series_order=4) == 1
    assert oracle.iterated_residue(expr, (y2, y1), series_order=4) == 0


def test_rank4_literal_residue_checks_nonsimple_root_interaction():
    y1, y2, y3 = sp.symbols("Y1 Y2 Y3")
    expr = 1 / (y3 * (y2 + y3) * (y1 + y2 + y3))

    assert oracle.iterated_residue(expr, (y3, y2, y1), series_order=8) == 1
    assert oracle.iterated_residue(expr, (y1, y2, y3), series_order=8) == 0


def test_rank3_exact_transition_matches_literal_sympy_expansion():
    cases = [
        ((0, 0), (0, 0), sp.Rational(-53, 4_898_880)),
        ((1, 0), (0, 1), sp.Rational(409, 816_480)),
    ]

    for alpha, derivative_orders, expected in cases:
        literal = oracle.literal_residue_sympy(
            3,
            alpha,
            derivative_orders,
            series_order=8,
        )
        exact_oracle = oracle.exact_transition_residue(3, alpha, derivative_orders)

        assert literal == expected
        assert exact_oracle == expected


def test_rank7_special_coefficients_are_fixed_exact_values():
    cases = [
        (0, 0, -1, sp.Rational(5, 14)),
        (1, 1, -1, sp.Rational(25, 588)),
        (2, 2, 0, sp.Rational(9, 49)),
        (5, 3, 2, sp.Rational(36, 7)),
    ]

    for var_idx, derivative_order, y_exponent, expected in cases:
        assert (
            oracle.special_coefficient_exact(7, var_idx, derivative_order, y_exponent)
            == expected
        )


def test_rank5_exact_oracle_matches_modular_residue_transition():
    cases = [
        ((0, 0, 0, 0), (0, 0, 0, 0), 1487712662349434581),
        ((2, 0, 0, 0), (0, 0, 0, 0), 1394772177027177008),
        ((0, 0, 0, 2), (0, 0, 0, 1), 987425446863991084),
        ((3, 2, 1, 0), (1, 0, 0, 0), 1639863168027850062),
        ((4, 3, 2, 1), (0, 1, 0, 0), 357148846229076161),
    ]

    for alpha, derivative_orders, expected in cases:
        observed_oracle = oracle.jk_residue_mod(
            5,
            alpha,
            derivative_orders,
            prime=RANK5_PRIMARY_PRIME,
        )
        observed_transition = slow_evaluator._residue_poly_mod(  # noqa: SLF001
            {alpha: 1},
            derivative_orders,
            RANK5_PRIMARY_PRIME,
        )
        assert observed_oracle == expected
        assert observed_transition == expected


def test_rational_mod_rejects_bad_moduli_and_zero_denominators():
    assert oracle.rational_mod(sp.Rational(-3, 7), 11) == 9

    try:
        oracle.rational_mod(sp.Rational(1, 3), 9)
    except ValueError:
        pass
    else:
        raise AssertionError("non-prime modulus should be rejected")

    try:
        oracle.rational_mod(sp.Rational(1, 5), 5)
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("zero denominator modulo prime should be rejected")
