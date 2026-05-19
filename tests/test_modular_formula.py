import sympy as sp

from rank7_jk import formula_ref, modular_formula
from rank7_jk.config import FormulaConfig
from rank7_jk.mod_arith import rational_mod


def _point_subs(config, point):
    return dict(zip(formula_ref.y_symbols(config), point))


def _expr_mod(expr, config, point, prime):
    return rational_mod(sp.simplify(expr.subs(_point_subs(config, point))), prime)


def _poly_value(poly_items, point, prime):
    return modular_formula.evaluate_sparse(dict(poly_items), point, prime=prime)


def test_rank_generic_tau_matches_formula_ref_at_fixed_points():
    prime = 1_000_033
    cases = [
        (FormulaConfig(rank=5, genus=2), (2, 3, 5, 7)),
        (FormulaConfig(rank=7, genus=2), (2, 3, 5, 7, 11, 13)),
    ]

    for config, point in cases:
        for r in config.class_ranks:
            observed = _poly_value(modular_formula.tau_mod(config, r, prime), point, prime)
            expected = _expr_mod(formula_ref.tau(config, r), config, point, prime)
            assert observed == expected


def test_rank7_gradients_and_hessians_match_formula_ref_at_fixed_point():
    config = FormulaConfig(rank=7, genus=2)
    prime = 1_000_033
    point = (2, 3, 5, 7, 11, 13)
    y = formula_ref.y_symbols(config)

    for r in (2, 3, 5, 7):
        grad = modular_formula.tau_grad_mod(config, r, prime)
        hessian = modular_formula.tau_hessian_mod(config, r, prime)
        tau_expr = formula_ref.tau(config, r)
        for i, yi in enumerate(y):
            assert _poly_value(grad[i], point, prime) == _expr_mod(
                sp.diff(tau_expr, yi),
                config,
                point,
                prime,
            )
            for j, yj in enumerate(y):
                assert _poly_value(hessian[i][j], point, prime) == _expr_mod(
                    sp.diff(tau_expr, yi, yj),
                    config,
                    point,
                    prime,
                )


def test_rank7_b_and_ctilde_delta_coefficients_match_formula_ref():
    config = FormulaConfig(rank=7, genus=2)
    prime = 1_000_033
    point = (2, 3, 5, 7, 11, 13)

    for r in config.delta_ranks:
        c_observed = _poly_value(
            modular_formula.c_tilde_delta_coeff_mod(config, r, prime),
            point,
            prime,
        )
        c_expected = _expr_mod(
            formula_ref.directional_derivative(
                config,
                formula_ref.tau(config, r),
                formula_ref.c_tilde_direction_y(config),
            ),
            config,
            point,
            prime,
        )
        assert c_observed == c_expected

        for j, direction in enumerate(formula_ref.simple_coroot_directions(config), start=1):
            b_observed = _poly_value(
                modular_formula.b_perturbation_mod(config, r, j, prime),
                point,
                prime,
            )
            b_expected = _expr_mod(
                -formula_ref.directional_derivative(
                    config,
                    formula_ref.tau(config, r),
                    direction,
                ),
                config,
                point,
                prime,
            )
            assert b_observed == b_expected


def test_rank7_hat_pair_coefficients_match_formula_ref_at_fixed_point():
    config = FormulaConfig(rank=7, genus=2)
    prime = 1_000_033
    point = (2, 3, 5, 7, 11, 13)

    zero_observed = _poly_value(
        modular_formula.hat_pair_delta_zero_mod(config, 3, 5, prime),
        point,
        prime,
    )
    zero_expected = _expr_mod(
        formula_ref.hat_pair_coefficient_at_delta_zero(config, 3, 5),
        config,
        point,
        prime,
    )
    assert zero_observed == zero_expected

    first_observed = _poly_value(
        modular_formula.hat_pair_first_delta_mod(config, 3, 5, 4, prime),
        point,
        prime,
    )
    first_expected = _expr_mod(
        formula_ref.hat_pair_first_delta_coefficient_at_zero(config, 3, 5, 4),
        config,
        point,
        prime,
    )
    assert first_observed == first_expected
