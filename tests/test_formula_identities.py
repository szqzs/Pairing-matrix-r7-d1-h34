import sympy as sp

from rank7_jk.config import FormulaConfig, PairingProblem
from rank7_jk import formula_ref as ref


def test_rank7_simple_root_coordinates():
    config = FormulaConfig()
    y = ref.y_symbols(config)
    xs = ref.x_coordinates(config)

    assert sp.simplify(sum(xs)) == 0
    for idx in range(config.y_count):
        assert sp.simplify(xs[idx] - xs[idx + 1] - y[idx]) == 0


def test_rank7_tau_homogeneity_and_tau1_zero():
    config = FormulaConfig()
    y = ref.y_symbols(config)

    assert sp.simplify(ref.tau(config, 1)) == 0
    for r in range(2, config.rank + 1):
        poly = sp.Poly(ref.tau(config, r), *y)
        assert {sum(monom) for monom, _coeff in poly.terms()} == {r}


def test_rank7_b_components_at_delta_zero():
    config = FormulaConfig()
    y = ref.y_symbols(config)
    zero = ref.delta_zero_subs(config)

    components = [sp.simplify(item.subs(zero)) for item in ref.B_map_components(config)]
    assert components == list(y)


def test_rank7_b_delta_coefficients_have_negative_derivative_sign():
    config = FormulaConfig()
    components = ref.B_map_components(config)

    for delta_symbol, r in zip(ref.delta_symbols(config), config.delta_ranks):
        for idx, direction in enumerate(ref.simple_coroot_directions(config)):
            expected = -ref.directional_derivative(config, ref.tau(config, r), direction)
            assert sp.simplify(sp.diff(components[idx], delta_symbol) - expected) == 0


def test_rank7_positive_roots_and_denominator_power():
    config = FormulaConfig()
    y = ref.y_symbols(config)
    roots = ref.positive_roots(config)

    assert len(roots) == 21
    assert config.root_denominator_power == 2
    assert ref.positive_root_intervals(config) == (
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
        (3, 4), (3, 5), (3, 6), (3, 7),
        (4, 5), (4, 6), (4, 7),
        (5, 6), (5, 7),
        (6, 7),
    )
    assert roots[0] == y[0]
    assert roots[-1] == y[-1]
    assert sp.simplify(roots[5] - sum(y)) == 0


def test_rank7_ctilde_exponent_and_prefactor():
    config = FormulaConfig()
    y = ref.y_symbols(config)
    zero = ref.delta_zero_subs(config)

    assert ref.c_tilde_direction_y(config) == (0, 0, 0, 0, 0, 1)

    exponent = sp.simplify(ref.c_tilde_exponent(config).subs(zero))
    expected = -sum((idx + 1) * y[idx] for idx in range(config.y_count)) / config.rank
    assert sp.simplify(exponent - expected) == 0
    assert config.collapsed_prefactor == -7


def test_rank7_ctilde_y_direction_is_derived_from_x_direction():
    config = FormulaConfig()

    assert ref.c_tilde_direction_y(config) == ref.x_direction_to_y_direction(
        config,
        ref.c_tilde_x_coordinates(config),
    )


def test_rank7_ctilde_delta_coefficients_match_tau_derivatives():
    config = FormulaConfig()
    exponent = ref.c_tilde_exponent(config)
    direction = ref.c_tilde_direction_y(config)

    for delta_symbol, r in zip(ref.delta_symbols(config), config.delta_ranks):
        expected = ref.directional_derivative(config, ref.tau(config, r), direction)
        assert sp.simplify(sp.diff(exponent, delta_symbol) - expected) == 0


def test_rank7_q_at_delta_zero_is_tau2():
    config = FormulaConfig()

    assert sp.simplify(
        ref.q_polynomial(config).subs(ref.delta_zero_subs(config)) - ref.tau(config, 2)
    ) == 0


def test_rank7_hessian_ratio_at_delta_zero():
    config = FormulaConfig()

    assert ref.hessian_ratio_at_delta_zero(config) == 1


def _ranked_y_point_subs(config):
    return {var: idx + 1 for idx, var in enumerate(ref.y_symbols(config))}


def _matrix_at_y(matrix, y_subs):
    return matrix.applyfunc(lambda item: sp.simplify(item.subs(y_subs)))


def _hat_pair_delta_zero_value(config, r, s):
    y_subs = _ranked_y_point_subs(config)
    hessian = ref.hessian_y_basis(config, ref.tau(config, 2))
    gr = sp.Matrix([item.subs(y_subs) for item in ref.tau_grad_y(config, r)])
    gs = sp.Matrix([item.subs(y_subs) for item in ref.tau_grad_y(config, s)])
    return sp.simplify(-(gr.T * hessian.inv() * gs)[0])


def _hat_pair_first_delta_value(config, r, s, delta_rank):
    y_subs = _ranked_y_point_subs(config)
    h0_inv = ref.hessian_y_basis(config, ref.tau(config, 2)).inv()
    h_delta = _matrix_at_y(ref.hessian_y_basis(config, ref.tau(config, delta_rank)), y_subs)
    gr = sp.Matrix([item.subs(y_subs) for item in ref.tau_grad_y(config, r)])
    gs = sp.Matrix([item.subs(y_subs) for item in ref.tau_grad_y(config, s)])
    return sp.simplify((gr.T * h0_inv * h_delta * h0_inv * gs)[0])


def test_rank7_hat_tau_coefficients_at_delta_zero():
    config = FormulaConfig()
    cases = [
        (2, 2, sp.Integer(364)),
        (2, 3, sp.Integer(1560)),
        (3, 5, sp.Integer(-945444)),
        (6, 7, sp.Integer(1652676480)),
    ]

    for r, s, expected in cases:
        assert _hat_pair_delta_zero_value(config, r, s) == expected


def test_rank7_hat_tau_first_delta_coefficients_have_correct_sign():
    config = FormulaConfig()
    cases = [
        (2, 2, 3, sp.Integer(-3120)),
        (2, 3, 3, sp.Integer(-38376)),
        (3, 5, 4, sp.Rational(-619473348, 7)),
        (6, 7, 7, sp.Integer(-63072322267968)),
    ]

    for r, s, delta_rank, expected in cases:
        assert _hat_pair_first_delta_value(config, r, s, delta_rank) == expected


def test_delta_coefficient_and_f_factorial_helpers():
    config = FormulaConfig()
    d3, d4, *_rest = ref.delta_symbols(config)
    expr = (1 + d3) ** 3 * (2 + d4)

    assert ref.delta_coefficient_at_zero(config, expr, (2, 1, 0, 0, 0)) == 3
    assert ref.f_factorial_scale((0, 2, 3, 1, 0, 4)) == (
        sp.factorial(2) * sp.factorial(3) * sp.factorial(4)
    )


def test_iterated_residue_order_is_visible():
    config = FormulaConfig(rank=3, genus=2)
    y1, y2 = ref.y_symbols(config)
    expr = 1 / (y1 * (y1 + y2))

    assert ref.iterated_residue(expr, (y1, y2), series_order=4) == 1
    assert ref.iterated_residue(expr, (y2, y1), series_order=4) == 0


def test_rank7_pairing_problem_degrees_match_top_degree():
    problem = PairingProblem()

    assert problem.source_degree == 34
    assert problem.test_degree == 62
    assert problem.source_degree + problem.test_degree == problem.formula.top_degree
    assert problem.expected_relation_chern_degree == 18


def test_rank5_compatibility_basic_identities():
    config = FormulaConfig(rank=5, genus=2)
    y = ref.y_symbols(config)
    zero = ref.delta_zero_subs(config)

    assert config.top_degree == 48
    assert config.collapsed_prefactor == 5
    assert ref.positive_root_intervals(config) == (
        (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5),
        (3, 4), (3, 5),
        (4, 5),
    )
    assert [sp.simplify(item.subs(zero)) for item in ref.B_map_components(config)] == list(
        y
    )

    exponent = sp.simplify(ref.c_tilde_exponent(config).subs(zero))
    expected = -sum((idx + 1) * y[idx] for idx in range(config.y_count)) / config.rank
    assert sp.simplify(exponent - expected) == 0
    assert ref.hessian_ratio_at_delta_zero(config) == 1
