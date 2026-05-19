from rank7_jk.residue_functional import ResidueFunctional
from rank7_jk.residue_transition import residue_monomial_mod, residue_poly_mod


def test_residue_functional_matches_transition_for_rank5_monomials():
    p = 101
    cases = [
        ((0, 0, 0, 0), (0, 0, 0, 0)),
        ((2, 0, 0, 0), (0, 0, 0, 0)),
        ((0, 0, 0, 2), (0, 0, 0, 1)),
        ((3, 2, 1, 0), (1, 0, 0, 0)),
    ]

    for alpha, deriv in cases:
        functional = ResidueFunctional(
            rank=5,
            derivative_orders=deriv,
            root_power=2,
            prime=p,
        )
        assert functional.evaluate_poly_terms({alpha: 1}) == residue_monomial_mod(
            5,
            alpha,
            deriv,
            prime=p,
            root_power=2,
        )


def test_residue_functional_matches_transition_for_rank7_sparse_poly():
    p = 101
    deriv = (0, 0, 0, 0, 0, 0)
    poly = {
        (0, 0, 0, 0, 0, 0): 3,
        (0, 0, 0, 0, 0, 2): 5,
        (4, 2, 1, 0, 0, 0): 7,
        (8, 1, 0, 0, 0, 2): 11,
    }
    functional = ResidueFunctional(rank=7, derivative_orders=deriv, root_power=2, prime=p)

    assert functional.evaluate_poly_terms(poly) == residue_poly_mod(
        7,
        poly,
        deriv,
        prime=p,
        root_power=2,
    )


def test_residue_functional_profile_records_sliced_stages():
    functional = ResidueFunctional(
        rank=7,
        derivative_orders=(0, 0, 0, 0, 0, 0),
        root_power=2,
        prime=101,
    )
    profile = functional.profile_poly_terms({(0, 0, 0, 0, 0, 0): 1})

    assert profile.input_terms == 1
    assert profile.result == residue_monomial_mod(
        7,
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        prime=101,
        root_power=2,
    )
    assert tuple(stage.var_idx for stage in profile.stages) == (5, 4, 3, 2, 1, 0)
    assert all(stage.elapsed_seconds >= 0 for stage in profile.stages)
