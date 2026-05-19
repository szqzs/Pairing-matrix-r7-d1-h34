from rank7_jk.config import FormulaConfig
from rank7_jk.exterior import ExteriorAlgebra, gamma_exponent


def term_dict(exterior, poly):
    return {
        tuple(exterior.labels_from_mask(mask)): coeff
        for mask, coeff in poly.items()
    }


def test_wedge_repeated_variable_is_zero():
    exterior = ExteriorAlgebra(FormulaConfig())
    b21 = exterior.mask_for_b_label((2, 1))

    assert exterior.wedge_masks(b21, b21) is None


def test_gamma_symmetry_rank7_genus2():
    exterior = ExteriorAlgebra(FormulaConfig())

    for r in range(2, 8):
        for s in range(2, 8):
            assert exterior.gamma_as_mask_poly(r, s) == exterior.gamma_as_mask_poly(s, r)


def test_gamma_rr_doubled_terms_rank7_genus2():
    exterior = ExteriorAlgebra(FormulaConfig())

    assert term_dict(exterior, exterior.gamma_as_mask_poly(2, 2)) == {
        ((2, 1), (2, 3)): 2,
        ((2, 2), (2, 4)): 2,
    }
    assert term_dict(exterior, exterior.gamma_as_mask_poly(7, 7)) == {
        ((7, 1), (7, 3)): 2,
        ((7, 2), (7, 4)): 2,
    }


def test_gamma_power_nilpotence_rank7_genus2():
    config = FormulaConfig()
    exterior = ExteriorAlgebra(config)

    # gamma_22 lives in the four variables b2_1,...,b2_4, so its cube is zero.
    gamma22_cubed = exterior.gamma_product_to_mask_poly(gamma_exponent(config, (2, 2), 3))
    assert gamma22_cubed == {}


def test_rank5_gamma_convention_matches_same_formula():
    config = FormulaConfig(rank=5, genus=2)
    exterior = ExteriorAlgebra(config)

    assert term_dict(exterior, exterior.gamma_as_mask_poly(2, 2)) == {
        ((2, 1), (2, 3)): 2,
        ((2, 2), (2, 4)): 2,
    }
