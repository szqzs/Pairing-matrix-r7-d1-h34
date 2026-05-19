from rank7_jk.all_a_pairing import all_a_pairing_total_mod
from rank7_jk.config import FormulaConfig
from rank7_jk.invariants import InvariantMonomial
from rank7_jk.slow_evaluator import pairing_mod_prime


RANK5 = FormulaConfig(rank=5, genus=2)
PRIME = 1_000_033


def test_one_defect_all_a_evaluator_matches_rank5_reference_for_f2_defect():
    total = InvariantMonomial.from_string(RANK5, "a4^2 a5^3 f2")

    assert all_a_pairing_total_mod(RANK5, total, prime=PRIME) == _rank5_reference(total)


def test_one_defect_all_a_evaluator_matches_rank5_reference_for_nonzero_delta_defect():
    total = InvariantMonomial.from_string(RANK5, "a2 a5^4 f3")

    assert all_a_pairing_total_mod(RANK5, total, prime=PRIME) == _rank5_reference(total)


def test_one_defect_all_a_evaluator_matches_rank5_reference_for_gamma_defects():
    cases = [
        "a2 a4 a5^3 gamma22",
        "a2 a5^3 gamma35",
    ]

    for text in cases:
        total = InvariantMonomial.from_string(RANK5, text)
        assert all_a_pairing_total_mod(RANK5, total, prime=PRIME) == _rank5_reference(total)


def _rank5_reference(total):
    return pairing_mod_prime(
        RANK5,
        total,
        InvariantMonomial.identity(RANK5),
        prime=PRIME,
    )
