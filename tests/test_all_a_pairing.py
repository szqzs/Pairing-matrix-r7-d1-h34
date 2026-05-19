from rank7_jk.all_a_pairing import (
    all_a_cache_info,
    all_a_pairing_total_mod,
    all_a_pairing_total_moment_mod,
    clear_all_a_caches,
    precompute_all_a_defect_kernels,
)
from rank7_jk import all_a_pairing
from rank7_jk.c18_basis import a_exp_from_parts, c18_source_rows, restricted_partitions
from rank7_jk.config import FormulaConfig
from rank7_jk.invariants import InvariantMonomial
from rank7_jk.slow_evaluator import pairing_mod_prime


RANK5 = FormulaConfig(rank=5, genus=2)
PRIME = 1_000_033


def test_one_defect_all_a_evaluator_matches_rank5_reference_for_f2_defect():
    total = InvariantMonomial.from_string(RANK5, "a4^2 a5^3 f2")

    _assert_generic_and_moment_match_rank5_reference(total)


def test_one_defect_all_a_evaluator_matches_rank5_reference_for_nonzero_delta_defect():
    total = InvariantMonomial.from_string(RANK5, "a2 a5^4 f3")

    _assert_generic_and_moment_match_rank5_reference(total)


def test_one_defect_all_a_evaluator_matches_rank5_reference_for_gamma_defects():
    cases = [
        "a2 a4 a5^3 gamma22",
        "a2 a5^3 gamma35",
    ]

    for text in cases:
        total = InvariantMonomial.from_string(RANK5, text)
        _assert_generic_and_moment_match_rank5_reference(total)


def test_moment_evaluator_matches_rank5_reference_for_every_f_defect():
    for r in RANK5.class_ranks:
        _assert_generic_and_moment_match_rank5_reference(_rank5_total_with_f(r))


def test_moment_evaluator_matches_rank5_reference_for_every_gamma_defect():
    for r, s in RANK5.gamma_labels:
        _assert_generic_and_moment_match_rank5_reference(_rank5_total_with_gamma(r, s))


def test_all_a_cache_info_and_kernel_precompute_are_probe_visible():
    clear_all_a_caches()
    defects = precompute_all_a_defect_kernels(
        FormulaConfig(rank=7, genus=2),
        c18_source_rows()[:20],
        prime=101,
    )
    total = InvariantMonomial.from_string(RANK5, "a4^2 a5^3 f2")
    all_a_pairing_total_moment_mod(RANK5, total, prime=PRIME)
    info = all_a_cache_info()

    assert defects == ("f2",)
    assert {"batch_evaluator", "kernel_terms", "moment", "monomial_residue", "tau_power"} <= set(info)
    assert "bounded_tau_power" in info
    assert info["kernel_terms"]["misses"] >= 1
    assert info["moment"]["misses"] >= 1
    assert info["monomial_residue"]["misses"] >= 1


def test_bounded_tau_power_matches_filtered_full_tau_power_for_small_case():
    config = FormulaConfig(rank=7, genus=2)
    prime = 101
    a_exp = (2, 1, 0, 0, 0, 0)
    caps = (None, None, None, None, None, 2)

    full = dict(all_a_pairing._tau_power_mod(config, a_exp, prime))  # noqa: SLF001
    bounded = dict(
        all_a_pairing._tau_power_bounded_mod(config, a_exp, caps, prime)  # noqa: SLF001
    )
    expected = {
        alpha: coeff
        for alpha, coeff in full.items()
        if alpha[-1] <= 2
    }

    assert bounded == expected


def test_residue_exponent_caps_capture_backward_transition_bounds():
    assert all_a_pairing._residue_exponent_caps(5, (0, 0, 0, 0), 2) == (30, 14, 6, 2)  # noqa: SLF001
    assert all_a_pairing._residue_exponent_caps(7, (0, 0, 0, 0, 0, 0), 2) == (126, 62, 30, 14, 6, 2)  # noqa: SLF001


def _assert_generic_and_moment_match_rank5_reference(total):
    expected = _rank5_reference(total)

    assert all_a_pairing_total_mod(RANK5, total, prime=PRIME) == expected
    assert all_a_pairing_total_moment_mod(RANK5, total, prime=PRIME) == expected


def _rank5_reference(total):
    return pairing_mod_prime(
        RANK5,
        total,
        InvariantMonomial.identity(RANK5),
        prime=PRIME,
    )


def _rank5_total_with_f(r):
    parts = restricted_partitions(25 - r, RANK5.class_ranks)[0]
    f_exp = [0 for _ in RANK5.class_ranks]
    f_exp[r - 2] = 1
    return InvariantMonomial.from_exponents(
        RANK5,
        a_exp=a_exp_from_parts(RANK5, parts),
        f_exp=f_exp,
    )


def _rank5_total_with_gamma(r, s):
    parts = restricted_partitions(25 - r - s, RANK5.class_ranks)[0]
    gamma_exp = [0 for _ in RANK5.gamma_labels]
    gamma_exp[RANK5.gamma_labels.index((r, s))] = 1
    return InvariantMonomial.from_exponents(
        RANK5,
        a_exp=a_exp_from_parts(RANK5, parts),
        gamma_exp=gamma_exp,
    )
