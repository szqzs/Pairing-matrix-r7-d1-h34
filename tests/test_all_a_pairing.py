from rank7_jk.all_a_pairing import (
    all_a_cache_info,
    all_a_pairing_total_batched_mod,
    all_a_pairing_total_mod,
    all_a_pairing_total_moment_mod,
    b_mask_pairing_total_batched_mod,
    b_mask_pairing_total_moment_mod,
    clear_all_a_caches,
    f2_power_pairing_total_batched_mod,
    f2_power_pairing_total_moment_mod,
    f_only_pairing_total_batched_mod,
    f_only_pairing_total_moment_mod,
    f_gamma_pairing_total_batched_mod,
    f_gamma_pairing_total_moment_mod,
    precompute_all_a_defect_kernels,
)
from rank7_jk import all_a_pairing
from rank7_jk import slow_evaluator
from rank7_jk.c18_basis import (
    a_exp_from_parts,
    c18_source_rows,
    h62_all_a_test_columns,
    restricted_partitions,
)
from rank7_jk.config import FormulaConfig
from rank7_jk.exterior import ExteriorAlgebra
from rank7_jk.invariants import InvariantMonomial
from rank7_jk.rank5_regression import scalar_fixture_by_name
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
    assert info["residue_functional"]["misses"] >= 1


def test_batched_total_matches_moment_for_rank7_f2_probe_entry():
    config = FormulaConfig(rank=7, genus=2)
    row = next(row for row in c18_source_rows(config) if row.defect == "f2")
    column = h62_all_a_test_columns(config)[0]
    total = row.monomial * column.monomial

    assert all_a_pairing_total_batched_mod(config, total, prime=101) == (
        all_a_pairing_total_moment_mod(config, total, prime=101)
    )


def test_threaded_derivative_groups_match_moment_for_rank5_delta(monkeypatch):
    total = InvariantMonomial.from_string(RANK5, "a2 a5^4 f3")
    monkeypatch.setenv("RANK7_JK_DERIVATIVE_THREADS", "2")

    assert all_a_pairing_total_batched_mod(RANK5, total, prime=101) == (
        all_a_pairing_total_moment_mod(RANK5, total, prime=101)
    )


def test_generic_degree_two_delta_kernel_matches_rank5_reference_kernel():
    targets = [
        (2, 0, 0),
        (1, 1, 0),
        (0, 2, 0),
        (0, 0, 2),
    ]

    for target in targets:
        observed = all_a_pairing._even_kernel_terms_delta_generic(  # noqa: SLF001
            RANK5,
            target,
            101,
        )
        expected = tuple(
            (deriv, items)
            for delta, deriv, items in slow_evaluator._even_kernel_terms_mod(  # noqa: SLF001
                target,
                101,
            )
            if delta == target
        )

        assert observed == expected


def test_f_only_two_defect_scaffold_matches_rank5_reference():
    for r, s in ((2, 2), (2, 3), (3, 3), (3, 4), (5, 5)):
        total = _rank5_total_with_two_f(r, s)
        expected = _rank5_reference(total)

        assert f_only_pairing_total_moment_mod(RANK5, total, prime=PRIME) == expected
        assert f_only_pairing_total_batched_mod(RANK5, total, prime=PRIME) == expected


def test_gamma_delta_kernel_matches_rank5_reference_kernel():
    gamma_exp = [0 for _ in RANK5.gamma_labels]
    gamma_exp[RANK5.gamma_labels.index((2, 2))] = 1
    for target in ((0, 0, 0), (1, 0, 0), (0, 1, 0)):
        observed = all_a_pairing._gamma_delta_kernel_terms(  # noqa: SLF001
            RANK5,
            target,
            tuple(gamma_exp),
            101,
        )
        expected = slow_evaluator._pairing_kernel_gamma_products_mod(  # noqa: SLF001
            target,
            tuple(gamma_exp),
            101,
        )

        assert observed == expected


def test_f_gamma_scaffold_matches_rank5_reference():
    cases = (
        (2, (2, 2)),
        (3, (2, 3)),
        (5, (4, 5)),
    )
    for f_rank, gamma_label in cases:
        total = _rank5_total_with_f_gamma(f_rank, gamma_label)
        expected = _rank5_reference(total)

        assert f_gamma_pairing_total_moment_mod(RANK5, total, prime=PRIME) == expected
        assert f_gamma_pairing_total_batched_mod(RANK5, total, prime=PRIME) == expected


def test_f2_power_scaffold_matches_rank5_public_high_f2_fixtures():
    for name in (
        "f2_11__f2_13",
        "f2_8_f4__f2_13",
        "f2_8_gamma22__f2_13",
    ):
        fixture = scalar_fixture_by_name(name)
        total = fixture.product

        assert (
            f2_power_pairing_total_moment_mod(
                RANK5,
                total,
                prime=fixture.prime,
            )
            == fixture.expected_mod
        )
        assert (
            f2_power_pairing_total_batched_mod(
                RANK5,
                total,
                prime=fixture.prime,
            )
            == fixture.expected_mod
        )


def test_direct_b_mask_kernel_matches_rank5_reference_kernel():
    mask = ExteriorAlgebra(RANK5).b_product_to_mask(((2, 1), (2, 3)))[1]

    for target in ((0, 0, 0), (1, 0, 0)):
        observed = all_a_pairing._shared_b_mask_kernel_terms(  # noqa: SLF001
            RANK5,
            target,
            mask,
            101,
        )
        b_delta = {
            delta: dict(poly_items)
            for delta, poly_items in slow_evaluator._b_hat_mask_mod(  # noqa: SLF001
                mask,
                target,
                101,
            )
        }
        expected = []
        for kernel_delta, deriv_orders, kernel_items in slow_evaluator._even_kernel_terms_mod(  # noqa: SLF001
            target,
            101,
        ):
            mask_delta = slow_evaluator._delta_sub(target, kernel_delta)  # noqa: SLF001
            if mask_delta is None or mask_delta not in b_delta:
                continue
            shared_poly = slow_evaluator._sparse_mul(  # noqa: SLF001
                dict(kernel_items),
                b_delta[mask_delta],
                101,
            )
            if shared_poly:
                expected.append((deriv_orders, tuple(sorted(shared_poly.items()))))

        assert observed == tuple(expected)


def test_direct_b_mask_scaffold_moment_and_batched_agree():
    mask = ExteriorAlgebra(RANK5).b_product_to_mask(((2, 1), (3, 3)))[1]
    a_exp = a_exp_from_parts(RANK5, restricted_partitions(26 - 3 - 2 - 3, RANK5.class_ranks)[0])
    f_exp = [0 for _ in RANK5.class_ranks]
    f_exp[3 - 2] = 1

    assert b_mask_pairing_total_batched_mod(
        RANK5,
        a_exp=a_exp,
        f_exp=f_exp,
        b_mask=mask,
        prime=101,
    ) == b_mask_pairing_total_moment_mod(
        RANK5,
        a_exp=a_exp,
        f_exp=f_exp,
        b_mask=mask,
        prime=101,
    )


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


def test_finite_cap_bounded_tau_power_matches_filtered_full_tau_power():
    config = FormulaConfig(rank=7, genus=2)
    prime = 101
    a_exp = (2, 1, 0, 0, 0, 0)
    caps = (126, 62, 30, 14, 6, 2)

    full = dict(all_a_pairing._tau_power_mod(config, a_exp, prime))  # noqa: SLF001
    bounded = dict(
        all_a_pairing._tau_power_bounded_mod(config, a_exp, caps, prime)  # noqa: SLF001
    )
    expected = {
        alpha: coeff
        for alpha, coeff in full.items()
        if all(alpha[idx] <= caps[idx] for idx in range(config.y_count))
    }

    assert bounded == expected


def test_high_first_bounded_tau_handles_f5_f7_caps_that_used_to_fail():
    config = FormulaConfig(rank=7, genus=2)
    prime = 101
    cases = [
        ((5, 1, 0, 0, 4, 1), (126, 62, 30, 14, 4, 1)),
        ((4, 1, 0, 0, 4, 1), (126, 62, 30, 14, 2, 1)),
    ]

    for a_exp, caps in cases:
        arrays = all_a_pairing._tau_power_bounded_numpy_arrays_mod(  # noqa: SLF001
            config,
            a_exp,
            caps,
            prime,
        )
        assert arrays is not None
        alpha, coeff = arrays
        assert coeff.size
        assert all(
            int(alpha[row_idx, col_idx]) <= caps[col_idx]
            for row_idx in range(alpha.shape[0])
            for col_idx in range(config.y_count)
        )
        assert set(int(value) for value in coeff) <= set(range(prime))


def test_array_kernel_product_matches_dict_product_for_small_manual_kernel():
    config = FormulaConfig(rank=7, genus=2)
    prime = 101
    a_exp = (2, 1, 0, 0, 0, 0)
    deriv_orders = (0, 0, 0, 0, 0, 0)
    residue_caps = all_a_pairing._residue_exponent_caps(  # noqa: SLF001
        config.rank,
        deriv_orders,
        config.root_denominator_power,
    )
    shared_items = (
        ((0, 0, 0, 0, 0, 0), 2),
        ((1, 0, 0, 0, 0, 0), 3),
        ((0, 1, 0, 0, 0, 0), 5),
    )

    actual = all_a_pairing._kernel_product_moment_sum_arrays_mod(  # noqa: SLF001
        config,
        a_exp,
        deriv_orders,
        shared_items,
        residue_caps,
        prime,
        max_chunk_terms=100,
    )
    product = all_a_pairing._bounded_tau_kernel_product_mod(  # noqa: SLF001
        config,
        a_exp,
        dict(shared_items),
        residue_caps,
        prime,
    )
    functional = all_a_pairing._residue_functional_cached(  # noqa: SLF001
        config.rank,
        deriv_orders,
        config.root_denominator_power,
        prime,
    )

    assert product is not None
    assert actual == functional.evaluate_poly_terms(product)


def test_array_kernel_product_splits_when_full_tau_cap_build_fails(monkeypatch):
    config = FormulaConfig(rank=7, genus=2)
    prime = 101
    a_exp = (2, 1, 0, 0, 0, 0)
    deriv_orders = (0, 0, 0, 0, 0, 0)
    residue_caps = all_a_pairing._residue_exponent_caps(  # noqa: SLF001
        config.rank,
        deriv_orders,
        config.root_denominator_power,
    )
    shared_items = (
        ((0, 0, 0, 0, 0, 0), 2),
        ((1, 0, 0, 0, 0, 0), 3),
        ((0, 1, 0, 0, 0, 0), 5),
    )
    real_tau_arrays = all_a_pairing._tau_power_bounded_numpy_arrays_mod  # noqa: SLF001
    calls = []

    def flaky_tau_arrays(config_arg, a_exp_arg, caps_arg, prime_arg):
        calls.append(tuple(caps_arg))
        if len(calls) == 1:
            return None
        return real_tau_arrays(config_arg, a_exp_arg, caps_arg, prime_arg)

    monkeypatch.setattr(
        all_a_pairing,
        "_tau_power_bounded_numpy_arrays_mod",
        flaky_tau_arrays,
    )

    actual = all_a_pairing._kernel_product_moment_sum_arrays_mod(  # noqa: SLF001
        config,
        a_exp,
        deriv_orders,
        shared_items,
        residue_caps,
        prime,
        max_chunk_terms=100,
    )
    product = all_a_pairing._bounded_tau_kernel_product_mod(  # noqa: SLF001
        config,
        a_exp,
        dict(shared_items),
        residue_caps,
        prime,
    )
    functional = all_a_pairing._residue_functional_cached(  # noqa: SLF001
        config.rank,
        deriv_orders,
        config.root_denominator_power,
        prime,
    )

    assert len(calls) >= 3
    assert product is not None
    assert actual == functional.evaluate_poly_terms(product)


def test_shifted_beta_cluster_reduces_duplicate_final_alpha_terms():
    import numpy as np

    prime = 101
    alpha_terms = np.asarray(
        (
            (1, 0, 0, 0, 0, 0),
            (0, 1, 0, 0, 0, 0),
        ),
        dtype=np.uint16,
    )
    coeff_terms = np.asarray((2, 3), dtype=np.int64)
    beta_items = (
        ((0, 1, 0, 0, 0, 0), 5, (5, 0, 5, 5, 5, 5)),
        ((1, 0, 0, 0, 0, 0), 7, (0, 5, 5, 5, 5, 5)),
    )

    reduced = all_a_pairing._reduce_shifted_beta_cluster_6(  # noqa: SLF001
        alpha_terms,
        coeff_terms,
        beta_items,
        (5, 5, 5, 5, 5, 5),
        prime,
        np,
    )

    assert reduced is not None
    reduced_alpha, reduced_coeff = reduced
    assert {
        tuple(int(item) for item in alpha): int(coeff)
        for alpha, coeff in zip(reduced_alpha, reduced_coeff)
    } == {
        (1, 1, 0, 0, 0, 0): (2 * 5 + 3 * 7) % prime,
    }


def test_dense_shifted_beta_reducer_matches_bincount_reducer():
    import numpy as np

    prime = 101
    alpha_terms = np.asarray(
        (
            (1, 1, 0, 0, 0, 0),
            (0, 1, 1, 0, 0, 0),
            (0, 0, 1, 1, 0, 0),
        ),
        dtype=np.uint16,
    )
    coeff_terms = np.asarray((2, 3, 4), dtype=np.int64)
    residue_caps = (5, 3, 2, 2, 1, 1)
    beta_items = (
        ((0, 0, 0, 0, 1, 0), 5, (5, 3, 2, 2, 0, 1)),
        ((0, 0, 0, 1, 0, 0), 7, (5, 3, 2, 1, 1, 1)),
        ((0, 0, 1, 0, 0, 0), 11, (5, 3, 1, 2, 1, 1)),
    )

    dense = all_a_pairing._reduce_shifted_beta_cluster_6_dense_grid(  # noqa: SLF001
        alpha_terms,
        coeff_terms,
        beta_items,
        residue_caps,
        prime,
        np,
    )
    bincount = all_a_pairing._reduce_shifted_beta_cluster_6_bincount(  # noqa: SLF001
        alpha_terms,
        coeff_terms,
        beta_items,
        residue_caps,
        prime,
        np,
    )

    assert dense is not None
    assert bincount is not None
    dense_alpha, dense_coeff = dense
    bincount_alpha, bincount_coeff = bincount
    assert {
        tuple(int(item) for item in alpha): int(coeff)
        for alpha, coeff in zip(dense_alpha, dense_coeff)
    } == {
        tuple(int(item) for item in alpha): int(coeff)
        for alpha, coeff in zip(bincount_alpha, bincount_coeff)
    }


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


def _rank5_total_with_two_f(r, s):
    parts = restricted_partitions(26 - r - s, RANK5.class_ranks)[0]
    f_exp = [0 for _ in RANK5.class_ranks]
    f_exp[r - 2] += 1
    f_exp[s - 2] += 1
    return InvariantMonomial.from_exponents(
        RANK5,
        a_exp=a_exp_from_parts(RANK5, parts),
        f_exp=f_exp,
    )


def _rank5_total_with_f_gamma(f_rank, gamma_label):
    r, s = gamma_label
    parts = restricted_partitions(26 - f_rank - r - s, RANK5.class_ranks)[0]
    f_exp = [0 for _ in RANK5.class_ranks]
    f_exp[f_rank - 2] = 1
    gamma_exp = [0 for _ in RANK5.gamma_labels]
    gamma_exp[RANK5.gamma_labels.index(gamma_label)] = 1
    return InvariantMonomial.from_exponents(
        RANK5,
        a_exp=a_exp_from_parts(RANK5, parts),
        f_exp=f_exp,
        gamma_exp=gamma_exp,
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
