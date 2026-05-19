import json

from rank7_jk import formula_ref, residue_oracle, residue_transition, slow_evaluator
from rank7_jk.config import FormulaConfig, RANK7_G2_D1
from rank7_jk.mod_arith import rational_mod
from rank7_jk.rank5_regression import RANK5_PRIMARY_PRIME
from rank7_jk.rank7_smoke import RANK7_RESIDUE_SMOKE_CASES, run_residue_smoke_cases
from rank7_jk.root_system import type_a_roots
from rank7_jk import repro, sparse_poly


def test_type_a_root_system_documents_interval_conventions():
    roots = type_a_roots(7)
    formula_config = FormulaConfig(rank=7, genus=2)

    assert roots.y_count == 6
    assert roots.positive_root_count == 21
    assert roots.positive_intervals_one_based == formula_ref.positive_root_intervals(
        formula_config
    )
    assert roots.positive_intervals_zero_based == residue_oracle.root_intervals(7)
    assert roots.transition_schedule[0] == ((0, -1),)
    assert roots.transition_schedule[-1][-1] == (20, -1)


def test_sparse_poly_core_operations_are_dimension_generic():
    p = 101
    y1 = sparse_poly.monomial((1, 0, 0, 0, 0, 0), prime=p)
    y2 = sparse_poly.monomial((0, 1, 0, 0, 0, 0), 3, prime=p)
    poly = sparse_poly.add(y1, y2, prime=p)

    assert sparse_poly.mul(poly, poly, prime=p) == {
        (2, 0, 0, 0, 0, 0): 1,
        (1, 1, 0, 0, 0, 0): 6,
        (0, 2, 0, 0, 0, 0): 9,
    }
    assert sparse_poly.derivative(poly, 1, prime=p) == {
        (0, 0, 0, 0, 0, 0): 3,
    }


def test_gate_c_special_coefficients_match_literal_oracle_coefficients():
    cases = [
        (7, 0, 0, 0),
        (7, 1, 1, 0),
        (7, 2, 2, -1),
        (7, 5, 3, -3),
        (5, 3, 2, 1),
    ]

    for args in cases:
        assert residue_transition._special_coeff_exact_from_bernoulli(  # noqa: SLF001
            *args
        ) == residue_oracle._special_coeff(*args)  # noqa: SLF001


def test_gate_c_transition_matches_rank5_reference_transition():
    cases = [
        ((0, 0, 0, 0), (0, 0, 0, 0)),
        ((2, 0, 0, 0), (0, 0, 0, 0)),
        ((0, 0, 0, 2), (0, 0, 0, 1)),
        ((3, 2, 1, 0), (1, 0, 0, 0)),
        ((4, 3, 2, 1), (0, 1, 0, 0)),
    ]

    for alpha, derivative_orders in cases:
        observed = residue_transition.residue_monomial_mod(
            5,
            alpha,
            derivative_orders,
            prime=RANK5_PRIMARY_PRIME,
            root_power=2,
        )
        expected = slow_evaluator._residue_poly_mod(  # noqa: SLF001
            {alpha: 1},
            derivative_orders,
            RANK5_PRIMARY_PRIME,
        )
        assert observed == expected


def test_gate_c_rank7_tiny_cases_match_exact_transition_oracle():
    cases = [
        ((0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0), 0),
        ((0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 1), 0),
        ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0), 1),
        ((0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 1), 1),
    ]

    for alpha, derivative_orders, root_power in cases:
        exact = residue_oracle.exact_transition_residue(
            7,
            alpha,
            derivative_orders,
            root_power=root_power,
        )
        observed = residue_transition.residue_monomial_mod(
            7,
            alpha,
            derivative_orders,
            prime=RANK7_G2_D1.primary_prime,
            root_power=root_power,
        )
        assert observed == rational_mod(exact, RANK7_G2_D1.primary_prime)


def test_gate_c_rank7_root_power2_smoke_cases_match_two_prime_fixtures():
    results = run_residue_smoke_cases()

    assert len(results) == len(RANK7_RESIDUE_SMOKE_CASES) == 5
    assert all(item["passed"] for item in results)


def test_gate_c_payload_and_artifact_shape():
    payload = repro.gate_c_payload()

    assert payload["status"] == "passed"
    assert payload["github_repository"] == "https://github.com/szqzs/Pairing-matrix-r7-d1-h34"
    assert len(payload["residue_transition_smoke"]) == 5

    artifact = repro.PROJECT_ROOT / "artifacts/math_gates/gate_C_rank7_smoke.json"
    if artifact.exists():
        saved = json.loads(artifact.read_text(encoding="utf-8"))
        assert saved["gate"] == payload["gate"]
        assert len(saved["residue_transition_smoke"]) == 5
