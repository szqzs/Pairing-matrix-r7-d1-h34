"""Command-line structural checks for the Step 1 formula layer."""

from __future__ import annotations

import json
from typing import Any, Dict

import sympy as sp

from .config import JKConfig, PairingProblem, RANK7_H34_H62
from .exterior import ExteriorAlgebra
from . import formula_ref as ref


def _ok(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_structural_checks(
    config: JKConfig = JKConfig(),
    problem: PairingProblem | None = None,
) -> Dict[str, Any]:
    if problem is not None:
        _ok(problem.formula == config, "pairing problem formula does not match config")

    y = ref.y_symbols(config)
    deltas_zero = ref.delta_zero_subs(config)
    xs = ref.x_coordinates(config)

    coordinate_checks = {
        "sum_x_zero": sp.simplify(sum(xs)) == 0,
        "successive_differences": [
            sp.simplify(xs[idx] - xs[idx + 1] - y[idx]) == 0
            for idx in range(config.y_count)
        ],
        "x_coordinates": [str(item) for item in xs],
    }
    _ok(coordinate_checks["sum_x_zero"], "sum_i x_i is not zero")
    _ok(all(coordinate_checks["successive_differences"]), "x_i - x_{i+1} check failed")

    tau_checks = {
        "tau1_zero": sp.simplify(ref.tau(config, 1)) == 0,
        "tau_degrees": {},
    }
    _ok(tau_checks["tau1_zero"], "tau_1 is not zero")
    for r in range(2, config.rank + 1):
        poly = sp.Poly(ref.tau(config, r), *y)
        degrees = {sum(monom) for monom, _coeff in poly.terms()}
        tau_checks["tau_degrees"][str(r)] = sorted(degrees)
        _ok(degrees == {r}, f"tau_{r} is not homogeneous of degree {r}")

    b_components = ref.B_map_components(config)
    b_delta_zero = [sp.simplify(component.subs(deltas_zero)) for component in b_components]
    b_checks = {
        "b_components_at_delta_zero": [str(item) for item in b_delta_zero],
        "b_j_equals_y_j_at_delta_zero": [
            sp.simplify(item - y[idx]) == 0
            for idx, item in enumerate(b_delta_zero)
        ],
        "delta_coefficients_match_negative_tau_derivatives": {},
    }
    _ok(all(b_checks["b_j_equals_y_j_at_delta_zero"]), "B_j != Y_j at delta zero")
    for delta_symbol, r in zip(ref.delta_symbols(config), config.delta_ranks):
        for idx, direction in enumerate(ref.simple_coroot_directions(config), start=1):
            expected = -ref.directional_derivative(config, ref.tau(config, r), direction)
            observed = sp.diff(b_components[idx - 1], delta_symbol)
            passed = sp.simplify(observed - expected) == 0
            b_checks["delta_coefficients_match_negative_tau_derivatives"][f"d{r}:B{idx}"] = passed
            _ok(passed, f"B_{idx} delta coefficient for tau_{r} has wrong sign")

    roots = ref.positive_roots(config)
    root_intervals = ref.positive_root_intervals(config)
    root_checks = {
        "positive_root_count": len(roots),
        "expected_positive_root_count": config.positive_root_count,
        "root_denominator_power": config.root_denominator_power,
        "positive_root_intervals": root_intervals,
    }
    _ok(len(roots) == config.positive_root_count, "wrong positive-root count")

    exponent_delta_zero = sp.simplify(ref.c_tilde_exponent(config).subs(deltas_zero))
    expected_exponent = -sum((idx + 1) * y[idx] for idx in range(config.y_count)) / config.rank
    c_tilde_checks = {
        "c_tilde_x_coordinates": [str(item) for item in ref.c_tilde_x_coordinates(config)],
        "c_tilde_direction_y": [str(item) for item in ref.c_tilde_direction_y(config)],
        "exponent_at_delta_zero": str(exponent_delta_zero),
        "expected_exponent_at_delta_zero": str(expected_exponent),
        "passed": sp.simplify(exponent_delta_zero - expected_exponent) == 0,
    }
    _ok(c_tilde_checks["passed"], "c_tilde exponent check failed")

    prefactor_checks = {
        "collapsed_prefactor": str(config.collapsed_prefactor),
        "expected_rank7_genus2_prefactor": "-7" if config.rank == 7 and config.genus == 2 else None,
    }
    if config.rank == 7 and config.genus == 2:
        _ok(config.collapsed_prefactor == -7, "rank 7 genus 2 collapsed prefactor is not -7")

    hessian_checks = {
        "q_at_delta_zero_equals_tau2": sp.simplify(
            ref.q_polynomial(config).subs(deltas_zero) - ref.tau(config, 2)
        ) == 0,
        "det_hq_over_det_htau2_at_delta_zero": str(ref.hessian_ratio_at_delta_zero(config)),
        "passed": ref.hessian_ratio_at_delta_zero(config) == 1,
    }
    _ok(hessian_checks["q_at_delta_zero_equals_tau2"], "q at delta zero does not equal tau_2")
    _ok(hessian_checks["passed"], "Hessian determinant ratio is not 1 at delta zero")

    exterior = ExteriorAlgebra(config)
    gamma_rr = exterior.gamma_as_mask_poly(2, 2)
    gamma_rr_terms = {
        " ".join(f"b{r}_{j}" for r, j in exterior.labels_from_mask(mask)): coeff
        for mask, coeff in gamma_rr.items()
    }
    expected_gamma_rr_terms = {
        "b2_1 b2_3": 2,
        "b2_2 b2_4": 2,
    } if config.genus == 2 else None
    gamma_checks = {
        "gamma22_terms": gamma_rr_terms,
        "gamma22_expected_terms": expected_gamma_rr_terms,
        "gamma_symmetry_23_32": exterior.gamma_as_mask_poly(2, 3) == exterior.gamma_as_mask_poly(3, 2),
    }
    _ok(gamma_checks["gamma_symmetry_23_32"], "gamma symmetry check failed")
    if expected_gamma_rr_terms is not None:
        _ok(gamma_rr_terms == expected_gamma_rr_terms, "gamma22 doubled-term check failed")

    config_payload: Dict[str, Any] = {
        "rank": config.rank,
        "genus": config.genus,
        "determinant_degree": config.determinant_degree,
        "top_degree": config.top_degree,
    }
    if problem is not None:
        config_payload["pairing_problem"] = {
            "source_degree": problem.source_degree,
            "test_degree": problem.test_degree,
            "expected_relation_chern_degree": problem.expected_relation_chern_degree,
        }

    return {
        "status": "passed",
        "config": config_payload,
        "coordinates": coordinate_checks,
        "tau": tau_checks,
        "B": b_checks,
        "roots": root_checks,
        "c_tilde": c_tilde_checks,
        "prefactor": prefactor_checks,
        "hessian": hessian_checks,
        "gamma": gamma_checks,
    }


def main() -> None:
    payload = run_structural_checks(problem=RANK7_H34_H62)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
