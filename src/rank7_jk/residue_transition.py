"""Generic modular residue transition for Gate C smoke tests."""

from __future__ import annotations

from functools import lru_cache
from math import comb, factorial
from typing import Dict, Sequence, Tuple

import sympy as sp

from .mod_arith import rational_mod, require_prime
from .root_system import TypeARootSystem, type_a_roots
from .sparse_poly import Alpha, SparsePoly, clean

DenominatorPowers = Tuple[int, ...]
ResidueStateKey = Tuple[Alpha, DenominatorPowers]


def residue_monomial_mod(
    rank: int,
    alpha: Sequence[int],
    derivative_orders: Sequence[int],
    *,
    prime: int,
    root_power: int = 2,
) -> int:
    y_count = int(rank) - 1
    alpha_t = _validate_alpha(alpha, y_count, "alpha")
    return residue_poly_mod(
        rank,
        {alpha_t: 1},
        derivative_orders,
        prime=prime,
        root_power=root_power,
    )


def residue_poly_mod(
    rank: int,
    poly: SparsePoly,
    derivative_orders: Sequence[int],
    *,
    prime: int,
    root_power: int = 2,
) -> int:
    """Evaluate an iterated JK residue of a sparse numerator modulo ``prime``."""

    p = require_prime(prime)
    roots = type_a_roots(rank)
    derivative_t = _validate_alpha(derivative_orders, roots.y_count, "derivative order")
    if root_power < 0:
        raise ValueError("root_power must be nonnegative")

    states = _initial_states(poly, roots, root_power, p)
    zero_alpha = tuple(0 for _ in range(roots.y_count))
    zero_denoms = tuple(0 for _ in range(roots.positive_root_count))

    for var_idx in reversed(range(roots.y_count)):
        states = _eliminate_variable_mod(states, roots, var_idx, derivative_t, p)
        if not states:
            return 0

    return states.get((zero_alpha, zero_denoms), 0) % p


def _validate_alpha(values: Sequence[int], expected_len: int, label: str) -> Alpha:
    if len(values) != expected_len:
        raise ValueError(f"{label} length must be {expected_len}")
    out = tuple(int(item) for item in values)
    if any(item < 0 for item in out):
        raise ValueError(f"{label} entries must be nonnegative")
    return out


def _initial_states(
    poly: SparsePoly,
    roots: TypeARootSystem,
    root_power: int,
    prime: int,
) -> Dict[ResidueStateKey, int]:
    cleaned = clean(poly, prime)
    root_powers = tuple(int(root_power) for _ in range(roots.positive_root_count))
    states: Dict[ResidueStateKey, int] = {}
    for alpha, coeff in cleaned.items():
        _validate_alpha(alpha, roots.y_count, "alpha")
        key = (alpha, root_powers)
        states[key] = (states.get(key, 0) + coeff) % prime
    return {key: value for key, value in states.items() if value}


def _eliminate_variable_mod(
    states: Dict[ResidueStateKey, int],
    roots: TypeARootSystem,
    var_idx: int,
    derivative_orders: Alpha,
    prime: int,
) -> Dict[ResidueStateKey, int]:
    next_terms: Dict[ResidueStateKey, int] = {}
    for (cur_alpha, denom_powers), coeff in states.items():
        local: Dict[Tuple[int, DenominatorPowers], int] = {
            (cur_alpha[var_idx], denom_powers): coeff
        }
        for pos, lower_pos in roots.transition_schedule[var_idx]:
            local = _expand_root_denominator_mod(
                local,
                roots,
                var_idx,
                derivative_orders,
                pos,
                lower_pos,
                prime,
            )
            if not local:
                break

        for (cur_y_exp, dtuple), state_coeff in local.items():
            special_exponent = -1 - cur_y_exp
            special = _special_coeff_mod(
                roots.rank,
                var_idx,
                derivative_orders[var_idx],
                special_exponent,
                prime,
            )
            if not special:
                continue
            next_alpha = list(cur_alpha)
            next_alpha[var_idx] = 0
            key = (tuple(next_alpha), dtuple)
            next_terms[key] = (next_terms.get(key, 0) + state_coeff * special) % prime
    return {key: value for key, value in next_terms.items() if value}


def _expand_root_denominator_mod(
    local: Dict[Tuple[int, DenominatorPowers], int],
    roots: TypeARootSystem,
    var_idx: int,
    derivative_orders: Alpha,
    pos: int,
    lower_pos: int,
    prime: int,
) -> Dict[Tuple[int, DenominatorPowers], int]:
    expanded: Dict[Tuple[int, DenominatorPowers], int] = {}
    for (cur_y_exp, dtuple), state_coeff in local.items():
        current_power = int(dtuple[pos])
        if not current_power:
            key = (cur_y_exp, dtuple)
            expanded[key] = (expanded.get(key, 0) + state_coeff) % prime
            continue

        base_den = list(dtuple)
        base_den[pos] = 0
        base_den_t = tuple(base_den)
        y_bound = _max_survivable_y_exp(
            roots,
            var_idx,
            derivative_orders,
            base_den_t,
            pos,
        )
        if lower_pos < 0:
            next_y_exp = cur_y_exp - current_power
            if next_y_exp <= y_bound:
                key = (next_y_exp, base_den_t)
                expanded[key] = (expanded.get(key, 0) + state_coeff) % prime
            continue

        max_m = y_bound - cur_y_exp
        if max_m < 0:
            continue
        for m in range(max_m + 1):
            expanded_den = list(base_den_t)
            expanded_den[lower_pos] += current_power + m
            binom = comb(current_power + m - 1, m)
            if m % 2:
                binom = -binom
            key = (cur_y_exp + m, tuple(expanded_den))
            expanded[key] = (expanded.get(key, 0) + state_coeff * binom) % prime
    return {key: value for key, value in expanded.items() if value}


def _max_survivable_y_exp(
    roots: TypeARootSystem,
    var_idx: int,
    derivative_orders: Alpha,
    denom_powers: DenominatorPowers,
    current_root_pos: int,
) -> int:
    simple_pos = roots.interval_index[(var_idx, var_idx + 1)]
    simple_drop = int(denom_powers[simple_pos]) if current_root_pos < simple_pos else 0
    return int(derivative_orders[var_idx]) + simple_drop


@lru_cache(maxsize=None)
def _special_coefficients_mod(
    rank: int,
    var_idx: int,
    derivative_order: int,
    cutoff: int,
    prime: int,
) -> Tuple[Tuple[int, int], ...]:
    min_exponent = -int(derivative_order) - 1
    out: dict[int, int] = {}
    for exponent in range(min_exponent, cutoff + 1):
        coeff = _special_coeff_exact_from_bernoulli(
            rank,
            var_idx,
            derivative_order,
            exponent,
        )
        if coeff:
            out[exponent] = rational_mod(coeff, prime)
    return tuple(sorted((exponent, coeff) for exponent, coeff in out.items() if coeff))


def _special_coeff_mod(
    rank: int,
    var_idx: int,
    derivative_order: int,
    exponent: int,
    prime: int,
) -> int:
    if exponent < -(derivative_order + 1):
        return 0
    cutoff = max(0, int(exponent))
    coeffs = dict(
        _special_coefficients_mod(rank, var_idx, derivative_order, cutoff, prime)
    )
    return coeffs.get(int(exponent), 0) % prime


@lru_cache(maxsize=None)
def _special_coeff_exact_from_bernoulli(
    rank: int,
    var_idx: int,
    derivative_order: int,
    exponent: int,
) -> sp.Rational:
    """Coefficient of z^exponent in exp(-lambda z) d^k/dz^k 1/(1-exp(-z))."""

    derivative_order = int(derivative_order)
    exponent = int(exponent)
    min_exponent = -derivative_order - 1
    if exponent < min_exponent:
        return sp.Rational(0)

    lam = sp.Rational(var_idx + 1, rank)
    total = sp.Rational(0)
    for exp_order in range(exponent - min_exponent + 1):
        base_exponent = exponent - exp_order + derivative_order
        base_coeff = _base_special_coeff(base_exponent)
        derivative_coeff = base_coeff * _falling_factorial(
            base_exponent,
            derivative_order,
        )
        exponential_coeff = (-lam) ** exp_order / factorial(exp_order)
        total += derivative_coeff * exponential_coeff
    return sp.Rational(total)


@lru_cache(maxsize=None)
def _base_special_coeff(exponent: int) -> sp.Rational:
    """Coefficient of z^exponent in 1/(1-exp(-z))."""

    exponent = int(exponent)
    if exponent < -1:
        return sp.Rational(0)
    n = exponent + 1
    return sp.Rational(sp.bernoulli(n, 1), factorial(n))


def _falling_factorial(value: int, order: int) -> int:
    out = 1
    for offset in range(int(order)):
        out *= int(value) - offset
    return out
