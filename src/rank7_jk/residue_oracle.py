"""Residue witnesses for small JK checks.

This module intentionally keeps two paths separate:

* ``literal_residue_sympy`` expands the meromorphic expression directly with
  SymPy Laurent series.  It is slow but genuinely independent, and should only
  be used for tiny ranks.
* ``exact_transition_residue`` is a rational transition oracle.  It mirrors the
  residue-elimination mathematics used by the modular rank-5 reference, but it
  works over exact fractions and is useful for larger spot checks.
"""

from __future__ import annotations

from functools import lru_cache
from fractions import Fraction
from math import comb
from typing import Dict, Sequence, Tuple

import sympy as sp


def y_symbols(rank: int) -> Tuple[sp.Symbol, ...]:
    if rank < 2:
        raise ValueError("rank must be at least 2")
    return sp.symbols(f"Y1:{rank}")


def special_factor(
    var: sp.Symbol,
    *,
    rank: int,
    var_index: int,
    derivative_order: int = 0,
) -> sp.Expr:
    """Return exp(-i Y_i/rank) times the derivative of 1/(1-exp(-Y_i))."""

    if derivative_order < 0:
        raise ValueError("derivative_order must be nonnegative")
    if var_index < 0 or var_index >= rank - 1:
        raise ValueError("var_index is outside the simple-root coordinate range")
    base = 1 / (1 - sp.exp(-var))
    if derivative_order:
        base = sp.diff(base, var, derivative_order)
    return sp.exp(sp.Rational(-(var_index + 1), rank) * var) * base


def root_denominator(rank: int, root_power: int = 2) -> sp.Expr:
    if root_power < 0:
        raise ValueError("root_power must be nonnegative")
    y = y_symbols(rank)
    out = sp.Integer(1)
    for start in range(rank - 1):
        running = sp.Integer(0)
        for end in range(start, rank - 1):
            running += y[end]
            out *= running**root_power
    return out


def root_intervals(rank: int) -> Tuple[Tuple[int, int], ...]:
    if rank < 2:
        raise ValueError("rank must be at least 2")
    return tuple((i, j) for i in range(rank - 1) for j in range(i + 1, rank))


def residue_monomial_expr(
    rank: int,
    alpha: Sequence[int],
    derivative_orders: Sequence[int],
    *,
    root_power: int = 2,
) -> sp.Expr:
    y = y_symbols(rank)
    if len(alpha) != len(y):
        raise ValueError("alpha length must be rank - 1")
    if len(derivative_orders) != len(y):
        raise ValueError("derivative order length must be rank - 1")

    numerator = sp.Integer(1)
    for var, power in zip(y, alpha):
        if power < 0:
            raise ValueError("alpha entries must be nonnegative")
        numerator *= var ** int(power)
    for idx, (var, order) in enumerate(zip(y, derivative_orders)):
        numerator *= special_factor(
            var,
            rank=rank,
            var_index=idx,
            derivative_order=int(order),
        )
    return numerator / root_denominator(rank, root_power=root_power)


def one_variable_residue(expr: sp.Expr, var: sp.Symbol, series_order: int) -> sp.Expr:
    if series_order < 1:
        raise ValueError("series_order must be positive")
    series = sp.series(expr, var, 0, series_order).removeO()
    return sp.expand(series.coeff(var, -1))


def iterated_residue(
    expr: sp.Expr,
    variables: Sequence[sp.Symbol],
    *,
    series_order: int = 24,
) -> sp.Expr:
    out = expr
    for var in variables:
        out = one_variable_residue(out, var, series_order)
    return sp.simplify(out)


def literal_residue_sympy(
    rank: int,
    alpha: Sequence[int],
    derivative_orders: Sequence[int],
    *,
    root_power: int = 2,
    series_order: int = 24,
) -> sp.Expr:
    """Compute a literal SymPy Laurent-series JK residue in reverse Y order."""

    expr = residue_monomial_expr(
        rank,
        alpha,
        derivative_orders,
        root_power=root_power,
    )
    return iterated_residue(
        expr,
        tuple(reversed(y_symbols(rank))),
        series_order=series_order,
    )


@lru_cache(maxsize=None)
def _root_transition_schedule(rank: int) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    index = {interval: idx for idx, interval in enumerate(root_intervals(rank))}
    by_var = [[] for _ in range(rank - 1)]
    for var_idx in range(rank - 1):
        for interval, pos in index.items():
            if interval[1] != var_idx + 1:
                continue
            lower_pos = -1 if interval[0] == var_idx else index[(interval[0], var_idx)]
            by_var[var_idx].append((pos, lower_pos))
    return tuple(tuple(items) for items in by_var)


def _max_survivable_y_exp(
    rank: int,
    var_idx: int,
    derivative_orders: Tuple[int, ...],
    denom_powers: Tuple[int, ...],
    current_root_pos: int,
) -> int:
    index = {interval: idx for idx, interval in enumerate(root_intervals(rank))}
    simple_pos = index[(var_idx, var_idx + 1)]
    simple_drop = int(denom_powers[simple_pos]) if current_root_pos < simple_pos else 0
    return int(derivative_orders[var_idx]) + simple_drop


def _sympy_rational_to_fraction(value: sp.Expr) -> Fraction:
    rational = sp.Rational(value)
    return Fraction(int(rational.p), int(rational.q))


@lru_cache(maxsize=None)
def _special_coefficients(
    rank: int,
    var_idx: int,
    derivative_order: int,
    cutoff: int,
) -> Dict[int, Fraction]:
    z = sp.Symbol("z")
    expr = special_factor(
        z,
        rank=rank,
        var_index=var_idx,
        derivative_order=derivative_order,
    )
    series = sp.expand(sp.series(expr, z, 0, cutoff + 1).removeO())
    out: Dict[int, Fraction] = {}
    for term in sp.Add.make_args(series):
        coeff, exponent = term.as_coeff_exponent(z)
        if exponent.is_Integer:
            out[int(exponent)] = out.get(int(exponent), Fraction(0)) + (
                _sympy_rational_to_fraction(coeff)
            )
    return {exponent: coeff for exponent, coeff in out.items() if coeff}


def _special_coeff(
    rank: int,
    var_idx: int,
    derivative_order: int,
    exponent: int,
) -> Fraction:
    if exponent < -(derivative_order + 1):
        return Fraction(0)
    cutoff = max(0, exponent)
    return _special_coefficients(rank, var_idx, derivative_order, cutoff).get(
        exponent,
        Fraction(0),
    )


def special_coefficient_exact(
    rank: int,
    var_idx: int,
    derivative_order: int,
    y_exponent: int,
) -> sp.Rational:
    """Return the exact local residue coefficient for a current Y exponent.

    During one variable elimination, a current monomial ``Y_i^k`` needs the
    coefficient of ``Y_i^(-1-k)`` in the special factor.
    """

    coeff = _special_coeff(rank, var_idx, derivative_order, -1 - y_exponent)
    return sp.Rational(coeff.numerator, coeff.denominator)


def exact_transition_residue(
    rank: int,
    alpha: Sequence[int],
    derivative_orders: Sequence[int],
    *,
    root_power: int = 2,
) -> sp.Expr:
    y_count = rank - 1
    if len(alpha) != y_count:
        raise ValueError("alpha length must be rank - 1")
    if len(derivative_orders) != y_count:
        raise ValueError("derivative order length must be rank - 1")
    if root_power < 0:
        raise ValueError("root_power must be nonnegative")

    alpha_t = tuple(int(item) for item in alpha)
    derivative_t = tuple(int(item) for item in derivative_orders)
    if any(item < 0 for item in alpha_t) or any(item < 0 for item in derivative_t):
        raise ValueError("alpha and derivative orders must be nonnegative")

    root_powers = tuple(root_power for _ in root_intervals(rank))
    zero_alpha = tuple(0 for _ in range(y_count))
    zero_denoms = tuple(0 for _ in root_intervals(rank))
    states: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Fraction] = {
        (alpha_t, root_powers): Fraction(1)
    }

    for var_idx in reversed(range(y_count)):
        next_terms: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Fraction] = {}
        for (cur_alpha, denom_powers), coeff in states.items():
            local: Dict[Tuple[int, Tuple[int, ...]], Fraction] = {
                (cur_alpha[var_idx], denom_powers): coeff
            }
            for pos, lower_pos in _root_transition_schedule(rank)[var_idx]:
                expanded: Dict[Tuple[int, Tuple[int, ...]], Fraction] = {}
                for (cur_y_exp, dtuple), state_coeff in local.items():
                    current_power = int(dtuple[pos])
                    if not current_power:
                        key = (cur_y_exp, dtuple)
                        expanded[key] = expanded.get(key, Fraction(0)) + state_coeff
                        continue
                    base_den = list(dtuple)
                    base_den[pos] = 0
                    base_den_t = tuple(base_den)
                    y_bound = _max_survivable_y_exp(
                        rank,
                        var_idx,
                        derivative_t,
                        base_den_t,
                        pos,
                    )
                    if lower_pos < 0:
                        next_y_exp = cur_y_exp - current_power
                        if next_y_exp > y_bound:
                            continue
                        key = (next_y_exp, base_den_t)
                        expanded[key] = expanded.get(key, Fraction(0)) + state_coeff
                        continue

                    max_m = y_bound - cur_y_exp
                    if max_m < 0:
                        continue
                    for m in range(max_m + 1):
                        expanded_den = list(base_den_t)
                        expanded_den[lower_pos] += current_power + m
                        binom = Fraction(((-1) ** m) * comb(current_power + m - 1, m))
                        key = (cur_y_exp + m, tuple(expanded_den))
                        expanded[key] = expanded.get(key, Fraction(0)) + state_coeff * binom
                local = {key: value for key, value in expanded.items() if value}
                if not local:
                    break

            for (cur_y_exp, dtuple), state_coeff in local.items():
                special_exponent = -1 - cur_y_exp
                special = _special_coeff(
                    rank,
                    var_idx,
                    derivative_t[var_idx],
                    special_exponent,
                )
                if not special:
                    continue
                next_alpha = list(cur_alpha)
                next_alpha[var_idx] = 0
                key = (tuple(next_alpha), dtuple)
                next_terms[key] = next_terms.get(key, Fraction(0)) + state_coeff * special
        states = {key: value for key, value in next_terms.items() if value}
        if not states:
            return sp.Integer(0)

    result = states.get((zero_alpha, zero_denoms), Fraction(0))
    return sp.Rational(result.numerator, result.denominator)


def jk_residue_exact(
    rank: int,
    alpha: Sequence[int],
    derivative_orders: Sequence[int],
    *,
    root_power: int = 2,
) -> sp.Expr:
    """Compatibility wrapper for the exact transition residue."""

    return exact_transition_residue(
        rank,
        alpha,
        derivative_orders,
        root_power=root_power,
    )


def rational_mod(value: sp.Expr, prime: int) -> int:
    if prime <= 1 or not sp.isprime(prime):
        raise ValueError("prime must be prime")
    rational = sp.Rational(value)
    denominator = int(rational.q) % prime
    if denominator == 0:
        raise ZeroDivisionError("rational denominator is zero modulo prime")
    return int(rational.p) % prime * pow(denominator, prime - 2, prime) % prime


def jk_residue_mod(
    rank: int,
    alpha: Sequence[int],
    derivative_orders: Sequence[int],
    *,
    prime: int,
    root_power: int = 2,
) -> int:
    exact = exact_transition_residue(
        rank,
        alpha,
        derivative_orders,
        root_power=root_power,
    )
    return rational_mod(exact, prime)
