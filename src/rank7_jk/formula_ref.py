"""Readable symbolic Jeffrey-Kirwan formula components.

This module is intentionally a reference layer, not the production fast path.
It should stay close to the paper formula and to the written verification
plan.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from typing import Sequence, Tuple

import sympy as sp

from .config import JKConfig


@lru_cache(maxsize=None)
def y_symbols(config: JKConfig) -> Tuple[sp.Symbol, ...]:
    return sp.symbols(f"Y1:{config.rank}")


@lru_cache(maxsize=None)
def delta_symbols(config: JKConfig) -> Tuple[sp.Symbol, ...]:
    if config.rank < 3:
        return ()
    return sp.symbols(f"d3:{config.rank + 1}")


@lru_cache(maxsize=None)
def x_coordinates(config: JKConfig) -> Tuple[sp.Expr, ...]:
    """Return x_1,...,x_n in simple-root coordinates."""

    y = y_symbols(config)
    n = config.rank
    out = []
    for i in range(1, n + 1):
        numerator = sp.Integer(0)
        for k in range(i, n):
            numerator += (n - k) * y[k - 1]
        for k in range(1, i):
            numerator -= k * y[k - 1]
        out.append(sp.expand(numerator / n))
    return tuple(out)


@lru_cache(maxsize=None)
def tau(config: JKConfig, r: int) -> sp.Expr:
    if r < 1 or r > config.rank:
        raise ValueError(f"tau index must be between 1 and {config.rank}, got {r}")
    acc = sp.Integer(0)
    for combo in combinations(x_coordinates(config), r):
        term = sp.Integer(1)
        for item in combo:
            term *= item
        acc += term
    return sp.expand(acc)


@lru_cache(maxsize=None)
def tau_grad_y(config: JKConfig, r: int) -> Tuple[sp.Expr, ...]:
    tr = tau(config, r)
    return tuple(sp.expand(sp.diff(tr, var)) for var in y_symbols(config))


@lru_cache(maxsize=None)
def q_polynomial(config: JKConfig) -> sp.Expr:
    q = tau(config, 2)
    for symbol, r in zip(delta_symbols(config), config.delta_ranks):
        q += symbol * tau(config, r)
    return sp.expand(q)


def simple_coroot_directions(config: JKConfig) -> Tuple[Tuple[int, ...], ...]:
    """Simple coroot directions represented as dY coordinate directions."""

    directions = []
    for j in range(config.y_count):
        row = [0 for _ in range(config.y_count)]
        row[j] = 2
        if j > 0:
            row[j - 1] = -1
        if j + 1 < config.y_count:
            row[j + 1] = -1
        directions.append(tuple(row))
    return tuple(directions)


def directional_derivative(
    config: JKConfig,
    expr: sp.Expr,
    direction_y: Sequence[int | sp.Expr],
) -> sp.Expr:
    y = y_symbols(config)
    if len(direction_y) != len(y):
        raise ValueError("direction length does not match the number of Y variables")
    return sp.expand(sum(direction_y[idx] * sp.diff(expr, y[idx]) for idx in range(len(y))))


def B_map_components(config: JKConfig, q: sp.Expr | None = None) -> Tuple[sp.Expr, ...]:
    q = q_polynomial(config) if q is None else q
    return tuple(
        sp.expand(-directional_derivative(config, q, direction))
        for direction in simple_coroot_directions(config)
    )


def b_map_components(config: JKConfig, q: sp.Expr | None = None) -> Tuple[sp.Expr, ...]:
    """Compatibility alias for the paper's B_j map components."""

    return B_map_components(config, q)


def x_direction_to_y_direction(
    config: JKConfig,
    direction_x: Sequence[int | sp.Expr],
) -> Tuple[sp.Expr, ...]:
    if len(direction_x) != config.rank:
        raise ValueError("x direction length must equal the rank")
    return tuple(
        sp.simplify(direction_x[idx] - direction_x[idx + 1])
        for idx in range(config.y_count)
    )


def c_tilde_x_coordinates(config: JKConfig) -> Tuple[sp.Rational, ...]:
    n = config.rank
    return tuple([sp.Rational(1, n) for _ in range(n - 1)] + [sp.Rational(-(n - 1), n)])


def c_tilde_direction_y(config: JKConfig) -> Tuple[sp.Expr, ...]:
    return x_direction_to_y_direction(config, c_tilde_x_coordinates(config))


def c_tilde_exponent(config: JKConfig, q: sp.Expr | None = None) -> sp.Expr:
    q = q_polynomial(config) if q is None else q
    return directional_derivative(config, q, c_tilde_direction_y(config))


def positive_root_intervals(config: JKConfig) -> Tuple[Tuple[int, int], ...]:
    """Return positive roots x_i-x_j as 1-based intervals (i,j)."""

    intervals = []
    for start in range(1, config.rank):
        for end in range(start + 1, config.rank + 1):
            intervals.append((start, end))
    return tuple(intervals)


def positive_roots(config: JKConfig) -> Tuple[sp.Expr, ...]:
    y = y_symbols(config)
    roots = []
    for start, end in positive_root_intervals(config):
        running = sp.Integer(0)
        for idx in range(start - 1, end - 1):
            running += y[idx]
        roots.append(sp.expand(running))
    return tuple(roots)


def denominator_root_product(config: JKConfig) -> sp.Expr:
    prod = sp.Integer(1)
    for root in positive_roots(config):
        prod *= root ** config.root_denominator_power
    return sp.expand(prod)


def hessian_y_basis(config: JKConfig, expr: sp.Expr) -> sp.Matrix:
    y = y_symbols(config)
    return sp.Matrix([[sp.diff(expr, left, right) for right in y] for left in y])


def hessian_ratio_at_delta_zero(config: JKConfig) -> sp.Expr:
    zero = delta_zero_subs(config)
    hq0 = hessian_y_basis(config, q_polynomial(config)).applyfunc(
        lambda item: sp.simplify(item.subs(zero))
    )
    h0 = hessian_y_basis(config, tau(config, 2))
    return sp.simplify(hq0.det() / h0.det())


def determinant_ratio(config: JKConfig) -> sp.Expr:
    """Full symbolic Hessian determinant ratio.

    This may be expensive for rank 7 and is intended for small checks, not for
    production pairing evaluation.
    """

    hq = hessian_y_basis(config, q_polynomial(config))
    h0 = hessian_y_basis(config, tau(config, 2))
    return sp.factor(hq.det() / h0.det())


def hat_pair_coefficient(config: JKConfig, r: int, s: int) -> sp.Expr:
    q = q_polynomial(config)
    h_inv = hessian_y_basis(config, q).inv()
    gr = sp.Matrix(tau_grad_y(config, r))
    gs = sp.Matrix(tau_grad_y(config, s))
    return sp.factor(-(gr.T * h_inv * gs)[0])


def hat_pair_coefficient_at_delta_zero(config: JKConfig, r: int, s: int) -> sp.Expr:
    h_inv = hessian_y_basis(config, tau(config, 2)).inv()
    gr = sp.Matrix(tau_grad_y(config, r))
    gs = sp.Matrix(tau_grad_y(config, s))
    return sp.expand(-(gr.T * h_inv * gs)[0])


def hat_pair_first_delta_coefficient_at_zero(
    config: JKConfig,
    r: int,
    s: int,
    delta_rank: int,
) -> sp.Expr:
    """Coefficient of one delta in -grad(tau_r)^T H_q^{-1} grad(tau_s)."""

    if delta_rank not in config.delta_ranks:
        raise ValueError(f"delta_rank must be one of {config.delta_ranks}")
    h0_inv = hessian_y_basis(config, tau(config, 2)).inv()
    h_delta = hessian_y_basis(config, tau(config, delta_rank))
    gr = sp.Matrix(tau_grad_y(config, r))
    gs = sp.Matrix(tau_grad_y(config, s))
    return sp.expand((gr.T * h0_inv * h_delta * h0_inv * gs)[0])


def delta_zero_subs(config: JKConfig) -> dict[sp.Symbol, int]:
    return {symbol: 0 for symbol in delta_symbols(config)}


def delta_coefficient_at_zero(
    config: JKConfig,
    expr: sp.Expr,
    orders: Sequence[int],
) -> sp.Expr:
    """Extract the delta monomial coefficient using derivatives at zero."""

    deltas = delta_symbols(config)
    if len(orders) != len(deltas):
        raise ValueError(f"expected {len(deltas)} delta orders, got {len(orders)}")
    out = expr
    scale = sp.Integer(1)
    for symbol, order in zip(deltas, orders):
        if order < 0:
            raise ValueError("delta orders must be nonnegative")
        if order:
            out = sp.diff(out, symbol, order)
        scale *= sp.factorial(order)
    return sp.simplify(out.subs(delta_zero_subs(config)) / scale)


def f_factorial_scale(f_exponents: Sequence[int]) -> sp.Integer:
    """Return the product of factorials for f_2,...,f_n exponents."""

    scale = sp.Integer(1)
    for exp in f_exponents:
        if exp < 0:
            raise ValueError("f exponents must be nonnegative")
        scale *= sp.factorial(exp)
    return scale


def one_variable_residue(expr: sp.Expr, var: sp.Symbol, series_order: int = 8) -> sp.Expr:
    if series_order < 1:
        raise ValueError("series_order must be positive")
    expanded = sp.series(expr, var, 0, series_order).removeO()
    return sp.simplify(expanded.coeff(var, -1))


def iterated_residue(
    expr: sp.Expr,
    variables: Sequence[sp.Symbol],
    series_order: int = 8,
) -> sp.Expr:
    out = expr
    for var in variables:
        out = one_variable_residue(out, var, series_order=series_order)
    return sp.simplify(out)


def residue_elimination_order(config: JKConfig) -> Tuple[sp.Symbol, ...]:
    return tuple(reversed(y_symbols(config)))
