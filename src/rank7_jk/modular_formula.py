"""Rank-generic modular JK formula components.

This is the first layer above the raw residue-transition kernel.  It builds
the formula polynomials over a prime field for any rank supported by
``FormulaConfig``.  Full pairings are intentionally not here yet.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from typing import Sequence, Tuple

from .config import FormulaConfig
from .mod_arith import mod_inv, require_prime
from .sparse_poly import (
    Alpha,
    SparsePoly,
    add,
    constant,
    derivative,
    directional_derivative,
    monomial,
    mul,
)

SparseMatrix = Tuple[Tuple[SparsePoly, ...], ...]


def evaluate_sparse(poly: SparsePoly, point: Sequence[int], *, prime: int) -> int:
    p = require_prime(prime)
    point_t = tuple(int(item) % p for item in point)
    out = 0
    for alpha, coeff in poly.items():
        if len(alpha) != len(point_t):
            raise ValueError("point dimension does not match sparse polynomial")
        term = coeff % p
        for value, power in zip(point_t, alpha):
            if power:
                term = term * pow(value, power, p) % p
        out = (out + term) % p
    return out


@lru_cache(maxsize=None)
def x_polys_mod(config: FormulaConfig, prime: int) -> Tuple[Tuple[Tuple[Alpha, int], ...], ...]:
    p = require_prime(prime)
    n = config.rank
    inv_rank = mod_inv(n, p)
    polys = []
    for i in range(1, n + 1):
        poly: SparsePoly = {}
        for k in range(i, n):
            coeff = (n - k) * inv_rank % p
            poly = add(poly, monomial(_unit(k - 1, config.y_count), coeff, prime=p), prime=p)
        for k in range(1, i):
            coeff = -k * inv_rank % p
            poly = add(poly, monomial(_unit(k - 1, config.y_count), coeff, prime=p), prime=p)
        polys.append(tuple(sorted(poly.items())))
    return tuple(polys)


@lru_cache(maxsize=None)
def tau_mod(config: FormulaConfig, r: int, prime: int) -> Tuple[Tuple[Alpha, int], ...]:
    p = require_prime(prime)
    if r < 1 or r > config.rank:
        raise ValueError(f"tau index must be between 1 and {config.rank}, got {r}")
    acc: SparsePoly = {}
    x_polys = [dict(items) for items in x_polys_mod(config, p)]
    for combo in combinations(x_polys, r):
        term = constant(config.y_count, 1, prime=p)
        for poly in combo:
            term = mul(term, poly, prime=p)
        acc = add(acc, term, prime=p)
    return tuple(sorted(acc.items()))


@lru_cache(maxsize=None)
def tau_grad_mod(
    config: FormulaConfig,
    r: int,
    prime: int,
) -> Tuple[Tuple[Tuple[Alpha, int], ...], ...]:
    p = require_prime(prime)
    tau_poly = dict(tau_mod(config, r, p))
    return tuple(
        tuple(sorted(derivative(tau_poly, idx, prime=p).items()))
        for idx in range(config.y_count)
    )


@lru_cache(maxsize=None)
def tau_hessian_mod(
    config: FormulaConfig,
    r: int,
    prime: int,
) -> Tuple[Tuple[Tuple[Tuple[Alpha, int], ...], ...], ...]:
    p = require_prime(prime)
    tau_poly = dict(tau_mod(config, r, p))
    rows = []
    for i in range(config.y_count):
        first = derivative(tau_poly, i, prime=p)
        row = []
        for j in range(config.y_count):
            row.append(tuple(sorted(derivative(first, j, prime=p).items())))
        rows.append(tuple(row))
    return tuple(rows)


def simple_coroot_directions(config: FormulaConfig) -> Tuple[Alpha, ...]:
    directions = []
    for idx in range(config.y_count):
        row = [0 for _ in range(config.y_count)]
        row[idx] = 2
        if idx > 0:
            row[idx - 1] = -1
        if idx + 1 < config.y_count:
            row[idx + 1] = -1
        directions.append(tuple(row))
    return tuple(directions)


def b_perturbation_mod(
    config: FormulaConfig,
    r: int,
    j: int,
    prime: int,
) -> Tuple[Tuple[Alpha, int], ...]:
    p = require_prime(prime)
    if j < 1 or j > config.y_count:
        raise ValueError(f"B index must be 1,...,{config.y_count}")
    tau_poly = dict(tau_mod(config, r, p))
    deriv = directional_derivative(
        tau_poly,
        simple_coroot_directions(config)[j - 1],
        prime=p,
    )
    return tuple(sorted((alpha, (-coeff) % p) for alpha, coeff in deriv.items()))


def c_tilde_delta_coeff_mod(
    config: FormulaConfig,
    r: int,
    prime: int,
) -> Tuple[Tuple[Alpha, int], ...]:
    p = require_prime(prime)
    # For determinant degree 1, c_tilde has Y-direction (0,...,0,1).
    return tau_grad_mod(config, r, p)[-1]


@lru_cache(maxsize=None)
def hessian_tau2_inverse_mod(config: FormulaConfig, prime: int) -> Tuple[Tuple[int, ...], ...]:
    p = require_prime(prime)
    matrix = [
        [_constant_sparse(dict(tau_hessian_mod(config, 2, p)[i][j])) % p for j in range(config.y_count)]
        for i in range(config.y_count)
    ]
    return tuple(tuple(row) for row in _matrix_inverse_mod(matrix, p))


def hat_pair_delta_zero_mod(
    config: FormulaConfig,
    r: int,
    s: int,
    prime: int,
) -> Tuple[Tuple[Alpha, int], ...]:
    p = require_prime(prime)
    h0_inv = hessian_tau2_inverse_mod(config, p)
    return _gradient_matrix_gradient(config, r, s, h0_inv, p, scale=-1)


def hat_pair_first_delta_mod(
    config: FormulaConfig,
    r: int,
    s: int,
    delta_rank: int,
    prime: int,
) -> Tuple[Tuple[Alpha, int], ...]:
    p = require_prime(prime)
    h0_inv = hessian_tau2_inverse_mod(config, p)
    h_delta = tau_hessian_mod(config, delta_rank, p)
    middle = _const_sparse_matrix_mul(h0_inv, h_delta, p)
    middle = _sparse_matrix_const_mul(middle, h0_inv, p)
    return _gradient_sparse_matrix_gradient(config, r, s, middle, p)


def _gradient_matrix_gradient(
    config: FormulaConfig,
    r: int,
    s: int,
    matrix: Sequence[Sequence[int]],
    prime: int,
    *,
    scale: int,
) -> Tuple[Tuple[Alpha, int], ...]:
    p = require_prime(prime)
    gr = tau_grad_mod(config, r, p)
    gs = tau_grad_mod(config, s, p)
    acc: SparsePoly = {}
    for i in range(config.y_count):
        for j in range(config.y_count):
            coeff = matrix[i][j] * scale % p
            if coeff:
                product = mul(dict(gr[i]), dict(gs[j]), prime=p)
                acc = add(acc, product, prime=p, scale=coeff)
    return tuple(sorted(acc.items()))


def _gradient_sparse_matrix_gradient(
    config: FormulaConfig,
    r: int,
    s: int,
    matrix: SparseMatrix,
    prime: int,
) -> Tuple[Tuple[Alpha, int], ...]:
    p = require_prime(prime)
    gr = tau_grad_mod(config, r, p)
    gs = tau_grad_mod(config, s, p)
    acc: SparsePoly = {}
    for i in range(config.y_count):
        for j in range(config.y_count):
            coeff_poly = mul(dict(gr[i]), dict(gs[j]), prime=p)
            product = mul(coeff_poly, matrix[i][j], prime=p)
            acc = add(acc, product, prime=p)
    return tuple(sorted(acc.items()))


def _const_sparse_matrix_mul(
    left: Sequence[Sequence[int]],
    right: Sequence[Sequence[Tuple[Tuple[Alpha, int], ...]]],
    prime: int,
) -> SparseMatrix:
    p = require_prime(prime)
    size = len(left)
    rows = []
    for i in range(size):
        row = []
        for j in range(size):
            acc: SparsePoly = {}
            for k in range(size):
                if left[i][k] % p:
                    acc = add(acc, dict(right[k][j]), prime=p, scale=left[i][k])
            row.append(acc)
        rows.append(tuple(row))
    return tuple(rows)


def _sparse_matrix_const_mul(
    left: SparseMatrix,
    right: Sequence[Sequence[int]],
    prime: int,
) -> SparseMatrix:
    p = require_prime(prime)
    size = len(right)
    rows = []
    for i in range(size):
        row = []
        for j in range(size):
            acc: SparsePoly = {}
            for k in range(size):
                if right[k][j] % p:
                    acc = add(acc, left[i][k], prime=p, scale=right[k][j])
            row.append(acc)
        rows.append(tuple(row))
    return tuple(rows)


def _matrix_inverse_mod(matrix: Sequence[Sequence[int]], prime: int) -> list[list[int]]:
    p = require_prime(prime)
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("matrix must be square")
    aug = [
        [int(value) % p for value in row] + [1 if idx == col else 0 for col in range(size)]
        for idx, row in enumerate(matrix)
    ]
    for col in range(size):
        pivot = None
        for row in range(col, size):
            if aug[row][col] % p:
                pivot = row
                break
        if pivot is None:
            raise ZeroDivisionError("matrix is singular modulo prime")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        inv = mod_inv(aug[col][col], p)
        aug[col] = [value * inv % p for value in aug[col]]
        for row in range(size):
            if row == col:
                continue
            coeff = aug[row][col] % p
            if coeff:
                aug[row] = [
                    (aug[row][idx] - coeff * aug[col][idx]) % p
                    for idx in range(2 * size)
                ]
    return [row[size:] for row in aug]


def _constant_sparse(poly: SparsePoly) -> int:
    out = 0
    for alpha, coeff in poly.items():
        if any(alpha):
            raise ValueError("expected a constant sparse polynomial")
        out += coeff
    return out


def _unit(idx: int, size: int) -> Alpha:
    row = [0 for _ in range(size)]
    row[idx] = 1
    return tuple(row)
