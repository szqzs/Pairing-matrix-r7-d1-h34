"""Semi-symbolic reference JK evaluator for Gate B.

The implementation is intentionally scoped to rank 5, genus 2.  It follows the
rank-5 specialization of the JK formula using modular sparse polynomials,
truncated delta series, exterior expansion of the gamma classes, and an
explicit residue transition.  This is not the final rank-7 fast path; it is the
external regression oracle that must reproduce the public rank-5 values.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations, permutations
from math import comb, factorial
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import FormulaConfig
from .exterior import ExteriorAlgebra
from .invariants import InvariantMonomial

Alpha = Tuple[int, int, int, int]
DeltaKey = Tuple[int, int, int]
DerivOrders = Tuple[int, int, int, int]
SparsePoly = Dict[Alpha, int]
DeltaPoly = Dict[DeltaKey, SparsePoly]
KernelTerms = Dict[Tuple[DeltaKey, DerivOrders], SparsePoly]

RANK5_CONFIG = FormulaConfig(rank=5, genus=2)
RANK5_ROOT_INTERVALS = tuple((i, j) for i in range(4) for j in range(i + 1, 5))
RANK5_ROOT_INDEX = {interval: idx for idx, interval in enumerate(RANK5_ROOT_INTERVALS)}
RANK5_ROOT_POWERS = tuple(2 for _ in RANK5_ROOT_INTERVALS)
RANK5_ZERO_DENOM = tuple(0 for _ in RANK5_ROOT_INTERVALS)
RANK5_BASE_LAMBDA_NUMS = (-1, -2, -3, -4)

ZERO_ALPHA: Alpha = (0, 0, 0, 0)
ZERO_DELTA: DeltaKey = (0, 0, 0)
ZERO_DERIV: DerivOrders = (0, 0, 0, 0)
DELTA_UNITS: Tuple[DeltaKey, ...] = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
SIMPLE_COROOT_DIRECTIONS: Tuple[Alpha, ...] = (
    (2, -1, 0, 0),
    (-1, 2, -1, 0),
    (0, -1, 2, -1),
    (0, 0, -1, 2),
)
B_LABELS = RANK5_CONFIG.b_labels
B_INDEX = {label: idx for idx, label in enumerate(B_LABELS)}


def pairing_mod_prime(
    config: FormulaConfig,
    left: InvariantMonomial,
    right: InvariantMonomial,
    *,
    prime: int | None = None,
) -> int:
    """Evaluate the rank-5 genus-2 JK pairing modulo ``prime``."""

    p = config.primary_prime if prime is None else int(prime)
    _validate_rank5_inputs(config, left, right, p)
    total = left * right
    return _pairing_total_mod(total, p)


def pairing_matrix_mod_prime(
    config: FormulaConfig,
    rows: Sequence[InvariantMonomial],
    columns: Sequence[InvariantMonomial],
    *,
    prime: int | None = None,
) -> Tuple[Tuple[int, ...], ...]:
    p = config.primary_prime if prime is None else int(prime)
    return tuple(
        tuple(pairing_mod_prime(config, row, column, prime=p) for column in columns)
        for row in rows
    )


def determinant_mod(matrix: Sequence[Sequence[int]], prime: int) -> int:
    """Determinant by modular Gaussian elimination."""

    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("determinant_mod expects a square matrix")
    mat = [[int(value) % prime for value in row] for row in matrix]
    det = 1
    for col in range(size):
        pivot = None
        for row in range(col, size):
            if mat[row][col] % prime:
                pivot = row
                break
        if pivot is None:
            return 0
        if pivot != col:
            mat[col], mat[pivot] = mat[pivot], mat[col]
            det = -det % prime
        pivot_value = mat[col][col] % prime
        det = det * pivot_value % prime
        inv = mod_inv(pivot_value, prime)
        for row in range(col + 1, size):
            factor = mat[row][col] * inv % prime
            if factor:
                mat[row] = [
                    (mat[row][idx] - factor * mat[col][idx]) % prime
                    for idx in range(size)
                ]
    return det % prime


def _validate_rank5_inputs(
    config: FormulaConfig,
    left: InvariantMonomial,
    right: InvariantMonomial,
    prime: int,
) -> None:
    if config != RANK5_CONFIG:
        raise NotImplementedError("Gate B evaluator currently supports rank 5 genus 2 only")
    if prime <= 1:
        raise ValueError("prime must be greater than 1")
    if left.rank != config.rank or right.rank != config.rank:
        raise ValueError("pairing inputs must have the same rank as the formula config")
    if left.ordinary_degree + right.ordinary_degree != config.top_degree:
        raise ValueError(
            "pairing inputs must have ordinary degrees summing to the top degree"
        )


def mod_inv(value: int, p: int) -> int:
    value %= p
    if value == 0:
        raise ZeroDivisionError(f"denominator is 0 modulo prime {p}")
    return pow(value, p - 2, p)


@lru_cache(maxsize=None)
def mod_factorial(n: int, p: int) -> int:
    out = 1
    for k in range(2, n + 1):
        out = out * k % p
    return out


@lru_cache(maxsize=None)
def _f_factorial_scale_mod(f_exp: Tuple[int, int, int, int], p: int) -> int:
    out = 1
    for exp in f_exp:
        out = out * mod_factorial(int(exp), p) % p
    return out


def _sparse_clean(poly: SparsePoly, p: int) -> SparsePoly:
    return {alpha: coeff % p for alpha, coeff in poly.items() if coeff % p}


def _sparse_add(left: SparsePoly, right: SparsePoly, p: int, scale: int = 1) -> SparsePoly:
    out = dict(left)
    scale %= p
    if not scale:
        return _sparse_clean(out, p)
    for alpha, coeff in right.items():
        value = (out.get(alpha, 0) + scale * coeff) % p
        if value:
            out[alpha] = value
        else:
            out.pop(alpha, None)
    return out


def _sparse_scale(poly: SparsePoly, scale: int, p: int) -> SparsePoly:
    scale %= p
    if not scale:
        return {}
    return {alpha: coeff * scale % p for alpha, coeff in poly.items() if coeff * scale % p}


def _sparse_mul(left: SparsePoly, right: SparsePoly, p: int) -> SparsePoly:
    if not left or not right:
        return {}
    out: SparsePoly = {}
    for a1, c1 in left.items():
        for a2, c2 in right.items():
            alpha = tuple(a1[idx] + a2[idx] for idx in range(4))
            out[alpha] = (out.get(alpha, 0) + c1 * c2) % p
    return _sparse_clean(out, p)


def _sparse_pow(base: SparsePoly, exp: int, p: int) -> SparsePoly:
    out: SparsePoly = {ZERO_ALPHA: 1}
    cur = base
    n = int(exp)
    while n:
        if n & 1:
            out = _sparse_mul(out, cur, p)
        n >>= 1
        if n:
            cur = _sparse_mul(cur, cur, p)
    return out


def _sparse_derivative(poly: SparsePoly, var_idx: int, p: int) -> SparsePoly:
    out: SparsePoly = {}
    for alpha, coeff in poly.items():
        power = alpha[var_idx]
        if not power:
            continue
        next_alpha = list(alpha)
        next_alpha[var_idx] -= 1
        key = tuple(next_alpha)
        out[key] = (out.get(key, 0) + coeff * power) % p
    return _sparse_clean(out, p)


def _sparse_directional_derivative(
    poly: SparsePoly,
    direction: Sequence[int],
    p: int,
) -> SparsePoly:
    out: SparsePoly = {}
    for idx, coeff in enumerate(direction):
        if coeff:
            out = _sparse_add(out, _sparse_derivative(poly, idx, p), p, scale=coeff)
    return out


def _linear_poly(coeffs: Sequence[int], p: int) -> SparsePoly:
    inv5 = mod_inv(5, p)
    out: SparsePoly = {}
    for idx, coeff in enumerate(coeffs):
        alpha = [0, 0, 0, 0]
        alpha[idx] = 1
        out[tuple(alpha)] = coeff * inv5 % p
    return _sparse_clean(out, p)


@lru_cache(maxsize=None)
def _rank5_x_polys_mod(p: int) -> Tuple[Tuple[Tuple[Alpha, int], ...], ...]:
    xs = (
        (4, 3, 2, 1),
        (-1, 3, 2, 1),
        (-1, -2, 2, 1),
        (-1, -2, -3, 1),
        (-1, -2, -3, -4),
    )
    return tuple(tuple(sorted(_linear_poly(coeffs, p).items())) for coeffs in xs)


@lru_cache(maxsize=None)
def _rank5_tau_mod(r: int, p: int) -> Tuple[Tuple[Alpha, int], ...]:
    if r < 2 or r > 5:
        raise ValueError(f"rank-5 tau index must be 2,...,5, got {r}")
    x_polys = [dict(items) for items in _rank5_x_polys_mod(p)]
    acc: SparsePoly = {}
    for combo in combinations(x_polys, r):
        term: SparsePoly = {ZERO_ALPHA: 1}
        for poly in combo:
            term = _sparse_mul(term, poly, p)
        acc = _sparse_add(acc, term, p)
    return tuple(sorted(acc.items()))


@lru_cache(maxsize=None)
def _tau_power_mod(a_exp: Tuple[int, int, int, int], p: int) -> Tuple[Tuple[Alpha, int], ...]:
    out: SparsePoly = {ZERO_ALPHA: 1}
    for offset, exp in enumerate(a_exp):
        if exp:
            out = _sparse_mul(out, _sparse_pow(dict(_rank5_tau_mod(offset + 2, p)), exp, p), p)
    return tuple(sorted(out.items()))


@lru_cache(maxsize=None)
def _tau_grad_mod(r: int, p: int) -> Tuple[Tuple[Tuple[Alpha, int], ...], ...]:
    tau_poly = dict(_rank5_tau_mod(r, p))
    return tuple(
        tuple(sorted(_sparse_derivative(tau_poly, idx, p).items()))
        for idx in range(4)
    )


@lru_cache(maxsize=None)
def _tau_hessian_mod(
    r: int,
    p: int,
) -> Tuple[Tuple[Tuple[Tuple[Alpha, int], ...], ...], ...]:
    tau_poly = dict(_rank5_tau_mod(r, p))
    rows = []
    for i in range(4):
        first = _sparse_derivative(tau_poly, i, p)
        row = []
        for j in range(4):
            row.append(tuple(sorted(_sparse_derivative(first, j, p).items())))
        rows.append(tuple(row))
    return tuple(rows)


def _constant_sparse_value(poly_items: Tuple[Tuple[Alpha, int], ...], p: int) -> int:
    poly = dict(poly_items)
    if any(alpha != ZERO_ALPHA and coeff % p for alpha, coeff in poly.items()):
        raise ValueError("expected a constant sparse polynomial")
    return poly.get(ZERO_ALPHA, 0) % p


@lru_cache(maxsize=None)
def _hessian_tau2_inverse_const_mod(p: int) -> Tuple[Tuple[int, ...], ...]:
    matrix = [
        [_constant_sparse_value(_tau_hessian_mod(2, p)[i][j], p) for j in range(4)]
        for i in range(4)
    ]
    aug = [row[:] + [1 if i == j else 0 for j in range(4)] for i, row in enumerate(matrix)]
    for col in range(4):
        pivot = None
        for row in range(col, 4):
            if aug[row][col] % p:
                pivot = row
                break
        if pivot is None:
            raise ZeroDivisionError("tau2 Hessian is singular modulo p")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        inv = mod_inv(aug[col][col], p)
        aug[col] = [value * inv % p for value in aug[col]]
        for row in range(4):
            if row == col:
                continue
            coeff = aug[row][col] % p
            if coeff:
                aug[row] = [(aug[row][idx] - coeff * aug[col][idx]) % p for idx in range(8)]
    return tuple(tuple(row[4:]) for row in aug)


def _sparse_matrix_left_const_mul(
    left: Sequence[Sequence[int]],
    right: Sequence[Sequence[Tuple[Tuple[Alpha, int], ...]]],
    p: int,
) -> Tuple[Tuple[Tuple[Tuple[Alpha, int], ...], ...], ...]:
    rows = []
    for i in range(4):
        row = []
        for j in range(4):
            acc: SparsePoly = {}
            for k in range(4):
                if left[i][k] % p:
                    acc = _sparse_add(acc, dict(right[k][j]), p, scale=left[i][k])
            row.append(tuple(sorted(acc.items())))
        rows.append(tuple(row))
    return tuple(rows)


@lru_cache(maxsize=None)
def _hessian_perturbation_mod(
    p: int,
) -> Tuple[Tuple[Tuple[Tuple[Tuple[Alpha, int], ...], ...], ...], ...]:
    h0_inv = _hessian_tau2_inverse_const_mod(p)
    return tuple(
        _sparse_matrix_left_const_mul(h0_inv, _tau_hessian_mod(r, p), p)
        for r in (3, 4, 5)
    )


@lru_cache(maxsize=None)
def _c_direction_term_mod(r: int, p: int) -> Tuple[Tuple[Alpha, int], ...]:
    if r < 3 or r > 5:
        raise ValueError(f"expected r=3,...,5, got {r}")
    return _tau_grad_mod(r, p)[3]


@lru_cache(maxsize=None)
def _b_perturbation_mod(r: int, j: int, p: int) -> Tuple[Tuple[Alpha, int], ...]:
    if r < 3 or r > 5 or j < 1 or j > 4:
        raise ValueError(f"expected r=3,...,5 and j=1,...,4, got {(r, j)}")
    poly = dict(_rank5_tau_mod(r, p))
    deriv = _sparse_directional_derivative(poly, SIMPLE_COROOT_DIRECTIONS[j - 1], p)
    return tuple(sorted(_sparse_scale(deriv, -1, p).items()))


def _delta_leq(left: DeltaKey, right: DeltaKey) -> bool:
    return all(left[i] <= right[i] for i in range(3))


def _delta_add(left: DeltaKey, right: DeltaKey) -> DeltaKey:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _delta_sub(left: DeltaKey, right: DeltaKey) -> Optional[DeltaKey]:
    out = (left[0] - right[0], left[1] - right[1], left[2] - right[2])
    if min(out) < 0:
        return None
    return out


def _delta_poly_add(left: DeltaPoly, right: DeltaPoly, p: int, scale: int = 1) -> DeltaPoly:
    out: DeltaPoly = {delta: dict(poly) for delta, poly in left.items()}
    for delta, poly in right.items():
        out[delta] = _sparse_add(out.get(delta, {}), poly, p, scale=scale)
        if not out[delta]:
            del out[delta]
    return out


def _delta_poly_mul(left: DeltaPoly, right: DeltaPoly, max_delta: DeltaKey, p: int) -> DeltaPoly:
    out: DeltaPoly = {}
    for d1, p1 in left.items():
        for d2, p2 in right.items():
            delta = _delta_add(d1, d2)
            if not _delta_leq(delta, max_delta):
                continue
            prod = _sparse_mul(p1, p2, p)
            if prod:
                out[delta] = _sparse_add(out.get(delta, {}), prod, p)
    return out


def _delta_poly_scale(poly: DeltaPoly, scale: int, p: int) -> DeltaPoly:
    return {
        delta: scaled
        for delta, value in poly.items()
        if (scaled := _sparse_scale(value, scale, p))
    }


def _delta_poly_pow(base: DeltaPoly, exp: int, max_delta: DeltaKey, p: int) -> DeltaPoly:
    out: DeltaPoly = {ZERO_DELTA: {ZERO_ALPHA: 1}}
    cur = base
    n = int(exp)
    while n:
        if n & 1:
            out = _delta_poly_mul(out, cur, max_delta, p)
        n >>= 1
        if n:
            cur = _delta_poly_mul(cur, cur, max_delta, p)
    return out


def _delta_poly_exp_linear_mod(
    linear: Dict[DeltaKey, SparsePoly],
    max_delta: DeltaKey,
    p: int,
) -> DeltaPoly:
    out: DeltaPoly = {}
    powers: Dict[DeltaKey, Tuple[SparsePoly, ...]] = {}
    for unit, poly in linear.items():
        axis = unit.index(1)
        unit_powers = [{ZERO_ALPHA: 1}]
        for _exp in range(1, max_delta[axis] + 1):
            unit_powers.append(_sparse_mul(unit_powers[-1], poly, p))
        powers[unit] = tuple(unit_powers)

    inv_factorials = {
        n: mod_inv(mod_factorial(n, p), p)
        for n in range(sum(max_delta) + 1)
    }
    for e3 in range(max_delta[0] + 1):
        for e4 in range(max_delta[1] + 1):
            for e5 in range(max_delta[2] + 1):
                term: SparsePoly = {ZERO_ALPHA: 1}
                for exp, unit in ((e3, (1, 0, 0)), (e4, (0, 1, 0)), (e5, (0, 0, 1))):
                    if exp:
                        term = _sparse_mul(term, powers[unit][exp], p)
                        term = _sparse_scale(term, inv_factorials[exp], p)
                if term:
                    out[(e3, e4, e5)] = term
    return out


def _kernel_terms_mul_delta_mod(
    terms: KernelTerms,
    poly: DeltaPoly,
    max_delta: DeltaKey,
    p: int,
) -> KernelTerms:
    out: KernelTerms = {}
    for (kd, deriv), val in terms.items():
        for pd, pval in poly.items():
            nd = _delta_add(kd, pd)
            if not _delta_leq(nd, max_delta):
                continue
            product = _sparse_mul(val, pval, p)
            if product:
                out[(nd, deriv)] = _sparse_add(out.get((nd, deriv), {}), product, p)
    return {key: value for key, value in out.items() if value}


def _delta_matrix_identity(size: int) -> Tuple[Tuple[DeltaPoly, ...], ...]:
    return tuple(
        tuple({ZERO_DELTA: {ZERO_ALPHA: 1}} if i == j else {} for j in range(size))
        for i in range(size)
    )


def _delta_matrix_mul(
    left: Sequence[Sequence[DeltaPoly]],
    right: Sequence[Sequence[DeltaPoly]],
    max_delta: DeltaKey,
    p: int,
) -> Tuple[Tuple[DeltaPoly, ...], ...]:
    rows = len(left)
    cols = len(right[0])
    inner = len(right)
    out: List[List[DeltaPoly]] = [[{} for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            acc: DeltaPoly = {}
            for k in range(inner):
                acc = _delta_poly_add(
                    acc,
                    _delta_poly_mul(left[i][k], right[k][j], max_delta, p),
                    p,
                )
            out[i][j] = acc
    return tuple(tuple(row) for row in out)


@lru_cache(maxsize=None)
def _hessian_inverse_delta_mod(
    max_delta: DeltaKey,
    p: int,
) -> Tuple[Tuple[Tuple[Tuple[DeltaKey, Tuple[Tuple[Alpha, int], ...]], ...], ...], ...]:
    perturb = _hessian_perturbation_mod(p)

    a_mat: List[List[DeltaPoly]] = [[{} for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            entry: DeltaPoly = {}
            for unit, mat in zip(DELTA_UNITS, perturb):
                if _delta_leq(unit, max_delta):
                    poly = dict(mat[i][j])
                    if poly:
                        entry[unit] = _sparse_add(entry.get(unit, {}), poly, p)
            a_mat[i][j] = entry

    series = _delta_matrix_identity(4)
    power = _delta_matrix_identity(4)
    for order in range(1, sum(max_delta) + 1):
        power = _delta_matrix_mul(power, a_mat, max_delta, p)
        sign = -1 if order % 2 else 1
        series = tuple(
            tuple(
                _delta_poly_add(series[i][j], power[i][j], p, scale=sign)
                for j in range(4)
            )
            for i in range(4)
        )

    h0_inv = _hessian_tau2_inverse_const_mod(p)
    out: List[List[DeltaPoly]] = [[{} for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            acc: DeltaPoly = {}
            for k in range(4):
                if h0_inv[k][j] % p:
                    acc = _delta_poly_add(
                        acc,
                        _delta_poly_scale(series[i][k], h0_inv[k][j], p),
                        p,
                    )
            out[i][j] = acc

    return tuple(
        tuple(
            tuple(
                sorted((delta, tuple(sorted(poly.items()))) for delta, poly in cell.items() if poly)
            )
            for cell in row
        )
        for row in out
    )


def _hessian_inverse_cell_mod(max_delta: DeltaKey, i: int, j: int, p: int) -> DeltaPoly:
    return {
        delta: dict(poly_items)
        for delta, poly_items in _hessian_inverse_delta_mod(max_delta, p)[i][j]
    }


@lru_cache(maxsize=None)
def _hat_pair_delta_mod(
    r: int,
    s: int,
    max_delta: DeltaKey,
    p: int,
) -> Tuple[Tuple[DeltaKey, Tuple[Tuple[Alpha, int], ...]], ...]:
    gr = _tau_grad_mod(r, p)
    gs = _tau_grad_mod(s, p)
    acc: DeltaPoly = {}
    for i in range(4):
        for j in range(4):
            cell = _hessian_inverse_cell_mod(max_delta, i, j, p)
            if not cell:
                continue
            coeff_poly = _sparse_mul(dict(gr[i]), dict(gs[j]), p)
            coeff_poly = _sparse_scale(coeff_poly, -1, p)
            for delta, poly in cell.items():
                product = _sparse_mul(coeff_poly, poly, p)
                if product:
                    acc[delta] = _sparse_add(acc.get(delta, {}), product, p)
    return tuple(
        sorted((delta, tuple(sorted(poly.items()))) for delta, poly in acc.items() if poly)
    )


@lru_cache(maxsize=None)
def _det_ratio_delta_power_mod(
    max_delta: DeltaKey,
    power: int,
    p: int,
) -> Tuple[Tuple[DeltaKey, Tuple[Tuple[Alpha, int], ...]], ...]:
    perturb = _hessian_perturbation_mod(p)
    matrix: List[List[DeltaPoly]] = []
    for i in range(4):
        row: List[DeltaPoly] = []
        for j in range(4):
            entry: DeltaPoly = {ZERO_DELTA: {ZERO_ALPHA: 1}} if i == j else {}
            for unit, mat in zip(DELTA_UNITS, perturb):
                if _delta_leq(unit, max_delta):
                    poly = dict(mat[i][j])
                    if poly:
                        entry[unit] = _sparse_add(entry.get(unit, {}), poly, p)
            row.append(entry)
        matrix.append(row)

    det_poly: DeltaPoly = {}
    for perm in permutations(range(4)):
        inversions = sum(1 for i in range(4) for j in range(i + 1, 4) if perm[i] > perm[j])
        term: DeltaPoly = {ZERO_DELTA: {ZERO_ALPHA: -1 % p if inversions % 2 else 1}}
        for i, j in enumerate(perm):
            term = _delta_poly_mul(term, matrix[i][j], max_delta, p)
            if not term:
                break
        det_poly = _delta_poly_add(det_poly, term, p)
    result = _delta_poly_pow(det_poly, power, max_delta, p)
    return tuple(
        sorted((delta, tuple(sorted(poly.items()))) for delta, poly in result.items() if poly)
    )


def _denominator_taylor_terms_mod(max_delta: DeltaKey, p: int) -> KernelTerms:
    if max_delta == ZERO_DELTA:
        return {(ZERO_DELTA, ZERO_DERIV): {ZERO_ALPHA: 1}}
    terms: KernelTerms = {(ZERO_DELTA, ZERO_DERIV): {ZERO_ALPHA: 1}}
    max_order = sum(max_delta)
    unit_to_rank = {(1, 0, 0): 3, (0, 1, 0): 4, (0, 0, 1): 5}
    for j in range(1, 5):
        eps: DeltaPoly = {
            key: dict(_b_perturbation_mod(rank, j, p))
            for key, rank in unit_to_rank.items()
            if _delta_leq(key, max_delta)
        }
        factor: KernelTerms = {}
        for order in range(max_order + 1):
            eps_power = _delta_poly_pow(eps, order, max_delta, p)
            if not eps_power:
                continue
            deriv = [0, 0, 0, 0]
            deriv[j - 1] = order
            scale = mod_inv(mod_factorial(order, p), p)
            for kd, poly in eps_power.items():
                if not poly:
                    continue
                key = (kd, tuple(deriv))
                factor[key] = _sparse_add(factor.get(key, {}), poly, p, scale=scale)

        new_terms: KernelTerms = {}
        for (d1, der1), v1 in terms.items():
            for (d2, der2), v2 in factor.items():
                nd = _delta_add(d1, d2)
                if not _delta_leq(nd, max_delta):
                    continue
                nder = tuple(der1[i] + der2[i] for i in range(4))
                product = _sparse_mul(v1, v2, p)
                if product:
                    new_terms[(nd, nder)] = _sparse_add(new_terms.get((nd, nder), {}), product, p)
        terms = {key: value for key, value in new_terms.items() if value}
    return terms


@lru_cache(maxsize=None)
def _even_kernel_terms_mod(
    target_delta: DeltaKey,
    p: int,
) -> Tuple[Tuple[DeltaKey, DerivOrders, Tuple[Tuple[Alpha, int], ...]], ...]:
    linear = {
        (1, 0, 0): dict(_c_direction_term_mod(3, p)),
        (0, 1, 0): dict(_c_direction_term_mod(4, p)),
        (0, 0, 1): dict(_c_direction_term_mod(5, p)),
    }
    exp_delta = _delta_poly_exp_linear_mod(linear, target_delta, p)
    det_delta = {
        delta: dict(poly)
        for delta, poly in _det_ratio_delta_power_mod(target_delta, RANK5_CONFIG.genus, p)
    }
    terms = _denominator_taylor_terms_mod(target_delta, p)
    terms = _kernel_terms_mul_delta_mod(terms, exp_delta, target_delta, p)
    terms = _kernel_terms_mul_delta_mod(terms, det_delta, target_delta, p)
    return tuple(
        (kd, deriv, tuple(sorted(poly.items())))
        for (kd, deriv), poly in sorted(terms.items())
        if poly
    )


def _bit_for_label(label: Tuple[int, int]) -> int:
    return 1 << B_INDEX[label]


def _labels_from_mask(mask: int) -> Tuple[Tuple[int, int], ...]:
    return tuple(label for idx, label in enumerate(B_LABELS) if mask & (1 << idx))


def _wedge_masks(left: int, right: int) -> Optional[Tuple[int, int]]:
    if left & right:
        return None
    inversions = 0
    for idx in range(len(B_LABELS)):
        if not (left & (1 << idx)):
            continue
        inversions += sum(1 for j in range(idx) if right & (1 << j))
    return (-1 if inversions % 2 else 1, left | right)


@lru_cache(maxsize=None)
def _gamma_mask_expansion(
    gamma_exp: Tuple[int, ...],
    p: int,
) -> Tuple[Tuple[int, int], ...]:
    exterior = ExteriorAlgebra(RANK5_CONFIG)
    out = []
    for mask, coeff in exterior.gamma_product_to_mask_poly(gamma_exp).items():
        if coeff % p:
            out.append((mask, coeff % p))
    return tuple(sorted(out))


def _ext_delta_mul_pruned(
    left: Dict[int, DeltaPoly],
    right: Dict[int, DeltaPoly],
    max_delta: DeltaKey,
    target_mask: int,
    target_len: int,
    p: int,
) -> Dict[int, DeltaPoly]:
    out: Dict[int, DeltaPoly] = {}
    for m1, d1 in left.items():
        for m2, d2 in right.items():
            wedge = _wedge_masks(m1, m2)
            if wedge is None:
                continue
            sign, mask = wedge
            if mask.bit_count() > target_len or (mask | target_mask) != target_mask:
                continue
            prod = _delta_poly_mul(d1, d2, max_delta, p)
            if prod:
                out[mask] = _delta_poly_add(out.get(mask, {}), prod, p, scale=sign)
    return {mask: poly for mask, poly in out.items() if poly}


@lru_cache(maxsize=None)
def _b_hat_mask_mod(
    mask: int,
    max_delta: DeltaKey,
    p: int,
) -> Tuple[Tuple[DeltaKey, Tuple[Tuple[Alpha, int], ...]], ...]:
    target = _labels_from_mask(mask)
    if len(target) % 2:
        return ()
    if not target:
        return ((ZERO_DELTA, ((ZERO_ALPHA, 1),)),)

    pair_terms: Dict[int, DeltaPoly] = {}
    target_set = set(target)
    for left_side, right_side in ((1, 3), (2, 4)):
        left_labels = [label for label in target if label[1] == left_side]
        right_labels = [label for label in target if label[1] == right_side]
        for left_label in left_labels:
            for right_label in right_labels:
                pair_mask = _bit_for_label(left_label) | _bit_for_label(right_label)
                if any(label not in target_set for label in _labels_from_mask(pair_mask)):
                    continue
                wedge = _wedge_masks(_bit_for_label(left_label), _bit_for_label(right_label))
                if wedge is None:
                    continue
                sign, odd_mask = wedge
                coeff: DeltaPoly = {
                    delta: dict(items)
                    for delta, items in _hat_pair_delta_mod(
                        left_label[0],
                        right_label[0],
                        max_delta,
                        p,
                    )
                }
                pair_terms[odd_mask] = _delta_poly_add(
                    pair_terms.get(odd_mask, {}),
                    coeff,
                    p,
                    scale=sign,
                )

    pair_count = len(target) // 2
    power: Dict[int, DeltaPoly] = {0: {ZERO_DELTA: {ZERO_ALPHA: 1}}}
    for _ in range(pair_count):
        power = _ext_delta_mul_pruned(power, pair_terms, max_delta, mask, len(target), p)
        if not power:
            return ()

    scale = mod_inv(factorial(pair_count), p)
    result = {
        delta: _sparse_scale(poly, scale, p)
        for delta, poly in power.get(mask, {}).items()
    }
    return tuple(
        sorted((delta, tuple(sorted(poly.items()))) for delta, poly in result.items() if poly)
    )


@lru_cache(maxsize=None)
def _gamma_hat_mod(
    gamma_exp: Tuple[int, ...],
    target_delta: DeltaKey,
    p: int,
) -> Tuple[Tuple[DeltaKey, Tuple[Tuple[Alpha, int], ...]], ...]:
    out: DeltaPoly = {}
    for mask, coeff in _gamma_mask_expansion(gamma_exp, p):
        b_delta = {
            delta: dict(poly_items)
            for delta, poly_items in _b_hat_mask_mod(mask, target_delta, p)
        }
        out = _delta_poly_add(out, b_delta, p, scale=coeff)
    return tuple(
        sorted((delta, tuple(sorted(poly.items()))) for delta, poly in out.items() if poly)
    )


@lru_cache(maxsize=None)
def _h_coeffs_mod(nmax: int, p: int) -> Tuple[int, ...]:
    out = []
    fact = 1
    for n in range(nmax + 1):
        fact = fact * (n + 1) % p
        coeff = mod_inv(fact, p)
        if n % 2:
            coeff = (-coeff) % p
        out.append(coeff)
    return tuple(out)


@lru_cache(maxsize=None)
def _poly_power_coeffs_mod(base: Tuple[int, ...], power: int, nmax: int, p: int) -> Tuple[int, ...]:
    coeffs = [0 for _ in range(nmax + 1)]
    coeffs[0] = 1
    for _ in range(power):
        next_coeffs = [0 for _ in range(nmax + 1)]
        for i, left in enumerate(coeffs):
            if not left:
                continue
            for j, right in enumerate(base[: nmax + 1 - i]):
                if right:
                    next_coeffs[i + j] = (next_coeffs[i + j] + left * right) % p
        coeffs = next_coeffs
    return tuple(coeffs)


@lru_cache(maxsize=None)
def _exp_coeffs_mod(lam_num: int, nmax: int, p: int) -> Tuple[int, ...]:
    lam = lam_num * mod_inv(5, p) % p
    out = []
    fact = 1
    pow_lam = 1
    for n in range(nmax + 1):
        if n:
            fact = fact * n % p
            pow_lam = pow_lam * lam % p
        out.append(pow_lam * mod_inv(fact, p) % p)
    return tuple(out)


@lru_cache(maxsize=None)
def _special_series_mod(
    power: int,
    lam_num: int,
    cutoff: int,
    p: int,
) -> Tuple[Tuple[int, int], ...]:
    nmax = cutoff + power
    if nmax < 0:
        return ()
    if power == 0:
        return tuple(
            (n, coeff)
            for n, coeff in enumerate(_exp_coeffs_mod(lam_num, cutoff, p))
            if coeff
        )
    h_power = _poly_power_coeffs_mod(_h_coeffs_mod(nmax, p), power, nmax, p)
    exp_coeffs = _exp_coeffs_mod(lam_num, nmax, p)
    quotient = [0 for _ in range(nmax + 1)]
    for n in range(nmax + 1):
        coeff = exp_coeffs[n]
        for i in range(1, n + 1):
            coeff = (coeff - h_power[i] * quotient[n - i]) % p
        quotient[n] = coeff
    return tuple((n - power, coeff) for n, coeff in enumerate(quotient) if coeff)


@lru_cache(maxsize=None)
def _stirling2(nmax: int) -> Tuple[Tuple[int, ...], ...]:
    table = [[0 for _ in range(nmax + 1)] for _ in range(nmax + 1)]
    table[0][0] = 1
    for n in range(1, nmax + 1):
        for k in range(1, n + 1):
            table[n][k] = table[n - 1][k - 1] + k * table[n - 1][k]
    return tuple(tuple(row) for row in table)


@lru_cache(maxsize=None)
def _special_derivative_dict_mod(order: int, lam_num: int, cutoff: int, p: int) -> Dict[int, int]:
    if order == 0:
        return dict(_special_series_mod(1, lam_num, cutoff, p))
    accum: Dict[int, int] = {}
    sign = -1 if order % 2 else 1
    st = _stirling2(order)
    for k in range(1, order + 1):
        s2 = st[order][k]
        if not s2:
            continue
        scale = sign * factorial(k) * s2
        for exp, coeff in _special_series_mod(k + 1, lam_num - 5 * k, cutoff, p):
            accum[exp] = (accum.get(exp, 0) + scale * coeff) % p
    return {exp: coeff for exp, coeff in accum.items() if coeff % p}


@lru_cache(maxsize=None)
def _binomial_series_mod(root_power: int, cutoff: int, p: int) -> Tuple[int, ...]:
    return tuple(((-1) ** m * comb(root_power + m - 1, m)) % p for m in range(cutoff + 1))


@lru_cache(maxsize=None)
def _root_transition_schedule() -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    by_var = [[] for _ in range(4)]
    for var_idx in range(4):
        for interval, pos in RANK5_ROOT_INDEX.items():
            if interval[1] != var_idx + 1:
                continue
            lower_pos = -1 if interval[0] == var_idx else RANK5_ROOT_INDEX[(interval[0], var_idx)]
            by_var[var_idx].append((pos, lower_pos))
    return tuple(tuple(items) for items in by_var)


def _max_survivable_y_exp(
    var_idx: int,
    deriv_orders: DerivOrders,
    denom_powers: Tuple[int, ...],
    current_root_pos: int,
) -> int:
    simple_pos = RANK5_ROOT_INDEX[(var_idx, var_idx + 1)]
    simple_drop = int(denom_powers[simple_pos]) if current_root_pos < simple_pos else 0
    return int(deriv_orders[var_idx]) + simple_drop


@lru_cache(maxsize=None)
def _variable_transition_mod(
    var_idx: int,
    deriv_orders: DerivOrders,
    y_exp: int,
    denom_powers: Tuple[int, ...],
    p: int,
) -> Tuple[Tuple[Tuple[int, ...], int], ...]:
    states: Dict[Tuple[int, Tuple[int, ...]], int] = {(int(y_exp), denom_powers): 1}
    for pos, lower_pos in _root_transition_schedule()[var_idx]:
        next_states: Dict[Tuple[int, Tuple[int, ...]], int] = {}
        for (cur_y_exp, dtuple), state_coeff in states.items():
            root_power = int(dtuple[pos])
            if not root_power:
                key = (cur_y_exp, dtuple)
                next_states[key] = (next_states.get(key, 0) + state_coeff) % p
                continue
            base_den = list(dtuple)
            base_den[pos] = 0
            base_den_tuple = tuple(base_den)
            y_bound = _max_survivable_y_exp(var_idx, deriv_orders, base_den_tuple, pos)
            if lower_pos < 0:
                next_y_exp = cur_y_exp - root_power
                if next_y_exp > y_bound:
                    continue
                key = (next_y_exp, base_den_tuple)
                next_states[key] = (next_states.get(key, 0) + state_coeff) % p
                continue
            max_m = y_bound - cur_y_exp
            if max_m < 0:
                continue
            binoms = _binomial_series_mod(root_power, max_m, p)
            for m in range(max_m + 1):
                expanded_den = list(base_den_tuple)
                expanded_den[lower_pos] += root_power + m
                key = (cur_y_exp + m, tuple(expanded_den))
                next_states[key] = (next_states.get(key, 0) + state_coeff * binoms[m]) % p
        states = {key: value for key, value in next_states.items() if value % p}
        if not states:
            return ()

    needed_cutoff = max(max(0, -1 - cur_y_exp) for cur_y_exp, _dtuple in states)
    special = _special_derivative_dict_mod(
        deriv_orders[var_idx],
        RANK5_BASE_LAMBDA_NUMS[var_idx],
        needed_cutoff,
        p,
    )
    out: Dict[Tuple[int, ...], int] = {}
    for (cur_y_exp, dtuple), state_coeff in states.items():
        special_coeff = special.get(-1 - cur_y_exp)
        if special_coeff:
            out[dtuple] = (out.get(dtuple, 0) + state_coeff * special_coeff) % p
    return tuple(sorted((dtuple, coeff) for dtuple, coeff in out.items() if coeff % p))


def _residue_poly_mod(poly: SparsePoly, deriv_orders: DerivOrders, p: int) -> int:
    terms: Dict[Tuple[Alpha, Tuple[int, ...]], int] = {
        (alpha, RANK5_ROOT_POWERS): int(coeff) % p
        for alpha, coeff in poly.items()
        if int(coeff) % p
    }
    for var_idx in reversed(range(4)):
        new_terms: Dict[Tuple[Alpha, Tuple[int, ...]], int] = {}
        for (cur_alpha, denom_powers), coeff in terms.items():
            next_alpha = list(cur_alpha)
            next_alpha[var_idx] = 0
            next_alpha_t = tuple(next_alpha)
            for dtuple, trans_coeff in _variable_transition_mod(
                var_idx,
                deriv_orders,
                cur_alpha[var_idx],
                denom_powers,
                p,
            ):
                key = (next_alpha_t, dtuple)
                new_terms[key] = (new_terms.get(key, 0) + coeff * trans_coeff) % p
        terms = {key: value for key, value in new_terms.items() if value % p}
        if not terms:
            return 0

    total = 0
    for (cur_alpha, denom_powers), coeff in terms.items():
        if cur_alpha == ZERO_ALPHA and denom_powers == RANK5_ZERO_DENOM:
            total = (total + coeff) % p
    return total


@lru_cache(maxsize=None)
def _pairing_kernel_gamma_products_mod(
    target_delta: DeltaKey,
    gamma_exp: Tuple[int, ...],
    p: int,
) -> Tuple[Tuple[DerivOrders, Tuple[Tuple[Alpha, int], ...]], ...]:
    b_delta = {
        delta: dict(poly)
        for delta, poly in _gamma_hat_mod(gamma_exp, target_delta, p)
    }
    if not b_delta:
        return ()
    out: List[Tuple[DerivOrders, Tuple[Tuple[Alpha, int], ...]]] = []
    for kd, deriv_orders, kval_items in _even_kernel_terms_mod(target_delta, p):
        bd = _delta_sub(target_delta, kd)
        if bd is None or bd not in b_delta:
            continue
        shared_poly = _sparse_mul(dict(kval_items), b_delta[bd], p)
        if shared_poly:
            out.append((deriv_orders, tuple(sorted(shared_poly.items()))))
    return tuple(out)


@lru_cache(maxsize=None)
def _pairing_total_mod_cached(
    a_exp: Tuple[int, int, int, int],
    f_exp: Tuple[int, int, int, int],
    gamma_exp: Tuple[int, ...],
    p: int,
) -> int:
    target_delta: DeltaKey = (f_exp[1], f_exp[2], f_exp[3])
    shared_terms = _pairing_kernel_gamma_products_mod(target_delta, gamma_exp, p)
    if not shared_terms:
        return 0
    a_poly = dict(_tau_power_mod(a_exp, p))
    value = 0
    for deriv_orders, shared_items in shared_terms:
        full_poly = _sparse_mul(a_poly, dict(shared_items), p)
        if full_poly:
            value = (value + _residue_poly_mod(full_poly, deriv_orders, p)) % p

    scale = int(RANK5_CONFIG.collapsed_prefactor) % p
    scale = scale * _f_factorial_scale_mod(f_exp, p) % p
    return scale * value % p


def _pairing_total_mod(total: InvariantMonomial, p: int) -> int:
    return _pairing_total_mod_cached(
        tuple(total.a_exp),
        tuple(total.f_exp),
        tuple(total.gamma_exp),
        p,
    )


def pairing_values_mod_prime(
    config: FormulaConfig,
    pairs: Iterable[Tuple[InvariantMonomial, InvariantMonomial]],
    *,
    prime: int | None = None,
) -> Tuple[int, ...]:
    p = config.primary_prime if prime is None else int(prime)
    return tuple(pairing_mod_prime(config, left, right, prime=p) for left, right in pairs)
