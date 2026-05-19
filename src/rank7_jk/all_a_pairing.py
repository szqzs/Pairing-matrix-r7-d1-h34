"""Specialized all-a pairing evaluator for the one-defect c18 probe.

This is not the full production evaluator.  It implements the narrow formula
needed when the H62 test column has only ``a`` classes and the H34 c18 source
row has exactly one defect: either one ``f_r`` or one ``gamma_rs``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from math import factorial
from typing import Sequence, Tuple

from . import modular_formula
from .c18_basis import C18SourceRow, H62TestColumn
from .config import FormulaConfig, RANK7_G2_D1
from .exterior import ExteriorAlgebra
from .invariants import InvariantMonomial
from .mod_arith import require_prime
from .residue_transition import residue_monomial_mod, residue_poly_mod
from .root_system import type_a_roots
from .sparse_poly import SparsePoly, add, clean, constant, mul, pow_poly

DerivOrders = Tuple[int, ...]


def c18_all_a_pairing_column(
    index: int,
    column: H62TestColumn,
    rows: Sequence[C18SourceRow],
    prime: int,
    *,
    config: FormulaConfig = RANK7_G2_D1,
) -> Tuple[int, ...]:
    """Evaluate one actual all-a probe column against all c18 source rows."""

    del index
    if column.kind != "all_a":
        raise ValueError("c18_all_a_pairing_column expects an all-a test column")
    return tuple(
        all_a_pairing_value(config, row.monomial, column.monomial, prime=prime)
        for row in rows
    )


def c18_all_a_pairing_column_moment(
    index: int,
    column: H62TestColumn,
    rows: Sequence[C18SourceRow],
    prime: int,
    *,
    config: FormulaConfig = RANK7_G2_D1,
) -> Tuple[int, ...]:
    """Evaluate one all-a column with the cached residue-moment engine."""

    evaluator = _cached_moment_batch_evaluator(config, tuple(rows), require_prime(prime))
    return evaluator.column_vector(index, column)


def all_a_pairing_value(
    config: FormulaConfig,
    source: InvariantMonomial,
    test: InvariantMonomial,
    *,
    prime: int | None = None,
) -> int:
    """Evaluate a one-defect source monomial against an all-a test monomial."""

    if any(test.f_exp) or any(test.gamma_exp):
        raise ValueError("the specialized all-a evaluator requires an all-a test")
    return all_a_pairing_total_mod(config, source * test, prime=prime)


def all_a_pairing_value_moment(
    config: FormulaConfig,
    source: InvariantMonomial,
    test: InvariantMonomial,
    *,
    prime: int | None = None,
) -> int:
    """Evaluate a one-defect source against an all-a test with cached moments."""

    if any(test.f_exp) or any(test.gamma_exp):
        raise ValueError("the specialized all-a evaluator requires an all-a test")
    return all_a_pairing_total_moment_mod(config, source * test, prime=prime)


def all_a_pairing_total_mod(
    config: FormulaConfig,
    total: InvariantMonomial,
    *,
    prime: int | None = None,
) -> int:
    """Evaluate a top-degree all-a times one-defect total monomial modulo a prime."""

    p = require_prime(config.primary_prime if prime is None else prime)
    _validate_total(config, total)
    target_delta = tuple(int(exp) for exp in total.f_exp[1:])
    gamma_exp = tuple(int(exp) for exp in total.gamma_exp)

    shared_terms = _shared_kernel_terms(config, target_delta, gamma_exp, p)
    if not shared_terms:
        return 0

    a_poly = dict(_tau_power_mod(config, tuple(total.a_exp), p))
    value = 0
    for deriv_orders, shared_items in shared_terms:
        full_poly = mul(a_poly, dict(shared_items), prime=p)
        if full_poly:
            value = (
                value
                + residue_poly_mod(
                    config.rank,
                    full_poly,
                    deriv_orders,
                    prime=p,
                    root_power=config.root_denominator_power,
                )
            ) % p

    scale_factor = int(config.collapsed_prefactor) % p
    for exp in total.f_exp:
        scale_factor = scale_factor * factorial(int(exp)) % p
    return scale_factor * value % p


def all_a_pairing_total_moment_mod(
    config: FormulaConfig,
    total: InvariantMonomial,
    *,
    prime: int | None = None,
) -> int:
    """Evaluate a one-defect all-a total monomial via cached residue moments."""

    p = require_prime(config.primary_prime if prime is None else prime)
    _validate_total(config, total)
    return _pairing_from_parts_moment(
        config,
        tuple(total.a_exp),
        tuple(total.f_exp),
        tuple(total.gamma_exp),
        p,
    )


@dataclass
class AllAMomentBatchEvaluator:
    """Batch all-a column evaluator with shared row plans and cached kernels."""

    config: FormulaConfig
    rows: Tuple[C18SourceRow, ...]
    prime: int
    row_plans: Tuple["AllARowPlan", ...] = field(init=False)
    rows_by_defect: dict[str, Tuple[int, ...]] = field(init=False)

    def __post_init__(self) -> None:
        p = require_prime(self.prime)
        object.__setattr__(self, "prime", p)
        plans = tuple(
            AllARowPlan.from_row(self.config, row_index, row)
            for row_index, row in enumerate(self.rows)
        )
        object.__setattr__(self, "row_plans", plans)

        grouped: dict[str, list[int]] = defaultdict(list)
        for plan in plans:
            grouped[plan.defect_id].append(plan.row_index)
            _shared_kernel_terms(self.config, plan.target_delta, plan.gamma_exp, p)
        object.__setattr__(
            self,
            "rows_by_defect",
            {key: tuple(indices) for key, indices in sorted(grouped.items())},
        )

    def column_vector(self, index: int, column: H62TestColumn) -> Tuple[int, ...]:
        del index
        if column.kind != "all_a":
            raise ValueError("AllAMomentBatchEvaluator expects an all-a test column")
        column_a_exp = tuple(column.monomial.a_exp)
        return tuple(
            _pairing_from_parts_moment(
                self.config,
                _add_exp(plan.a_exp, column_a_exp),
                plan.f_exp,
                plan.gamma_exp,
                self.prime,
            )
            for plan in self.row_plans
        )


@dataclass(frozen=True)
class AllARowPlan:
    row_index: int
    a_exp: Tuple[int, ...]
    f_exp: Tuple[int, ...]
    gamma_exp: Tuple[int, ...]
    target_delta: Tuple[int, ...]
    defect_id: str

    @classmethod
    def from_row(
        cls,
        config: FormulaConfig,
        row_index: int,
        row: C18SourceRow,
    ) -> "AllARowPlan":
        monomial = row.monomial
        f_exp = tuple(monomial.f_exp)
        gamma_exp = tuple(monomial.gamma_exp)
        if sum(f_exp) + sum(gamma_exp) != 1:
            raise NotImplementedError("all-a row plans require exactly one defect")
        return cls(
            row_index=int(row_index),
            a_exp=tuple(monomial.a_exp),
            f_exp=f_exp,
            gamma_exp=gamma_exp,
            target_delta=tuple(f_exp[1:]),
            defect_id=_defect_id(config, f_exp, gamma_exp),
        )


def precompute_all_a_defect_kernels(
    config: FormulaConfig,
    rows: Sequence[C18SourceRow],
    *,
    prime: int | None = None,
) -> Tuple[str, ...]:
    """Warm the one-defect kernel cache for the defects appearing in ``rows``."""

    p = require_prime(config.primary_prime if prime is None else prime)
    defect_ids = []
    seen = set()
    for row in rows:
        plan = AllARowPlan.from_row(config, len(defect_ids), row)
        if plan.defect_id in seen:
            continue
        _shared_kernel_terms(config, plan.target_delta, plan.gamma_exp, p)
        seen.add(plan.defect_id)
        defect_ids.append(plan.defect_id)
    return tuple(defect_ids)


def all_a_cache_info() -> dict[str, dict[str, int]]:
    """Return JSON-friendly cache counters for all-a performance probes."""

    return {
        "batch_evaluator": _cache_info_dict(_cached_moment_batch_evaluator.cache_info()),
        "bounded_tau_power": _cache_info_dict(_tau_power_bounded_mod.cache_info()),
        "kernel_terms": _cache_info_dict(_shared_kernel_terms.cache_info()),
        "moment": _cache_info_dict(_moment_mod.cache_info()),
        "monomial_residue": _cache_info_dict(_residue_monomial_cached.cache_info()),
        "tau_power": _cache_info_dict(_tau_power_mod.cache_info()),
    }


def clear_all_a_caches() -> None:
    """Clear all-a evaluator caches for controlled profiling tests."""

    _cached_moment_batch_evaluator.cache_clear()
    _shared_kernel_terms.cache_clear()
    _moment_mod.cache_clear()
    _residue_monomial_cached.cache_clear()
    _tau_power_bounded_mod.cache_clear()
    _tau_power_mod.cache_clear()


@lru_cache(maxsize=16)
def _cached_moment_batch_evaluator(
    config: FormulaConfig,
    rows: Tuple[C18SourceRow, ...],
    prime: int,
) -> AllAMomentBatchEvaluator:
    return AllAMomentBatchEvaluator(config=config, rows=rows, prime=prime)


def _pairing_from_parts_moment(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
    f_exp: Tuple[int, ...],
    gamma_exp: Tuple[int, ...],
    prime: int,
) -> int:
    p = require_prime(prime)
    target_delta = tuple(int(exp) for exp in f_exp[1:])
    shared_terms = _shared_kernel_terms(config, target_delta, tuple(gamma_exp), p)
    if not shared_terms:
        return 0

    value = 0
    for deriv_orders, shared_items in shared_terms:
        for beta, coeff in shared_items:
            if coeff % p:
                value = (
                    value
                    + coeff * _moment_mod(config, tuple(a_exp), beta, deriv_orders, p)
                ) % p

    scale_factor = int(config.collapsed_prefactor) % p
    for exp in f_exp:
        scale_factor = scale_factor * factorial(int(exp)) % p
    return scale_factor * value % p


@lru_cache(maxsize=None)
def _moment_mod(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
    beta: Tuple[int, ...],
    deriv_orders: DerivOrders,
    prime: int,
) -> int:
    p = require_prime(prime)
    if len(beta) != config.y_count:
        raise ValueError("kernel monomial dimension does not match the formula config")
    caps = _moment_tau_caps(config, beta, deriv_orders)
    if caps is None:
        return 0
    value = 0
    for alpha, coeff in _tau_power_bounded_mod(config, tuple(a_exp), caps, p):
        shifted = tuple(alpha[idx] + beta[idx] for idx in range(config.y_count))
        value = (
            value
            + coeff
            * _residue_monomial_cached(
                config.rank,
                shifted,
                tuple(deriv_orders),
                config.root_denominator_power,
                p,
            )
        ) % p
    return value


def _moment_tau_caps(
    config: FormulaConfig,
    beta: Tuple[int, ...],
    deriv_orders: DerivOrders,
) -> Tuple[int | None, ...] | None:
    if len(deriv_orders) != config.y_count:
        raise ValueError("derivative order length does not match the formula config")

    residue_caps = _residue_exponent_caps(
        config.rank,
        tuple(int(item) for item in deriv_orders),
        config.root_denominator_power,
    )
    caps: list[int | None] = []
    for idx, cap in enumerate(residue_caps):
        tau_cap = cap - int(beta[idx])
        if tau_cap < 0:
            return None
        caps.append(tau_cap)
    return tuple(caps)


@lru_cache(maxsize=None)
def _residue_monomial_cached(
    rank: int,
    alpha: Tuple[int, ...],
    deriv_orders: DerivOrders,
    root_power: int,
    prime: int,
) -> int:
    return residue_monomial_mod(
        int(rank),
        alpha,
        deriv_orders,
        prime=require_prime(prime),
        root_power=int(root_power),
    )


@lru_cache(maxsize=None)
def _residue_exponent_caps(
    rank: int,
    deriv_orders: DerivOrders,
    root_power: int,
) -> Tuple[int, ...]:
    """Conservative monomial exponent caps for nonzero residue contributions."""

    roots = type_a_roots(rank)
    if len(deriv_orders) != roots.y_count:
        raise ValueError("derivative order length does not match rank")
    max_den = [int(root_power) for _ in range(roots.positive_root_count)]
    caps = [0 for _ in range(roots.y_count)]
    for var_idx in reversed(range(roots.y_count)):
        simple_pos = roots.interval_index[(var_idx, var_idx + 1)]
        cap = int(deriv_orders[var_idx]) + max_den[simple_pos]
        caps[var_idx] = cap
        for pos, lower_pos in roots.transition_schedule[var_idx]:
            current_power = max_den[pos]
            max_den[pos] = 0
            if lower_pos >= 0:
                max_den[lower_pos] += current_power + cap
    return tuple(caps)


def _validate_total(config: FormulaConfig, total: InvariantMonomial) -> None:
    if total.rank != config.rank:
        raise ValueError("pairing monomial rank must match the formula config")
    if total.ordinary_degree != config.top_degree:
        raise ValueError("pairing monomial must have top ordinary degree")
    if sum(total.f_exp) + sum(total.gamma_exp) != 1:
        raise NotImplementedError(
            "the all-a specialized evaluator currently supports exactly one defect"
        )
    if sum(total.f_exp[1:]) > 1:
        raise NotImplementedError("at most one non-f2 delta defect is supported")


@lru_cache(maxsize=None)
def _shared_kernel_terms(
    config: FormulaConfig,
    target_delta: Tuple[int, ...],
    gamma_exp: Tuple[int, ...],
    prime: int,
) -> Tuple[Tuple[DerivOrders, Tuple[Tuple[Tuple[int, ...], int], ...]], ...]:
    p = require_prime(prime)
    if len(target_delta) != len(config.delta_ranks):
        raise ValueError("target delta length does not match the formula config")
    if len(gamma_exp) != len(config.gamma_labels):
        raise ValueError("gamma exponent length does not match the formula config")

    gamma_count = sum(gamma_exp)
    if gamma_count == 0:
        return _even_kernel_terms(config, target_delta, p)
    if gamma_count == 1 and not any(target_delta):
        return _gamma_zero_delta_terms(config, gamma_exp, p)
    raise NotImplementedError(
        "the first all-a evaluator supports either one f defect or one gamma defect"
    )


@lru_cache(maxsize=None)
def _even_kernel_terms(
    config: FormulaConfig,
    target_delta: Tuple[int, ...],
    prime: int,
) -> Tuple[Tuple[DerivOrders, Tuple[Tuple[Tuple[int, ...], int], ...]], ...]:
    p = require_prime(prime)
    zero_deriv = _zero_deriv(config)
    if not any(target_delta):
        unit = tuple(sorted(constant(config.y_count, 1, prime=p).items()))
        return ((zero_deriv, unit),)
    if sum(target_delta) != 1:
        raise NotImplementedError("only zero or one delta defect is supported")

    delta_rank = config.delta_ranks[target_delta.index(1)]
    poly0 = add(
        dict(modular_formula.c_tilde_delta_coeff_mod(config, delta_rank, p)),
        dict(_det_ratio_first_delta_mod(config, delta_rank, p)),
        prime=p,
        scale=config.genus,
    )

    terms = []
    if poly0:
        terms.append((zero_deriv, tuple(sorted(poly0.items()))))
    for j in range(1, config.y_count + 1):
        deriv = [0 for _ in range(config.y_count)]
        deriv[j - 1] = 1
        poly = dict(modular_formula.b_perturbation_mod(config, delta_rank, j, p))
        if poly:
            terms.append((tuple(deriv), tuple(sorted(clean(poly, p).items()))))
    return tuple(terms)


@lru_cache(maxsize=None)
def _gamma_zero_delta_terms(
    config: FormulaConfig,
    gamma_exp: Tuple[int, ...],
    prime: int,
) -> Tuple[Tuple[DerivOrders, Tuple[Tuple[Tuple[int, ...], int], ...]], ...]:
    p = require_prime(prime)
    exterior = ExteriorAlgebra(config)
    acc: SparsePoly = {}
    for mask, coeff in exterior.gamma_product_to_mask_poly(gamma_exp).items():
        term = dict(_b_hat_mask_delta_zero_mod(config, mask, p))
        if term:
            acc = add(acc, term, prime=p, scale=coeff)
    if not acc:
        return ()
    return ((_zero_deriv(config), tuple(sorted(acc.items()))),)


@lru_cache(maxsize=None)
def _b_hat_mask_delta_zero_mod(
    config: FormulaConfig,
    mask: int,
    prime: int,
) -> Tuple[Tuple[Tuple[int, ...], int], ...]:
    p = require_prime(prime)
    exterior = ExteriorAlgebra(config)
    target = exterior.labels_from_mask(int(mask))
    if len(target) % 2:
        return ()
    if not target:
        return tuple(sorted(constant(config.y_count, 1, prime=p).items()))
    if len(target) != 2:
        raise NotImplementedError("only one gamma factor is supported in the all-a probe")

    acc: SparsePoly = {}
    for i in range(1, config.genus + 1):
        left_side = i
        right_side = i + config.genus
        left_labels = [label for label in target if label[1] == left_side]
        right_labels = [label for label in target if label[1] == right_side]
        for left_label in left_labels:
            for right_label in right_labels:
                wedge = exterior.wedge_masks(
                    exterior.mask_for_b_label(left_label),
                    exterior.mask_for_b_label(right_label),
                )
                if wedge is None:
                    continue
                sign, pair_mask = wedge
                if pair_mask != mask:
                    continue
                pair_poly = dict(
                    modular_formula.hat_pair_delta_zero_mod(
                        config,
                        left_label[0],
                        right_label[0],
                        p,
                    )
                )
                acc = add(acc, pair_poly, prime=p, scale=sign)
    return tuple(sorted(acc.items()))


@lru_cache(maxsize=None)
def _det_ratio_first_delta_mod(
    config: FormulaConfig,
    delta_rank: int,
    prime: int,
) -> Tuple[Tuple[Tuple[int, ...], int], ...]:
    p = require_prime(prime)
    h0_inv = modular_formula.hessian_tau2_inverse_mod(config, p)
    h_delta = modular_formula.tau_hessian_mod(config, delta_rank, p)
    acc: SparsePoly = {}
    for i in range(config.y_count):
        for k in range(config.y_count):
            coeff = h0_inv[i][k] % p
            if coeff:
                acc = add(acc, dict(h_delta[k][i]), prime=p, scale=coeff)
    return tuple(sorted(clean(acc, p).items()))


@lru_cache(maxsize=None)
def _tau_power_mod(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
    prime: int,
) -> Tuple[Tuple[Tuple[int, ...], int], ...]:
    p = require_prime(prime)
    if len(a_exp) != len(config.class_ranks):
        raise ValueError("a exponent length does not match the formula config")
    out = constant(config.y_count, 1, prime=p)
    for exp, r in zip(a_exp, config.class_ranks):
        if exp:
            factor = dict(modular_formula.tau_mod(config, r, p))
            out = mul(out, pow_poly(factor, int(exp), prime=p), prime=p)
    return tuple(sorted(out.items()))


@lru_cache(maxsize=None)
def _tau_power_bounded_mod(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
    caps: Tuple[int | None, ...],
    prime: int,
) -> Tuple[Tuple[Tuple[int, ...], int], ...]:
    p = require_prime(prime)
    if len(a_exp) != len(config.class_ranks):
        raise ValueError("a exponent length does not match the formula config")
    if len(caps) != config.y_count:
        raise ValueError("cap length does not match the formula config")
    if all(cap is None for cap in caps):
        return _tau_power_mod(config, a_exp, p)

    out = constant(config.y_count, 1, prime=p)
    for exp, r in zip(a_exp, config.class_ranks):
        if exp:
            factor = dict(modular_formula.tau_mod(config, r, p))
            out = _bounded_mul(
                out,
                _bounded_pow(factor, int(exp), caps, p),
                caps,
                p,
            )
    return tuple(sorted(out.items()))


def _bounded_pow(
    base: SparsePoly,
    exponent: int,
    caps: Tuple[int | None, ...],
    prime: int,
) -> SparsePoly:
    p = require_prime(prime)
    exp = int(exponent)
    if exp < 0:
        raise ValueError("polynomial exponent must be nonnegative")
    out = constant(len(caps), 1, prime=p)
    factor = _bounded_clean(base, caps, p)
    for _ in range(exp):
        out = _bounded_mul(out, factor, caps, p)
    return out


def _bounded_mul(
    left: SparsePoly,
    right: SparsePoly,
    caps: Tuple[int | None, ...],
    prime: int,
) -> SparsePoly:
    p = require_prime(prime)
    if not left or not right:
        return {}
    if len(caps) == 6:
        return _bounded_mul_6(left, right, caps, p)
    out: SparsePoly = {}
    for a1, c1 in left.items():
        for a2, c2 in right.items():
            alpha = tuple(a1[idx] + a2[idx] for idx in range(len(caps)))
            if _exceeds_caps(alpha, caps):
                continue
            out[alpha] = (out.get(alpha, 0) + c1 * c2) % p
    return clean(out, p)


def _bounded_mul_6(
    left: SparsePoly,
    right: SparsePoly,
    caps: Tuple[int | None, ...],
    prime: int,
) -> SparsePoly:
    cap0 = 10**9 if caps[0] is None else int(caps[0])
    cap1 = 10**9 if caps[1] is None else int(caps[1])
    cap2 = 10**9 if caps[2] is None else int(caps[2])
    cap3 = 10**9 if caps[3] is None else int(caps[3])
    cap4 = 10**9 if caps[4] is None else int(caps[4])
    cap5 = 10**9 if caps[5] is None else int(caps[5])
    right_items = tuple(right.items())
    out: SparsePoly = {}
    p = prime
    for a1, c1 in left.items():
        a10, a11, a12, a13, a14, a15 = a1
        for a2, c2 in right_items:
            s0 = a10 + a2[0]
            if s0 > cap0:
                continue
            s1 = a11 + a2[1]
            if s1 > cap1:
                continue
            s2 = a12 + a2[2]
            if s2 > cap2:
                continue
            s3 = a13 + a2[3]
            if s3 > cap3:
                continue
            s4 = a14 + a2[4]
            if s4 > cap4:
                continue
            s5 = a15 + a2[5]
            if s5 > cap5:
                continue
            alpha = (s0, s1, s2, s3, s4, s5)
            value = (out.get(alpha, 0) + c1 * c2) % p
            if value:
                out[alpha] = value
            else:
                out.pop(alpha, None)
    return out


def _bounded_clean(
    poly: SparsePoly,
    caps: Tuple[int | None, ...],
    prime: int,
) -> SparsePoly:
    return {
        alpha: coeff
        for alpha, coeff in clean(poly, prime).items()
        if not _exceeds_caps(alpha, caps)
    }


def _exceeds_caps(alpha: Tuple[int, ...], caps: Tuple[int | None, ...]) -> bool:
    return any(cap is not None and alpha[idx] > cap for idx, cap in enumerate(caps))


def _zero_deriv(config: FormulaConfig) -> DerivOrders:
    return tuple(0 for _ in range(config.y_count))


def _add_exp(left: Tuple[int, ...], right: Tuple[int, ...]) -> Tuple[int, ...]:
    if len(left) != len(right):
        raise ValueError("exponent tuples must have the same length")
    return tuple(int(a) + int(b) for a, b in zip(left, right))


def _defect_id(
    config: FormulaConfig,
    f_exp: Tuple[int, ...],
    gamma_exp: Tuple[int, ...],
) -> str:
    for exp, r in zip(f_exp, config.class_ranks):
        if exp:
            return f"f{r}"
    for exp, (r, s) in zip(gamma_exp, config.gamma_labels):
        if exp:
            return f"gamma{r}{s}"
    return "none"


def _cache_info_dict(info) -> dict[str, int]:
    return {
        "hits": int(info.hits),
        "misses": int(info.misses),
        "maxsize": -1 if info.maxsize is None else int(info.maxsize),
        "currsize": int(info.currsize),
    }
