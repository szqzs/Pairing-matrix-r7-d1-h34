"""Specialized all-a pairing evaluator for the one-defect c18 probe.

This is not the full production evaluator.  It implements the narrow formula
needed when the H62 test column has only ``a`` classes and the H34 c18 source
row has exactly one defect: either one ``f_r`` or one ``gamma_rs``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import permutations
import json
from math import factorial
import os
import sys
from time import perf_counter
from typing import Sequence, Tuple

from . import modular_formula
from .c18_basis import C18SourceRow, H62TestColumn
from .config import FormulaConfig, RANK7_G2_D1
from .exterior import ExteriorAlgebra
from .invariants import InvariantMonomial
from .mod_arith import require_prime
from .residue_functional import ResidueFunctional, clear_global_transition_spmat_cache
from .residue_transition import residue_monomial_mod, residue_poly_mod
from .root_system import type_a_roots
from .sparse_poly import SparsePoly, add, clean, constant, mul, pow_poly

DerivOrders = Tuple[int, ...]
DeltaKey = Tuple[int, ...]
DeltaPoly = dict[DeltaKey, SparsePoly]
DeltaKernelTerms = dict[Tuple[DeltaKey, DerivOrders], SparsePoly]
_MOMENT_CACHE_MAXSIZE = 262_144
_DEFAULT_SEMANTIC_CACHE_MAXSIZE = 8_192
_ARRAY_MUL_EMISSION_MAX = 40_000_000
_DENSE_REDUCER_RANGE_MAX = 5_000_000
_TauArrayHint = Tuple[Tuple[int, ...], Tuple[object, object]]


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


def c18_all_a_pairing_column_semantic(
    index: int,
    column: H62TestColumn,
    rows: Sequence[C18SourceRow],
    prime: int,
    *,
    config: FormulaConfig = RANK7_G2_D1,
    method: str = "moment",
    semantic_cache_maxsize: int | None = None,
    moment_cache_clear_size: int | None = None,
) -> Tuple[int, ...]:
    """Evaluate one all-a column, caching whole semantic row-column values."""

    evaluator = _cached_semantic_batch_evaluator(
        config,
        tuple(rows),
        require_prime(prime),
        method,
        _normalize_semantic_cache_maxsize(semantic_cache_maxsize),
        _normalize_optional_cache_size(moment_cache_clear_size),
    )
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


def all_a_pairing_total_batched_mod(
    config: FormulaConfig,
    total: InvariantMonomial,
    *,
    prime: int | None = None,
    beta_chunk_size: int = 2,
    max_chunk_terms: int = 200_000,
) -> int:
    """Evaluate a one-defect all-a total monomial with chunked beta batches."""

    p = require_prime(config.primary_prime if prime is None else prime)
    _validate_total(config, total)
    return _pairing_from_parts_batched(
        config,
        tuple(total.a_exp),
        tuple(total.f_exp),
        tuple(total.gamma_exp),
        p,
        beta_chunk_size=beta_chunk_size,
        max_chunk_terms=max_chunk_terms,
    )


def f_only_pairing_total_moment_mod(
    config: FormulaConfig,
    total: InvariantMonomial,
    *,
    prime: int | None = None,
) -> int:
    """Evaluate a top-degree f-only total monomial via residue moments.

    This is the correctness-first scaffold for c18 even rows tested against
    one-f H62 columns.  It supports no gamma factors and at most two f factors.
    """

    p = require_prime(config.primary_prime if prime is None else prime)
    _validate_f_only_total(config, total)
    return _pairing_from_parts_moment(
        config,
        tuple(total.a_exp),
        tuple(total.f_exp),
        tuple(total.gamma_exp),
        p,
    )


def f_only_pairing_total_batched_mod(
    config: FormulaConfig,
    total: InvariantMonomial,
    *,
    prime: int | None = None,
    beta_chunk_size: int = 2,
    max_chunk_terms: int = 200_000,
) -> int:
    """Evaluate a top-degree f-only total monomial with batched moments."""

    p = require_prime(config.primary_prime if prime is None else prime)
    _validate_f_only_total(config, total)
    return _pairing_from_parts_batched(
        config,
        tuple(total.a_exp),
        tuple(total.f_exp),
        tuple(total.gamma_exp),
        p,
        beta_chunk_size=beta_chunk_size,
        max_chunk_terms=max_chunk_terms,
    )


def f_gamma_pairing_total_moment_mod(
    config: FormulaConfig,
    total: InvariantMonomial,
    *,
    prime: int | None = None,
) -> int:
    """Evaluate a top-degree total monomial with one f and one gamma."""

    p = require_prime(config.primary_prime if prime is None else prime)
    _validate_f_gamma_total(config, total)
    return _pairing_from_parts_moment(
        config,
        tuple(total.a_exp),
        tuple(total.f_exp),
        tuple(total.gamma_exp),
        p,
    )


def f_gamma_pairing_total_batched_mod(
    config: FormulaConfig,
    total: InvariantMonomial,
    *,
    prime: int | None = None,
    beta_chunk_size: int = 2,
    max_chunk_terms: int = 200_000,
) -> int:
    """Evaluate a one-f/one-gamma total monomial with batched moments."""

    p = require_prime(config.primary_prime if prime is None else prime)
    _validate_f_gamma_total(config, total)
    return _pairing_from_parts_batched(
        config,
        tuple(total.a_exp),
        tuple(total.f_exp),
        tuple(total.gamma_exp),
        p,
        beta_chunk_size=beta_chunk_size,
        max_chunk_terms=max_chunk_terms,
    )


def f2_power_pairing_total_moment_mod(
    config: FormulaConfig,
    total: InvariantMonomial,
    *,
    prime: int | None = None,
) -> int:
    """Evaluate a top-degree total with arbitrary ``f2`` power.

    This wrapper is for the defect-rich H62 probes.  Powers of ``f2`` do not
    change the delta target; they only contribute the usual factorial scale.
    We still cap the non-f2/gamma part to shapes supported by the shared JK
    kernel.
    """

    p = require_prime(config.primary_prime if prime is None else prime)
    _validate_f2_power_total(config, total)
    return _pairing_from_parts_moment(
        config,
        tuple(total.a_exp),
        tuple(total.f_exp),
        tuple(total.gamma_exp),
        p,
    )


def f2_power_pairing_total_batched_mod(
    config: FormulaConfig,
    total: InvariantMonomial,
    *,
    prime: int | None = None,
    beta_chunk_size: int = 2,
    max_chunk_terms: int = 200_000,
) -> int:
    """Evaluate a top-degree arbitrary-``f2`` total with batched moments."""

    p = require_prime(config.primary_prime if prime is None else prime)
    _validate_f2_power_total(config, total)
    return _pairing_from_parts_batched(
        config,
        tuple(total.a_exp),
        tuple(total.f_exp),
        tuple(total.gamma_exp),
        p,
        beta_chunk_size=beta_chunk_size,
        max_chunk_terms=max_chunk_terms,
    )


def b_mask_pairing_total_moment_mod(
    config: FormulaConfig,
    *,
    a_exp: Sequence[int],
    f_exp: Sequence[int],
    b_mask: int,
    prime: int | None = None,
) -> int:
    """Evaluate a top-degree total with f defects and an explicit b-mask."""

    p = require_prime(config.primary_prime if prime is None else prime)
    a_tuple = tuple(int(item) for item in a_exp)
    f_tuple = tuple(int(item) for item in f_exp)
    _validate_b_mask_total(config, a_tuple, f_tuple, int(b_mask))
    return _pairing_from_parts_b_mask_moment(config, a_tuple, f_tuple, int(b_mask), p)


def b_mask_pairing_total_batched_mod(
    config: FormulaConfig,
    *,
    a_exp: Sequence[int],
    f_exp: Sequence[int],
    b_mask: int,
    prime: int | None = None,
    beta_chunk_size: int = 2,
    max_chunk_terms: int = 200_000,
) -> int:
    """Evaluate a top-degree f/b-mask total with batched moments."""

    p = require_prime(config.primary_prime if prime is None else prime)
    a_tuple = tuple(int(item) for item in a_exp)
    f_tuple = tuple(int(item) for item in f_exp)
    _validate_b_mask_total(config, a_tuple, f_tuple, int(b_mask))
    return _pairing_from_parts_b_mask_batched(
        config,
        a_tuple,
        f_tuple,
        int(b_mask),
        p,
        beta_chunk_size=beta_chunk_size,
        max_chunk_terms=max_chunk_terms,
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


@dataclass
class AllASemanticBatchEvaluator:
    """Column evaluator that memoizes whole ``(defect, total_a)`` values."""

    config: FormulaConfig
    rows: Tuple[C18SourceRow, ...]
    prime: int
    method: str = "moment"
    semantic_cache_maxsize: int | None = _DEFAULT_SEMANTIC_CACHE_MAXSIZE
    moment_cache_clear_size: int | None = None
    row_plans: Tuple["AllARowPlan", ...] = field(init=False)
    semantic_cache: OrderedDict[Tuple[str, Tuple[int, ...]], int] = field(
        default_factory=OrderedDict,
        init=False,
    )
    semantic_cache_hits: int = 0
    semantic_cache_misses: int = 0

    def __post_init__(self) -> None:
        p = require_prime(self.prime)
        method = self.method.lower()
        if method not in {"moment", "batched"}:
            raise ValueError("semantic all-a method must be moment or batched")
        object.__setattr__(self, "prime", p)
        object.__setattr__(self, "method", method)
        object.__setattr__(
            self,
            "semantic_cache_maxsize",
            _normalize_semantic_cache_maxsize(self.semantic_cache_maxsize),
        )
        object.__setattr__(
            self,
            "moment_cache_clear_size",
            _normalize_optional_cache_size(self.moment_cache_clear_size),
        )
        object.__setattr__(
            self,
            "row_plans",
            tuple(
                AllARowPlan.from_row(self.config, row_index, row)
                for row_index, row in enumerate(self.rows)
            ),
        )

    def column_vector(self, index: int, column: H62TestColumn) -> Tuple[int, ...]:
        del index
        if column.kind != "all_a":
            raise ValueError("AllASemanticBatchEvaluator expects an all-a test column")
        column_a_exp = tuple(column.monomial.a_exp)
        return tuple(
            self.semantic_value(plan, _add_exp(plan.a_exp, column_a_exp))
            for plan in self.row_plans
        )

    def semantic_value(
        self,
        plan: "AllARowPlan",
        total_a_exp: Tuple[int, ...],
    ) -> int:
        key = (plan.defect_id, tuple(total_a_exp))
        if key in self.semantic_cache:
            self.semantic_cache_hits += 1
            value = self.semantic_cache.pop(key)
            self.semantic_cache[key] = value
            return value

        self.semantic_cache_misses += 1
        if self.method == "batched":
            value = _pairing_from_parts_batched(
                self.config,
                key[1],
                plan.f_exp,
                plan.gamma_exp,
                self.prime,
            )
        else:
            value = _pairing_from_parts_moment(
                self.config,
                key[1],
                plan.f_exp,
                plan.gamma_exp,
                self.prime,
            )
        self._store_semantic_value(key, value)
        self._maybe_clear_moment_cache()
        return value

    def cache_info(self) -> dict[str, int]:
        return {
            "hits": int(self.semantic_cache_hits),
            "misses": int(self.semantic_cache_misses),
            "maxsize": -1
            if self.semantic_cache_maxsize is None
            else int(self.semantic_cache_maxsize),
            "currsize": int(len(self.semantic_cache)),
        }

    def _store_semantic_value(
        self,
        key: Tuple[str, Tuple[int, ...]],
        value: int,
    ) -> None:
        maxsize = self.semantic_cache_maxsize
        if maxsize == 0:
            return
        self.semantic_cache[key] = int(value) % self.prime
        if maxsize is not None:
            while len(self.semantic_cache) > maxsize:
                self.semantic_cache.popitem(last=False)

    def _maybe_clear_moment_cache(self) -> None:
        threshold = self.moment_cache_clear_size
        if threshold is None or threshold <= 0:
            return
        if _moment_mod.cache_info().currsize >= threshold:
            _moment_mod.cache_clear()


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
        "bounded_tau_arrays": _cache_info_dict(
            _tau_power_bounded_numpy_arrays_mod.cache_info()
        ),
        "bounded_tau_power": _cache_info_dict(_tau_power_bounded_mod.cache_info()),
        "kernel_terms": _cache_info_dict(_shared_kernel_terms.cache_info()),
        "moment": _cache_info_dict(_moment_mod.cache_info()),
        "monomial_residue": _cache_info_dict(_residue_monomial_cached.cache_info()),
        "residue_functional": _cache_info_dict(_residue_functional_cached.cache_info()),
        "semantic_batch_evaluator": _cache_info_dict(
            _cached_semantic_batch_evaluator.cache_info()
        ),
        "tau_power": _cache_info_dict(_tau_power_mod.cache_info()),
    }


def clear_all_a_caches() -> None:
    """Clear all-a evaluator caches for controlled profiling tests."""

    _cached_moment_batch_evaluator.cache_clear()
    _cached_semantic_batch_evaluator.cache_clear()
    _shared_kernel_terms.cache_clear()
    _moment_mod.cache_clear()
    _residue_monomial_cached.cache_clear()
    _residue_functional_cached.cache_clear()
    clear_global_transition_spmat_cache()
    _tau_power_bounded_numpy_arrays_mod.cache_clear()
    _tau_power_bounded_mod.cache_clear()
    _tau_power_mod.cache_clear()


@lru_cache(maxsize=16)
def _cached_moment_batch_evaluator(
    config: FormulaConfig,
    rows: Tuple[C18SourceRow, ...],
    prime: int,
) -> AllAMomentBatchEvaluator:
    return AllAMomentBatchEvaluator(config=config, rows=rows, prime=prime)


@lru_cache(maxsize=16)
def _cached_semantic_batch_evaluator(
    config: FormulaConfig,
    rows: Tuple[C18SourceRow, ...],
    prime: int,
    method: str,
    semantic_cache_maxsize: int | None,
    moment_cache_clear_size: int | None,
) -> AllASemanticBatchEvaluator:
    return AllASemanticBatchEvaluator(
        config=config,
        rows=rows,
        prime=prime,
        method=method,
        semantic_cache_maxsize=semantic_cache_maxsize,
        moment_cache_clear_size=moment_cache_clear_size,
    )


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


def _pairing_from_parts_batched(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
    f_exp: Tuple[int, ...],
    gamma_exp: Tuple[int, ...],
    prime: int,
    *,
    beta_chunk_size: int = 2,
    max_chunk_terms: int = 200_000,
) -> int:
    p = require_prime(prime)
    target_delta = tuple(int(exp) for exp in f_exp[1:])
    shared_terms = _shared_kernel_terms(config, target_delta, tuple(gamma_exp), p)
    if not shared_terms:
        return 0

    defect_id = _defect_id(config, tuple(f_exp), tuple(gamma_exp))
    tau_array_hint = _shared_tau_array_hint_for_terms(
        config,
        tuple(a_exp),
        shared_terms,
        p,
    )
    worker_count = _derivative_thread_count(len(shared_terms))
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            values = tuple(
                executor.map(
                    lambda item: _batched_shared_term_value(
                        config,
                        tuple(a_exp),
                        item[0],
                        item[1],
                        p,
                        beta_chunk_size=beta_chunk_size,
                        max_chunk_terms=max_chunk_terms,
                        defect_id=defect_id,
                        tau_array_hint=tau_array_hint,
                    ),
                    shared_terms,
                )
            )
        value = sum(values) % p
    else:
        value = 0
        for deriv_orders, shared_items in shared_terms:
            value = (
                value
                + _batched_shared_term_value(
                    config,
                    tuple(a_exp),
                    deriv_orders,
                    shared_items,
                    p,
                    beta_chunk_size=beta_chunk_size,
                    max_chunk_terms=max_chunk_terms,
                    defect_id=defect_id,
                    tau_array_hint=tau_array_hint,
                )
            ) % p

    scale_factor = int(config.collapsed_prefactor) % p
    for exp in f_exp:
        scale_factor = scale_factor * factorial(int(exp)) % p
    return scale_factor * value % p


def _pairing_from_parts_b_mask_moment(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
    f_exp: Tuple[int, ...],
    b_mask: int,
    prime: int,
) -> int:
    p = require_prime(prime)
    target_delta = tuple(int(exp) for exp in f_exp[1:])
    shared_terms = _shared_b_mask_kernel_terms(config, target_delta, int(b_mask), p)
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


def _pairing_from_parts_b_mask_batched(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
    f_exp: Tuple[int, ...],
    b_mask: int,
    prime: int,
    *,
    beta_chunk_size: int = 2,
    max_chunk_terms: int = 200_000,
) -> int:
    p = require_prime(prime)
    target_delta = tuple(int(exp) for exp in f_exp[1:])
    shared_terms = _shared_b_mask_kernel_terms(config, target_delta, int(b_mask), p)
    if not shared_terms:
        return 0

    tau_array_hint = _shared_tau_array_hint_for_terms(
        config,
        tuple(a_exp),
        shared_terms,
        p,
    )
    defect_id = f"bmask{int(b_mask)}"
    worker_count = _derivative_thread_count(len(shared_terms))
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            values = tuple(
                executor.map(
                    lambda item: _batched_shared_term_value(
                        config,
                        tuple(a_exp),
                        item[0],
                        item[1],
                        p,
                        beta_chunk_size=beta_chunk_size,
                        max_chunk_terms=max_chunk_terms,
                        defect_id=defect_id,
                        tau_array_hint=tau_array_hint,
                    ),
                    shared_terms,
                )
            )
        value = sum(values) % p
    else:
        value = 0
        for deriv_orders, shared_items in shared_terms:
            value = (
                value
                + _batched_shared_term_value(
                    config,
                    tuple(a_exp),
                    deriv_orders,
                    shared_items,
                    p,
                    beta_chunk_size=beta_chunk_size,
                    max_chunk_terms=max_chunk_terms,
                    defect_id=defect_id,
                    tau_array_hint=tau_array_hint,
                )
            ) % p

    scale_factor = int(config.collapsed_prefactor) % p
    for exp in f_exp:
        scale_factor = scale_factor * factorial(int(exp)) % p
    return scale_factor * value % p


def _batched_shared_term_value(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
    deriv_orders: DerivOrders,
    shared_items: Tuple[Tuple[Tuple[int, ...], int], ...],
    prime: int,
    *,
    beta_chunk_size: int,
    max_chunk_terms: int,
    defect_id: str | None,
    tau_array_hint: _TauArrayHint | None,
) -> int:
    return _batched_moment_sum_mod(
        config,
        tuple(a_exp),
        deriv_orders,
        shared_items,
        prime,
        beta_chunk_size=beta_chunk_size,
        max_chunk_terms=max_chunk_terms,
        defect_id=defect_id,
        tau_array_hint=tau_array_hint,
    )


def _derivative_thread_count(term_count: int) -> int:
    raw = os.environ.get("RANK7_JK_DERIVATIVE_THREADS")
    if raw is None or raw == "":
        return 1
    count = max(1, int(raw))
    return min(count, max(1, int(term_count)))

def _batched_moment_sum_mod(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
    deriv_orders: DerivOrders,
    shared_items: Tuple[Tuple[Tuple[int, ...], int], ...],
    prime: int,
    *,
    beta_chunk_size: int,
    max_chunk_terms: int,
    defect_id: str | None = None,
    tau_array_hint: _TauArrayHint | None = None,
) -> int:
    if config.y_count != 6 or prime > 2_000_000:
        return _scalar_moment_sum_mod(config, a_exp, deriv_orders, shared_items, prime)
    try:
        import numpy as np
    except ImportError:
        return _scalar_moment_sum_mod(config, a_exp, deriv_orders, shared_items, prime)

    residue_caps = _residue_exponent_caps(
        config.rank,
        tuple(int(item) for item in deriv_orders),
        config.root_denominator_power,
    )
    product_value = _kernel_product_moment_sum_mod(
        config,
        tuple(a_exp),
        deriv_orders,
        shared_items,
        residue_caps,
        prime,
        max_chunk_terms=max_chunk_terms,
        defect_id=defect_id,
        tau_array_hint=tau_array_hint,
    )
    if product_value is not None:
        return product_value

    beta_items = _valid_beta_cap_items(shared_items, residue_caps, prime)
    if not beta_items:
        return 0

    clusters = _beta_cap_clusters(beta_items, beta_chunk_size)
    functional = _residue_functional_cached(
        config.rank,
        tuple(deriv_orders),
        config.root_denominator_power,
        prime,
    )
    value = 0
    for cluster in clusters:
        value = (
            value
            + _batched_moment_cluster_mod(
                config,
                tuple(a_exp),
                deriv_orders,
                cluster,
                residue_caps,
                prime,
                functional,
                np,
                max_chunk_terms=max_chunk_terms,
                tau_array_hint=tau_array_hint,
            )
        ) % prime
    return value


def _kernel_product_moment_sum_mod(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
    deriv_orders: DerivOrders,
    shared_items: Tuple[Tuple[Tuple[int, ...], int], ...],
    residue_caps: Tuple[int, ...],
    prime: int,
    *,
    max_chunk_terms: int = 200_000,
    defect_id: str | None = None,
    tau_array_hint: _TauArrayHint | None = None,
) -> int | None:
    array_value = _kernel_product_moment_sum_arrays_mod(
        config,
        tuple(a_exp),
        deriv_orders,
        shared_items,
        residue_caps,
        prime,
        max_chunk_terms=max_chunk_terms,
        defect_id=defect_id,
        tau_array_hint=tau_array_hint,
    )
    if array_value is not None:
        return array_value

    kernel = clean(dict(shared_items), prime)
    if not kernel:
        return 0
    product = _bounded_tau_kernel_product_mod(
        config,
        tuple(a_exp),
        kernel,
        tuple(int(item) for item in residue_caps),
        prime,
    )
    if product is None:
        return None
    if not product:
        return 0
    functional = _residue_functional_cached(
        config.rank,
        tuple(deriv_orders),
        config.root_denominator_power,
        prime,
    )
    return functional.evaluate_poly_terms(product)


def _kernel_product_moment_sum_arrays_mod(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
    deriv_orders: DerivOrders,
    shared_items: Tuple[Tuple[Tuple[int, ...], int], ...],
    residue_caps: Tuple[int, ...],
    prime: int,
    *,
    max_chunk_terms: int,
    defect_id: str | None = None,
    tau_array_hint: _TauArrayHint | None = None,
) -> int | None:
    if config.y_count != 6 or prime > 2_000_000:
        return None
    try:
        import numpy as np
    except ImportError:
        return None

    beta_items = _valid_beta_cap_items(shared_items, residue_caps, prime)
    if not beta_items:
        return 0

    functional = _residue_functional_cached(
        config.rank,
        tuple(deriv_orders),
        config.root_denominator_power,
        prime,
    )
    profile = _new_product_profile(
        defect_id=defect_id,
        a_exp=a_exp,
        deriv_orders=deriv_orders,
        shared_items=shared_items,
        beta_items=beta_items,
        residue_caps=residue_caps,
    )
    start = perf_counter()
    value = _array_moment_cluster_mod(
        config,
        tuple(a_exp),
        deriv_orders,
        beta_items,
        residue_caps,
        prime,
        functional,
        np,
        max_chunk_terms=max_chunk_terms,
        profile=profile,
        depth=0,
        tau_array_hint=tau_array_hint,
    )
    if profile is not None:
        profile["elapsed_seconds"] = perf_counter() - start
        profile["result"] = int(value) % prime
        _emit_product_profile(profile)
    return value


def _bounded_tau_kernel_product_mod(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
    kernel: SparsePoly,
    caps: Tuple[int, ...],
    prime: int,
) -> SparsePoly | None:
    if config.y_count != len(caps):
        return None
    p = require_prime(prime)
    out = constant(config.y_count, 1, prime=p)
    factor_cache: dict[int, SparsePoly] = {}
    for r in _bounded_tau_factor_order(config, a_exp):
        factor = factor_cache.get(r)
        if factor is None:
            factor = dict(modular_formula.tau_mod(config, r, p))
            factor_cache[r] = factor
        out = _bounded_mul(out, factor, caps, p)
        if not out:
            return {}
    return _bounded_mul(out, kernel, caps, p)


def _shared_tau_array_hint_for_terms(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
    shared_terms: Tuple[
        Tuple[DerivOrders, Tuple[Tuple[Tuple[int, ...], int], ...]],
        ...,
    ],
    prime: int,
) -> _TauArrayHint | None:
    if config.y_count != 6 or prime > 2_000_000:
        return None
    caps_list = []
    for deriv_orders, shared_items in shared_terms:
        residue_caps = _residue_exponent_caps(
            config.rank,
            tuple(int(item) for item in deriv_orders),
            config.root_denominator_power,
        )
        beta_items = _valid_beta_cap_items(shared_items, residue_caps, prime)
        if beta_items:
            caps_list.append(_union_caps(item[2] for item in beta_items))
    if len(caps_list) <= 1:
        return None
    tau_caps = _union_caps(caps_list)
    tau_arrays = _tau_power_bounded_numpy_arrays_mod(
        config,
        tuple(a_exp),
        tau_caps,
        prime,
    )
    if tau_arrays is None:
        return None
    return tau_caps, tau_arrays


def _tau_arrays_from_hint(
    tau_array_hint: _TauArrayHint | None,
    requested_caps: Tuple[int, ...],
):
    if tau_array_hint is None:
        return None
    hint_caps, tau_arrays = tau_array_hint
    if len(hint_caps) != len(requested_caps):
        return None
    if any(int(hint_cap) < int(cap) for hint_cap, cap in zip(hint_caps, requested_caps)):
        return None
    return tau_arrays


def _batched_moment_cluster_mod(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
    deriv_orders: DerivOrders,
    beta_items,
    residue_caps: Tuple[int, ...],
    prime: int,
    functional: ResidueFunctional,
    np,
    *,
    max_chunk_terms: int,
    tau_array_hint: _TauArrayHint | None = None,
) -> int:
    return _array_moment_cluster_mod(
        config,
        a_exp,
        deriv_orders,
        beta_items,
        residue_caps,
        prime,
        functional,
        np,
        max_chunk_terms=max_chunk_terms,
        profile=None,
        depth=0,
        tau_array_hint=tau_array_hint,
    )


def _array_moment_cluster_mod(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
    deriv_orders: DerivOrders,
    beta_items,
    residue_caps: Tuple[int, ...],
    prime: int,
    functional: ResidueFunctional,
    np,
    *,
    max_chunk_terms: int,
    profile: dict[str, object] | None,
    depth: int,
    tau_array_hint: _TauArrayHint | None,
) -> int:
    cluster_caps = _union_caps(item[2] for item in beta_items)
    _profile_cluster_attempt(profile, beta_items, cluster_caps, depth)
    tau_start = perf_counter()
    tau_arrays = _tau_arrays_from_hint(tau_array_hint, cluster_caps)
    if tau_arrays is None:
        tau_arrays = _tau_power_bounded_numpy_arrays_mod(
            config,
            a_exp,
            cluster_caps,
            prime,
        )
    tau_elapsed = perf_counter() - tau_start
    _profile_tau_build(profile, tau_arrays, tau_elapsed)
    if tau_arrays is None:
        if len(beta_items) == 1:
            beta, coeff_mod, _caps = beta_items[0]
            scalar_start = perf_counter()
            value = (
                coeff_mod * _moment_mod(config, a_exp, beta, deriv_orders, prime)
            ) % prime
            _profile_scalar_fallback(profile, perf_counter() - scalar_start)
            return value
        left_items, right_items = _split_beta_items_by_cap_spread(beta_items)
        _profile_split(profile)
        return (
            _array_moment_cluster_mod(
                config,
                a_exp,
                deriv_orders,
                left_items,
                residue_caps,
                prime,
                functional,
                np,
                max_chunk_terms=max_chunk_terms,
                profile=profile,
                depth=depth + 1,
                tau_array_hint=tau_array_hint,
            )
            + _array_moment_cluster_mod(
                config,
                a_exp,
                deriv_orders,
                right_items,
                residue_caps,
                prime,
                functional,
                np,
                max_chunk_terms=max_chunk_terms,
                profile=profile,
                depth=depth + 1,
                tau_array_hint=tau_array_hint,
            )
        ) % prime

    alpha_terms, coeff_terms = tau_arrays
    if not coeff_terms.size:
        return 0

    reduce_start = perf_counter()
    reduced = _reduce_shifted_beta_cluster_6(
        alpha_terms,
        coeff_terms,
        beta_items,
        residue_caps,
        prime,
        np,
    )
    reduce_elapsed = perf_counter() - reduce_start
    if reduced is not None:
        reduced_alpha, reduced_coeff = reduced
        _profile_reduction(profile, reduced_coeff.size, reduce_elapsed)
        if not reduced_coeff.size:
            return 0
        residue_start = perf_counter()
        value = functional.evaluate_unique_array_terms(reduced_alpha, reduced_coeff)
        _profile_residue(profile, perf_counter() - residue_start)
        return value

    _profile_reduce_fallback(profile, reduce_elapsed)
    if len(beta_items) > 1:
        left_items, right_items = _split_beta_items_by_cap_spread(beta_items)
        _profile_split(profile)
        return (
            _array_moment_cluster_mod(
                config,
                a_exp,
                deriv_orders,
                left_items,
                residue_caps,
                prime,
                functional,
                np,
                max_chunk_terms=max_chunk_terms,
                profile=profile,
                depth=depth + 1,
                tau_array_hint=tau_array_hint,
            )
            + _array_moment_cluster_mod(
                config,
                a_exp,
                deriv_orders,
                right_items,
                residue_caps,
                prime,
                functional,
                np,
                max_chunk_terms=max_chunk_terms,
                profile=profile,
                depth=depth + 1,
                tau_array_hint=tau_array_hint,
            )
        ) % prime

    alpha_i64 = alpha_terms.astype(np.int64, copy=False)
    value = 0
    alpha_chunks = []
    coeff_chunks = []
    pending_terms = 0

    for beta, coeff_mod, caps in beta_items:
        beta_array = np.asarray(beta, dtype=np.int64)
        limits = np.asarray(caps, dtype=np.int64)
        mask = np.all(alpha_i64 <= limits, axis=1)
        if not np.any(mask):
            continue
        shifted = (alpha_i64[mask] + beta_array).astype(np.uint16, copy=False)
        scaled_coeff = (coeff_terms[mask] * coeff_mod) % prime
        alpha_chunks.append(shifted)
        coeff_chunks.append(scaled_coeff.astype(np.int64, copy=False))
        pending_terms += int(scaled_coeff.size)
        if pending_terms >= max_chunk_terms:
            chunk_start = perf_counter()
            value = (
                value
                + _evaluate_array_chunks(functional, alpha_chunks, coeff_chunks, np)
            ) % prime
            _profile_manual_chunk(profile, pending_terms, perf_counter() - chunk_start)
            alpha_chunks = []
            coeff_chunks = []
            pending_terms = 0

    if alpha_chunks:
        chunk_start = perf_counter()
        value = (
            value
            + _evaluate_array_chunks(functional, alpha_chunks, coeff_chunks, np)
        ) % prime
        _profile_manual_chunk(profile, pending_terms, perf_counter() - chunk_start)
    return value


def _valid_beta_cap_items(
    shared_items: Tuple[Tuple[Tuple[int, ...], int], ...],
    residue_caps: Tuple[int, ...],
    prime: int,
):
    out = []
    for beta, coeff in shared_items:
        coeff_mod = int(coeff) % prime
        if not coeff_mod:
            continue
        caps = tuple(int(cap) - int(beta_i) for cap, beta_i in zip(residue_caps, beta))
        if min(caps) < 0:
            continue
        out.append((tuple(int(item) for item in beta), coeff_mod, caps))
    return tuple(out)


def _split_beta_items_by_cap_spread(beta_items):
    if len(beta_items) <= 1:
        return tuple(beta_items), ()
    width = len(beta_items[0][2])
    spreads = []
    for idx in range(width):
        values = tuple(int(item[2][idx]) for item in beta_items)
        spreads.append(max(values) - min(values))
    split_idx = max(range(width), key=lambda idx: (spreads[idx], idx))
    ordered = tuple(sorted(beta_items, key=lambda item: (item[2][split_idx], item[2])))
    mid = len(ordered) // 2
    return ordered[:mid], ordered[mid:]


def _new_product_profile(
    *,
    defect_id: str | None,
    a_exp: Tuple[int, ...],
    deriv_orders: DerivOrders,
    shared_items: Tuple[Tuple[Tuple[int, ...], int], ...],
    beta_items,
    residue_caps: Tuple[int, ...],
) -> dict[str, object] | None:
    if not _product_profile_enabled():
        return None
    return {
        "event": "all_a_kernel_product",
        "defect": "" if defect_id is None else str(defect_id),
        "a_exp": [int(item) for item in a_exp],
        "deriv_orders": [int(item) for item in deriv_orders],
        "kernel_items": int(len(shared_items)),
        "valid_beta_count": int(len(beta_items)),
        "residue_caps": [int(item) for item in residue_caps],
        "clusters": 0,
        "splits": 0,
        "scalar_fallbacks": 0,
        "reduce_fallbacks": 0,
        "manual_chunk_flushes": 0,
        "manual_chunk_terms": 0,
        "tau_builds": 0,
        "tau_build_failures": 0,
        "tau_build_seconds": 0.0,
        "max_tau_terms": 0,
        "total_tau_terms": 0,
        "reduce_seconds": 0.0,
        "max_reduced_terms": 0,
        "total_reduced_terms": 0,
        "residue_seconds": 0.0,
        "scalar_seconds": 0.0,
        "manual_chunk_seconds": 0.0,
        "max_depth": 0,
        "max_cluster_beta_count": 0,
        "tau_caps_attempts": [],
    }


def _product_profile_enabled() -> bool:
    raw = os.environ.get("RANK7_JK_PRODUCT_PROFILE")
    return raw not in {None, "", "0", "false", "False", "FALSE"}


def _emit_product_profile(profile: dict[str, object]) -> None:
    payload = json.dumps(profile, sort_keys=True)
    raw = os.environ.get("RANK7_JK_PRODUCT_PROFILE")
    if raw in {None, "", "1", "true", "True", "TRUE", "stderr"}:
        print(payload, file=sys.stderr)
        return
    with open(raw, "a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.write("\n")


def _profile_cluster_attempt(
    profile: dict[str, object] | None,
    beta_items,
    cluster_caps: Tuple[int, ...],
    depth: int,
) -> None:
    if profile is None:
        return
    profile["clusters"] = int(profile["clusters"]) + 1
    profile["max_depth"] = max(int(profile["max_depth"]), int(depth))
    profile["max_cluster_beta_count"] = max(
        int(profile["max_cluster_beta_count"]),
        int(len(beta_items)),
    )
    attempts = profile["tau_caps_attempts"]
    if isinstance(attempts, list) and len(attempts) < 16:
        attempts.append(
            {
                "depth": int(depth),
                "beta_count": int(len(beta_items)),
                "caps": [int(item) for item in cluster_caps],
            }
        )


def _profile_tau_build(
    profile: dict[str, object] | None,
    tau_arrays,
    elapsed_seconds: float,
) -> None:
    if profile is None:
        return
    profile["tau_builds"] = int(profile["tau_builds"]) + 1
    profile["tau_build_seconds"] = float(profile["tau_build_seconds"]) + elapsed_seconds
    if tau_arrays is None:
        profile["tau_build_failures"] = int(profile["tau_build_failures"]) + 1
        return
    _alpha_terms, coeff_terms = tau_arrays
    term_count = int(coeff_terms.size)
    profile["total_tau_terms"] = int(profile["total_tau_terms"]) + term_count
    profile["max_tau_terms"] = max(int(profile["max_tau_terms"]), term_count)


def _profile_split(profile: dict[str, object] | None) -> None:
    if profile is None:
        return
    profile["splits"] = int(profile["splits"]) + 1


def _profile_scalar_fallback(
    profile: dict[str, object] | None,
    elapsed_seconds: float,
) -> None:
    if profile is None:
        return
    profile["scalar_fallbacks"] = int(profile["scalar_fallbacks"]) + 1
    profile["scalar_seconds"] = float(profile["scalar_seconds"]) + elapsed_seconds


def _profile_reduction(
    profile: dict[str, object] | None,
    reduced_terms: int,
    elapsed_seconds: float,
) -> None:
    if profile is None:
        return
    term_count = int(reduced_terms)
    profile["reduce_seconds"] = float(profile["reduce_seconds"]) + elapsed_seconds
    profile["total_reduced_terms"] = int(profile["total_reduced_terms"]) + term_count
    profile["max_reduced_terms"] = max(int(profile["max_reduced_terms"]), term_count)


def _profile_reduce_fallback(
    profile: dict[str, object] | None,
    elapsed_seconds: float,
) -> None:
    if profile is None:
        return
    profile["reduce_fallbacks"] = int(profile["reduce_fallbacks"]) + 1
    profile["reduce_seconds"] = float(profile["reduce_seconds"]) + elapsed_seconds


def _profile_residue(
    profile: dict[str, object] | None,
    elapsed_seconds: float,
) -> None:
    if profile is None:
        return
    profile["residue_seconds"] = float(profile["residue_seconds"]) + elapsed_seconds


def _profile_manual_chunk(
    profile: dict[str, object] | None,
    term_count: int,
    elapsed_seconds: float,
) -> None:
    if profile is None:
        return
    profile["manual_chunk_flushes"] = int(profile["manual_chunk_flushes"]) + 1
    profile["manual_chunk_terms"] = int(profile["manual_chunk_terms"]) + int(term_count)
    profile["manual_chunk_seconds"] = (
        float(profile["manual_chunk_seconds"]) + elapsed_seconds
    )


def _reduce_shifted_beta_cluster_6(
    alpha_terms,
    coeff_terms,
    beta_items,
    residue_caps: Tuple[int, ...],
    prime: int,
    np,
):
    dense = _reduce_shifted_beta_cluster_6_dense_grid(
        alpha_terms,
        coeff_terms,
        beta_items,
        residue_caps,
        prime,
        np,
    )
    if dense is not None:
        return dense
    return _reduce_shifted_beta_cluster_6_bincount(
        alpha_terms,
        coeff_terms,
        beta_items,
        residue_caps,
        prime,
        np,
    )


def _reduce_shifted_beta_cluster_6_dense_grid(
    alpha_terms,
    coeff_terms,
    beta_items,
    residue_caps: Tuple[int, ...],
    prime: int,
    np,
):
    if _dense_reducer_disabled():
        return None
    if alpha_terms.ndim != 2 or alpha_terms.shape[1] != 6:
        return None
    if not beta_items:
        return (
            np.empty((0, 6), dtype=np.uint16),
            np.empty(0, dtype=np.int64),
        )
    if not coeff_terms.size:
        return (
            np.empty((0, 6), dtype=np.uint16),
            np.empty(0, dtype=np.int64),
        )

    alpha_degree = int(alpha_terms[0].sum())
    alpha_i64 = alpha_terms.astype(np.int64, copy=False)
    alpha_coord_max = alpha_i64.max(axis=0)
    by_degree: dict[int, list[tuple[Tuple[int, ...], int, Tuple[int, ...]]]] = {}
    for item in beta_items:
        beta, _coeff_mod, _caps = item
        degree = alpha_degree + sum(int(beta_i) for beta_i in beta)
        by_degree.setdefault(degree, []).append(item)

    grid_cache = {}
    alpha_chunks = []
    coeff_chunks = []
    for degree, items in by_degree.items():
        context = _homogeneous_final_code_context_6(residue_caps, degree, np)
        if context is None:
            return None
        drop_idx, keep_indices, keep_caps, strides, range_size = context
        if range_size > _DENSE_REDUCER_RANGE_MAX:
            return None
        if any(int(caps[drop_idx]) < int(alpha_coord_max[drop_idx]) for _b, _c, caps in items):
            return None

        cache_key = (
            int(drop_idx),
            tuple(int(idx) for idx in keep_indices),
            tuple(int(cap) for cap in keep_caps),
        )
        cached_grid = grid_cache.get(cache_key)
        if cached_grid is None:
            tau_grid = _dense_tau_grid_for_context(
                alpha_i64,
                coeff_terms,
                context,
                prime,
                np,
            )
            grid_cache[cache_key] = tau_grid
        else:
            tau_grid = cached_grid

        out_grid = np.zeros(tuple(int(cap) + 1 for cap in keep_caps), dtype=np.int64)
        for item_idx, (beta, coeff_mod, caps) in enumerate(items):
            beta_keep = tuple(int(beta[idx]) for idx in keep_indices)
            source_limits = []
            for global_idx, cap in zip(keep_indices, keep_caps):
                source_limits.append(
                    min(
                        int(caps[global_idx]),
                        int(alpha_coord_max[global_idx]),
                        int(cap),
                    )
                )
            if any(limit < 0 for limit in source_limits):
                continue
            source_slices = tuple(slice(0, limit + 1) for limit in source_limits)
            target_slices = tuple(
                slice(shift, shift + limit + 1)
                for shift, limit in zip(beta_keep, source_limits)
            )
            out_grid[target_slices] += (
                tau_grid[source_slices] * int(coeff_mod)
            ) % prime
            if item_idx % 16 == 15:
                out_grid %= prime
        out_grid %= prime

        flat = out_grid.ravel()
        keep = np.nonzero(flat)[0]
        if not keep.size:
            continue
        alpha_keep_out = _decode_final_keep_codes_6(keep, context, np)
        alpha = np.empty((keep.size, 6), dtype=np.uint16)
        keep_sum = np.zeros(keep.size, dtype=np.int64)
        for local_idx, global_idx in enumerate(keep_indices):
            values = alpha_keep_out[:, local_idx]
            alpha[:, global_idx] = values
            keep_sum += values
        drop_values = int(degree) - keep_sum
        if np.any(drop_values < 0) or np.any(drop_values > residue_caps[drop_idx]):
            return None
        alpha[:, drop_idx] = drop_values.astype(np.uint16)
        alpha_chunks.append(alpha)
        coeff_chunks.append(flat[keep].astype(np.int64, copy=False))

    if not alpha_chunks:
        return (
            np.empty((0, 6), dtype=np.uint16),
            np.empty(0, dtype=np.int64),
        )
    if len(alpha_chunks) == 1:
        return alpha_chunks[0], coeff_chunks[0]
    return np.concatenate(alpha_chunks, axis=0), np.concatenate(coeff_chunks)


def _dense_tau_grid_for_context(
    alpha_i64,
    coeff_terms,
    context,
    prime: int,
    np,
):
    _drop_idx, keep_indices, keep_caps, strides, _range_size = context
    shape = tuple(int(cap) + 1 for cap in keep_caps)
    grid = np.zeros(shape, dtype=np.int64)
    alpha_keep = alpha_i64[:, keep_indices]
    keep_mask = np.all(alpha_keep <= keep_caps, axis=1)
    if not np.any(keep_mask):
        return grid
    alpha_keep = alpha_keep[keep_mask]
    coeff_terms = coeff_terms[keep_mask]
    codes = (alpha_keep @ strides).astype(np.int64, copy=False)
    grid.ravel()[codes] = coeff_terms.astype(np.int64, copy=False) % prime
    return grid


def _dense_reducer_disabled() -> bool:
    raw = os.environ.get("RANK7_JK_DISABLE_DENSE_REDUCER")
    return raw not in {None, "", "0", "false", "False", "FALSE"}


def _reduce_shifted_beta_cluster_6_bincount(
    alpha_terms,
    coeff_terms,
    beta_items,
    residue_caps: Tuple[int, ...],
    prime: int,
    np,
):
    if alpha_terms.ndim != 2 or alpha_terms.shape[1] != 6:
        return None
    if not beta_items:
        return (
            np.empty((0, 6), dtype=np.uint16),
            np.empty(0, dtype=np.int64),
        )

    alpha_degree = int(alpha_terms[0].sum()) if alpha_terms.shape[0] else 0
    by_degree: dict[int, list[tuple[Tuple[int, ...], int, Tuple[int, ...]]]] = {}
    for item in beta_items:
        beta, _coeff_mod, _caps = item
        degree = alpha_degree + sum(int(beta_i) for beta_i in beta)
        by_degree.setdefault(degree, []).append(item)

    alpha_i64 = alpha_terms.astype(np.int64, copy=False)
    if alpha_i64.shape[0]:
        alpha_coord_max = alpha_i64.max(axis=0)
    else:
        alpha_coord_max = np.zeros(6, dtype=np.int64)
    alpha_chunks = []
    coeff_chunks = []
    for degree, items in by_degree.items():
        context = _homogeneous_final_code_context_6(residue_caps, degree, np)
        if context is None:
            return None
        drop_idx, keep_indices, keep_caps, strides, range_size = context
        sums = np.zeros(range_size, dtype=np.int64)
        alpha_keep = alpha_i64[:, keep_indices]
        for item_idx, (beta, coeff_mod, caps) in enumerate(items):
            limits = np.asarray(caps, dtype=np.int64)
            active = np.nonzero(limits < alpha_coord_max)[0]
            if active.size:
                mask = np.all(alpha_i64[:, active] <= limits[active], axis=1)
                if not np.any(mask):
                    continue
                selected_keep = alpha_keep[mask]
                selected_coeff = coeff_terms[mask]
            else:
                selected_keep = alpha_keep
                selected_coeff = coeff_terms
            beta_keep = np.asarray(
                tuple(int(beta[idx]) for idx in keep_indices),
                dtype=np.int64,
            )
            shifted_keep = selected_keep + beta_keep
            if np.any(shifted_keep > keep_caps):
                return None
            codes = (shifted_keep @ strides).astype(np.int64, copy=False)
            scaled = (selected_coeff * int(coeff_mod)) % prime
            # The small-prime production backend keeps these sums below 2^53
            # per cluster, so float64 bincount is exact before the mod reduce.
            sums += np.bincount(
                codes,
                weights=scaled.astype(np.float64, copy=False),
                minlength=range_size,
            ).astype(np.int64, copy=False)
            if item_idx % 16 == 15:
                sums %= prime
        sums %= prime
        keep = np.nonzero(sums)[0]
        if not keep.size:
            continue
        alpha_keep_out = _decode_final_keep_codes_6(keep, context, np)
        alpha = np.empty((keep.size, 6), dtype=np.uint16)
        keep_sum = np.zeros(keep.size, dtype=np.int64)
        for local_idx, global_idx in enumerate(keep_indices):
            values = alpha_keep_out[:, local_idx]
            alpha[:, global_idx] = values
            keep_sum += values
        drop_values = int(degree) - keep_sum
        if np.any(drop_values < 0) or np.any(drop_values > residue_caps[drop_idx]):
            return None
        alpha[:, drop_idx] = drop_values.astype(np.uint16)
        alpha_chunks.append(alpha)
        coeff_chunks.append(sums[keep].astype(np.int64, copy=False))

    if not alpha_chunks:
        return (
            np.empty((0, 6), dtype=np.uint16),
            np.empty(0, dtype=np.int64),
        )
    if len(alpha_chunks) == 1:
        return alpha_chunks[0], coeff_chunks[0]
    return np.concatenate(alpha_chunks, axis=0), np.concatenate(coeff_chunks)


def _homogeneous_final_code_context_6(
    caps: Tuple[int, ...],
    degree: int,
    np,
):
    cap_array = np.asarray(tuple(int(cap) for cap in caps), dtype=np.int64)
    candidates = [idx for idx, cap in enumerate(cap_array) if cap >= int(degree)]
    if not candidates:
        return None
    best = None
    for drop_idx in candidates:
        keep_indices = tuple(idx for idx in range(6) if idx != drop_idx)
        keep_caps = cap_array[list(keep_indices)]
        bases = keep_caps + 1
        range_size = int(np.prod(bases, dtype=object))
        if range_size > 20_000_000:
            continue
        if best is None or range_size < best[0]:
            best = (range_size, int(drop_idx), keep_indices, keep_caps)
    if best is None:
        return None
    range_size, drop_idx, keep_indices, keep_caps = best
    strides = np.ones(len(keep_indices), dtype=np.int64)
    bases = keep_caps + 1
    for idx in range(len(keep_indices) - 2, -1, -1):
        strides[idx] = strides[idx + 1] * bases[idx + 1]
    return drop_idx, keep_indices, keep_caps, strides, int(range_size)


def _decode_final_keep_codes_6(codes, context, np):
    _drop_idx, keep_indices, _keep_caps, strides, _range_size = context
    alpha_keep = np.empty((codes.size, len(keep_indices)), dtype=np.uint16)
    remainder = codes.astype(np.int64, copy=True)
    for idx in range(len(keep_indices)):
        alpha_keep[:, idx] = (remainder // strides[idx]).astype(np.uint16)
        remainder %= strides[idx]
    return alpha_keep


def _beta_cap_clusters(beta_items, beta_chunk_size: int):
    if not beta_items:
        return ()
    chunk_size = max(1, int(beta_chunk_size))
    ordered = sorted(
        beta_items,
        key=lambda item: (
            item[2][-1],
            item[2][-2],
            item[2][-3],
            item[2],
        ),
    )
    return tuple(
        tuple(ordered[start : start + chunk_size])
        for start in range(0, len(ordered), chunk_size)
    )


def _union_caps(caps_iter) -> Tuple[int, ...]:
    caps_list = list(caps_iter)
    if not caps_list:
        return ()
    width = len(caps_list[0])
    return tuple(max(caps[idx] for caps in caps_list) for idx in range(width))


def _scalar_moment_sum_mod(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
    deriv_orders: DerivOrders,
    shared_items: Tuple[Tuple[Tuple[int, ...], int], ...],
    prime: int,
) -> int:
    value = 0
    for beta, coeff in shared_items:
        coeff_mod = int(coeff) % prime
        if coeff_mod:
            value = (
                value
                + coeff_mod * _moment_mod(config, tuple(a_exp), beta, deriv_orders, prime)
            ) % prime
    return value


def _evaluate_array_chunks(functional: ResidueFunctional, alpha_chunks, coeff_chunks, np) -> int:
    if not alpha_chunks:
        return 0
    return functional.evaluate_array_terms(
        np.concatenate(alpha_chunks, axis=0),
        np.concatenate(coeff_chunks),
    )


@lru_cache(maxsize=_MOMENT_CACHE_MAXSIZE)
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
    functional = _residue_functional_cached(
        config.rank,
        tuple(deriv_orders),
        config.root_denominator_power,
        p,
    )
    tau_arrays = _tau_power_bounded_numpy_arrays_mod(config, tuple(a_exp), caps, p)
    if tau_arrays is not None:
        alpha_terms, coeff_terms = tau_arrays
        shifted_arrays = _shift_alpha_array(alpha_terms, beta)
        if shifted_arrays is not None:
            return functional.evaluate_array_terms(shifted_arrays, coeff_terms)

    tau_poly = dict(_tau_power_bounded_mod(config, tuple(a_exp), caps, p))
    if not tau_poly:
        return 0
    return functional.evaluate_poly_terms(_shift_poly(tau_poly, beta, p))


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


@lru_cache(maxsize=32)
def _residue_functional_cached(
    rank: int,
    deriv_orders: DerivOrders,
    root_power: int,
    prime: int,
) -> ResidueFunctional:
    return ResidueFunctional(
        rank=int(rank),
        derivative_orders=tuple(int(item) for item in deriv_orders),
        root_power=int(root_power),
        prime=require_prime(prime),
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


def _validate_f_only_total(config: FormulaConfig, total: InvariantMonomial) -> None:
    if total.rank != config.rank:
        raise ValueError("pairing monomial rank must match the formula config")
    if total.ordinary_degree != config.top_degree:
        raise ValueError("pairing monomial must have top ordinary degree")
    if any(total.gamma_exp):
        raise NotImplementedError("the f-only scaffold does not support gamma factors")
    defect_count = sum(total.f_exp)
    if defect_count < 1 or defect_count > 2:
        raise NotImplementedError("the f-only scaffold supports one or two f factors")
    if sum(total.f_exp[1:]) > 2:
        raise NotImplementedError("the f-only scaffold supports at most two delta defects")


def _validate_f_gamma_total(config: FormulaConfig, total: InvariantMonomial) -> None:
    if total.rank != config.rank:
        raise ValueError("pairing monomial rank must match the formula config")
    if total.ordinary_degree != config.top_degree:
        raise ValueError("pairing monomial must have top ordinary degree")
    if sum(total.f_exp) != 1 or sum(total.gamma_exp) != 1:
        raise NotImplementedError("the f-gamma scaffold supports exactly one f and one gamma")
    if sum(total.f_exp[1:]) > 1:
        raise NotImplementedError("the f-gamma scaffold supports at most one delta defect")


def _validate_f2_power_total(config: FormulaConfig, total: InvariantMonomial) -> None:
    if total.rank != config.rank:
        raise ValueError("pairing monomial rank must match the formula config")
    if total.ordinary_degree != config.top_degree:
        raise ValueError("pairing monomial must have top ordinary degree")
    if total.f_exp[0] < 1:
        raise NotImplementedError("the f2-power scaffold requires at least one f2 factor")
    if sum(total.gamma_exp) > 1:
        raise NotImplementedError("the f2-power scaffold supports at most one gamma factor")
    if sum(total.f_exp[1:]) > 1:
        raise NotImplementedError("the f2-power scaffold supports at most one non-f2 delta")


def _validate_b_mask_total(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
    f_exp: Tuple[int, ...],
    b_mask: int,
) -> None:
    if len(a_exp) != len(config.class_ranks):
        raise ValueError("a exponent length does not match the formula config")
    if len(f_exp) != len(config.class_ranks):
        raise ValueError("f exponent length does not match the formula config")
    if any(item < 0 for item in a_exp) or any(item < 0 for item in f_exp):
        raise ValueError("a and f exponents must be nonnegative")
    labels = ExteriorAlgebra(config).labels_from_mask(int(b_mask))
    if len(labels) != 2:
        raise NotImplementedError("the direct b-mask scaffold expects exactly two b labels")
    if sum(f_exp) != 1:
        raise NotImplementedError("the direct b-mask scaffold expects exactly one f factor")
    if sum(f_exp[1:]) > 1:
        raise NotImplementedError("the direct b-mask scaffold supports at most one delta defect")

    degree = 0
    for exp, r in zip(a_exp, config.class_ranks):
        degree += int(exp) * 2 * r
    for exp, r in zip(f_exp, config.class_ranks):
        degree += int(exp) * (2 * r - 2)
    for r, _j in labels:
        degree += 2 * r - 1
    if degree != config.top_degree:
        raise ValueError("b-mask pairing monomial must have top ordinary degree")


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
    if gamma_count <= 1 and sum(target_delta) <= 1:
        return _gamma_delta_kernel_terms(config, target_delta, gamma_exp, p)
    raise NotImplementedError(
        "the shared evaluator currently supports at most one gamma and one delta defect"
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
        if sum(target_delta) <= 2:
            return _even_kernel_terms_delta_generic(config, target_delta, p)
        raise NotImplementedError("only up to two delta defects are supported")

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
def _even_kernel_terms_delta_generic(
    config: FormulaConfig,
    target_delta: DeltaKey,
    prime: int,
) -> Tuple[Tuple[DerivOrders, Tuple[Tuple[Tuple[int, ...], int], ...]], ...]:
    return tuple(
        (deriv, items)
        for delta, deriv, items in _even_kernel_terms_delta_all(config, target_delta, prime)
        if delta == target_delta
    )


@lru_cache(maxsize=None)
def _even_kernel_terms_delta_all(
    config: FormulaConfig,
    target_delta: DeltaKey,
    prime: int,
) -> Tuple[Tuple[DeltaKey, DerivOrders, Tuple[Tuple[Tuple[int, ...], int], ...]], ...]:
    p = require_prime(prime)
    if len(target_delta) != len(config.delta_ranks):
        raise ValueError("target delta length does not match the formula config")
    if any(item < 0 for item in target_delta):
        raise ValueError("target delta entries must be nonnegative")
    if sum(target_delta) > 2:
        raise NotImplementedError("generic f-only kernel is currently capped at delta degree 2")

    linear = {
        _delta_unit(config, idx): dict(
            modular_formula.c_tilde_delta_coeff_mod(config, delta_rank, p)
        )
        for idx, delta_rank in enumerate(config.delta_ranks)
        if target_delta[idx]
    }
    exp_delta = _delta_poly_exp_linear_mod(config, linear, target_delta, p)
    det_delta = {
        delta: dict(poly)
        for delta, poly in _det_ratio_delta_power_mod(
            config,
            target_delta,
            config.genus,
            p,
        )
    }
    terms = _denominator_taylor_terms_mod(config, target_delta, p)
    terms = _kernel_terms_mul_delta_mod(config, terms, exp_delta, target_delta, p)
    terms = _kernel_terms_mul_delta_mod(config, terms, det_delta, target_delta, p)
    return tuple(
        (delta, deriv, tuple(sorted(poly.items())))
        for (delta, deriv), poly in sorted(terms.items())
        if poly
    )


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
def _gamma_delta_kernel_terms(
    config: FormulaConfig,
    target_delta: DeltaKey,
    gamma_exp: Tuple[int, ...],
    prime: int,
) -> Tuple[Tuple[DerivOrders, Tuple[Tuple[Tuple[int, ...], int], ...]], ...]:
    p = require_prime(prime)
    b_delta = {
        delta: dict(poly_items)
        for delta, poly_items in _gamma_hat_delta_mod(config, gamma_exp, target_delta, p)
    }
    if not b_delta:
        return ()
    out = []
    for kernel_delta, deriv_orders, kernel_items in _even_kernel_terms_delta_all(
        config,
        target_delta,
        p,
    ):
        gamma_delta = _delta_sub(target_delta, kernel_delta)
        if gamma_delta is None or gamma_delta not in b_delta:
            continue
        shared_poly = mul(dict(kernel_items), b_delta[gamma_delta], prime=p)
        if shared_poly:
            out.append((deriv_orders, tuple(sorted(shared_poly.items()))))
    return tuple(out)


@lru_cache(maxsize=None)
def _shared_b_mask_kernel_terms(
    config: FormulaConfig,
    target_delta: DeltaKey,
    b_mask: int,
    prime: int,
) -> Tuple[Tuple[DerivOrders, Tuple[Tuple[Tuple[int, ...], int], ...]], ...]:
    p = require_prime(prime)
    b_delta = {
        delta: dict(poly_items)
        for delta, poly_items in _b_hat_mask_delta_mod(config, int(b_mask), target_delta, p)
    }
    if not b_delta:
        return ()
    out = []
    for kernel_delta, deriv_orders, kernel_items in _even_kernel_terms_delta_all(
        config,
        target_delta,
        p,
    ):
        mask_delta = _delta_sub(target_delta, kernel_delta)
        if mask_delta is None or mask_delta not in b_delta:
            continue
        shared_poly = mul(dict(kernel_items), b_delta[mask_delta], prime=p)
        if shared_poly:
            out.append((deriv_orders, tuple(sorted(shared_poly.items()))))
    return tuple(out)


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


def _delta_zero(config: FormulaConfig) -> DeltaKey:
    return tuple(0 for _ in config.delta_ranks)


def _delta_unit(config: FormulaConfig, axis: int) -> DeltaKey:
    out = [0 for _ in config.delta_ranks]
    out[int(axis)] = 1
    return tuple(out)


def _delta_leq(left: DeltaKey, right: DeltaKey) -> bool:
    if len(left) != len(right):
        raise ValueError("delta keys must have the same length")
    return all(left[idx] <= right[idx] for idx in range(len(left)))


def _delta_add(left: DeltaKey, right: DeltaKey) -> DeltaKey:
    if len(left) != len(right):
        raise ValueError("delta keys must have the same length")
    return tuple(left[idx] + right[idx] for idx in range(len(left)))


def _delta_sub(left: DeltaKey, right: DeltaKey) -> DeltaKey | None:
    if len(left) != len(right):
        raise ValueError("delta keys must have the same length")
    out = tuple(left[idx] - right[idx] for idx in range(len(left)))
    if any(item < 0 for item in out):
        return None
    return out


def _delta_poly_add(
    left: DeltaPoly,
    right: DeltaPoly,
    prime: int,
    *,
    scale: int = 1,
) -> DeltaPoly:
    p = require_prime(prime)
    out: DeltaPoly = {delta: dict(poly) for delta, poly in left.items()}
    scale_mod = int(scale) % p
    if not scale_mod:
        return {delta: clean(poly, p) for delta, poly in out.items() if clean(poly, p)}
    for delta, poly in right.items():
        out[delta] = add(out.get(delta, {}), poly, prime=p, scale=scale_mod)
        if not out[delta]:
            del out[delta]
    return out


def _delta_poly_scale(poly: DeltaPoly, factor: int, prime: int) -> DeltaPoly:
    p = require_prime(prime)
    factor_mod = int(factor) % p
    if not factor_mod:
        return {}
    return {
        delta: scaled
        for delta, value in poly.items()
        if (scaled := {
            alpha: coeff * factor_mod % p
            for alpha, coeff in value.items()
            if coeff * factor_mod % p
        })
    }


def _delta_poly_mul(
    left: DeltaPoly,
    right: DeltaPoly,
    max_delta: DeltaKey,
    prime: int,
) -> DeltaPoly:
    p = require_prime(prime)
    out: DeltaPoly = {}
    for d1, p1 in left.items():
        for d2, p2 in right.items():
            delta = _delta_add(d1, d2)
            if not _delta_leq(delta, max_delta):
                continue
            product = mul(p1, p2, prime=p)
            if product:
                out[delta] = add(out.get(delta, {}), product, prime=p)
    return out


def _delta_poly_pow(
    config: FormulaConfig,
    base: DeltaPoly,
    exponent: int,
    max_delta: DeltaKey,
    prime: int,
) -> DeltaPoly:
    p = require_prime(prime)
    out: DeltaPoly = {_delta_zero(config): constant(config.y_count, 1, prime=p)}
    cur = base
    n = int(exponent)
    while n:
        if n & 1:
            out = _delta_poly_mul(out, cur, max_delta, p)
        n >>= 1
        if n:
            cur = _delta_poly_mul(cur, cur, max_delta, p)
    return out


def _delta_poly_exp_linear_mod(
    config: FormulaConfig,
    linear: dict[DeltaKey, SparsePoly],
    max_delta: DeltaKey,
    prime: int,
) -> DeltaPoly:
    p = require_prime(prime)
    powers: dict[DeltaKey, Tuple[SparsePoly, ...]] = {}
    for unit, poly in linear.items():
        axis = unit.index(1)
        unit_powers = [constant(config.y_count, 1, prime=p)]
        for _exp in range(1, max_delta[axis] + 1):
            unit_powers.append(mul(unit_powers[-1], poly, prime=p))
        powers[unit] = tuple(unit_powers)

    inv_factorials = {
        n: pow(factorial(n) % p, p - 2, p)
        for n in range(sum(max_delta) + 1)
    }
    out: DeltaPoly = {}

    def visit(axis: int, delta_prefix: list[int], term: SparsePoly) -> None:
        if axis == len(max_delta):
            out[tuple(delta_prefix)] = term
            return
        unit = _delta_unit(config, axis)
        for exp in range(max_delta[axis] + 1):
            next_term = term
            if exp:
                next_term = mul(next_term, powers[unit][exp], prime=p)
                next_term = {
                    alpha: coeff * inv_factorials[exp] % p
                    for alpha, coeff in next_term.items()
                    if coeff * inv_factorials[exp] % p
                }
            if next_term:
                visit(axis + 1, delta_prefix + [exp], next_term)

    visit(0, [], constant(config.y_count, 1, prime=p))
    return out


def _kernel_terms_mul_delta_mod(
    config: FormulaConfig,
    terms: DeltaKernelTerms,
    poly: DeltaPoly,
    max_delta: DeltaKey,
    prime: int,
) -> DeltaKernelTerms:
    p = require_prime(prime)
    out: DeltaKernelTerms = {}
    del config
    for (kd, deriv), val in terms.items():
        for pd, pval in poly.items():
            nd = _delta_add(kd, pd)
            if not _delta_leq(nd, max_delta):
                continue
            product = mul(val, pval, prime=p)
            if product:
                out[(nd, deriv)] = add(out.get((nd, deriv), {}), product, prime=p)
    return {key: value for key, value in out.items() if value}


def _delta_matrix_identity(
    config: FormulaConfig,
    prime: int,
) -> Tuple[Tuple[DeltaPoly, ...], ...]:
    p = require_prime(prime)
    zero_delta = _delta_zero(config)
    one = constant(config.y_count, 1, prime=p)
    return tuple(
        tuple({zero_delta: dict(one)} if i == j else {} for j in range(config.y_count))
        for i in range(config.y_count)
    )


def _delta_matrix_mul(
    left: Sequence[Sequence[DeltaPoly]],
    right: Sequence[Sequence[DeltaPoly]],
    max_delta: DeltaKey,
    prime: int,
) -> Tuple[Tuple[DeltaPoly, ...], ...]:
    rows = len(left)
    cols = len(right[0])
    inner = len(right)
    out: list[list[DeltaPoly]] = [[{} for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            acc: DeltaPoly = {}
            for k in range(inner):
                acc = _delta_poly_add(
                    acc,
                    _delta_poly_mul(left[i][k], right[k][j], max_delta, prime),
                    prime,
                )
            out[i][j] = acc
    return tuple(tuple(row) for row in out)


def _ext_delta_mul_pruned(
    config: FormulaConfig,
    left: dict[int, DeltaPoly],
    right: dict[int, DeltaPoly],
    max_delta: DeltaKey,
    target_mask: int,
    target_len: int,
    prime: int,
) -> dict[int, DeltaPoly]:
    p = require_prime(prime)
    exterior = ExteriorAlgebra(config)
    out: dict[int, DeltaPoly] = {}
    for m1, d1 in left.items():
        for m2, d2 in right.items():
            wedge = exterior.wedge_masks(m1, m2)
            if wedge is None:
                continue
            sign, mask = wedge
            if mask.bit_count() > target_len or (mask | target_mask) != target_mask:
                continue
            product = _delta_poly_mul(d1, d2, max_delta, p)
            if product:
                out[mask] = _delta_poly_add(out.get(mask, {}), product, p, scale=sign)
    return {mask: poly for mask, poly in out.items() if poly}


@lru_cache(maxsize=None)
def _hessian_perturbation_delta_mod(
    config: FormulaConfig,
    max_delta: DeltaKey,
    prime: int,
) -> Tuple[Tuple[Tuple[DeltaPoly, ...], ...], ...]:
    p = require_prime(prime)
    h0_inv = modular_formula.hessian_tau2_inverse_mod(config, p)
    mats = []
    for axis, delta_rank in enumerate(config.delta_ranks):
        unit = _delta_unit(config, axis)
        if not _delta_leq(unit, max_delta):
            continue
        left_mul = modular_formula._const_sparse_matrix_mul(  # noqa: SLF001
            h0_inv,
            modular_formula.tau_hessian_mod(config, delta_rank, p),
            p,
        )
        rows = []
        for i in range(config.y_count):
            row = []
            for j in range(config.y_count):
                poly = clean(left_mul[i][j], p)
                row.append({unit: poly} if poly else {})
            rows.append(tuple(row))
        mats.append(tuple(rows))
    return tuple(mats)


@lru_cache(maxsize=None)
def _hessian_inverse_delta_mod(
    config: FormulaConfig,
    max_delta: DeltaKey,
    prime: int,
) -> Tuple[Tuple[Tuple[Tuple[DeltaKey, Tuple[Tuple[Tuple[int, ...], int], ...]], ...], ...], ...]:
    p = require_prime(prime)
    perturb = _hessian_perturbation_delta_mod(config, max_delta, p)

    a_mat: list[list[DeltaPoly]] = [
        [{} for _ in range(config.y_count)] for _ in range(config.y_count)
    ]
    for i in range(config.y_count):
        for j in range(config.y_count):
            entry: DeltaPoly = {}
            for mat in perturb:
                entry = _delta_poly_add(entry, mat[i][j], p)
            a_mat[i][j] = entry

    series = _delta_matrix_identity(config, p)
    power = _delta_matrix_identity(config, p)
    for order in range(1, sum(max_delta) + 1):
        power = _delta_matrix_mul(power, a_mat, max_delta, p)
        sign = -1 if order % 2 else 1
        series = tuple(
            tuple(
                _delta_poly_add(series[i][j], power[i][j], p, scale=sign)
                for j in range(config.y_count)
            )
            for i in range(config.y_count)
        )

    h0_inv = modular_formula.hessian_tau2_inverse_mod(config, p)
    out: list[list[DeltaPoly]] = [
        [{} for _ in range(config.y_count)] for _ in range(config.y_count)
    ]
    for i in range(config.y_count):
        for j in range(config.y_count):
            acc: DeltaPoly = {}
            for k in range(config.y_count):
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
                sorted(
                    (delta, tuple(sorted(poly.items())))
                    for delta, poly in cell.items()
                    if poly
                )
            )
            for cell in row
        )
        for row in out
    )


def _hessian_inverse_cell_delta_mod(
    config: FormulaConfig,
    max_delta: DeltaKey,
    i: int,
    j: int,
    prime: int,
) -> DeltaPoly:
    return {
        delta: dict(poly_items)
        for delta, poly_items in _hessian_inverse_delta_mod(config, max_delta, prime)[i][j]
    }


@lru_cache(maxsize=None)
def _hat_pair_delta_mod(
    config: FormulaConfig,
    r: int,
    s: int,
    max_delta: DeltaKey,
    prime: int,
) -> Tuple[Tuple[DeltaKey, Tuple[Tuple[Tuple[int, ...], int], ...]], ...]:
    p = require_prime(prime)
    gr = modular_formula.tau_grad_mod(config, r, p)
    gs = modular_formula.tau_grad_mod(config, s, p)
    acc: DeltaPoly = {}
    for i in range(config.y_count):
        for j in range(config.y_count):
            cell = _hessian_inverse_cell_delta_mod(config, max_delta, i, j, p)
            if not cell:
                continue
            coeff_poly = mul(dict(gr[i]), dict(gs[j]), prime=p)
            coeff_poly = {
                alpha: (-coeff) % p for alpha, coeff in coeff_poly.items() if coeff % p
            }
            for delta, poly in cell.items():
                product = mul(coeff_poly, poly, prime=p)
                if product:
                    acc[delta] = add(acc.get(delta, {}), product, prime=p)
    return tuple(
        sorted((delta, tuple(sorted(poly.items()))) for delta, poly in acc.items() if poly)
    )


@lru_cache(maxsize=None)
def _b_hat_mask_delta_mod(
    config: FormulaConfig,
    mask: int,
    max_delta: DeltaKey,
    prime: int,
) -> Tuple[Tuple[DeltaKey, Tuple[Tuple[Tuple[int, ...], int], ...]], ...]:
    p = require_prime(prime)
    exterior = ExteriorAlgebra(config)
    target = exterior.labels_from_mask(int(mask))
    if len(target) % 2:
        return ()
    if not target:
        return ((_delta_zero(config), tuple(sorted(constant(config.y_count, 1, prime=p).items()))),)

    pair_terms: dict[int, DeltaPoly] = {}
    target_set = set(target)
    for left_side in range(1, config.genus + 1):
        right_side = left_side + config.genus
        left_labels = [label for label in target if label[1] == left_side]
        right_labels = [label for label in target if label[1] == right_side]
        for left_label in left_labels:
            for right_label in right_labels:
                pair_mask = (
                    exterior.mask_for_b_label(left_label)
                    | exterior.mask_for_b_label(right_label)
                )
                if any(label not in target_set for label in exterior.labels_from_mask(pair_mask)):
                    continue
                wedge = exterior.wedge_masks(
                    exterior.mask_for_b_label(left_label),
                    exterior.mask_for_b_label(right_label),
                )
                if wedge is None:
                    continue
                sign, odd_mask = wedge
                coeff: DeltaPoly = {
                    delta: dict(items)
                    for delta, items in _hat_pair_delta_mod(
                        config,
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
    power: dict[int, DeltaPoly] = {
        0: {_delta_zero(config): constant(config.y_count, 1, prime=p)}
    }
    for _ in range(pair_count):
        power = _ext_delta_mul_pruned(
            config,
            power,
            pair_terms,
            max_delta,
            int(mask),
            len(target),
            p,
        )
        if not power:
            return ()

    scale = pow(factorial(pair_count) % p, p - 2, p)
    result = _delta_poly_scale(power.get(int(mask), {}), scale, p)
    return tuple(
        sorted((delta, tuple(sorted(poly.items()))) for delta, poly in result.items() if poly)
    )


@lru_cache(maxsize=None)
def _gamma_hat_delta_mod(
    config: FormulaConfig,
    gamma_exp: Tuple[int, ...],
    target_delta: DeltaKey,
    prime: int,
) -> Tuple[Tuple[DeltaKey, Tuple[Tuple[Tuple[int, ...], int], ...]], ...]:
    p = require_prime(prime)
    exterior = ExteriorAlgebra(config)
    out: DeltaPoly = {}
    for mask, coeff in exterior.gamma_product_to_mask_poly(gamma_exp).items():
        if not coeff % p:
            continue
        b_delta = {
            delta: dict(poly_items)
            for delta, poly_items in _b_hat_mask_delta_mod(
                config,
                int(mask),
                target_delta,
                p,
            )
        }
        out = _delta_poly_add(out, b_delta, p, scale=coeff)
    return tuple(
        sorted((delta, tuple(sorted(poly.items()))) for delta, poly in out.items() if poly)
    )


@lru_cache(maxsize=None)
def _det_ratio_delta_power_mod(
    config: FormulaConfig,
    max_delta: DeltaKey,
    power: int,
    prime: int,
) -> Tuple[Tuple[DeltaKey, Tuple[Tuple[Tuple[int, ...], int], ...]], ...]:
    p = require_prime(prime)
    zero_delta = _delta_zero(config)
    perturb = _hessian_perturbation_delta_mod(config, max_delta, p)
    matrix: list[list[DeltaPoly]] = []
    for i in range(config.y_count):
        row: list[DeltaPoly] = []
        for j in range(config.y_count):
            entry: DeltaPoly = (
                {zero_delta: constant(config.y_count, 1, prime=p)} if i == j else {}
            )
            for mat in perturb:
                entry = _delta_poly_add(entry, mat[i][j], p)
            row.append(entry)
        matrix.append(row)

    det_poly: DeltaPoly = {}
    for perm in permutations(range(config.y_count)):
        inversions = sum(
            1
            for i in range(config.y_count)
            for j in range(i + 1, config.y_count)
            if perm[i] > perm[j]
        )
        sign = -1 if inversions % 2 else 1
        term: DeltaPoly = {
            zero_delta: constant(config.y_count, sign, prime=p)
        }
        for i, j in enumerate(perm):
            term = _delta_poly_mul(term, matrix[i][j], max_delta, p)
            if not term:
                break
        det_poly = _delta_poly_add(det_poly, term, p)
    result = _delta_poly_pow(config, det_poly, int(power), max_delta, p)
    return tuple(
        sorted(
            (delta, tuple(sorted(poly.items())))
            for delta, poly in result.items()
            if poly
        )
    )


def _denominator_taylor_terms_mod(
    config: FormulaConfig,
    max_delta: DeltaKey,
    prime: int,
) -> DeltaKernelTerms:
    p = require_prime(prime)
    zero_delta = _delta_zero(config)
    zero_deriv = _zero_deriv(config)
    terms: DeltaKernelTerms = {
        (zero_delta, zero_deriv): constant(config.y_count, 1, prime=p)
    }
    max_order = sum(max_delta)
    if max_order == 0:
        return terms

    for j in range(1, config.y_count + 1):
        eps: DeltaPoly = {}
        for axis, delta_rank in enumerate(config.delta_ranks):
            unit = _delta_unit(config, axis)
            if _delta_leq(unit, max_delta):
                poly = dict(modular_formula.b_perturbation_mod(config, delta_rank, j, p))
                if poly:
                    eps[unit] = add(eps.get(unit, {}), poly, prime=p)

        factor: DeltaKernelTerms = {}
        for order in range(max_order + 1):
            eps_power = _delta_poly_pow(config, eps, order, max_delta, p)
            if not eps_power:
                continue
            deriv = [0 for _ in range(config.y_count)]
            deriv[j - 1] = order
            scale = pow(factorial(order) % p, p - 2, p)
            for delta, poly in eps_power.items():
                if poly:
                    key = (delta, tuple(deriv))
                    factor[key] = add(factor.get(key, {}), poly, prime=p, scale=scale)

        next_terms: DeltaKernelTerms = {}
        for (d1, der1), v1 in terms.items():
            for (d2, der2), v2 in factor.items():
                nd = _delta_add(d1, d2)
                if not _delta_leq(nd, max_delta):
                    continue
                nder = tuple(der1[idx] + der2[idx] for idx in range(config.y_count))
                product = mul(v1, v2, prime=p)
                if product:
                    next_terms[(nd, nder)] = add(
                        next_terms.get((nd, nder), {}),
                        product,
                        prime=p,
                    )
        terms = {key: value for key, value in next_terms.items() if value}
    return terms


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
    numpy_result = _tau_power_bounded_numpy_mod(config, a_exp, caps, p)
    if numpy_result is not None:
        return numpy_result

    out = constant(config.y_count, 1, prime=p)
    factor_cache: dict[int, SparsePoly] = {}
    for r in _bounded_tau_factor_order(config, a_exp):
        factor = factor_cache.get(r)
        if factor is None:
            factor = dict(modular_formula.tau_mod(config, r, p))
            factor_cache[r] = factor
        out = _bounded_mul(out, factor, caps, p)
    return tuple(sorted(out.items()))


def _tau_power_bounded_numpy_mod(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
    caps: Tuple[int | None, ...],
    prime: int,
) -> Tuple[Tuple[Tuple[int, ...], int], ...] | None:
    arrays = _tau_power_bounded_numpy_arrays_mod(config, a_exp, caps, prime)
    if arrays is None:
        return None
    alpha, coeff = arrays
    return tuple(sorted(
        (tuple(int(item) for item in alpha_key), int(coeff_value))
        for alpha_key, coeff_value in zip(alpha, coeff)
    ))


def _bounded_tau_factor_order(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
) -> Tuple[int, ...]:
    if len(a_exp) != len(config.class_ranks):
        raise ValueError("a exponent length does not match the formula config")
    factors = []
    for exp, r in zip(a_exp, config.class_ranks):
        factors.extend(int(r) for _ in range(int(exp)))
    return tuple(sorted(factors, reverse=True))


@lru_cache(maxsize=256)
def _tau_power_bounded_numpy_arrays_mod(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
    caps: Tuple[int | None, ...],
    prime: int,
):
    if config.y_count != 6 or prime > 2_000_000:
        return None
    finite_caps = tuple(cap for cap in caps if cap is not None)
    if finite_caps and max(int(cap) for cap in finite_caps) > 65_535:
        return None
    try:
        import numpy as np
    except ImportError:
        return None

    homogeneous_context = _homogeneous_code_context_6(config, a_exp, caps, np)
    if homogeneous_context is not None:
        homogeneous_result = _tau_power_bounded_homogeneous_numpy_mod(
            config,
            a_exp,
            prime,
            np,
            homogeneous_context,
        )
        if homogeneous_result is not None:
            return homogeneous_result

    code_context = _bounded_code_context_6(caps, np)
    if code_context is not None:
        coded_result = _tau_power_bounded_coded_numpy_mod(
            config,
            a_exp,
            prime,
            np,
            code_context,
        )
        if coded_result is not None:
            return coded_result

    alpha = np.zeros((1, 6), dtype=np.uint16)
    coeff = np.ones(1, dtype=np.int64)
    factor_cache: dict[int, tuple[object, object]] = {}
    for r in _bounded_tau_factor_order(config, a_exp):
        factor = factor_cache.get(r)
        if factor is None:
            factor_terms = dict(modular_formula.tau_mod(config, r, prime))
            factor = (
                np.asarray(tuple(factor_terms.keys()), dtype=np.uint16),
                np.asarray(tuple(factor_terms.values()), dtype=np.int64),
            )
            factor_cache[r] = factor
        multiplied = _bounded_array_mul_6(
            alpha,
            coeff,
            factor[0],
            factor[1],
            caps,
            prime,
            np,
        )
        if multiplied is None:
            return None
        alpha, coeff = multiplied
        if not coeff.size:
            return alpha, coeff

    return alpha, coeff


def _tau_power_bounded_coded_numpy_mod(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
    prime: int,
    np,
    code_context,
):
    alpha = np.zeros((1, 6), dtype=np.uint16)
    _, _, code_dtype = code_context
    codes = np.zeros(1, dtype=code_dtype)
    coeff = np.ones(1, dtype=np.int64)
    factor_cache: dict[int, object] = {}
    for r in _bounded_tau_factor_order(config, a_exp):
        factor = factor_cache.get(r)
        if factor is None:
            factor = _bounded_factor_code_data(
                dict(modular_formula.tau_mod(config, r, prime)),
                np,
                code_context,
            )
            factor_cache[r] = factor
        multiplied = _bounded_array_mul_6_coded(
            alpha,
            codes,
            coeff,
            factor,
            code_context,
            prime,
            np,
        )
        if multiplied is None:
            return None
        alpha, codes, coeff = multiplied
        if not coeff.size:
            return alpha, coeff

    return alpha, coeff


def _tau_power_bounded_homogeneous_numpy_mod(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
    prime: int,
    np,
    homogeneous_context,
):
    keep_indices = homogeneous_context[1]
    alpha_keep = np.zeros((1, len(keep_indices)), dtype=np.uint16)
    codes = np.zeros(1, dtype=np.uint32)
    coeff = np.ones(1, dtype=np.int64)
    degree = 0
    factor_cache: dict[int, object] = {}
    for r in _bounded_tau_factor_order(config, a_exp):
        factor = factor_cache.get(r)
        if factor is None:
            factor = _homogeneous_factor_code_data(
                dict(modular_formula.tau_mod(config, r, prime)),
                np,
                homogeneous_context,
            )
            factor_cache[r] = factor
        multiplied = _bounded_homogeneous_array_mul_6(
            alpha_keep,
            codes,
            coeff,
            degree,
            int(r),
            factor,
            homogeneous_context,
            prime,
            np,
        )
        if multiplied is None:
            return None
        alpha_keep, codes, coeff, degree = multiplied
        if not coeff.size:
            return _decode_homogeneous_codes_6(
                codes,
                degree,
                homogeneous_context,
                np,
            ), coeff

    return (
        _decode_homogeneous_codes_6(codes, degree, homogeneous_context, np),
        coeff,
    )


def _homogeneous_code_context_6(
    config: FormulaConfig,
    a_exp: Tuple[int, ...],
    caps: Tuple[int | None, ...],
    np,
):
    if any(cap is None for cap in caps):
        return None
    final_degree = sum(int(exp) * int(r) for exp, r in zip(a_exp, config.class_ranks))
    cap_array = np.asarray(tuple(int(cap) for cap in caps), dtype=np.int64)
    candidate_drops = [idx for idx, cap in enumerate(cap_array) if cap >= final_degree]
    if not candidate_drops:
        return None

    best = None
    for drop_idx in candidate_drops:
        keep_indices = tuple(idx for idx in range(6) if idx != drop_idx)
        bases = cap_array[list(keep_indices)] + 1
        range_size = int(np.prod(bases, dtype=object))
        if best is None or range_size < best[0]:
            best = (range_size, drop_idx, keep_indices, bases)
    if best is None:
        return None

    range_size, drop_idx, keep_indices, bases = best
    if range_size >= 5_000_000:
        return None
    strides = np.ones(len(keep_indices), dtype=np.int64)
    for idx in range(len(keep_indices) - 2, -1, -1):
        strides[idx] = strides[idx + 1] * bases[idx + 1]
    return (
        int(drop_idx),
        keep_indices,
        cap_array,
        cap_array[list(keep_indices)],
        strides,
        int(range_size),
    )


def _homogeneous_factor_code_data(poly: SparsePoly, np, homogeneous_context):
    _, keep_indices, _, keep_caps, strides, _ = homogeneous_context
    factor = []
    for alpha, coeff in poly.items():
        shift = np.asarray(tuple(alpha[idx] for idx in keep_indices), dtype=np.int64)
        if np.any(shift > keep_caps):
            continue
        active = np.nonzero(shift)[0]
        factor.append(
            (
                int(coeff),
                np.uint32(int(shift @ strides)),
                active,
                keep_caps[active] - shift[active],
            )
        )
    return tuple(factor)


def _bounded_homogeneous_array_mul_6(
    left_alpha_keep,
    left_codes,
    left_coeff,
    degree: int,
    factor_degree: int,
    factor,
    homogeneous_context,
    prime: int,
    np,
):
    if int(left_alpha_keep.shape[0]) * len(factor) > _ARRAY_MUL_EMISSION_MAX:
        return None
    if not factor:
        return (
            np.empty((0, left_alpha_keep.shape[1]), dtype=np.uint16),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.int64),
            degree + factor_degree,
        )

    alpha_i64 = left_alpha_keep.astype(np.int64, copy=False)
    code_chunks = []
    coeff_chunks = []
    for coeff_scale, shift_code, active, limits in factor:
        if active.size:
            mask = np.all(alpha_i64[:, active] <= limits, axis=1)
            if not np.any(mask):
                continue
            code_chunks.append(left_codes[mask] + shift_code)
            coeff_chunks.append((left_coeff[mask] * coeff_scale) % prime)
        else:
            code_chunks.append(left_codes + shift_code)
            coeff_chunks.append((left_coeff * coeff_scale) % prime)

    next_degree = degree + factor_degree
    if not code_chunks:
        return (
            np.empty((0, left_alpha_keep.shape[1]), dtype=np.uint16),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.int64),
            next_degree,
        )

    codes = np.concatenate(code_chunks)
    coeff = np.concatenate(coeff_chunks)
    reduced_codes, reduced_coeff = _reduce_dense_codes_mod(
        codes,
        coeff,
        homogeneous_context,
        prime,
        np,
    )
    return (
        _decode_homogeneous_keep_codes(reduced_codes, homogeneous_context, np),
        reduced_codes,
        reduced_coeff,
        next_degree,
    )


def _reduce_dense_codes_mod(codes, coeff, homogeneous_context, prime: int, np):
    range_size = homogeneous_context[5]
    # With the emission cap and small-prime backend, these integer sums are
    # below 2^53, so float64 bincount is exact before reducing modulo p.
    sums = np.bincount(
        codes.astype(np.int64, copy=False),
        weights=coeff.astype(np.float64, copy=False),
        minlength=range_size,
    )
    sums_mod = sums.astype(np.int64, copy=False) % prime
    keep = np.nonzero(sums_mod)[0]
    return keep.astype(np.uint32), sums_mod[keep].astype(np.int64, copy=False)


def _decode_homogeneous_keep_codes(codes, homogeneous_context, np):
    keep_indices = homogeneous_context[1]
    strides = homogeneous_context[4]
    alpha_keep = np.empty((codes.size, len(keep_indices)), dtype=np.uint16)
    remainder = codes.astype(np.int64, copy=True)
    for idx in range(len(keep_indices)):
        alpha_keep[:, idx] = (remainder // strides[idx]).astype(np.uint16)
        remainder %= strides[idx]
    return alpha_keep


def _decode_homogeneous_codes_6(codes, degree: int, homogeneous_context, np):
    drop_idx, keep_indices, _, _, _, _ = homogeneous_context
    alpha_keep = _decode_homogeneous_keep_codes(codes, homogeneous_context, np)
    alpha = np.empty((codes.size, 6), dtype=np.uint16)
    keep_sum = np.zeros(codes.size, dtype=np.int64)
    for local_idx, global_idx in enumerate(keep_indices):
        values = alpha_keep[:, local_idx]
        alpha[:, global_idx] = values
        keep_sum += values
    drop_values = int(degree) - keep_sum
    if np.any(drop_values < 0) or np.any(drop_values > 65_535):
        raise ValueError("homogeneous tau code produced an invalid exponent")
    alpha[:, drop_idx] = drop_values.astype(np.uint16)
    return alpha


def _bounded_code_context_6(caps: Tuple[int | None, ...], np):
    if any(cap is None for cap in caps):
        return None
    cap_array = np.asarray(tuple(int(cap) for cap in caps), dtype=np.int64)
    if np.any(cap_array < 0) or np.any(cap_array > 65_535):
        return None
    bases = cap_array + 1
    total_size = int(np.prod(bases, dtype=object))
    if total_size >= 2**62:
        return None
    code_dtype = np.uint32 if total_size < 2**32 else np.int64
    strides = np.ones(6, dtype=np.int64)
    for idx in range(4, -1, -1):
        strides[idx] = strides[idx + 1] * bases[idx + 1]
    return cap_array, strides, code_dtype


def _bounded_factor_code_data(poly: SparsePoly, np, code_context):
    cap_array, strides, code_dtype = code_context
    code_type = np.dtype(code_dtype).type
    factor = []
    for alpha, coeff in poly.items():
        shift = np.asarray(alpha, dtype=np.int64)
        if np.any(shift > cap_array):
            continue
        active = np.nonzero(shift)[0]
        factor.append(
            (
                int(coeff),
                code_type(int(shift @ strides)),
                active,
                cap_array[active] - shift[active],
            )
        )
    return tuple(factor)


def _bounded_array_mul_6_coded(
    left_alpha,
    left_codes,
    left_coeff,
    factor,
    code_context,
    prime: int,
    np,
):
    _, _, code_dtype = code_context
    if int(left_alpha.shape[0]) * len(factor) > _ARRAY_MUL_EMISSION_MAX:
        return None
    if not factor:
        return (
            np.empty((0, 6), dtype=np.uint16),
            np.empty(0, dtype=code_dtype),
            np.empty(0, dtype=np.int64),
        )

    alpha_i64 = left_alpha.astype(np.int64, copy=False)
    code_chunks = []
    coeff_chunks = []
    for coeff_scale, shift_code, active, limits in factor:
        if active.size:
            mask = np.all(alpha_i64[:, active] <= limits, axis=1)
            if not np.any(mask):
                continue
            code_chunks.append(left_codes[mask] + shift_code)
            coeff_chunks.append((left_coeff[mask] * coeff_scale) % prime)
        else:
            code_chunks.append(left_codes + shift_code)
            coeff_chunks.append((left_coeff * coeff_scale) % prime)

    if not code_chunks:
        return (
            np.empty((0, 6), dtype=np.uint16),
            np.empty(0, dtype=code_dtype),
            np.empty(0, dtype=np.int64),
        )

    codes = np.concatenate(code_chunks)
    coeff = np.concatenate(coeff_chunks)
    reduced_codes, reduced_coeff = _reduce_codes_mod(codes, coeff, prime, np)
    return (
        _decode_codes_6(reduced_codes, code_context, np),
        reduced_codes,
        reduced_coeff,
    )


def _reduce_codes_mod(codes, coeff, prime: int, np):
    order = np.argsort(codes)
    sorted_codes = codes[order]
    sorted_coeff = coeff[order]
    changes = np.empty(sorted_codes.size, dtype=np.bool_)
    changes[0] = True
    changes[1:] = sorted_codes[1:] != sorted_codes[:-1]
    starts = np.nonzero(changes)[0]
    sums = np.add.reduceat(sorted_coeff, starts) % prime
    keep = sums != 0
    return (
        sorted_codes[starts][keep].copy(),
        sums[keep].astype(np.int64, copy=False),
    )


def _decode_codes_6(codes, code_context, np):
    _, strides, _ = code_context
    alpha = np.empty((codes.size, 6), dtype=np.uint16)
    remainder = codes.astype(np.int64, copy=True)
    for idx in range(6):
        alpha[:, idx] = (remainder // strides[idx]).astype(np.uint16)
        remainder %= strides[idx]
    return alpha


def _bounded_array_mul_6(
    left_alpha,
    left_coeff,
    right_alpha,
    right_coeff,
    caps: Tuple[int | None, ...],
    prime: int,
    np,
):
    if int(left_alpha.shape[0]) * int(right_alpha.shape[0]) > _ARRAY_MUL_EMISSION_MAX:
        return None
    finite_indices = tuple(idx for idx, cap in enumerate(caps) if cap is not None)
    finite_caps = np.asarray(
        tuple(int(caps[idx]) for idx in finite_indices),
        dtype=np.int64,
    )

    alpha_chunks = []
    coeff_chunks = []
    left_alpha_i64 = left_alpha.astype(np.int64, copy=False)
    for alpha_shift, coeff_scale in zip(right_alpha, right_coeff):
        shifted = left_alpha_i64 + alpha_shift.astype(np.int64, copy=False)
        if finite_indices:
            mask = np.all(shifted[:, finite_indices] <= finite_caps, axis=1)
            if not np.any(mask):
                continue
            shifted = shifted[mask]
            coeff = left_coeff[mask]
        else:
            coeff = left_coeff
        alpha_chunks.append(shifted.astype(np.uint16, copy=False))
        coeff_chunks.append((coeff * int(coeff_scale)) % prime)

    if not alpha_chunks:
        return (
            np.empty((0, 6), dtype=np.uint16),
            np.empty(0, dtype=np.int64),
        )

    alpha = np.concatenate(alpha_chunks, axis=0)
    coeff = np.concatenate(coeff_chunks, axis=0)
    order = np.lexsort(tuple(alpha[:, idx] for idx in range(5, -1, -1)))
    sorted_alpha = alpha[order]
    sorted_coeff = coeff[order]
    changes = np.empty(sorted_alpha.shape[0], dtype=np.bool_)
    changes[0] = True
    changes[1:] = np.any(sorted_alpha[1:] != sorted_alpha[:-1], axis=1)
    starts = np.nonzero(changes)[0]
    sums = np.add.reduceat(sorted_coeff, starts) % prime
    keep = sums != 0
    return (
        sorted_alpha[starts][keep].copy(),
        sums[keep].astype(np.int64, copy=False),
    )


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
    numpy_result = _bounded_mul_6_numpy(left, right, caps, prime)
    if numpy_result is not None:
        return numpy_result

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


def _bounded_mul_6_numpy(
    left: SparsePoly,
    right: SparsePoly,
    caps: Tuple[int | None, ...],
    prime: int,
) -> SparsePoly | None:
    if prime > 2_000_000:
        return None
    emission_bound = len(left) * len(right)
    if emission_bound > _ARRAY_MUL_EMISSION_MAX:
        return None
    try:
        import numpy as np
    except ImportError:
        return None

    left_alpha = np.asarray(tuple(left.keys()), dtype=np.uint16)
    left_coeff = np.asarray(tuple(left.values()), dtype=np.int64)
    finite_indices = tuple(idx for idx, cap in enumerate(caps) if cap is not None)
    finite_caps = np.asarray(
        tuple(int(caps[idx]) for idx in finite_indices),
        dtype=np.int64,
    )

    alpha_chunks = []
    coeff_chunks = []
    left_alpha_i64 = left_alpha.astype(np.int64, copy=False)
    for right_alpha, right_coeff in right.items():
        shifted = left_alpha_i64 + np.asarray(right_alpha, dtype=np.int64)
        if finite_indices:
            mask = np.all(shifted[:, finite_indices] <= finite_caps, axis=1)
            if not np.any(mask):
                continue
            shifted = shifted[mask]
            coeff = left_coeff[mask]
        else:
            coeff = left_coeff
        alpha_chunks.append(shifted.astype(np.uint16, copy=False))
        coeff_chunks.append((coeff * int(right_coeff)) % prime)

    if not alpha_chunks:
        return {}

    alpha = np.concatenate(alpha_chunks, axis=0)
    coeff = np.concatenate(coeff_chunks, axis=0)
    order = np.lexsort(tuple(alpha[:, idx] for idx in range(5, -1, -1)))
    sorted_alpha = alpha[order]
    sorted_coeff = coeff[order]
    changes = np.empty(sorted_alpha.shape[0], dtype=np.bool_)
    changes[0] = True
    changes[1:] = np.any(sorted_alpha[1:] != sorted_alpha[:-1], axis=1)
    starts = np.nonzero(changes)[0]
    sums = np.add.reduceat(sorted_coeff, starts) % prime
    keep = sums != 0
    return {
        tuple(int(item) for item in alpha_key): int(coeff_value)
        for alpha_key, coeff_value in zip(sorted_alpha[starts][keep], sums[keep])
    }


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


def _shift_alpha_array(alpha_terms, beta: Tuple[int, ...]):
    if not any(beta):
        return alpha_terms
    try:
        import numpy as np
    except ImportError:
        return None

    beta_array = np.asarray(beta, dtype=np.int64)
    shifted = alpha_terms.astype(np.int64, copy=False) + beta_array
    if np.any(shifted < 0) or np.any(shifted > 65_535):
        return None
    return shifted.astype(np.uint16, copy=False)


def _shift_poly(
    poly: SparsePoly,
    beta: Tuple[int, ...],
    prime: int,
) -> SparsePoly:
    if not any(beta):
        return poly
    p = require_prime(prime)
    out: SparsePoly = {}
    for alpha, coeff in poly.items():
        shifted = tuple(alpha[idx] + beta[idx] for idx in range(len(beta)))
        out[shifted] = (out.get(shifted, 0) + coeff) % p
    return clean(out, p)


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


def _normalize_semantic_cache_maxsize(value: int | None) -> int | None:
    if value is None:
        return _env_cache_maxsize()
    return _normalize_optional_cache_size(value)


def _normalize_optional_cache_size(value: int | None) -> int | None:
    if value is None:
        return None
    maxsize = int(value)
    return None if maxsize < 0 else maxsize


def _env_cache_maxsize() -> int | None:
    raw = os.environ.get("RANK7_JK_SEMANTIC_CACHE_MAX")
    if raw is None or raw == "":
        return _DEFAULT_SEMANTIC_CACHE_MAXSIZE
    value = int(raw)
    return None if value < 0 else value


def _cache_info_dict(info) -> dict[str, int]:
    return {
        "hits": int(info.hits),
        "misses": int(info.misses),
        "maxsize": -1 if info.maxsize is None else int(info.maxsize),
        "currsize": int(info.currsize),
    }
