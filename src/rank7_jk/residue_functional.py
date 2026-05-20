"""Packed/sliced exact residue functional.

This module keeps the same iterated JK transition as ``residue_transition`` but
stores only the alpha coordinates and denominator powers still alive at each
variable-elimination stage.  It is a speed-oriented representation, not a new
formula.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
import os
from threading import Lock
from time import perf_counter
from typing import Dict, Tuple

from .mod_arith import require_prime
from .residue_transition import _special_coeff_mod, _validate_alpha
from .root_system import TypeARootSystem, type_a_roots
from .sparse_poly import Alpha, SparsePoly, clean

DenominatorPowers = Tuple[int, ...]
SlicedStateKey = Tuple[Alpha, DenominatorPowers]

_ARRAY_BACKEND_PRIME_MAX = 1_000_000_000
_ARRAY_BACKEND_EXP_MAX = 65_535
_TRANSITION_SPMAT_CACHE_MAX = 128
_GLOBAL_TRANSITION_SPMAT_CACHE_MAX = 512
_GLOBAL_TRANSITION_SPMAT_CACHE: dict[object, Tuple[object, object]] = {}
_GLOBAL_TRANSITION_SPMAT_CACHE_LOCK = Lock()


def clear_global_transition_spmat_cache() -> None:
    with _GLOBAL_TRANSITION_SPMAT_CACHE_LOCK:
        _GLOBAL_TRANSITION_SPMAT_CACHE.clear()


@dataclass(frozen=True)
class ResidueStageProfile:
    var_idx: int
    input_states: int
    output_states: int
    elapsed_seconds: float
    local_cache_hits: int
    local_cache_misses: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "var_idx": self.var_idx,
            "input_states": self.input_states,
            "output_states": self.output_states,
            "elapsed_seconds": self.elapsed_seconds,
            "local_cache_hits": self.local_cache_hits,
            "local_cache_misses": self.local_cache_misses,
        }


@dataclass(frozen=True)
class ResidueProfile:
    input_terms: int
    result: int
    stages: Tuple[ResidueStageProfile, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "input_terms": self.input_terms,
            "result": self.result,
            "stages": [stage.to_dict() for stage in self.stages],
        }


@dataclass(frozen=True)
class _StageSpec:
    var_idx: int
    before_positions: Tuple[int, ...]
    after_positions: Tuple[int, ...]
    schedule: Tuple[Tuple[int, int], ...]
    full_to_before: dict[int, int]
    after_before_indices: Tuple[int, ...]
    simple_pos: int
    simple_before_idx: int


@dataclass(frozen=True)
class _ArrayStates:
    alpha: object
    denom: object
    coeff: object


@dataclass(frozen=True)
class _SpMatStates:
    alpha: object
    denom: object
    matrix: object


@dataclass(frozen=True)
class _LocalCodeAxis:
    present_codes: object
    inverse: object
    column_count: int
    direct: bool


@dataclass
class ResidueFunctional:
    rank: int
    derivative_orders: Tuple[int, ...]
    root_power: int
    prime: int
    backend: str = "auto"
    roots: TypeARootSystem = field(init=False)
    stage_specs: Tuple[_StageSpec, ...] = field(init=False)
    _local_cache: dict[
        Tuple[int, int, DenominatorPowers],
        Tuple[Tuple[DenominatorPowers, int], ...],
    ] = field(
        default_factory=dict,
        init=False,
    )
    _special_cache: dict[Tuple[int, int], int] = field(
        default_factory=dict,
        init=False,
    )
    _transition_spmat_cache: dict[object, Tuple[object, object]] = field(
        default_factory=dict,
        init=False,
    )
    local_cache_hits: int = 0
    local_cache_misses: int = 0

    def __post_init__(self) -> None:
        roots = type_a_roots(self.rank)
        derivative_orders = _validate_alpha(
            self.derivative_orders,
            roots.y_count,
            "derivative order",
        )
        if self.root_power < 0:
            raise ValueError("root_power must be nonnegative")
        backend = os.environ.get("RANK7_JK_RESIDUE_BACKEND", self.backend).lower()
        if backend not in {"auto", "python", "array", "spmat"}:
            raise ValueError(
                "residue backend must be auto, python, array, or spmat"
            )
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "derivative_orders", derivative_orders)
        object.__setattr__(self, "root_power", int(self.root_power))
        object.__setattr__(self, "prime", require_prime(self.prime))
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "roots", roots)
        object.__setattr__(self, "stage_specs", _stage_specs(roots))

    def evaluate_poly_terms(self, poly: SparsePoly) -> int:
        return self.profile_poly_terms(poly).result

    def evaluate_array_terms(self, alpha_terms, coeff_terms) -> int:
        return self.profile_array_terms(alpha_terms, coeff_terms).result

    def evaluate_unique_array_terms(self, alpha_terms, coeff_terms) -> int:
        return self.profile_unique_array_terms(alpha_terms, coeff_terms).result

    def profile_unique_array_terms(self, alpha_terms, coeff_terms) -> ResidueProfile:
        np = _require_numpy()
        input_terms = len(coeff_terms)
        if self._should_use_spmat_backend():
            sparse = _require_scipy_sparse()
            states = self._initial_spmat_states_from_unique_arrays(
                alpha_terms,
                coeff_terms,
                np,
                sparse,
            )
            return self._profile_spmat_states(
                states,
                input_terms,
                np,
                sparse,
            )
        return self.profile_array_terms(alpha_terms, coeff_terms)

    def profile_array_terms(self, alpha_terms, coeff_terms) -> ResidueProfile:
        np = _require_numpy()
        input_terms = len(coeff_terms)
        if self._should_use_spmat_backend():
            sparse = _require_scipy_sparse()
            states = self._initial_spmat_states_from_arrays(
                alpha_terms,
                coeff_terms,
                np,
                sparse,
            )
            return self._profile_spmat_states(
                states,
                input_terms,
                np,
                sparse,
            )

        states = self._initial_array_states_from_arrays(alpha_terms, coeff_terms, np)
        if self._should_use_array_backend():
            return self._profile_array_states(states, input_terms)
        poly: SparsePoly = {}
        for alpha, coeff in zip(states.alpha, states.coeff):
            value = int(coeff) % self.prime
            if not value:
                continue
            key = tuple(int(item) for item in alpha)
            poly[key] = (poly.get(key, 0) + value) % self.prime
        return self.profile_poly_terms(poly)

    def profile_poly_terms(self, poly: SparsePoly) -> ResidueProfile:
        if self._should_use_spmat_backend():
            return self._profile_poly_terms_spmat(poly)
        if self._should_use_array_backend():
            return self._profile_poly_terms_array(poly)

        states = self._initial_states(poly)
        stages = []
        for spec in reversed(self.stage_specs):
            hits_before = self.local_cache_hits
            misses_before = self.local_cache_misses
            start = perf_counter()
            input_states = len(states)
            states = self._eliminate_stage(states, spec)
            elapsed = perf_counter() - start
            stages.append(
                ResidueStageProfile(
                    var_idx=spec.var_idx,
                    input_states=input_states,
                    output_states=len(states),
                    elapsed_seconds=elapsed,
                    local_cache_hits=self.local_cache_hits - hits_before,
                    local_cache_misses=self.local_cache_misses - misses_before,
                )
            )
            if not states:
                return ResidueProfile(
                    input_terms=len(clean(poly, self.prime)),
                    result=0,
                    stages=tuple(stages),
                )

        return ResidueProfile(
            input_terms=len(clean(poly, self.prime)),
            result=states.get(((), ()), 0) % self.prime,
            stages=tuple(stages),
        )

    def _should_use_array_backend(self) -> bool:
        if self.backend in {"python", "spmat"}:
            return False
        if self.prime > _ARRAY_BACKEND_PRIME_MAX:
            if self.backend == "array":
                raise ValueError(
                    "array residue backend is limited to primes <= "
                    f"{_ARRAY_BACKEND_PRIME_MAX} to avoid int64 overflow"
                )
            return False
        if self.backend == "array":
            _require_numpy()
            return True
        return _numpy_available()

    def _should_use_spmat_backend(self) -> bool:
        if self.backend in {"python", "array"}:
            return False
        if self.prime > 2_000_000:
            if self.backend == "spmat":
                raise ValueError(
                    "spmat residue backend is limited to primes <= 2000000 "
                    "to avoid int64 sparse-product overflow"
                )
            return False
        if self.backend == "spmat":
            _require_scipy_sparse()
            return True
        return _scipy_available()

    def _profile_poly_terms_spmat(self, poly: SparsePoly) -> ResidueProfile:
        np = _require_numpy()
        sparse = _require_scipy_sparse()
        cleaned = clean(poly, self.prime)
        states = self._initial_spmat_states(cleaned, np, sparse)
        return self._profile_spmat_states(states, len(cleaned), np, sparse)

    def _profile_array_states_spmat(
        self,
        states: "_ArrayStates",
        input_terms: int,
    ) -> ResidueProfile:
        np = _require_numpy()
        sparse = _require_scipy_sparse()
        spmat_states = self._spmat_states_from_array_states(states, np, sparse)
        return self._profile_spmat_states(spmat_states, input_terms, np, sparse)

    def _profile_spmat_states(
        self,
        states: "_SpMatStates",
        input_terms: int,
        np,
        sparse,
    ) -> ResidueProfile:
        stages = []
        for spec in reversed(self.stage_specs):
            hits_before = self.local_cache_hits
            misses_before = self.local_cache_misses
            start = perf_counter()
            input_states = int(states.matrix.nnz)
            states = self._eliminate_stage_factored_spmat(states, spec, np, sparse)
            elapsed = perf_counter() - start
            stages.append(
                ResidueStageProfile(
                    var_idx=spec.var_idx,
                    input_states=input_states,
                    output_states=int(states.matrix.nnz),
                    elapsed_seconds=elapsed,
                    local_cache_hits=self.local_cache_hits - hits_before,
                    local_cache_misses=self.local_cache_misses - misses_before,
                )
            )
            if not states.matrix.nnz:
                return ResidueProfile(
                    input_terms=input_terms,
                    result=0,
                    stages=tuple(stages),
                )

        result = int(states.matrix.data.sum(dtype=np.int64) % self.prime)
        return ResidueProfile(
            input_terms=input_terms,
            result=result,
            stages=tuple(stages),
        )

    def _initial_spmat_states(
        self,
        cleaned: SparsePoly,
        np,
        sparse,
    ) -> "_SpMatStates":
        alpha_width = self.roots.y_count
        count = len(cleaned)
        alpha = np.empty((count, alpha_width), dtype=np.uint16)
        coeff = np.empty(count, dtype=np.int64)
        for row_idx, (alpha_key, value) in enumerate(cleaned.items()):
            alpha_t = _validate_alpha(alpha_key, alpha_width, "alpha")
            if any(item > _ARRAY_BACKEND_EXP_MAX for item in alpha_t):
                raise ValueError(
                    "array residue backend supports exponents at most "
                    f"{_ARRAY_BACKEND_EXP_MAX}"
                )
            alpha[row_idx, :] = alpha_t
            coeff[row_idx] = int(value) % self.prime
        return self._initial_spmat_states_from_arrays(alpha, coeff, np, sparse)

    def _initial_spmat_states_from_arrays(
        self,
        alpha_terms,
        coeff_terms,
        np,
        sparse,
    ) -> "_SpMatStates":
        alpha = np.asarray(alpha_terms, dtype=np.uint16)
        coeff = np.asarray(coeff_terms, dtype=np.int64) % self.prime
        if alpha.ndim != 2 or alpha.shape[1] != self.roots.y_count:
            raise ValueError(
                "alpha term array must have shape "
                f"(n, {self.roots.y_count})"
            )
        if coeff.ndim != 1 or coeff.shape[0] != alpha.shape[0]:
            raise ValueError("coefficient array length must match alpha terms")
        if self.root_power > _ARRAY_BACKEND_EXP_MAX:
            raise ValueError(
                "array residue backend supports denominator powers at most "
                f"{_ARRAY_BACKEND_EXP_MAX}"
            )
        keep = coeff != 0
        alpha = alpha[keep]
        coeff = coeff[keep].astype(np.int64, copy=False)
        if not coeff.size:
            return _empty_spmat_states(
                self.roots.y_count,
                self.roots.positive_root_count,
                np,
                sparse,
            )

        alpha_unique, alpha_inverse = _unique_uint16_rows(alpha, np)
        denom = np.full(
            (1, self.roots.positive_root_count),
            self.root_power,
            dtype=np.uint16,
        )
        matrix = sparse.coo_matrix(
            (
                coeff,
                (alpha_inverse, np.zeros(alpha_inverse.shape[0], dtype=np.int64)),
            ),
            shape=(alpha_unique.shape[0], 1),
            dtype=np.int64,
        ).tocsr()
        matrix.sum_duplicates()
        matrix.data %= self.prime
        matrix.eliminate_zeros()
        return _SpMatStates(alpha=alpha_unique, denom=denom, matrix=matrix)

    def _initial_spmat_states_from_unique_arrays(
        self,
        alpha_terms,
        coeff_terms,
        np,
        sparse,
    ) -> "_SpMatStates":
        alpha = np.asarray(alpha_terms, dtype=np.uint16)
        coeff = np.asarray(coeff_terms, dtype=np.int64) % self.prime
        if alpha.ndim != 2 or alpha.shape[1] != self.roots.y_count:
            raise ValueError(
                "alpha term array must have shape "
                f"(n, {self.roots.y_count})"
            )
        if coeff.ndim != 1 or coeff.shape[0] != alpha.shape[0]:
            raise ValueError("coefficient array length must match alpha terms")
        if self.root_power > _ARRAY_BACKEND_EXP_MAX:
            raise ValueError(
                "array residue backend supports denominator powers at most "
                f"{_ARRAY_BACKEND_EXP_MAX}"
            )
        keep = coeff != 0
        alpha = alpha[keep]
        coeff = coeff[keep].astype(np.int64, copy=False)
        if not coeff.size:
            return _empty_spmat_states(
                self.roots.y_count,
                self.roots.positive_root_count,
                np,
                sparse,
            )

        order = _lexsort_matrix(alpha, np)
        sorted_alpha = alpha[order]
        sorted_coeff = coeff[order]
        changes = np.empty(sorted_alpha.shape[0], dtype=np.bool_)
        changes[0] = True
        changes[1:] = np.any(sorted_alpha[1:] != sorted_alpha[:-1], axis=1)
        if np.all(changes):
            alpha_unique = sorted_alpha.copy()
            coeff_unique = sorted_coeff.astype(np.int64, copy=False)
        else:
            starts = np.nonzero(changes)[0]
            sums = np.add.reduceat(sorted_coeff, starts) % self.prime
            nonzero = sums != 0
            alpha_unique = sorted_alpha[starts][nonzero].copy()
            coeff_unique = sums[nonzero].astype(np.int64, copy=False)
            if not coeff_unique.size:
                return _empty_spmat_states(
                    self.roots.y_count,
                    self.roots.positive_root_count,
                    np,
                    sparse,
                )

        denom = np.full(
            (1, self.roots.positive_root_count),
            self.root_power,
            dtype=np.uint16,
        )
        row_count = int(coeff_unique.size)
        matrix = sparse.csr_matrix(
            (
                coeff_unique.astype(np.int64, copy=True),
                np.zeros(row_count, dtype=np.int32),
                np.arange(row_count + 1, dtype=np.int64),
            ),
            shape=(row_count, 1),
            dtype=np.int64,
        )
        return _SpMatStates(alpha=alpha_unique, denom=denom, matrix=matrix)

    def _spmat_states_from_array_states(
        self,
        states: "_ArrayStates",
        np,
        sparse,
    ) -> "_SpMatStates":
        if not states.coeff.size:
            return _empty_spmat_states(
                states.alpha.shape[1],
                states.denom.shape[1],
                np,
                sparse,
            )
        alpha_unique, alpha_inverse = _unique_uint16_rows(states.alpha, np)
        denom_unique, denom_inverse = _unique_uint16_rows(states.denom, np)
        matrix = sparse.coo_matrix(
            (
                states.coeff.astype(np.int64, copy=False),
                (alpha_inverse, denom_inverse),
            ),
            shape=(alpha_unique.shape[0], denom_unique.shape[0]),
            dtype=np.int64,
        ).tocsr()
        matrix.sum_duplicates()
        matrix.data %= self.prime
        matrix.eliminate_zeros()
        return _SpMatStates(alpha=alpha_unique, denom=denom_unique, matrix=matrix)

    def _eliminate_stage_factored_spmat(
        self,
        states: "_SpMatStates",
        spec: _StageSpec,
        np,
        sparse,
    ) -> "_SpMatStates":
        alpha_out_width = states.alpha.shape[1] - 1
        denom_out_width = len(spec.after_positions)
        if not states.matrix.nnz:
            return _empty_spmat_states(alpha_out_width, denom_out_width, np, sparse)

        denom_count = int(states.denom.shape[0])
        lower_unique, local_axis, state_matrix = _fold_spmat_by_last_alpha(
            states.alpha,
            states.matrix,
            denom_count,
            np,
            sparse,
        )
        if not state_matrix.nnz:
            return _empty_spmat_states(alpha_out_width, denom_out_width, np, sparse)

        transition_cache_key = _transition_spmat_cache_key(
            spec.var_idx,
            states.denom,
            local_axis,
        )
        cached_transition = self._transition_spmat_cache.get(transition_cache_key)
        global_transition_cache_key = _global_transition_spmat_cache_key(
            self.rank,
            self.root_power,
            self.prime,
            self.derivative_orders[spec.var_idx],
            transition_cache_key,
        )
        if cached_transition is None:
            with _GLOBAL_TRANSITION_SPMAT_CACHE_LOCK:
                cached_transition = _GLOBAL_TRANSITION_SPMAT_CACHE.get(
                    global_transition_cache_key
                )
            if cached_transition is not None:
                self._transition_spmat_cache[transition_cache_key] = cached_transition
        if cached_transition is None:
            transition_rows: list[int] = []
            transition_cols: list[int] = []
            transition_data: list[int] = []
            next_denom_index: dict[DenominatorPowers, int] = {}
            next_denoms: list[DenominatorPowers] = []
            for local_idx, code in enumerate(local_axis.present_codes):
                y_exp = int(code // denom_count)
                denom_idx = int(code % denom_count)
                denom_key = tuple(int(item) for item in states.denom[denom_idx])
                for next_denom, local_coeff in self._local_transition(
                    spec,
                    y_exp,
                    denom_key,
                ):
                    next_col = next_denom_index.get(next_denom)
                    if next_col is None:
                        next_col = len(next_denoms)
                        next_denom_index[next_denom] = next_col
                        next_denoms.append(next_denom)
                    transition_rows.append(
                        int(code) if local_axis.direct else int(local_idx)
                    )
                    transition_cols.append(next_col)
                    transition_data.append(int(local_coeff))

            if not transition_data:
                next_denom_array = np.empty((0, denom_out_width), dtype=np.uint16)
                transition_matrix = sparse.csr_matrix(
                    (local_axis.column_count, 0),
                    dtype=np.int64,
                )
            else:
                transition_matrix = _csr_from_grouped_rows(
                    np.asarray(transition_data, dtype=np.int64),
                    np.asarray(transition_rows, dtype=np.int64),
                    np.asarray(transition_cols, dtype=np.int64),
                    shape=(local_axis.column_count, len(next_denoms)),
                    np=np,
                    sparse=sparse,
                )
                transition_matrix.data %= self.prime
                transition_matrix.eliminate_zeros()
                next_denom_array = np.asarray(next_denoms, dtype=np.uint16)
                if next_denom_array.ndim == 1:
                    next_denom_array = next_denom_array.reshape(
                        (len(next_denoms), 0)
                    )

            if len(self._transition_spmat_cache) >= _TRANSITION_SPMAT_CACHE_MAX:
                self._transition_spmat_cache.clear()
            self._transition_spmat_cache[transition_cache_key] = (
                next_denom_array,
                transition_matrix,
            )
            with _GLOBAL_TRANSITION_SPMAT_CACHE_LOCK:
                if (
                    len(_GLOBAL_TRANSITION_SPMAT_CACHE)
                    >= _GLOBAL_TRANSITION_SPMAT_CACHE_MAX
                ):
                    _GLOBAL_TRANSITION_SPMAT_CACHE.clear()
                _GLOBAL_TRANSITION_SPMAT_CACHE[global_transition_cache_key] = (
                    next_denom_array,
                    transition_matrix,
                )
        else:
            next_denom_array, transition_matrix = cached_transition

        if not transition_matrix.nnz:
            return _empty_spmat_states(alpha_out_width, denom_out_width, np, sparse)

        product = state_matrix @ transition_matrix
        product.data %= self.prime
        product.eliminate_zeros()
        if not product.nnz:
            return _empty_spmat_states(alpha_out_width, denom_out_width, np, sparse)

        return _SpMatStates(
            alpha=lower_unique,
            denom=next_denom_array,
            matrix=product.tocsr(),
        )

    def _eliminate_stage_spmat(
        self,
        states: "_ArrayStates",
        spec: _StageSpec,
        np,
        sparse,
    ) -> "_ArrayStates":
        state_count = int(states.coeff.size)
        alpha_out_width = states.alpha.shape[1] - 1
        denom_out_width = len(spec.after_positions)
        if state_count == 0:
            return _empty_array_states(alpha_out_width, denom_out_width, np)

        lower_key = states.alpha[:, :alpha_out_width]
        local_key = np.concatenate((states.alpha[:, -1:], states.denom), axis=1)
        lower_unique, lower_inverse = _unique_uint16_rows(lower_key, np)
        local_unique, local_inverse = _unique_uint16_rows(local_key, np)

        state_matrix = sparse.coo_matrix(
            (
                states.coeff.astype(np.int64, copy=False),
                (lower_inverse, local_inverse),
            ),
            shape=(lower_unique.shape[0], local_unique.shape[0]),
            dtype=np.int64,
        ).tocsr()
        state_matrix.sum_duplicates()
        state_matrix.data %= self.prime
        state_matrix.eliminate_zeros()
        if not state_matrix.nnz:
            return _empty_array_states(alpha_out_width, denom_out_width, np)

        transition_rows: list[int] = []
        transition_cols: list[int] = []
        transition_data: list[int] = []
        next_denom_index: dict[DenominatorPowers, int] = {}
        next_denoms: list[DenominatorPowers] = []
        for local_idx, row in enumerate(local_unique):
            y_exp = int(row[0])
            denom_key = tuple(int(item) for item in row[1:])
            for next_denom, local_coeff in self._local_transition(
                spec,
                y_exp,
                denom_key,
            ):
                next_col = next_denom_index.get(next_denom)
                if next_col is None:
                    next_col = len(next_denoms)
                    next_denom_index[next_denom] = next_col
                    next_denoms.append(next_denom)
                transition_rows.append(local_idx)
                transition_cols.append(next_col)
                transition_data.append(int(local_coeff))

        if not transition_data:
            return _empty_array_states(alpha_out_width, denom_out_width, np)

        transition_matrix = sparse.coo_matrix(
            (
                np.asarray(transition_data, dtype=np.int64),
                (
                    np.asarray(transition_rows, dtype=np.int64),
                    np.asarray(transition_cols, dtype=np.int64),
                ),
            ),
            shape=(local_unique.shape[0], len(next_denoms)),
            dtype=np.int64,
        ).tocsr()
        transition_matrix.sum_duplicates()
        transition_matrix.data %= self.prime
        transition_matrix.eliminate_zeros()
        if not transition_matrix.nnz:
            return _empty_array_states(alpha_out_width, denom_out_width, np)

        product = state_matrix @ transition_matrix
        product.data %= self.prime
        product.eliminate_zeros()
        if not product.nnz:
            return _empty_array_states(alpha_out_width, denom_out_width, np)

        product_coo = product.tocoo()
        out_alpha = lower_unique[product_coo.row].astype(np.uint16, copy=True)
        next_denom_array = np.asarray(next_denoms, dtype=np.uint16)
        out_denom = next_denom_array[product_coo.col].astype(np.uint16, copy=True)
        out_coeff = product_coo.data.astype(np.int64, copy=False)
        return _ArrayStates(alpha=out_alpha, denom=out_denom, coeff=out_coeff)

    def _profile_poly_terms_array(self, poly: SparsePoly) -> ResidueProfile:
        np = _require_numpy()
        cleaned = clean(poly, self.prime)
        states = self._initial_array_states(cleaned, np)
        return self._profile_array_states(states, len(cleaned))

    def _profile_array_states(
        self,
        states: "_ArrayStates",
        input_terms: int,
    ) -> ResidueProfile:
        np = _require_numpy()
        stages = []
        for spec in reversed(self.stage_specs):
            hits_before = self.local_cache_hits
            misses_before = self.local_cache_misses
            start = perf_counter()
            input_states = int(states.coeff.size)
            states = self._eliminate_stage_array(states, spec, np)
            elapsed = perf_counter() - start
            stages.append(
                ResidueStageProfile(
                    var_idx=spec.var_idx,
                    input_states=input_states,
                    output_states=int(states.coeff.size),
                    elapsed_seconds=elapsed,
                    local_cache_hits=self.local_cache_hits - hits_before,
                    local_cache_misses=self.local_cache_misses - misses_before,
                )
            )
            if not states.coeff.size:
                return ResidueProfile(
                    input_terms=input_terms,
                    result=0,
                    stages=tuple(stages),
                )

        result = int(states.coeff[0] % self.prime) if states.coeff.size else 0
        return ResidueProfile(
            input_terms=input_terms,
            result=result,
            stages=tuple(stages),
        )

    def _initial_array_states(self, cleaned: SparsePoly, np) -> "_ArrayStates":
        alpha_width = self.roots.y_count
        denom_width = self.roots.positive_root_count
        count = len(cleaned)
        alpha = np.empty((count, alpha_width), dtype=np.uint16)
        denom = np.full((count, denom_width), self.root_power, dtype=np.uint16)
        coeff = np.empty(count, dtype=np.int64)

        for row_idx, (alpha_key, value) in enumerate(cleaned.items()):
            alpha_t = _validate_alpha(alpha_key, alpha_width, "alpha")
            if any(item > _ARRAY_BACKEND_EXP_MAX for item in alpha_t):
                raise ValueError(
                    "array residue backend supports exponents at most "
                    f"{_ARRAY_BACKEND_EXP_MAX}"
                )
            alpha[row_idx, :] = alpha_t
            coeff[row_idx] = int(value) % self.prime
        if self.root_power > _ARRAY_BACKEND_EXP_MAX:
            raise ValueError(
                "array residue backend supports denominator powers at most "
                f"{_ARRAY_BACKEND_EXP_MAX}"
            )
        return _ArrayStates(alpha=alpha, denom=denom, coeff=coeff)

    def _initial_array_states_from_arrays(
        self,
        alpha_terms,
        coeff_terms,
        np,
    ) -> "_ArrayStates":
        alpha = np.asarray(alpha_terms, dtype=np.uint16)
        coeff = np.asarray(coeff_terms, dtype=np.int64) % self.prime
        if alpha.ndim != 2 or alpha.shape[1] != self.roots.y_count:
            raise ValueError(
                "alpha term array must have shape "
                f"(n, {self.roots.y_count})"
            )
        if coeff.ndim != 1 or coeff.shape[0] != alpha.shape[0]:
            raise ValueError("coefficient array length must match alpha terms")
        if self.root_power > _ARRAY_BACKEND_EXP_MAX:
            raise ValueError(
                "array residue backend supports denominator powers at most "
                f"{_ARRAY_BACKEND_EXP_MAX}"
            )
        keep = coeff != 0
        alpha = alpha[keep].copy()
        coeff = coeff[keep].astype(np.int64, copy=False)
        denom = np.full(
            (alpha.shape[0], self.roots.positive_root_count),
            self.root_power,
            dtype=np.uint16,
        )
        return _ArrayStates(alpha=alpha, denom=denom, coeff=coeff)

    def _eliminate_stage_array(
        self,
        states: "_ArrayStates",
        spec: _StageSpec,
        np,
    ) -> "_ArrayStates":
        state_count = int(states.coeff.size)
        alpha_out_width = states.alpha.shape[1] - 1
        denom_out_width = len(spec.after_positions)
        if state_count == 0:
            return _empty_array_states(alpha_out_width, denom_out_width, np)

        local_key = np.concatenate(
            (states.alpha[:, -1:].astype(np.uint16, copy=False), states.denom),
            axis=1,
        )
        local_order = _lexsort_matrix(local_key, np)
        sorted_local_key = local_key[local_order]
        local_changes = np.empty(state_count, dtype=np.bool_)
        local_changes[0] = True
        local_changes[1:] = np.any(
            sorted_local_key[1:] != sorted_local_key[:-1],
            axis=1,
        )
        group_starts = np.nonzero(local_changes)[0]
        group_ends = np.empty_like(group_starts)
        group_ends[:-1] = group_starts[1:]
        group_ends[-1] = state_count

        group_plans = []
        total_emissions = 0
        for start, end in zip(group_starts, group_ends):
            first = int(local_order[start])
            y_exp = int(states.alpha[first, -1])
            denom_key = tuple(int(item) for item in states.denom[first])
            transition = self._local_transition(spec, y_exp, denom_key)
            group_size = end - start
            total_emissions += group_size * len(transition)
            group_plans.append((start, end, transition))

        if total_emissions == 0:
            return _empty_array_states(alpha_out_width, denom_out_width, np)

        out_alpha = np.empty((total_emissions, alpha_out_width), dtype=np.uint16)
        out_denom = np.empty((total_emissions, denom_out_width), dtype=np.uint16)
        out_coeff = np.empty(total_emissions, dtype=np.int64)

        cursor = 0
        for start, end, transition in group_plans:
            group_indices = local_order[start:end]
            group_alpha = states.alpha[group_indices, :alpha_out_width]
            group_coeff = states.coeff[group_indices]
            group_size = int(group_indices.size)
            for next_denom, local_coeff in transition:
                if any(item > _ARRAY_BACKEND_EXP_MAX for item in next_denom):
                    raise ValueError(
                        "array residue backend supports denominator powers at most "
                        f"{_ARRAY_BACKEND_EXP_MAX}"
                    )
                next_cursor = cursor + group_size
                if alpha_out_width:
                    out_alpha[cursor:next_cursor, :] = group_alpha
                if denom_out_width:
                    out_denom[cursor:next_cursor, :] = next_denom
                out_coeff[cursor:next_cursor] = (
                    group_coeff * int(local_coeff)
                ) % self.prime
                cursor = next_cursor

        return self._reduce_array_states(
            out_alpha,
            out_denom,
            out_coeff,
            np,
        )

    def _reduce_array_states(
        self,
        alpha,
        denom,
        coeff,
        np,
    ) -> "_ArrayStates":
        row_count = int(coeff.size)
        if row_count == 0:
            return _ArrayStates(alpha=alpha, denom=denom, coeff=coeff)

        key_width = alpha.shape[1] + denom.shape[1]
        if key_width == 0:
            value = int(np.sum(coeff, dtype=np.int64) % self.prime)
            if not value:
                return _empty_array_states(0, 0, np)
            return _ArrayStates(
                alpha=np.empty((1, 0), dtype=np.uint16),
                denom=np.empty((1, 0), dtype=np.uint16),
                coeff=np.array([value], dtype=np.int64),
            )

        key = np.concatenate((alpha, denom), axis=1)
        order = _lexsort_matrix(key, np)
        sorted_key = key[order]
        sorted_coeff = coeff[order]

        changes = np.empty(row_count, dtype=np.bool_)
        changes[0] = True
        changes[1:] = np.any(sorted_key[1:] != sorted_key[:-1], axis=1)
        starts = np.nonzero(changes)[0]
        sums = np.add.reduceat(sorted_coeff, starts) % self.prime
        keep = sums != 0
        if not np.any(keep):
            return _empty_array_states(alpha.shape[1], denom.shape[1], np)

        unique_key = sorted_key[starts[keep]]
        alpha_width = alpha.shape[1]
        return _ArrayStates(
            alpha=unique_key[:, :alpha_width].copy(),
            denom=unique_key[:, alpha_width:].copy(),
            coeff=sums[keep].astype(np.int64, copy=False),
        )

    def _initial_states(self, poly: SparsePoly) -> Dict[SlicedStateKey, int]:
        cleaned = clean(poly, self.prime)
        root_powers = tuple(self.root_power for _ in range(self.roots.positive_root_count))
        states: Dict[SlicedStateKey, int] = {}
        for alpha, coeff in cleaned.items():
            alpha_t = _validate_alpha(alpha, self.roots.y_count, "alpha")
            key = (alpha_t, root_powers)
            states[key] = (states.get(key, 0) + coeff) % self.prime
        return {key: value for key, value in states.items() if value}

    def _eliminate_stage(
        self,
        states: Dict[SlicedStateKey, int],
        spec: _StageSpec,
    ) -> Dict[SlicedStateKey, int]:
        next_terms: Dict[SlicedStateKey, int] = {}
        for (alpha, denom), coeff in states.items():
            local = self._local_transition(spec, alpha[-1], denom)
            next_alpha = alpha[:-1]
            for next_denom, local_coeff in local:
                key = (next_alpha, next_denom)
                next_terms[key] = (
                    next_terms.get(key, 0) + coeff * local_coeff
                ) % self.prime
        return {key: value for key, value in next_terms.items() if value}

    def _local_transition(
        self,
        spec: _StageSpec,
        y_exp: int,
        denom: DenominatorPowers,
    ) -> Tuple[Tuple[DenominatorPowers, int], ...]:
        key = (spec.var_idx, int(y_exp), denom)
        cached = self._local_cache.get(key)
        if cached is not None:
            self.local_cache_hits += 1
            return cached
        self.local_cache_misses += 1

        local: Dict[Tuple[int, DenominatorPowers], int] = {(int(y_exp), denom): 1}
        for pos, lower_pos in spec.schedule:
            local = self._expand_root(local, spec, pos, lower_pos)
            if not local:
                self._local_cache[key] = ()
                return ()

        out: Dict[DenominatorPowers, int] = {}
        for (next_y_exp, dtuple), coeff in local.items():
            special = self._special_coeff(
                spec.var_idx,
                -1 - next_y_exp,
            )
            if not special:
                continue
            projected = tuple(dtuple[idx] for idx in spec.after_before_indices)
            out[projected] = (out.get(projected, 0) + coeff * special) % self.prime

        result = tuple((dtuple, coeff) for dtuple, coeff in out.items() if coeff)
        self._local_cache[key] = result
        return result

    def _special_coeff(self, var_idx: int, exponent: int) -> int:
        key = (int(var_idx), int(exponent))
        cached = self._special_cache.get(key)
        if cached is not None:
            return cached
        value = _special_coeff_mod(
            self.rank,
            key[0],
            self.derivative_orders[key[0]],
            key[1],
            self.prime,
        )
        self._special_cache[key] = value
        return value

    def _expand_root(
        self,
        local: Dict[Tuple[int, DenominatorPowers], int],
        spec: _StageSpec,
        pos: int,
        lower_pos: int,
    ) -> Dict[Tuple[int, DenominatorPowers], int]:
        current_idx = spec.full_to_before[pos]
        lower_idx = spec.full_to_before.get(lower_pos)
        expanded: Dict[Tuple[int, DenominatorPowers], int] = {}
        for (cur_y_exp, dtuple), state_coeff in local.items():
            current_power = int(dtuple[current_idx])
            if not current_power:
                key = (cur_y_exp, dtuple)
                expanded[key] = (expanded.get(key, 0) + state_coeff) % self.prime
                continue

            base_den = list(dtuple)
            base_den[current_idx] = 0
            simple_drop = (
                int(base_den[spec.simple_before_idx])
                if pos < spec.simple_pos
                else 0
            )
            y_bound = self.derivative_orders[spec.var_idx] + simple_drop
            if lower_idx is None:
                next_y_exp = cur_y_exp - current_power
                if next_y_exp <= y_bound:
                    key = (next_y_exp, tuple(base_den))
                    expanded[key] = (expanded.get(key, 0) + state_coeff) % self.prime
                continue

            max_m = y_bound - cur_y_exp
            if max_m < 0:
                continue
            for m in range(max_m + 1):
                expanded_den = list(base_den)
                expanded_den[lower_idx] += current_power + m
                binom = comb(current_power + m - 1, m)
                if m % 2:
                    binom = -binom
                key = (cur_y_exp + m, tuple(expanded_den))
                expanded[key] = (
                    expanded.get(key, 0) + state_coeff * binom
                ) % self.prime
        return {key: value for key, value in expanded.items() if value}


def _stage_specs(roots: TypeARootSystem) -> Tuple[_StageSpec, ...]:
    specs = []
    intervals = roots.positive_intervals_zero_based
    for var_idx in range(roots.y_count):
        before_positions = tuple(
            pos for pos, interval in enumerate(intervals) if interval[1] <= var_idx + 1
        )
        after_positions = tuple(
            pos for pos, interval in enumerate(intervals) if interval[1] <= var_idx
        )
        full_to_before = {pos: idx for idx, pos in enumerate(before_positions)}
        simple_pos = roots.interval_index[(var_idx, var_idx + 1)]
        specs.append(
            _StageSpec(
                var_idx=var_idx,
                before_positions=before_positions,
                after_positions=after_positions,
                schedule=roots.transition_schedule[var_idx],
                full_to_before=full_to_before,
                after_before_indices=tuple(full_to_before[pos] for pos in after_positions),
                simple_pos=simple_pos,
                simple_before_idx=full_to_before[simple_pos],
            )
        )
    return tuple(specs)


def _numpy_available() -> bool:
    try:
        import numpy  # noqa: F401
    except ImportError:
        return False
    return True


def _scipy_available() -> bool:
    try:
        import scipy.sparse  # noqa: F401
    except ImportError:
        return False
    return _numpy_available()


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "array residue backend requires numpy; use backend='python' instead"
        ) from exc
    return np


def _require_scipy_sparse():
    try:
        import scipy.sparse as sparse
    except ImportError as exc:
        raise RuntimeError(
            "spmat residue backend requires scipy; use backend='array' instead"
        ) from exc
    return sparse


def _empty_array_states(alpha_width: int, denom_width: int, np) -> _ArrayStates:
    return _ArrayStates(
        alpha=np.empty((0, alpha_width), dtype=np.uint16),
        denom=np.empty((0, denom_width), dtype=np.uint16),
        coeff=np.empty(0, dtype=np.int64),
    )


def _empty_spmat_states(alpha_width: int, denom_width: int, np, sparse) -> _SpMatStates:
    return _SpMatStates(
        alpha=np.empty((0, alpha_width), dtype=np.uint16),
        denom=np.empty((0, denom_width), dtype=np.uint16),
        matrix=sparse.csr_matrix((0, 0), dtype=np.int64),
    )


def _fold_spmat_by_last_alpha(alpha, matrix, denom_count: int, np, sparse):
    row_count = int(alpha.shape[0])
    alpha_out_width = int(alpha.shape[1] - 1)
    if row_count == 0:
        local_axis = _LocalCodeAxis(
            present_codes=np.empty(0, dtype=np.int64),
            inverse=np.empty(0, dtype=np.int64),
            column_count=0,
            direct=True,
        )
        return (
            np.empty((0, alpha_out_width), dtype=np.uint16),
            local_axis,
            sparse.csr_matrix((0, 0), dtype=np.int64),
        )

    # The spmat pipeline keeps alpha rows lexicographically sorted, so rows with
    # the same lower alpha are consecutive.  Folding directly avoids a COO
    # round trip and a second lexicographic sort at every residue stage.
    if alpha_out_width:
        lower_alpha = alpha[:, :alpha_out_width]
        changes = np.empty(row_count, dtype=np.bool_)
        changes[0] = True
        changes[1:] = np.any(lower_alpha[1:] != lower_alpha[:-1], axis=1)
        group_starts = np.nonzero(changes)[0]
        lower_unique = lower_alpha[group_starts].copy()
    else:
        group_starts = np.array([0], dtype=np.int64)
        lower_unique = np.empty((1, 0), dtype=np.uint16)

    row_counts = np.diff(matrix.indptr).astype(np.int64, copy=False)
    group_counts = np.add.reduceat(row_counts, group_starts)
    indptr = np.empty(group_starts.size + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(group_counts, out=indptr[1:])

    row_offsets = alpha[:, -1].astype(np.int64) * int(denom_count)
    repeated_offsets = np.repeat(row_offsets, row_counts)
    folded_indices = matrix.indices.astype(np.int64, copy=False) + repeated_offsets
    local_axis = _local_code_axis(folded_indices, np)
    index_dtype = (
        np.int32 if local_axis.column_count <= np.iinfo(np.int32).max else np.int64
    )
    state_matrix = sparse.csr_matrix(
        (
            matrix.data.astype(np.int64, copy=True),
            local_axis.inverse.astype(index_dtype, copy=True),
            indptr,
        ),
        shape=(lower_unique.shape[0], local_axis.column_count),
        dtype=np.int64,
    )
    return lower_unique, local_axis, state_matrix


def _unique_uint16_rows(matrix, np):
    row_count, width = matrix.shape
    if row_count == 0:
        return matrix.copy(), np.empty(0, dtype=np.int64)
    if width == 0:
        return (
            np.empty((1, 0), dtype=np.uint16),
            np.zeros(row_count, dtype=np.int64),
        )

    order = _lexsort_matrix(matrix, np)
    sorted_matrix = matrix[order]
    changes = np.empty(row_count, dtype=np.bool_)
    changes[0] = True
    changes[1:] = np.any(sorted_matrix[1:] != sorted_matrix[:-1], axis=1)
    starts = np.nonzero(changes)[0]
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = row_count
    group_sizes = ends - starts
    sorted_inverse = np.repeat(np.arange(starts.size, dtype=np.int64), group_sizes)
    inverse = np.empty(row_count, dtype=np.int64)
    inverse[order] = sorted_inverse
    return sorted_matrix[starts].copy(), inverse


def _local_code_axis(codes, np) -> _LocalCodeAxis:
    if not codes.size:
        return _LocalCodeAxis(
            present_codes=codes.copy(),
            inverse=np.empty(0, dtype=np.int64),
            column_count=0,
            direct=True,
        )
    max_code = int(codes.max())
    if 0 <= max_code <= 5_000_000:
        present = np.zeros(max_code + 1, dtype=np.bool_)
        present[codes] = True
        return _LocalCodeAxis(
            present_codes=np.nonzero(present)[0].astype(codes.dtype, copy=False),
            inverse=codes.astype(np.int64, copy=False),
            column_count=max_code + 1,
            direct=True,
        )

    unique, inverse = np.unique(codes, return_inverse=True)
    return _LocalCodeAxis(
        present_codes=unique,
        inverse=inverse,
        column_count=int(unique.size),
        direct=False,
    )


def _transition_spmat_cache_key(
    var_idx: int,
    denom,
    local_axis: _LocalCodeAxis,
):
    return (
        int(var_idx),
        tuple(int(item) for item in denom.shape),
        denom.tobytes(),
        int(local_axis.column_count),
        bool(local_axis.direct),
        local_axis.present_codes.astype("int64", copy=False).tobytes(),
    )


def _global_transition_spmat_cache_key(
    rank: int,
    root_power: int,
    prime: int,
    derivative_order: int,
    transition_cache_key,
):
    return (
        int(rank),
        int(root_power),
        int(prime),
        int(derivative_order),
        transition_cache_key,
    )


def _csr_from_grouped_rows(data, rows, cols, *, shape, np, sparse):
    if not data.size:
        return sparse.csr_matrix(shape, dtype=np.int64)
    if np.any(rows[1:] < rows[:-1]):
        return sparse.coo_matrix((data, (rows, cols)), shape=shape, dtype=np.int64).tocsr()

    indptr = np.empty(shape[0] + 1, dtype=np.int64)
    indptr[0] = 0
    counts = np.bincount(rows, minlength=shape[0])
    np.cumsum(counts, out=indptr[1:])
    index_dtype = np.int32 if shape[1] <= np.iinfo(np.int32).max else np.int64
    return sparse.csr_matrix(
        (
            data.astype(np.int64, copy=True),
            cols.astype(index_dtype, copy=True),
            indptr,
        ),
        shape=shape,
        dtype=np.int64,
    )


def _lexsort_matrix(matrix, np):
    if matrix.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    if matrix.shape[1] == 0:
        return np.arange(matrix.shape[0], dtype=np.int64)
    return np.lexsort(tuple(matrix[:, idx] for idx in range(matrix.shape[1] - 1, -1, -1)))
