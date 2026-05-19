"""Packed/sliced exact residue functional.

This module keeps the same iterated JK transition as ``residue_transition`` but
stores only the alpha coordinates and denominator powers still alive at each
variable-elimination stage.  It is a speed-oriented representation, not a new
formula.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from time import perf_counter
from typing import Dict, Tuple

from .mod_arith import require_prime
from .residue_transition import _special_coeff_mod, _validate_alpha
from .root_system import TypeARootSystem, type_a_roots
from .sparse_poly import Alpha, SparsePoly, clean

DenominatorPowers = Tuple[int, ...]
SlicedStateKey = Tuple[Alpha, DenominatorPowers]


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


@dataclass
class ResidueFunctional:
    rank: int
    derivative_orders: Tuple[int, ...]
    root_power: int
    prime: int
    roots: TypeARootSystem = field(init=False)
    stage_specs: Tuple[_StageSpec, ...] = field(init=False)
    _local_cache: dict[
        Tuple[int, int, DenominatorPowers],
        Tuple[Tuple[DenominatorPowers, int], ...],
    ] = field(
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
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "derivative_orders", derivative_orders)
        object.__setattr__(self, "root_power", int(self.root_power))
        object.__setattr__(self, "prime", require_prime(self.prime))
        object.__setattr__(self, "roots", roots)
        object.__setattr__(self, "stage_specs", _stage_specs(roots))

    def evaluate_poly_terms(self, poly: SparsePoly) -> int:
        return self.profile_poly_terms(poly).result

    def profile_poly_terms(self, poly: SparsePoly) -> ResidueProfile:
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
            special = _special_coeff_mod(
                self.rank,
                spec.var_idx,
                self.derivative_orders[spec.var_idx],
                -1 - next_y_exp,
                self.prime,
            )
            if not special:
                continue
            projected = tuple(dtuple[idx] for idx in spec.after_before_indices)
            out[projected] = (out.get(projected, 0) + coeff * special) % self.prime

        result = tuple(sorted((dtuple, coeff) for dtuple, coeff in out.items() if coeff))
        self._local_cache[key] = result
        return result

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
