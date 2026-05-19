"""Rank-7 residue-transition smoke cases for Gate C."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from .config import RANK7_G2_D1
from .residue_transition import residue_monomial_mod


@dataclass(frozen=True)
class ResidueSmokeCase:
    name: str
    alpha: Tuple[int, ...]
    derivative_orders: Tuple[int, ...]
    root_power: int
    expected_primary_mod: int
    expected_second_mod: int

    def expected_for_prime(self, prime: int) -> int:
        if int(prime) == RANK7_G2_D1.primary_prime:
            return self.expected_primary_mod
        if int(prime) == RANK7_G2_D1.second_prime:
            return self.expected_second_mod
        raise ValueError(f"no frozen expected value for prime {prime}")


RANK7_RESIDUE_SMOKE_CASES: Tuple[ResidueSmokeCase, ...] = (
    ResidueSmokeCase(
        name="rank7_y3_root_power2",
        alpha=(0, 0, 1, 0, 0, 0),
        derivative_orders=(0, 0, 0, 0, 0, 0),
        root_power=2,
        expected_primary_mod=503975831935843257,
        expected_second_mod=599903,
    ),
    ResidueSmokeCase(
        name="rank7_y4_root_power2",
        alpha=(0, 0, 0, 1, 0, 0),
        derivative_orders=(0, 0, 0, 0, 0, 0),
        root_power=2,
        expected_primary_mod=2176777278537831732,
        expected_second_mod=133696,
    ),
    ResidueSmokeCase(
        name="rank7_y5_root_power2",
        alpha=(0, 0, 0, 0, 1, 0),
        derivative_orders=(0, 0, 0, 0, 0, 0),
        root_power=2,
        expected_primary_mod=395929348375758526,
        expected_second_mod=634489,
    ),
    ResidueSmokeCase(
        name="rank7_y6_squared_root_power2",
        alpha=(0, 0, 0, 0, 0, 2),
        derivative_orders=(0, 0, 0, 0, 0, 0),
        root_power=2,
        expected_primary_mod=2223454107303524158,
        expected_second_mod=580619,
    ),
    ResidueSmokeCase(
        name="rank7_y6_with_last_derivative_root_power2",
        alpha=(0, 0, 0, 0, 0, 1),
        derivative_orders=(0, 0, 0, 0, 0, 1),
        root_power=2,
        expected_primary_mod=20106287579861699,
        expected_second_mod=353916,
    ),
)


def run_residue_smoke_cases(
    *,
    primes: Sequence[int] = (RANK7_G2_D1.primary_prime, RANK7_G2_D1.second_prime),
) -> Tuple[dict[str, object], ...]:
    rows = []
    for case in RANK7_RESIDUE_SMOKE_CASES:
        observed_by_prime = {}
        expected_by_prime = {}
        passed_by_prime = {}
        for prime in primes:
            observed = residue_monomial_mod(
                RANK7_G2_D1.rank,
                case.alpha,
                case.derivative_orders,
                prime=int(prime),
                root_power=case.root_power,
            )
            expected = case.expected_for_prime(int(prime))
            observed_by_prime[str(prime)] = observed
            expected_by_prime[str(prime)] = expected
            passed_by_prime[str(prime)] = observed == expected
        rows.append(
            {
                "name": case.name,
                "alpha": list(case.alpha),
                "derivative_orders": list(case.derivative_orders),
                "root_power": case.root_power,
                "observed_mod": observed_by_prime,
                "expected_mod": expected_by_prime,
                "passed": all(passed_by_prime.values()),
                "passed_by_prime": passed_by_prime,
            }
        )
    return tuple(rows)
