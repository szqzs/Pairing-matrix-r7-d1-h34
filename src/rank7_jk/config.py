"""Configuration objects for Jeffrey-Kirwan pairing experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Tuple

import sympy as sp


@dataclass(frozen=True)
class FormulaConfig:
    """Formula-level JK configuration.

    This object intentionally has no source/test degrees.  The formula layer
    must be usable for rank-5 regression and small symbolic checks without
    accidentally inheriting the rank-7 production pairing degrees.
    """

    rank: int = 7
    genus: int = 2
    determinant_degree: int = 1
    primary_prime: int = 2305843009213693951
    second_prime: int = 1000033

    def __post_init__(self) -> None:
        if self.rank < 2:
            raise ValueError("rank must be at least 2")
        if self.genus < 1:
            raise ValueError("genus must be positive")
        if self.determinant_degree != 1:
            raise ValueError("Step 1 currently verifies determinant degree 1 only")

    @property
    def y_count(self) -> int:
        return self.rank - 1

    @property
    def top_degree(self) -> int:
        return 2 * (self.rank * self.rank - 1) * (self.genus - 1)

    @property
    def positive_root_count(self) -> int:
        return self.rank * (self.rank - 1) // 2

    @property
    def root_denominator_power(self) -> int:
        return 2 * self.genus - 2

    @property
    def delta_ranks(self) -> Tuple[int, ...]:
        return tuple(range(3, self.rank + 1))

    @property
    def class_ranks(self) -> Tuple[int, ...]:
        return tuple(range(2, self.rank + 1))

    @property
    def odd_indices(self) -> Tuple[int, ...]:
        return tuple(range(1, 2 * self.genus + 1))

    @property
    def gamma_labels(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(
            (r, s)
            for r in range(2, self.rank + 1)
            for s in range(r, self.rank + 1)
        )

    @property
    def b_labels(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(
            (r, j)
            for r in range(2, self.rank + 1)
            for j in range(1, 2 * self.genus + 1)
        )

    @property
    def collapsed_prefactor(self) -> sp.Integer:
        """Collapsed determinant-degree-1 JK scalar for the central Weyl sum."""

        n_plus = self.positive_root_count
        sign = -1 if (n_plus * (self.genus - 1)) % 2 else 1
        value = sp.Rational(self.rank**self.genus, factorial(self.rank))
        value *= factorial(self.rank - 1)
        return sp.Integer(sign) * sp.simplify(value)


@dataclass(frozen=True)
class PairingProblem:
    """Pairing-degree configuration attached to a formula configuration."""

    formula: FormulaConfig = FormulaConfig()
    source_degree: int = 34
    test_degree: int = 62
    expected_relation_chern_degree: int = 18

    def __post_init__(self) -> None:
        if self.source_degree + self.test_degree != self.formula.top_degree:
            raise ValueError(
                "source_degree + test_degree must equal top cohomological degree "
                f"{self.formula.top_degree}"
            )


# Backwards-compatible name for the formula layer while the project is small.
JKConfig = FormulaConfig

RANK7_G2_D1 = FormulaConfig()
RANK7_H34_H62 = PairingProblem()
