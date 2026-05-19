"""Frozen rank-5 public regression fixtures.

These fixtures are external truth data for Gate B.  They do not import or run
the old rank-5 code; they only record the public modular values that the new
slow evaluator must reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .config import FormulaConfig
from .invariants import InvariantMonomial


RANK5_FORMULA = FormulaConfig(rank=5, genus=2)
RANK5_SOURCE_DEGREE = 22
RANK5_TEST_DEGREE = 26
RANK5_TOP_DEGREE = 48
RANK5_PRIMARY_PRIME = 2305843009213693951


@dataclass(frozen=True)
class PairingFixture:
    name: str
    left_name: str
    right_name: str
    expected_mod: int
    prime: int = RANK5_PRIMARY_PRIME

    @property
    def left(self) -> InvariantMonomial:
        return InvariantMonomial.from_string(RANK5_FORMULA, self.left_name)

    @property
    def right(self) -> InvariantMonomial:
        return InvariantMonomial.from_string(RANK5_FORMULA, self.right_name)

    @property
    def product(self) -> InvariantMonomial:
        return self.left * self.right

    def degree_payload(self) -> dict[str, int]:
        return {
            "left_ordinary_degree": self.left.ordinary_degree,
            "left_chern_degree": self.left.chern_degree,
            "right_ordinary_degree": self.right.ordinary_degree,
            "right_chern_degree": self.right.chern_degree,
            "product_ordinary_degree": self.product.ordinary_degree,
            "product_chern_degree": self.product.chern_degree,
        }


@dataclass(frozen=True)
class MinorFixture:
    name: str
    row_names: Tuple[str, ...]
    column_names: Tuple[str, ...]
    expected_det_mod: int
    prime: int = RANK5_PRIMARY_PRIME
    chern_degree: int = 20

    @property
    def rows(self) -> Tuple[InvariantMonomial, ...]:
        return tuple(
            InvariantMonomial.from_string(RANK5_FORMULA, item)
            for item in self.row_names
        )

    @property
    def columns(self) -> Tuple[InvariantMonomial, ...]:
        return tuple(
            InvariantMonomial.from_string(RANK5_FORMULA, item)
            for item in self.column_names
        )

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self.row_names), len(self.column_names))


@dataclass(frozen=True)
class MinorSummary:
    chern_degree: int
    source_dimension: int
    rank: int
    expected_det_mod: int
    prime: int = RANK5_PRIMARY_PRIME


RANK5_PUBLIC_SCALAR_FIXTURES: Tuple[PairingFixture, ...] = (
    PairingFixture(
        name="f2_11__f2_13",
        left_name="f2^11",
        right_name="f2^13",
        expected_mod=1381783072775710288,
    ),
    PairingFixture(
        name="f2_9_f3__a2_f2_11",
        left_name="f2^9 f3",
        right_name="a2 f2^11",
        expected_mod=1438514327499689729,
    ),
    PairingFixture(
        name="f2_8_f4__f2_13",
        left_name="f2^8 f4",
        right_name="f2^13",
        expected_mod=513073332518773065,
    ),
    PairingFixture(
        name="f2_8_gamma22__f2_13",
        left_name="f2^8 gamma22",
        right_name="f2^13",
        expected_mod=967147192232714784,
    ),
    PairingFixture(
        name="f2_7_f3_2__f2_13",
        left_name="f2^7 f3^2",
        right_name="f2^13",
        expected_mod=825622484462206102,
    ),
)


RANK5_PUBLIC_MINOR_SUMMARIES: Tuple[MinorSummary, ...] = (
    MinorSummary(11, 7, 7, 630020914576076772),
    MinorSummary(13, 94, 94, 1268914876423577257),
    MinorSummary(14, 111, 111, 926543592233552319),
    MinorSummary(15, 81, 81, 247473739368847072),
    MinorSummary(16, 53, 53, 1822378321827871558),
    MinorSummary(17, 28, 28, 1424445965610867005),
    MinorSummary(18, 16, 16, 1996658450193783560),
    MinorSummary(19, 7, 7, 1343131481176977680),
    MinorSummary(20, 4, 4, 1674242889816756997),
    MinorSummary(21, 1, 1, 1438514327499689729),
    MinorSummary(22, 1, 1, 1381783072775710288),
)


RANK5_SMALL_PUBLIC_MINOR_FIXTURES: Tuple[MinorFixture, ...] = (
    MinorFixture(
        name="rank5_c19_public_minor",
        chern_degree=19,
        row_names=(
            "a2 f2^7 f3",
            "a3 f2^8",
            "f2^7 f5",
            "f2^7 gamma23",
            "f2^6 f3 f4",
            "f2^6 f3 gamma22",
            "f2^5 f3^3",
        ),
        column_names=(
            "f2^13",
            "a2 f2^11",
            "a3 f2^10",
            "a2^2 f2^9",
            "a4 f2^9",
            "a2 a3 f2^8",
            "a5 f2^8",
        ),
        expected_det_mod=1343131481176977680,
    ),
    MinorFixture(
        name="rank5_c20_public_minor",
        chern_degree=20,
        row_names=(
            "a2 f2^9",
            "f2^8 f4",
            "f2^8 gamma22",
            "f2^7 f3^2",
        ),
        column_names=(
            "f2^13",
            "a2 f2^11",
            "a3 f2^10",
            "a2^2 f2^9",
        ),
        expected_det_mod=1674242889816756997,
    ),
    MinorFixture(
        name="rank5_c21_public_minor",
        chern_degree=21,
        row_names=("f2^9 f3",),
        column_names=("a2 f2^11",),
        expected_det_mod=1438514327499689729,
    ),
    MinorFixture(
        name="rank5_c22_public_minor",
        chern_degree=22,
        row_names=("f2^11",),
        column_names=("f2^13",),
        expected_det_mod=1381783072775710288,
    ),
)

RANK5_C20_MINOR_FIXTURE = RANK5_SMALL_PUBLIC_MINOR_FIXTURES[1]


def scalar_fixture_by_name(name: str) -> PairingFixture:
    for fixture in RANK5_PUBLIC_SCALAR_FIXTURES:
        if fixture.name == name:
            return fixture
    raise KeyError(f"unknown rank-5 scalar fixture {name!r}")
