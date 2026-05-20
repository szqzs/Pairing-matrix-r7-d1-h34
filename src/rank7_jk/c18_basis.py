"""Specialized bases for the aggressive c18 relation probe.

The c18 source slice in H^34 has one defect factor because
``18 - 34/2 = 1``.  Thus every row is either one ``f_r`` times an
``a``-monomial, or one ``gamma_rs`` times an ``a``-monomial.  This module keeps
that combinatorics explicit for the fast c18-first plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Sequence, Tuple

from .config import FormulaConfig, RANK7_G2_D1
from .invariants import InvariantMonomial

SourceKind = Literal["even", "gamma"]
TestKind = Literal["all_a", "one_f", "one_gamma", "one_b_pair", "f2_power"]


@dataclass(frozen=True)
class C18SourceRow:
    kind: SourceKind
    monomial: InvariantMonomial
    defect: str

    @property
    def name(self) -> str:
        return str(self.monomial)


@dataclass(frozen=True)
class H62TestColumn:
    kind: TestKind
    monomial: InvariantMonomial
    defect: str | None = None
    b_labels: Tuple[Tuple[int, int], ...] = ()

    @property
    def name(self) -> str:
        if self.b_labels:
            b_part = " ".join(f"b{r}_{j}" for r, j in self.b_labels)
            monomial = str(self.monomial)
            return b_part if monomial == "1" else f"{monomial} {b_part}"
        return str(self.monomial)


def c18_source_rows(config: FormulaConfig = RANK7_G2_D1) -> Tuple[C18SourceRow, ...]:
    if config.rank != 7 or config.genus != 2:
        raise ValueError("the c18 fast basis is currently specialized to rank 7 genus 2")
    rows = []
    for r in config.class_ranks:
        for a_parts in restricted_partitions(18 - r, config.class_ranks):
            rows.append(
                C18SourceRow(
                    kind="even",
                    monomial=_monomial(config, a_parts, f_rank=r),
                    defect=f"f{r}",
                )
            )
    for r, s in config.gamma_labels:
        for a_parts in restricted_partitions(18 - r - s, config.class_ranks):
            rows.append(
                C18SourceRow(
                    kind="gamma",
                    monomial=_monomial(config, a_parts, gamma_label=(r, s)),
                    defect=f"gamma{r}{s}",
                )
            )
    return tuple(rows)


def c18_even_source_rows(config: FormulaConfig = RANK7_G2_D1) -> Tuple[C18SourceRow, ...]:
    return tuple(row for row in c18_source_rows(config) if row.kind == "even")


def c18_gamma_source_rows(config: FormulaConfig = RANK7_G2_D1) -> Tuple[C18SourceRow, ...]:
    return tuple(row for row in c18_source_rows(config) if row.kind == "gamma")


def h62_all_a_test_columns(config: FormulaConfig = RANK7_G2_D1) -> Tuple[H62TestColumn, ...]:
    if config.rank != 7 or config.genus != 2:
        raise ValueError("the H62 fast test basis is currently specialized to rank 7 genus 2")
    return tuple(
        H62TestColumn(kind="all_a", monomial=_monomial(config, a_parts))
        for a_parts in restricted_partitions(31, config.class_ranks)
    )


def h62_one_f_test_columns(config: FormulaConfig = RANK7_G2_D1) -> Tuple[H62TestColumn, ...]:
    if config.rank != 7 or config.genus != 2:
        raise ValueError("the H62 fast test basis is currently specialized to rank 7 genus 2")
    columns = []
    for r in config.class_ranks:
        # H62 has ordinary/2 = 31, and f_r contributes ordinary/2 = r - 1.
        for a_parts in restricted_partitions(32 - r, config.class_ranks):
            columns.append(
                H62TestColumn(
                    kind="one_f",
                    monomial=_monomial(config, a_parts, f_rank=r),
                    defect=f"f{r}",
                )
            )
    return tuple(columns)


def h62_f2_power_test_columns(config: FormulaConfig = RANK7_G2_D1) -> Tuple[H62TestColumn, ...]:
    """Return defect-rich H62 tests ``a^alpha f2^k`` with ``k >= 1``.

    The rank-5 regression minor shows that high powers of ``f2`` are often the
    first place where nonzero pairings appear.  We therefore list the highest
    ``f2`` powers first, while keeping the all-a ``k = 0`` block separate.
    """

    if config.rank != 7 or config.genus != 2:
        raise ValueError("the H62 fast test basis is currently specialized to rank 7 genus 2")
    columns = []
    for f2_power in range(31, 0, -1):
        for a_parts in restricted_partitions(31 - f2_power, config.class_ranks):
            f_exp = [0 for _ in config.class_ranks]
            f_exp[0] = f2_power
            columns.append(
                H62TestColumn(
                    kind="f2_power",
                    monomial=InvariantMonomial.from_exponents(
                        config,
                        a_exp=a_exp_from_parts(config, a_parts),
                        f_exp=f_exp,
                    ),
                    defect=f"f2^{f2_power}",
                )
            )
    return tuple(columns)


def h62_one_gamma_test_columns(config: FormulaConfig = RANK7_G2_D1) -> Tuple[H62TestColumn, ...]:
    if config.rank != 7 or config.genus != 2:
        raise ValueError("the H62 fast test basis is currently specialized to rank 7 genus 2")
    columns = []
    for r, s in config.gamma_labels:
        # H62 has ordinary/2 = 31, and gamma_rs contributes ordinary/2 = r+s-1.
        for a_parts in restricted_partitions(32 - r - s, config.class_ranks):
            columns.append(
                H62TestColumn(
                    kind="one_gamma",
                    monomial=_monomial(config, a_parts, gamma_label=(r, s)),
                    defect=f"gamma{r}{s}",
                )
            )
    return tuple(columns)


def h62_one_b_pair_test_columns(config: FormulaConfig = RANK7_G2_D1) -> Tuple[H62TestColumn, ...]:
    if config.rank != 7 or config.genus != 2:
        raise ValueError("the H62 fast test basis is currently specialized to rank 7 genus 2")
    columns = []
    labels = config.b_labels
    for left_idx, left in enumerate(labels):
        for right in labels[left_idx + 1 :]:
            r, _i = left
            s, _j = right
            # H62 has ordinary/2 = 31, and b_r b_s contributes r+s-1.
            for a_parts in restricted_partitions(32 - r - s, config.class_ranks):
                columns.append(
                    H62TestColumn(
                        kind="one_b_pair",
                        monomial=_monomial(config, a_parts),
                        defect=f"b{left[0]}_{left[1]}*b{right[0]}_{right[1]}",
                        b_labels=(left, right),
                    )
                )
    return tuple(columns)


@lru_cache(maxsize=None)
def restricted_partitions(total: int, parts: Tuple[int, ...], min_part: int | None = None) -> Tuple[Tuple[int, ...], ...]:
    """Return nondecreasing partitions of ``total`` using the given part set."""

    if total < 0:
        return ()
    min_allowed = parts[0] if min_part is None else int(min_part)
    if total == 0:
        return ((),)
    out = []
    for part in parts:
        if part < min_allowed:
            continue
        if part > total:
            break
        for rest in restricted_partitions(total - part, parts, part):
            out.append((part,) + rest)
    return tuple(out)


def a_exp_from_parts(config: FormulaConfig, a_parts: Sequence[int]) -> Tuple[int, ...]:
    counts = {r: 0 for r in config.class_ranks}
    for part in a_parts:
        if part not in counts:
            raise ValueError(f"a-part {part} is outside rank {config.rank}")
        counts[int(part)] += 1
    return tuple(counts[r] for r in config.class_ranks)


def _monomial(
    config: FormulaConfig,
    a_parts: Sequence[int],
    *,
    f_rank: int | None = None,
    gamma_label: tuple[int, int] | None = None,
) -> InvariantMonomial:
    f_exp = [0 for _ in config.class_ranks]
    gamma_exp = [0 for _ in config.gamma_labels]
    if f_rank is not None:
        f_exp[f_rank - 2] = 1
    if gamma_label is not None:
        gamma_exp[list(config.gamma_labels).index(gamma_label)] = 1
    return InvariantMonomial.from_exponents(
        config,
        a_exp=a_exp_from_parts(config, a_parts),
        f_exp=f_exp,
        gamma_exp=gamma_exp,
    )
