"""Type-A root-system indexing for residue transitions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple


@dataclass(frozen=True)
class TypeARootSystem:
    """Positive-root intervals for ``SL_rank`` in simple-root coordinates.

    Internally intervals are zero-based half-open pairs ``(start, end)``.  The
    display helper ``positive_intervals_one_based`` exposes the paper's
    one-based labels ``(i, j)``.
    """

    rank: int

    def __post_init__(self) -> None:
        if self.rank < 2:
            raise ValueError("rank must be at least 2")

    @property
    def y_count(self) -> int:
        return self.rank - 1

    @property
    def positive_intervals_zero_based(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(
            (start, end)
            for start in range(self.y_count)
            for end in range(start + 1, self.rank)
        )

    @property
    def positive_intervals_one_based(self) -> Tuple[Tuple[int, int], ...]:
        return tuple((start + 1, end + 1) for start, end in self.positive_intervals_zero_based)

    @property
    def positive_root_count(self) -> int:
        return len(self.positive_intervals_zero_based)

    @property
    def interval_index(self) -> dict[Tuple[int, int], int]:
        return {
            interval: idx
            for idx, interval in enumerate(self.positive_intervals_zero_based)
        }

    @property
    def transition_schedule(self) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
        """For each Y variable, list denominator transitions eliminated there.

        Each pair is ``(current_root_pos, lower_root_pos)``.  A ``lower_root_pos``
        of ``-1`` means the denominator is the simple root being eliminated.
        """

        index = self.interval_index
        by_var = []
        for var_idx in range(self.y_count):
            local = []
            for interval, pos in index.items():
                if interval[1] != var_idx + 1:
                    continue
                lower_pos = -1 if interval[0] == var_idx else index[(interval[0], var_idx)]
                local.append((pos, lower_pos))
            by_var.append(tuple(local))
        return tuple(by_var)


@lru_cache(maxsize=None)
def type_a_roots(rank: int) -> TypeARootSystem:
    return TypeARootSystem(rank=int(rank))
