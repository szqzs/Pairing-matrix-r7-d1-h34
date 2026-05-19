"""First all-a matrix probe for the c18 relation search.

This module is the thin orchestration layer between the c18/H62 basis
generators and the streamed modular rank tracker.  The synthetic backend is a
deliberately simple scaffold; the actual backend is the current narrow
one-defect JK evaluator and is kept behind an explicit backend choice.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

from .all_a_pairing import c18_all_a_pairing_column
from .c18_basis import C18SourceRow, H62TestColumn, c18_source_rows, h62_all_a_test_columns
from .config import FormulaConfig, RANK7_G2_D1
from .mod_arith import require_prime
from .rank_stream import ColumnRankTracker

AllAColumnEvaluator = Callable[
    [int, H62TestColumn, Sequence[C18SourceRow], int],
    Sequence[int],
]


@dataclass(frozen=True)
class AllAProbeResult:
    prime: int
    evaluator_name: str
    row_count: int
    column_count: int
    processed_columns: int
    rank: int
    left_nullity: int
    selected_column_indices: Tuple[int, ...]
    selected_column_names: Tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "prime": self.prime,
            "evaluator_name": self.evaluator_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "processed_columns": self.processed_columns,
            "rank": self.rank,
            "left_nullity": self.left_nullity,
            "selected_column_indices": list(self.selected_column_indices),
            "selected_column_names": list(self.selected_column_names),
        }


def run_all_a_probe(
    *,
    config: FormulaConfig = RANK7_G2_D1,
    prime: int | None = None,
    evaluator: AllAColumnEvaluator | None = None,
    stop_rank: int | None = None,
    rows: Sequence[C18SourceRow] | None = None,
    columns: Sequence[H62TestColumn] | None = None,
) -> AllAProbeResult:
    """Stream all-a H62 test columns into a modular column-rank tracker."""

    p = require_prime(config.primary_prime if prime is None else prime)
    column_evaluator = synthetic_all_a_column if evaluator is None else evaluator
    source_rows = tuple(c18_source_rows(config) if rows is None else rows)
    test_columns = tuple(h62_all_a_test_columns(config) if columns is None else columns)
    if any(column.kind != "all_a" for column in test_columns):
        raise ValueError("run_all_a_probe only accepts all-a test columns")

    tracker = ColumnRankTracker(row_count=len(source_rows), prime=p)
    for index, column in enumerate(test_columns):
        if stop_rank is not None and tracker.rank >= stop_rank:
            break
        vector = tuple(
            int(value) % p for value in column_evaluator(index, column, source_rows, p)
        )
        tracker.add_column(vector, index=index)

    selected_indices = tuple(tracker.selected_indices)
    return AllAProbeResult(
        prime=p,
        evaluator_name=_evaluator_name(column_evaluator),
        row_count=len(source_rows),
        column_count=len(test_columns),
        processed_columns=tracker.processed_columns,
        rank=tracker.rank,
        left_nullity=tracker.nullity_left,
        selected_column_indices=selected_indices,
        selected_column_names=tuple(test_columns[index].name for index in selected_indices),
    )


def synthetic_all_a_column(
    index: int,
    column: H62TestColumn,
    rows: Sequence[C18SourceRow],
    prime: int,
) -> Tuple[int, ...]:
    """Return a deterministic unit-column scaffold for rank plumbing tests."""

    if column.kind != "all_a":
        raise ValueError("synthetic_all_a_column expects an all-a test column")
    p = require_prime(prime)
    vector = [0 for _ in rows]
    if index < len(rows):
        vector[index] = 1 % p
    return tuple(vector)


def actual_all_a_column(
    index: int,
    column: H62TestColumn,
    rows: Sequence[C18SourceRow],
    prime: int,
) -> Tuple[int, ...]:
    """Evaluate the actual one-defect JK all-a pairing column."""

    return c18_all_a_pairing_column(index, column, rows, prime)


def evaluator_by_name(name: str) -> AllAColumnEvaluator:
    if name == "synthetic":
        return synthetic_all_a_column
    if name == "actual":
        return actual_all_a_column
    raise ValueError(f"unknown all-a evaluator backend {name!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("synthetic", "actual"),
        default="synthetic",
        help="column evaluator backend to use",
    )
    parser.add_argument(
        "--prime",
        type=int,
        default=RANK7_G2_D1.primary_prime,
        help="prime modulus for streamed rank arithmetic",
    )
    parser.add_argument(
        "--stop-rank",
        type=int,
        default=None,
        help="stop after this rank is reached",
    )
    args = parser.parse_args(argv)

    result = run_all_a_probe(
        prime=args.prime,
        evaluator=evaluator_by_name(args.backend),
        stop_rank=args.stop_rank,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _evaluator_name(evaluator: AllAColumnEvaluator) -> str:
    return getattr(evaluator, "__name__", evaluator.__class__.__name__)


if __name__ == "__main__":
    raise SystemExit(main())
