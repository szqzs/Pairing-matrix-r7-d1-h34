"""First all-a matrix probe for the c18 relation search.

This module is the thin orchestration layer between the c18/H62 basis
generators and the streamed modular rank tracker.  The synthetic backend is a
deliberately simple scaffold; the actual backend is the current narrow
one-defect JK evaluator and is kept behind an explicit backend choice.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, Tuple

from .all_a_pairing import (
    AllASemanticBatchEvaluator,
    all_a_cache_info,
    c18_all_a_pairing_column,
    c18_all_a_pairing_column_moment,
    c18_all_a_pairing_column_semantic,
)
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
    row_kind: str
    row_count: int
    column_count: int
    processed_columns: int
    source_row_indices: Tuple[int, ...]
    test_column_indices: Tuple[int, ...]
    rank: int
    left_nullity: int
    selected_column_indices: Tuple[int, ...]
    selected_column_names: Tuple[str, ...]
    elapsed_seconds: float
    column_seconds: Tuple[float, ...]
    cache_info: dict[str, dict[str, int]]
    git_head: str | None
    git_dirty: bool | None
    selected_column_vectors: dict[int, Tuple[int, ...]] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "prime": self.prime,
            "evaluator_name": self.evaluator_name,
            "row_kind": self.row_kind,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "processed_columns": self.processed_columns,
            "source_row_indices": list(self.source_row_indices),
            "test_column_indices": list(self.test_column_indices),
            "rank": self.rank,
            "left_nullity": self.left_nullity,
            "selected_column_indices": list(self.selected_column_indices),
            "selected_column_names": list(self.selected_column_names),
            "elapsed_seconds": self.elapsed_seconds,
            "column_seconds": list(self.column_seconds),
            "cache_info": self.cache_info,
            "git_head": self.git_head,
            "git_dirty": self.git_dirty,
            "selected_column_vectors": (
                None
                if self.selected_column_vectors is None
                else {
                    str(index): list(vector)
                    for index, vector in sorted(self.selected_column_vectors.items())
                }
            ),
        }


def run_all_a_probe(
    *,
    config: FormulaConfig = RANK7_G2_D1,
    prime: int | None = None,
    evaluator: AllAColumnEvaluator | None = None,
    stop_rank: int | None = None,
    row_kind: str = "all",
    start_row: int = 0,
    end_row: int | None = None,
    max_rows: int | None = None,
    start_column: int = 0,
    end_column: int | None = None,
    max_columns: int | None = None,
    store_selected_vectors: bool = False,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = 1,
    rows: Sequence[C18SourceRow] | None = None,
    columns: Sequence[H62TestColumn] | None = None,
) -> AllAProbeResult:
    """Stream all-a H62 test columns into a modular column-rank tracker."""

    p = require_prime(config.primary_prime if prime is None else prime)
    column_evaluator = synthetic_all_a_column if evaluator is None else evaluator
    source_rows, source_row_indices = _select_rows(
        config,
        row_kind,
        rows,
        start_row=start_row,
        end_row=end_row,
        max_rows=max_rows,
    )
    test_columns, test_column_indices = _select_columns(
        config,
        columns,
        start_column=start_column,
        end_column=end_column,
        max_columns=max_columns,
    )
    if any(column.kind != "all_a" for column in test_columns):
        raise ValueError("run_all_a_probe only accepts all-a test columns")

    tracker = ColumnRankTracker(row_count=len(source_rows), prime=p)
    selected_vectors: dict[int, Tuple[int, ...]] | None = {} if store_selected_vectors else None
    column_seconds = []
    start = time.perf_counter()
    column_name_by_index = dict(zip(test_column_indices, (column.name for column in test_columns)))
    last_checkpoint_processed = 0
    for index, column in zip(test_column_indices, test_columns):
        if stop_rank is not None and tracker.rank >= stop_rank:
            break
        column_start = time.perf_counter()
        vector = tuple(
            int(value) % p for value in column_evaluator(index, column, source_rows, p)
        )
        independent = tracker.add_column(vector, index=index)
        column_seconds.append(time.perf_counter() - column_start)
        if independent and selected_vectors is not None:
            selected_vectors[index] = vector
        if checkpoint_path is not None and checkpoint_interval > 0:
            if tracker.processed_columns % checkpoint_interval == 0:
                _write_probe_checkpoint(
                    checkpoint_path,
                    prime=p,
                    evaluator_name=_evaluator_name(column_evaluator),
                    row_kind=row_kind,
                    row_count=len(source_rows),
                    column_count=len(test_columns),
                    processed_columns=tracker.processed_columns,
                    source_row_indices=source_row_indices,
                    test_column_indices=test_column_indices,
                    tracker=tracker,
                    column_seconds=column_seconds,
                    column_name_by_index=column_name_by_index,
                    start_time=start,
                    selected_vectors=selected_vectors,
                )
                last_checkpoint_processed = tracker.processed_columns

    selected_indices = tuple(tracker.selected_indices)
    elapsed = time.perf_counter() - start
    if checkpoint_path is not None and tracker.processed_columns != last_checkpoint_processed:
        _write_probe_checkpoint(
            checkpoint_path,
            prime=p,
            evaluator_name=_evaluator_name(column_evaluator),
            row_kind=row_kind,
            row_count=len(source_rows),
            column_count=len(test_columns),
            processed_columns=tracker.processed_columns,
            source_row_indices=source_row_indices,
            test_column_indices=test_column_indices,
            tracker=tracker,
            column_seconds=column_seconds,
            column_name_by_index=column_name_by_index,
            start_time=start,
            selected_vectors=selected_vectors,
        )
    return AllAProbeResult(
        prime=p,
        evaluator_name=_evaluator_name(column_evaluator),
        row_kind=row_kind,
        row_count=len(source_rows),
        column_count=len(test_columns),
        processed_columns=tracker.processed_columns,
        source_row_indices=source_row_indices,
        test_column_indices=test_column_indices,
        rank=tracker.rank,
        left_nullity=tracker.nullity_left,
        selected_column_indices=selected_indices,
        selected_column_names=tuple(column_name_by_index[index] for index in selected_indices),
        elapsed_seconds=elapsed,
        column_seconds=tuple(column_seconds),
        cache_info=all_a_cache_info(),
        git_head=_git_head(),
        git_dirty=_git_dirty(),
        selected_column_vectors=selected_vectors,
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
    """Evaluate the actual one-defect JK all-a pairing column with moments."""

    return c18_all_a_pairing_column_moment(index, column, rows, prime)


def semantic_actual_all_a_column(
    index: int,
    column: H62TestColumn,
    rows: Sequence[C18SourceRow],
    prime: int,
) -> Tuple[int, ...]:
    """Evaluate one all-a column with semantic ``(defect,total_a)`` caching."""

    return c18_all_a_pairing_column_semantic(index, column, rows, prime)


def batched_semantic_all_a_column(
    index: int,
    column: H62TestColumn,
    rows: Sequence[C18SourceRow],
    prime: int,
) -> Tuple[int, ...]:
    """Evaluate one all-a column with semantic caching and beta batching."""

    return c18_all_a_pairing_column_semantic(
        index,
        column,
        rows,
        prime,
        method="batched",
    )


def make_semantic_all_a_column(
    *,
    method: str = "moment",
    semantic_cache_maxsize: int | None = None,
    moment_cache_clear_size: int | None = None,
) -> AllAColumnEvaluator:
    """Build a semantic evaluator closure with explicit cache controls."""

    def _semantic_column(
        index: int,
        column: H62TestColumn,
        rows: Sequence[C18SourceRow],
        prime: int,
    ) -> Tuple[int, ...]:
        return c18_all_a_pairing_column_semantic(
            index,
            column,
            rows,
            prime,
            method=method,
            semantic_cache_maxsize=semantic_cache_maxsize,
            moment_cache_clear_size=moment_cache_clear_size,
        )

    _semantic_column.__name__ = f"semantic_{method}_all_a_column"
    return _semantic_column


def slow_actual_all_a_column(
    index: int,
    column: H62TestColumn,
    rows: Sequence[C18SourceRow],
    prime: int,
) -> Tuple[int, ...]:
    """Evaluate one actual all-a column with the generic product/residue path."""

    return c18_all_a_pairing_column(index, column, rows, prime)


def evaluator_by_name(name: str) -> AllAColumnEvaluator:
    if name == "synthetic":
        return synthetic_all_a_column
    if name in {"actual", "moment"}:
        return actual_all_a_column
    if name == "semantic":
        return semantic_actual_all_a_column
    if name == "semantic-batched":
        return batched_semantic_all_a_column
    if name == "slow-actual":
        return slow_actual_all_a_column
    raise ValueError(f"unknown all-a evaluator backend {name!r}")


def benchmark_all_a_defects(
    *,
    config: FormulaConfig = RANK7_G2_D1,
    prime: int | None = None,
    defects: Sequence[str] = (),
    method: str = "moment",
    row_kind: str = "even",
    start_row: int = 0,
    end_row: int | None = None,
    max_rows: int | None = None,
    start_column: int = 0,
    end_column: int | None = None,
    max_columns: int | None = None,
    semantic_cache_maxsize: int | None = None,
    moment_cache_clear_size: int | None = None,
) -> dict[str, object]:
    """Benchmark semantic all-a evaluation one defect block at a time."""

    p = require_prime(config.primary_prime if prime is None else prime)
    source_rows, source_row_indices = _select_rows(
        config,
        row_kind,
        None,
        start_row=start_row,
        end_row=end_row,
        max_rows=max_rows,
    )
    test_columns, test_column_indices = _select_columns(
        config,
        None,
        start_column=start_column,
        end_column=end_column,
        max_columns=max_columns,
    )
    if defects:
        defect_order = tuple(defects)
    else:
        defect_order = tuple(dict.fromkeys(row.defect for row in source_rows))

    benchmark_payloads = []
    total_start = time.perf_counter()
    for defect in defect_order:
        indexed_rows = tuple(
            (idx, row)
            for idx, row in zip(source_row_indices, source_rows)
            if row.defect == defect
        )
        if not indexed_rows:
            benchmark_payloads.append(
                {
                    "defect": defect,
                    "row_count": 0,
                    "column_count": len(test_columns),
                    "processed_columns": 0,
                    "rank": 0,
                    "left_nullity": 0,
                    "elapsed_seconds": 0.0,
                    "column_seconds": [],
                    "semantic_cache": {
                        "hits": 0,
                        "misses": 0,
                        "maxsize": 0,
                        "currsize": 0,
                    },
                }
            )
            continue

        row_indices = tuple(idx for idx, _row in indexed_rows)
        rows = tuple(row for _idx, row in indexed_rows)
        evaluator = AllASemanticBatchEvaluator(
            config=config,
            rows=rows,
            prime=p,
            method=method,
            semantic_cache_maxsize=semantic_cache_maxsize,
            moment_cache_clear_size=moment_cache_clear_size,
        )
        tracker = ColumnRankTracker(row_count=len(rows), prime=p)
        column_seconds = []
        start = time.perf_counter()
        for index, column in zip(test_column_indices, test_columns):
            column_start = time.perf_counter()
            vector = tuple(int(value) % p for value in evaluator.column_vector(index, column))
            tracker.add_column(vector, index=index)
            column_seconds.append(time.perf_counter() - column_start)
        elapsed = time.perf_counter() - start
        benchmark_payloads.append(
            {
                "defect": defect,
                "row_count": len(rows),
                "row_indices": list(row_indices),
                "column_count": len(test_columns),
                "test_column_indices": list(test_column_indices),
                "processed_columns": tracker.processed_columns,
                "rank": tracker.rank,
                "left_nullity": tracker.nullity_left,
                "elapsed_seconds": elapsed,
                "column_seconds": column_seconds,
                "semantic_cache": evaluator.cache_info(),
            }
        )

    return {
        "prime": p,
        "method": method,
        "row_kind": row_kind,
        "defects": list(defect_order),
        "elapsed_seconds": time.perf_counter() - total_start,
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
        "cache_info": all_a_cache_info(),
        "benchmarks": benchmark_payloads,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=(
            "synthetic",
            "actual",
            "moment",
            "semantic",
            "semantic-batched",
            "slow-actual",
        ),
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
    parser.add_argument(
        "--row-kind",
        choices=("all", "even", "gamma"),
        default="all",
        help="source row slice to use",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="first row position inside the selected row-kind slice",
    )
    parser.add_argument(
        "--end-row",
        type=int,
        default=None,
        help="exclusive row position inside the selected row-kind slice",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="maximum number of rows after --start-row",
    )
    parser.add_argument(
        "--start-column",
        type=int,
        default=0,
        help="first all-a H62 column index to evaluate",
    )
    parser.add_argument(
        "--end-column",
        type=int,
        default=None,
        help="exclusive all-a H62 column index to stop at",
    )
    parser.add_argument(
        "--max-columns",
        type=int,
        default=None,
        help="maximum number of columns to evaluate after --start-column",
    )
    parser.add_argument(
        "--store-selected-vectors",
        action="store_true",
        help="include original selected column vectors in the JSON result",
    )
    parser.add_argument(
        "--semantic-cache-max-size",
        type=int,
        default=None,
        help="semantic value LRU size; use -1 for unbounded and 0 to disable",
    )
    parser.add_argument(
        "--moment-cache-clear-size",
        type=int,
        default=None,
        help="clear scalar moment cache whenever it reaches this many entries",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="write resumability metadata after streamed column batches",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="number of processed columns between checkpoint writes",
    )
    parser.add_argument(
        "--benchmark-defects",
        nargs="*",
        default=None,
        help="benchmark selected defects instead of running one combined probe",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the full probe JSON result to this path",
    )
    parser.add_argument(
        "--timing-json",
        type=Path,
        default=None,
        help="write a compact timing/cache JSON payload to this path",
    )
    args = parser.parse_args(argv)

    if args.backend == "semantic":
        evaluator = make_semantic_all_a_column(
            method="moment",
            semantic_cache_maxsize=args.semantic_cache_max_size,
            moment_cache_clear_size=args.moment_cache_clear_size,
        )
    elif args.backend == "semantic-batched":
        evaluator = make_semantic_all_a_column(
            method="batched",
            semantic_cache_maxsize=args.semantic_cache_max_size,
            moment_cache_clear_size=args.moment_cache_clear_size,
        )
    else:
        evaluator = evaluator_by_name(args.backend)

    if args.benchmark_defects is not None:
        payload = benchmark_all_a_defects(
            prime=args.prime,
            defects=tuple(args.benchmark_defects),
            method="batched" if args.backend == "semantic-batched" else "moment",
            row_kind=args.row_kind,
            start_row=args.start_row,
            end_row=args.end_row,
            max_rows=args.max_rows,
            start_column=args.start_column,
            end_column=args.end_column,
            max_columns=args.max_columns,
            semantic_cache_maxsize=args.semantic_cache_max_size,
            moment_cache_clear_size=args.moment_cache_clear_size,
        )
        if args.output is not None:
            _write_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    result = run_all_a_probe(
        prime=args.prime,
        evaluator=evaluator,
        stop_rank=args.stop_rank,
        row_kind=args.row_kind,
        start_row=args.start_row,
        end_row=args.end_row,
        max_rows=args.max_rows,
        start_column=args.start_column,
        end_column=args.end_column,
        max_columns=args.max_columns,
        store_selected_vectors=args.store_selected_vectors,
        checkpoint_path=args.checkpoint,
        checkpoint_interval=args.checkpoint_interval,
    )
    payload = result.to_dict()
    if args.output is not None:
        _write_json(args.output, payload)
    if args.timing_json is not None:
        _write_json(
            args.timing_json,
            {
                "elapsed_seconds": result.elapsed_seconds,
                "column_seconds": list(result.column_seconds),
                "cache_info": result.cache_info,
                "git_head": result.git_head,
                "git_dirty": result.git_dirty,
                "rank": result.rank,
                "left_nullity": result.left_nullity,
                "processed_columns": result.processed_columns,
            },
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _evaluator_name(evaluator: AllAColumnEvaluator) -> str:
    return getattr(evaluator, "__name__", evaluator.__class__.__name__)


def _select_rows(
    config: FormulaConfig,
    row_kind: str,
    rows: Sequence[C18SourceRow] | None,
    *,
    start_row: int,
    end_row: int | None,
    max_rows: int | None,
) -> Tuple[Tuple[C18SourceRow, ...], Tuple[int, ...]]:
    if row_kind not in {"all", "even", "gamma"}:
        raise ValueError("row_kind must be all, even, or gamma")
    if start_row < 0:
        raise ValueError("start_row must be nonnegative")
    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows must be nonnegative")

    if rows is not None:
        base_rows = tuple(rows)
        base_indices = tuple(range(len(base_rows)))
    else:
        all_rows = tuple(c18_source_rows(config))
        if row_kind == "all":
            base_indices = tuple(range(len(all_rows)))
        else:
            base_indices = tuple(idx for idx, row in enumerate(all_rows) if row.kind == row_kind)
        base_rows = tuple(all_rows[idx] for idx in base_indices)

    stop = len(base_rows) if end_row is None else int(end_row)
    if max_rows is not None:
        stop = min(stop, start_row + max_rows)
    if stop < start_row:
        raise ValueError("end_row must be greater than or equal to start_row")

    return base_rows[start_row:stop], base_indices[start_row:stop]


def _select_columns(
    config: FormulaConfig,
    columns: Sequence[H62TestColumn] | None,
    *,
    start_column: int,
    end_column: int | None,
    max_columns: int | None,
) -> Tuple[Tuple[H62TestColumn, ...], Tuple[int, ...]]:
    if start_column < 0:
        raise ValueError("start_column must be nonnegative")
    if max_columns is not None and max_columns < 0:
        raise ValueError("max_columns must be nonnegative")

    all_columns = tuple(h62_all_a_test_columns(config) if columns is None else columns)
    stop = len(all_columns) if end_column is None else int(end_column)
    if max_columns is not None:
        stop = min(stop, start_column + max_columns)
    if stop < start_column:
        raise ValueError("end_column must be greater than or equal to start_column")
    selected = all_columns[start_column:stop]
    return selected, tuple(range(start_column, start_column + len(selected)))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_probe_checkpoint(
    path: Path,
    *,
    prime: int,
    evaluator_name: str,
    row_kind: str,
    row_count: int,
    column_count: int,
    processed_columns: int,
    source_row_indices: Tuple[int, ...],
    test_column_indices: Tuple[int, ...],
    tracker: ColumnRankTracker,
    column_seconds: Sequence[float],
    column_name_by_index: dict[int, str],
    start_time: float,
    selected_vectors: dict[int, Tuple[int, ...]] | None,
) -> None:
    selected_indices = tuple(tracker.selected_indices)
    payload = {
        "prime": prime,
        "evaluator_name": evaluator_name,
        "row_kind": row_kind,
        "row_count": row_count,
        "column_count": column_count,
        "processed_columns": processed_columns,
        "source_row_indices": list(source_row_indices),
        "test_column_indices": list(test_column_indices),
        "rank": tracker.rank,
        "left_nullity": tracker.nullity_left,
        "selected_column_indices": list(selected_indices),
        "selected_column_names": [
            column_name_by_index[index] for index in selected_indices
        ],
        "elapsed_seconds": time.perf_counter() - start_time,
        "column_seconds": list(column_seconds),
        "cache_info": all_a_cache_info(),
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
        "selected_column_vectors": (
            None
            if selected_vectors is None
            else {
                str(index): list(vector)
                for index, vector in sorted(selected_vectors.items())
            }
        ),
    }
    _write_json(path, payload)


def _git_head() -> str | None:
    result = _git_command("rev-parse", "HEAD")
    return result.strip() if result else None


def _git_dirty() -> bool | None:
    result = _git_command("status", "--short")
    return None if result is None else bool(result.strip())


def _git_command(*args: str) -> str | None:
    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return None
    if result.returncode:
        return None
    return result.stdout


if __name__ == "__main__":
    raise SystemExit(main())
