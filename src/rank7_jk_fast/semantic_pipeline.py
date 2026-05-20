"""Fast c18 semantic-table and plateau-aware scout workflows.

The original ``rank7_jk`` modules remain the source of mathematical truth.
This module only changes orchestration:

* semantic-key chunks are the default production unit;
* rank assembly can stop after a dependent-column plateau;
* row order defaults favor cache locality;
* scout commands default to short, resumable, plateau-capped runs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from rank7_jk.c18_block_probe import UnsupportedBlockEntry, _entry_key_and_metadata
from rank7_jk.c18_block_semantic_table import (
    _columns_by_stored_indices,
    _git_dirty,
    _git_head,
    _normalize_table_unsupported,
    _rows_by_stored_indices,
    assemble_combined_block_rank_from_value_tables,
    merge_block_key_manifest_outputs,
    plan_c18_block_semantic_keys,
    run_block_key_manifest_chunk,
)
from rank7_jk.c18_even_worker import read_json_maybe_gzip, write_json_maybe_gzip
from rank7_jk.c18_f2_rank_growth import run_c18_f2_rank_growth
from rank7_jk.config import FormulaConfig, RANK7_G2_D1
from rank7_jk.exterior import ExteriorAlgebra
from rank7_jk.mod_arith import require_prime
from rank7_jk.rank_stream import ColumnRankTracker, left_nullspace_mod


DEFAULT_PRIME = 101
DEFAULT_METHOD = "semantic-batched"
DEFAULT_CHUNK_SIZE = 100
DEFAULT_SCOUT_DEPENDENT_PLATEAU = 32
DEFAULT_MAX_CHUNK_TERMS = 200_000
DEFAULT_BETA_CHUNK_SIZE = 2


def default_column_order(column_kind: str) -> str:
    """Choose a scout-friendly default order for a supported column family."""

    return "f2-power-balanced" if column_kind == "f2-power" else "balanced"


@dataclass(frozen=True)
class ScoutSpec:
    name: str
    row_kind: str
    column_kind: str
    max_columns: int
    row_order: str = "sequential"
    column_order: str = "balanced"
    unsupported: str = "error"


SCOUT_SPECS: tuple[ScoutSpec, ...] = (
    ScoutSpec(
        name="all_f2_balanced_128",
        row_kind="all",
        column_kind="f2-power",
        max_columns=128,
        column_order="f2-power-balanced",
    ),
    ScoutSpec(
        name="gamma_f2_balanced_256",
        row_kind="gamma",
        column_kind="f2-power",
        max_columns=256,
        column_order="f2-power-balanced",
    ),
    ScoutSpec(
        name="even_one_gamma_128",
        row_kind="even",
        column_kind="one-gamma",
        max_columns=128,
        column_order="balanced",
    ),
    ScoutSpec(
        name="gamma_one_f_128",
        row_kind="gamma",
        column_kind="one-f",
        max_columns=128,
        column_order="balanced",
    ),
    ScoutSpec(
        name="even_b_pair_64",
        row_kind="even",
        column_kind="b-pair",
        max_columns=64,
        column_order="balanced",
    ),
)


def set_worker_env_defaults(
    *,
    derivative_threads: int | None = None,
    residue_backend: str | None = None,
    product_profile: str | None = None,
) -> None:
    """Set conservative process-local defaults before a chunk/scout run."""

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if derivative_threads is not None:
        os.environ["RANK7_JK_DERIVATIVE_THREADS"] = str(int(derivative_threads))
    if residue_backend:
        os.environ["RANK7_JK_RESIDUE_BACKEND"] = residue_backend
    if product_profile:
        os.environ["RANK7_JK_PRODUCT_PROFILE"] = product_profile


def plan_fast_manifest(
    *,
    output_path: Path,
    output_dir: Path,
    config: FormulaConfig = RANK7_G2_D1,
    prime: int = DEFAULT_PRIME,
    row_kind: str = "all",
    column_kind: str = "f2-power",
    method: str = DEFAULT_METHOD,
    start_row: int = 0,
    end_row: int | None = None,
    max_rows: int | None = None,
    row_order: str = "sequential",
    row_random_seed: int = 0,
    start_column: int = 0,
    end_column: int | None = None,
    max_columns: int | None = None,
    column_order: str | None = None,
    column_random_seed: int = 0,
    unsupported: str = "error",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, object]:
    """Plan one semantic-key manifest with cache-local defaults."""

    effective_column_order = (
        default_column_order(column_kind) if column_order is None else column_order
    )
    return plan_c18_block_semantic_keys(
        config=config,
        prime=prime,
        row_kind=row_kind,
        column_kind=column_kind,
        method=method,
        start_row=start_row,
        end_row=end_row,
        max_rows=max_rows,
        row_order=row_order,
        row_random_seed=row_random_seed,
        start_column=start_column,
        end_column=end_column,
        max_columns=max_columns,
        column_order=effective_column_order,
        column_random_seed=column_random_seed,
        unsupported=unsupported,
        chunk_size=chunk_size,
        output_dir=output_dir,
        output_path=output_path,
    )


def plan_scout_suite(
    *,
    output_root: Path,
    config: FormulaConfig = RANK7_G2_D1,
    prime: int = DEFAULT_PRIME,
    method: str = DEFAULT_METHOD,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    specs: Sequence[ScoutSpec] = SCOUT_SPECS,
) -> dict[str, object]:
    """Create a small suite of scout manifests without evaluating entries."""

    manifests = []
    for spec in specs:
        manifest_dir = output_root / spec.name
        payload = plan_fast_manifest(
            config=config,
            prime=prime,
            method=method,
            row_kind=spec.row_kind,
            column_kind=spec.column_kind,
            max_columns=spec.max_columns,
            row_order=spec.row_order,
            column_order=spec.column_order,
            unsupported=spec.unsupported,
            chunk_size=chunk_size,
            output_dir=manifest_dir / "chunks",
            output_path=manifest_dir / "manifest.json",
        )
        manifests.append(_manifest_summary(payload, manifest_dir / "manifest.json"))
    return {
        "kind": "fast_c18_scout_suite",
        "prime": int(prime),
        "method": method,
        "chunk_size": int(chunk_size),
        "output_root": str(output_root),
        "manifest_count": len(manifests),
        "manifests": manifests,
    }


def assemble_rank_with_plateau(
    table_path: Path | str,
    *,
    config: FormulaConfig = RANK7_G2_D1,
    output_path: Path | None = None,
    stop_rank: int | None = None,
    max_dependent_columns: int | None = DEFAULT_SCOUT_DEPENDENT_PLATEAU,
    compute_left_nullspace: bool = False,
    store_matrix: bool = False,
) -> dict[str, object]:
    """Assemble a value table while stopping after a rank-growth plateau."""

    table = read_json_maybe_gzip(Path(table_path))
    if table.get("kind") != "c18_block_semantic_value_table":
        raise ValueError("value table kind must be c18_block_semantic_value_table")
    if not table.get("complete", False):
        raise ValueError("value table is not complete")
    if max_dependent_columns is not None and max_dependent_columns < 0:
        raise ValueError("max_dependent_columns must be nonnegative")

    p = require_prime(int(table["prime"]))
    row_kind = str(table["row_kind"])
    column_kind = str(table["column_kind"])
    unsupported = _normalize_table_unsupported(str(table.get("unsupported", "error")))
    rows, source_row_indices = _rows_by_stored_indices(
        config,
        row_kind,
        tuple(int(item) for item in table["source_row_indices"]),
    )
    columns, selected_column_indices = _columns_by_stored_indices(
        config,
        column_kind,
        tuple(int(item) for item in table["test_column_indices"]),
    )
    if len(rows) != int(table["row_count"]):
        raise ValueError("value table row selection does not match current basis")
    if len(columns) != int(table["column_count"]):
        raise ValueError("value table column selection does not match current basis")

    raw_values = table.get("values", {})
    if not isinstance(raw_values, dict):
        raise ValueError("value table values must be a dict")
    value_by_key = {str(key): int(value) % p for key, value in raw_values.items()}

    tracker = ColumnRankTracker(row_count=len(rows), prime=p)
    exterior = ExteriorAlgebra(config)
    matrix_rows = [[] for _ in rows] if compute_left_nullspace or store_matrix else None
    column_records = []
    missing_keys = set()
    dependent_columns_since_rank_gain = 0
    stop_reason = "exhausted_columns"
    start = time.perf_counter()

    for column_index, column in zip(selected_column_indices, columns):
        if stop_rank is not None and tracker.rank >= stop_rank:
            stop_reason = "stop_rank"
            break
        if (
            max_dependent_columns is not None
            and dependent_columns_since_rank_gain >= max_dependent_columns
        ):
            stop_reason = "max_dependent_columns"
            break

        column_start = time.perf_counter()
        vector = []
        nonzero_count = 0
        unsupported_count = 0
        for row_pos, row in enumerate(rows):
            try:
                key, _metadata = _entry_key_and_metadata(config, exterior, row, column)
            except UnsupportedBlockEntry:
                unsupported_count += 1
                if unsupported == "zero":
                    value = 0
                else:
                    raise
            else:
                value = value_by_key.get(key)
                if value is None:
                    missing_keys.add(key)
                    value = 0
            value %= p
            vector.append(value)
            if matrix_rows is not None:
                matrix_rows[row_pos].append(value)
            if value:
                nonzero_count += 1
        if missing_keys:
            raise ValueError(
                f"value table is missing semantic keys: {sorted(missing_keys)[:5]}"
            )

        independent = tracker.add_column(vector, index=column_index)
        if independent:
            dependent_columns_since_rank_gain = 0
        else:
            dependent_columns_since_rank_gain += 1
        column_records.append(
            {
                "position": int(tracker.processed_columns - 1),
                "index": int(column_index),
                "name": column.name,
                "kind": column.kind,
                "defect": column.defect,
                "nonzero_count": int(nonzero_count),
                "unsupported_count": int(unsupported_count),
                "independent": bool(independent),
                "rank_after": int(tracker.rank),
                "left_nullity_after": int(tracker.nullity_left),
                "dependent_columns_since_rank_gain": int(
                    dependent_columns_since_rank_gain
                ),
                "elapsed_seconds": time.perf_counter() - column_start,
            }
        )

    left_nullspace = None
    if compute_left_nullspace:
        assert matrix_rows is not None
        left_nullspace = [list(vector) for vector in left_nullspace_mod(matrix_rows, p)]

    selected_name_by_index = dict(
        zip(selected_column_indices, (column.name for column in columns))
    )
    payload = {
        "kind": "fast_c18_block_semantic_table_rank",
        "prime": p,
        "row_kind": row_kind,
        "column_kind": column_kind,
        "method": table.get("method"),
        "normalized_method": table.get("normalized_method"),
        "unsupported": unsupported,
        "value_table_path": str(table_path),
        "row_count": len(rows),
        "available_column_count": int(table["available_column_count"]),
        "column_count": len(columns),
        "processed_columns": int(tracker.processed_columns),
        "source_row_indices": list(source_row_indices),
        "source_row_names": [row.name for row in rows],
        "source_row_kinds": [row.kind for row in rows],
        "test_column_indices": list(selected_column_indices),
        "test_column_names": [column.name for column in columns],
        "test_column_kinds": [column.kind for column in columns],
        "test_column_defects": [column.defect for column in columns],
        "rank": int(tracker.rank),
        "left_nullity": int(tracker.nullity_left),
        "selected_column_indices": list(tracker.selected_indices),
        "selected_column_names": [
            selected_name_by_index[index] for index in tracker.selected_indices
        ],
        "max_dependent_columns": max_dependent_columns,
        "dependent_columns_since_rank_gain": int(dependent_columns_since_rank_gain),
        "stop_rank": stop_rank,
        "stop_reason": stop_reason,
        "elapsed_seconds": time.perf_counter() - start,
        "columns": column_records,
        "left_nullspace": left_nullspace,
        "matrix": matrix_rows if store_matrix else None,
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
    }
    if output_path is not None:
        write_json_maybe_gzip(output_path, payload)
    return payload


def run_rank_growth_scout(
    *,
    output_path: Path | None,
    checkpoint_path: Path | None,
    resume_from: Path | None = None,
    config: FormulaConfig = RANK7_G2_D1,
    prime: int = DEFAULT_PRIME,
    method: str = "batched",
    row_kind: str = "all",
    row_order: str = "sequential",
    column_order: str = "f2-power-balanced",
    max_columns: int = 64,
    max_dependent_columns: int = DEFAULT_SCOUT_DEPENDENT_PLATEAU,
    derivative_threads: int | None = 1,
    beta_chunk_size: int = DEFAULT_BETA_CHUNK_SIZE,
    max_chunk_terms: int = DEFAULT_MAX_CHUNK_TERMS,
    residue_backend: str | None = None,
    product_profile: str | None = None,
) -> dict[str, object]:
    """Run a short, checkpointed high-f2 scout with safer defaults."""

    set_worker_env_defaults(
        derivative_threads=derivative_threads,
        residue_backend=residue_backend,
        product_profile=product_profile,
    )
    effective_resume_from = resume_from
    if (
        effective_resume_from is None
        and checkpoint_path is not None
        and checkpoint_path.exists()
    ):
        effective_resume_from = checkpoint_path
    return run_c18_f2_rank_growth(
        config=config,
        prime=prime,
        method=method,
        row_kind=row_kind,
        row_order=row_order,
        column_order=column_order,
        max_columns=max_columns,
        target_left_nullity=None,
        max_dependent_columns=max_dependent_columns,
        beta_chunk_size=beta_chunk_size,
        max_chunk_terms=max_chunk_terms,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=1,
        resume_from=effective_resume_from,
        output_path=output_path,
    )


def _manifest_summary(payload: dict[str, object], path: Path | None = None) -> dict[str, object]:
    return {
        "path": None if path is None else str(path),
        "kind": payload.get("kind"),
        "prime": payload.get("prime"),
        "row_kind": payload.get("row_kind"),
        "column_kind": payload.get("column_kind"),
        "row_order": payload.get("row_order"),
        "column_order": payload.get("column_order"),
        "row_count": payload.get("row_count"),
        "column_count": payload.get("column_count"),
        "entry_count": payload.get("entry_count"),
        "key_count": payload.get("key_count"),
        "chunk_count": payload.get("chunk_count"),
        "reuse_factor": payload.get("reuse_factor"),
    }


def _rank_summary(payload: dict[str, object]) -> dict[str, object]:
    keys = (
        "kind",
        "prime",
        "row_kind",
        "column_kind",
        "row_count",
        "column_count",
        "processed_columns",
        "rank",
        "left_nullity",
        "stop_reason",
        "max_dependent_columns",
        "dependent_columns_since_rank_gain",
        "elapsed_seconds",
    )
    return {key: payload.get(key) for key in keys if key in payload}


def _growth_summary(payload: dict[str, object]) -> dict[str, object]:
    keys = (
        "kind",
        "complete",
        "prime",
        "method",
        "row_kind",
        "row_order",
        "column_order",
        "row_count",
        "scheduled_column_count",
        "processed_columns",
        "rank",
        "left_nullity",
        "nonzero_entries",
        "nonzero_columns",
        "semantic_cache_misses",
        "max_dependent_columns",
        "dependent_columns_since_rank_gain",
        "stop_reason",
        "elapsed_seconds",
    )
    return {key: payload.get(key) for key in keys if key in payload}


def _manifest_chunk_output_path(manifest_path: Path, task_id: int) -> Path:
    """Return the output path for one manifest task without changing core output."""

    manifest = read_json_maybe_gzip(manifest_path)
    if manifest.get("kind") != "c18_block_semantic_key_manifest":
        raise ValueError("manifest kind must be c18_block_semantic_key_manifest")
    chunks = manifest.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError("manifest chunks must be a list")
    index = int(task_id)
    if index < 0 or index >= len(chunks):
        raise ValueError("task_id is outside the manifest chunk range")
    return Path(str(chunks[index]["output_path"]))


def _print(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="plan one semantic-key manifest")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--prime", type=int, default=DEFAULT_PRIME)
    plan.add_argument("--method", default=DEFAULT_METHOD)
    plan.add_argument("--row-kind", choices=("all", "even", "gamma"), default="all")
    plan.add_argument(
        "--column-kind",
        choices=("all-a", "f2-power", "one-f", "one-gamma", "b-pair"),
        default="f2-power",
    )
    plan.add_argument("--start-row", type=int, default=0)
    plan.add_argument("--end-row", type=int, default=None)
    plan.add_argument("--max-rows", type=int, default=None)
    plan.add_argument(
        "--row-order",
        choices=("sequential", "random", "defect-balanced"),
        default="sequential",
    )
    plan.add_argument("--row-random-seed", type=int, default=0)
    plan.add_argument("--start-column", type=int, default=0)
    plan.add_argument("--end-column", type=int, default=None)
    plan.add_argument("--max-columns", type=int, default=None)
    plan.add_argument(
        "--column-order",
        choices=(
            "sequential",
            "random",
            "balanced",
            "f2-balanced",
            "f2-desc-balanced",
            "f2-power-balanced",
            "f2-power-desc-balanced",
        ),
        default=None,
    )
    plan.add_argument("--column-random-seed", type=int, default=0)
    plan.add_argument("--unsupported", choices=("error", "zero"), default="error")
    plan.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)

    suite = sub.add_parser("plan-suite", help="plan the default scout suite")
    suite.add_argument("--output-root", type=Path, required=True)
    suite.add_argument("--prime", type=int, default=DEFAULT_PRIME)
    suite.add_argument("--method", default=DEFAULT_METHOD)
    suite.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)

    run = sub.add_parser("run-chunk", help="evaluate one semantic-key chunk")
    run.add_argument("manifest", type=Path)
    run.add_argument("--task-id", type=int, default=0)
    run.add_argument("--recompute", action="store_true")
    run.add_argument("--derivative-threads", type=int, default=1)
    run.add_argument("--residue-backend", choices=("auto", "python", "array", "spmat"))
    run.add_argument("--product-profile", default=None)
    run.add_argument("--beta-chunk-size", type=int, default=DEFAULT_BETA_CHUNK_SIZE)
    run.add_argument("--max-chunk-terms", type=int, default=DEFAULT_MAX_CHUNK_TERMS)

    merge = sub.add_parser("merge", help="merge chunk outputs listed by a manifest")
    merge.add_argument("manifest", type=Path)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--allow-missing", action="store_true")

    rank = sub.add_parser("rank", help="assemble rank from one value table")
    rank.add_argument("table", type=Path)
    rank.add_argument("--output", type=Path, default=None)
    rank.add_argument("--stop-rank", type=int, default=None)
    rank.add_argument(
        "--max-dependent-columns",
        type=int,
        default=DEFAULT_SCOUT_DEPENDENT_PLATEAU,
    )
    rank.add_argument("--no-plateau-stop", action="store_true")
    rank.add_argument("--compute-left-nullspace", action="store_true")
    rank.add_argument("--store-matrix", action="store_true")

    combined = sub.add_parser(
        "combined-rank",
        help="assemble rank from several value tables with matching rows",
    )
    combined.add_argument("tables", type=Path, nargs="+")
    combined.add_argument("--output", type=Path, default=None)
    combined.add_argument("--stop-rank", type=int, default=None)
    combined.add_argument("--compute-left-nullspace", action="store_true")
    combined.add_argument("--store-matrix", action="store_true")

    scout = sub.add_parser("rank-growth-scout", help="run a short high-f2 scout")
    scout.add_argument("--output", type=Path, default=None)
    scout.add_argument("--checkpoint", type=Path, default=None)
    scout.add_argument("--resume-from", type=Path, default=None)
    scout.add_argument("--prime", type=int, default=DEFAULT_PRIME)
    scout.add_argument("--method", choices=("synthetic", "moment", "batched"), default="batched")
    scout.add_argument("--row-kind", choices=("all", "even", "gamma"), default="all")
    scout.add_argument(
        "--row-order",
        choices=("sequential", "random", "defect-balanced"),
        default="sequential",
    )
    scout.add_argument(
        "--column-order",
        choices=(
            "sequential",
            "random",
            "balanced",
            "f2-balanced",
            "f2-desc-balanced",
            "f2-power-balanced",
            "f2-power-desc-balanced",
        ),
        default="f2-power-balanced",
    )
    scout.add_argument("--max-columns", type=int, default=64)
    scout.add_argument(
        "--max-dependent-columns",
        type=int,
        default=DEFAULT_SCOUT_DEPENDENT_PLATEAU,
    )
    scout.add_argument("--derivative-threads", type=int, default=1)
    scout.add_argument("--residue-backend", choices=("auto", "python", "array", "spmat"))
    scout.add_argument("--product-profile", default=None)
    scout.add_argument("--beta-chunk-size", type=int, default=DEFAULT_BETA_CHUNK_SIZE)
    scout.add_argument("--max-chunk-terms", type=int, default=DEFAULT_MAX_CHUNK_TERMS)

    args = parser.parse_args(argv)

    if args.command == "plan":
        payload = plan_fast_manifest(
            output_path=args.output,
            output_dir=args.output_dir,
            prime=args.prime,
            method=args.method,
            row_kind=args.row_kind,
            column_kind=args.column_kind,
            start_row=args.start_row,
            end_row=args.end_row,
            max_rows=args.max_rows,
            row_order=args.row_order,
            row_random_seed=args.row_random_seed,
            start_column=args.start_column,
            end_column=args.end_column,
            max_columns=args.max_columns,
            column_order=args.column_order,
            column_random_seed=args.column_random_seed,
            unsupported=args.unsupported,
            chunk_size=args.chunk_size,
        )
        _print(_manifest_summary(payload, args.output))
        return 0

    if args.command == "plan-suite":
        payload = plan_scout_suite(
            output_root=args.output_root,
            prime=args.prime,
            method=args.method,
            chunk_size=args.chunk_size,
        )
        _print(payload)
        return 0

    if args.command == "run-chunk":
        set_worker_env_defaults(
            derivative_threads=args.derivative_threads,
            residue_backend=args.residue_backend,
            product_profile=args.product_profile,
        )
        payload = run_block_key_manifest_chunk(
            args.manifest,
            args.task_id,
            skip_existing=not args.recompute,
            beta_chunk_size=args.beta_chunk_size,
            max_chunk_terms=args.max_chunk_terms,
        )
        output_path = _manifest_chunk_output_path(args.manifest, args.task_id)
        _print(
            {
                "kind": payload.get("kind"),
                "complete": payload.get("complete"),
                "prime": payload.get("prime"),
                "chunk_id": payload.get("chunk_id"),
                "key_count": payload.get("key_count"),
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "output_path": str(output_path),
            }
        )
        return 0

    if args.command == "merge":
        payload = merge_block_key_manifest_outputs(
            args.manifest,
            output_path=args.output,
            require_complete=not args.allow_missing,
        )
        _print(
            {
                "kind": payload.get("kind"),
                "complete": payload.get("complete"),
                "prime": payload.get("prime"),
                "row_kind": payload.get("row_kind"),
                "column_kind": payload.get("column_kind"),
                "key_count": payload.get("key_count"),
                "missing_chunk_outputs": payload.get("missing_chunk_outputs"),
                "output": str(args.output),
            }
        )
        return 0

    if args.command == "rank":
        payload = assemble_rank_with_plateau(
            args.table,
            output_path=args.output,
            stop_rank=args.stop_rank,
            max_dependent_columns=(
                None if args.no_plateau_stop else args.max_dependent_columns
            ),
            compute_left_nullspace=args.compute_left_nullspace,
            store_matrix=args.store_matrix,
        )
        _print(_rank_summary(payload))
        return 0

    if args.command == "combined-rank":
        payload = assemble_combined_block_rank_from_value_tables(
            args.tables,
            output_path=args.output,
            stop_rank=args.stop_rank,
            compute_left_nullspace=args.compute_left_nullspace,
            store_matrix=args.store_matrix,
        )
        _print(_rank_summary(payload))
        return 0

    if args.command == "rank-growth-scout":
        payload = run_rank_growth_scout(
            output_path=args.output,
            checkpoint_path=args.checkpoint,
            resume_from=args.resume_from,
            prime=args.prime,
            method=args.method,
            row_kind=args.row_kind,
            row_order=args.row_order,
            column_order=args.column_order,
            max_columns=args.max_columns,
            max_dependent_columns=args.max_dependent_columns,
            derivative_threads=args.derivative_threads,
            beta_chunk_size=args.beta_chunk_size,
            max_chunk_terms=args.max_chunk_terms,
            residue_backend=args.residue_backend,
            product_profile=args.product_profile,
        )
        _print(_growth_summary(payload))
        return 0

    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
