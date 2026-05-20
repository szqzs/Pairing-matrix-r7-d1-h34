"""Production chunk worker for c18 all-a matrix generation.

The worker computes a slice of all-a H62 test columns against a selected c18
row block and writes every column vector to a compressed JSON artifact.  The
merge command streams one or more artifacts back through ``ColumnRankTracker``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from .all_a_pairing import AllASemanticBatchEvaluator, all_a_cache_info
from .c18_all_a_probe import (
    _git_dirty,
    _git_head,
    _select_columns,
    _select_rows,
    synthetic_all_a_column,
)
from .config import FormulaConfig, RANK7_G2_D1
from .mod_arith import require_prime
from .rank_stream import ColumnRankTracker

_SCHEMA_VERSION = 1


def run_c18_even_worker(
    *,
    config: FormulaConfig = RANK7_G2_D1,
    prime: int | None = None,
    row_kind: str = "even",
    method: str = "semantic-batched",
    start_row: int = 0,
    end_row: int | None = None,
    max_rows: int | None = None,
    start_column: int = 0,
    end_column: int | None = None,
    max_columns: int | None = None,
    semantic_cache_maxsize: int | None = None,
    moment_cache_clear_size: int | None = None,
    output_path: Path | None = None,
    chunk_id: int | None = None,
    manifest_path: Path | None = None,
    skip_existing: bool = False,
) -> dict[str, object]:
    """Compute one chunk of c18 all-a column vectors."""

    if skip_existing and output_path is not None and output_path.exists():
        payload = read_json_maybe_gzip(output_path)
        if payload.get("kind") == "c18_all_a_chunk" and payload.get("complete"):
            return payload

    p = require_prime(config.primary_prime if prime is None else prime)
    normalized_method = _normalize_method(method)
    rows, source_row_indices = _select_rows(
        config,
        row_kind,
        None,
        start_row=start_row,
        end_row=end_row,
        max_rows=max_rows,
    )
    columns, test_column_indices = _select_columns(
        config,
        None,
        start_column=start_column,
        end_column=end_column,
        max_columns=max_columns,
    )
    if any(column.kind != "all_a" for column in columns):
        raise ValueError("c18 even worker only accepts all-a test columns")

    evaluator = None
    if normalized_method != "synthetic":
        evaluator = AllASemanticBatchEvaluator(
            config=config,
            rows=tuple(rows),
            prime=p,
            method=normalized_method,
            semantic_cache_maxsize=semantic_cache_maxsize,
            moment_cache_clear_size=moment_cache_clear_size,
        )

    column_records = []
    column_seconds = []
    start = time.perf_counter()
    for index, column in zip(test_column_indices, columns):
        column_start = time.perf_counter()
        if normalized_method == "synthetic":
            vector = synthetic_all_a_column(index, column, rows, p)
        else:
            assert evaluator is not None
            vector = evaluator.column_vector(index, column)
        elapsed = time.perf_counter() - column_start
        column_seconds.append(elapsed)
        column_records.append(
            {
                "index": int(index),
                "name": column.name,
                "values": [int(value) % p for value in vector],
                "elapsed_seconds": elapsed,
            }
        )

    payload = {
        "kind": "c18_all_a_chunk",
        "schema_version": _SCHEMA_VERSION,
        "complete": True,
        "prime": p,
        "chunk_id": None if chunk_id is None else int(chunk_id),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "row_kind": row_kind,
        "method": method,
        "normalized_method": normalized_method,
        "row_count": len(rows),
        "column_count": len(columns),
        "processed_columns": len(column_records),
        "source_row_indices": list(source_row_indices),
        "source_row_names": [row.name for row in rows],
        "test_column_indices": list(test_column_indices),
        "test_column_names": [column.name for column in columns],
        "columns": column_records,
        "elapsed_seconds": time.perf_counter() - start,
        "column_seconds": column_seconds,
        "semantic_cache": None if evaluator is None else evaluator.cache_info(),
        "cache_info": all_a_cache_info(),
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
    }
    if output_path is not None:
        write_json_maybe_gzip(output_path, payload)
    return payload


def plan_c18_even_chunks(
    *,
    config: FormulaConfig = RANK7_G2_D1,
    prime: int | None = None,
    row_kind: str = "even",
    method: str = "semantic-batched",
    start_row: int = 0,
    end_row: int | None = None,
    max_rows: int | None = None,
    start_column: int = 0,
    end_column: int | None = None,
    max_columns: int | None = None,
    chunk_size: int = 1,
    output_dir: Path = Path("results/c18_even_chunks"),
    output_prefix: str = "chunk",
    output_suffix: str = ".json.gz",
    output_path: Path | None = None,
) -> dict[str, object]:
    """Create a deterministic column-chunk manifest."""

    p = require_prime(config.primary_prime if prime is None else prime)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    rows, source_row_indices = _select_rows(
        config,
        row_kind,
        None,
        start_row=start_row,
        end_row=end_row,
        max_rows=max_rows,
    )
    columns, test_column_indices = _select_columns(
        config,
        None,
        start_column=start_column,
        end_column=end_column,
        max_columns=max_columns,
    )
    normalized_method = _normalize_method(method)
    chunks = []
    for chunk_id, offset in enumerate(range(0, len(test_column_indices), chunk_size)):
        indices = test_column_indices[offset : offset + chunk_size]
        if not indices:
            continue
        chunk_start = int(indices[0])
        chunk_end = int(indices[-1]) + 1
        chunk_output = output_dir / (
            f"{output_prefix}_{chunk_id:04d}_cols_{chunk_start:04d}_{chunk_end:04d}"
            f"{output_suffix}"
        )
        chunks.append(
            {
                "chunk_id": chunk_id,
                "start_column": chunk_start,
                "end_column": chunk_end,
                "column_count": len(indices),
                "output_path": str(chunk_output),
            }
        )

    payload = {
        "kind": "c18_all_a_chunk_manifest",
        "schema_version": _SCHEMA_VERSION,
        "prime": p,
        "row_kind": row_kind,
        "method": method,
        "normalized_method": normalized_method,
        "start_row": int(start_row),
        "end_row": None if end_row is None else int(end_row),
        "max_rows": None if max_rows is None else int(max_rows),
        "start_column": int(start_column),
        "end_column": None if end_column is None else int(end_column),
        "max_columns": None if max_columns is None else int(max_columns),
        "row_count": len(rows),
        "column_count": len(columns),
        "chunk_size": int(chunk_size),
        "chunk_count": len(chunks),
        "source_row_indices": list(source_row_indices),
        "source_row_names": [row.name for row in rows],
        "test_column_indices": list(test_column_indices),
        "test_column_names": [column.name for column in columns],
        "output_dir": str(output_dir),
        "output_prefix": output_prefix,
        "output_suffix": output_suffix,
        "chunks": chunks,
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
    }
    if output_path is not None:
        write_json_maybe_gzip(output_path, payload)
    return payload


def run_manifest_chunk(
    manifest_path: Path,
    task_id: int,
    *,
    semantic_cache_maxsize: int | None = None,
    moment_cache_clear_size: int | None = None,
    skip_existing: bool = True,
) -> dict[str, object]:
    """Run one chunk from a manifest by zero-based task id."""

    manifest = read_json_maybe_gzip(manifest_path)
    if manifest.get("kind") != "c18_all_a_chunk_manifest":
        raise ValueError("manifest kind must be c18_all_a_chunk_manifest")
    chunks = manifest.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError("manifest chunks must be a list")
    index = int(task_id)
    if index < 0 or index >= len(chunks):
        raise ValueError("task_id is outside the manifest chunk range")
    chunk = chunks[index]
    return run_c18_even_worker(
        prime=int(manifest["prime"]),
        row_kind=str(manifest["row_kind"]),
        method=str(manifest["method"]),
        start_row=int(manifest.get("start_row", 0)),
        end_row=(
            None
            if manifest.get("end_row") is None
            else int(manifest["end_row"])
        ),
        max_rows=(
            None
            if manifest.get("max_rows") is None
            else int(manifest["max_rows"])
        ),
        start_column=int(chunk["start_column"]),
        end_column=int(chunk["end_column"]),
        semantic_cache_maxsize=semantic_cache_maxsize,
        moment_cache_clear_size=moment_cache_clear_size,
        output_path=Path(str(chunk["output_path"])),
        chunk_id=int(chunk["chunk_id"]),
        manifest_path=manifest_path,
        skip_existing=skip_existing,
    )


def merge_c18_worker_outputs(
    paths: Sequence[Path | str],
    *,
    output_path: Path | None = None,
    stop_rank: int | None = None,
) -> dict[str, object]:
    """Merge worker chunks and compute streamed column rank."""

    if not paths:
        raise ValueError("at least one worker output is required")
    chunks = [read_json_maybe_gzip(Path(path)) for path in paths]
    _validate_chunks(chunks)

    first = chunks[0]
    prime = require_prime(int(first["prime"]))
    row_count = int(first["row_count"])
    tracker = ColumnRankTracker(row_count=row_count, prime=prime)
    columns = []
    for chunk_id, chunk in enumerate(chunks):
        for record in chunk["columns"]:
            columns.append(
                {
                    "chunk_id": chunk_id,
                    "index": int(record["index"]),
                    "name": str(record["name"]),
                    "values": tuple(int(value) % prime for value in record["values"]),
                    "elapsed_seconds": float(record.get("elapsed_seconds", 0.0)),
                }
            )
    columns.sort(key=lambda record: (record["index"], record["chunk_id"]))

    duplicate_indices = _duplicate_column_indices(columns)
    start = time.perf_counter()
    for record in columns:
        if stop_rank is not None and tracker.rank >= stop_rank:
            break
        tracker.add_column(record["values"], index=record["index"])
    elapsed = time.perf_counter() - start

    selected_name_by_index = {record["index"]: record["name"] for record in columns}
    payload = {
        "kind": "c18_all_a_chunk_merge",
        "schema_version": _SCHEMA_VERSION,
        "prime": prime,
        "row_kind": first["row_kind"],
        "method": first.get("method"),
        "normalized_method": first.get("normalized_method"),
        "row_count": row_count,
        "chunk_count": len(chunks),
        "input_paths": [str(path) for path in paths],
        "source_row_indices": list(first["source_row_indices"]),
        "source_row_names": list(first.get("source_row_names", [])),
        "processed_columns": tracker.processed_columns,
        "available_columns": len(columns),
        "duplicate_column_indices": duplicate_indices,
        "rank": tracker.rank,
        "left_nullity": tracker.nullity_left,
        "selected_column_indices": list(tracker.selected_indices),
        "selected_column_names": [
            selected_name_by_index[index] for index in tracker.selected_indices
        ],
        "elapsed_seconds": elapsed,
        "chunk_elapsed_seconds": [
            float(chunk.get("elapsed_seconds", 0.0)) for chunk in chunks
        ],
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
    }
    if output_path is not None:
        write_json_maybe_gzip(output_path, payload)
    return payload


def merge_c18_manifest_outputs(
    manifest_path: Path,
    *,
    output_path: Path | None = None,
    stop_rank: int | None = None,
    require_complete: bool = True,
) -> dict[str, object]:
    """Merge all chunk outputs listed in a manifest."""

    manifest = read_json_maybe_gzip(manifest_path)
    if manifest.get("kind") != "c18_all_a_chunk_manifest":
        raise ValueError("manifest kind must be c18_all_a_chunk_manifest")
    chunk_paths = [Path(str(chunk["output_path"])) for chunk in manifest["chunks"]]
    missing = [str(path) for path in chunk_paths if not path.exists()]
    if missing and require_complete:
        raise ValueError(f"manifest has missing chunk outputs: {missing[:5]}")
    existing = [path for path in chunk_paths if path.exists()]
    payload = merge_c18_worker_outputs(
        existing,
        output_path=output_path,
        stop_rank=stop_rank,
    )
    payload["manifest_path"] = str(manifest_path)
    payload["manifest_chunk_count"] = int(manifest["chunk_count"])
    payload["missing_chunk_outputs"] = missing
    if output_path is not None:
        write_json_maybe_gzip(output_path, payload)
    return payload


def write_json_maybe_gzip(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if str(path).endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(data)
        return
    path.write_text(data, encoding="utf-8")


def read_json_maybe_gzip(path: Path) -> dict[str, object]:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_method(method: str) -> str:
    normalized = method.lower()
    if normalized == "semantic-batched":
        return "batched"
    if normalized in {"synthetic", "moment", "batched"}:
        return normalized
    raise ValueError("method must be synthetic, moment, batched, or semantic-batched")


def _validate_chunks(chunks: Sequence[dict[str, object]]) -> None:
    first = chunks[0]
    if first.get("kind") != "c18_all_a_chunk":
        raise ValueError("worker output kind must be c18_all_a_chunk")
    required = (
        "prime",
        "row_kind",
        "row_count",
        "source_row_indices",
        "normalized_method",
    )
    for chunk in chunks[1:]:
        if chunk.get("kind") != "c18_all_a_chunk":
            raise ValueError("worker output kind must be c18_all_a_chunk")
        for key in required:
            if chunk.get(key) != first.get(key):
                raise ValueError(f"worker outputs disagree on {key}")
        if len(chunk.get("source_row_names", [])) != len(first.get("source_row_names", [])):
            raise ValueError("worker outputs disagree on source_row_names")


def _duplicate_column_indices(columns: Sequence[dict[str, object]]) -> list[int]:
    seen = set()
    duplicates = set()
    for record in columns:
        index = int(record["index"])
        if index in seen:
            duplicates.add(index)
        seen.add(index)
    return sorted(duplicates)


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] == "plan":
        return _plan_main(args_list[1:])
    if args_list and args_list[0] == "run":
        args_list = args_list[1:]
    if args_list and args_list[0] == "run-manifest":
        return _run_manifest_main(args_list[1:])
    if args_list and args_list[0] == "merge":
        return _merge_main(args_list[1:])
    if args_list and args_list[0] == "merge-manifest":
        return _merge_manifest_main(args_list[1:])
    return _run_main(args_list)


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--row-kind",
        choices=("all", "even", "gamma"),
        default="even",
    )
    parser.add_argument(
        "--method",
        choices=("synthetic", "moment", "batched", "semantic-batched"),
        default="semantic-batched",
    )
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--start-column", type=int, default=0)
    parser.add_argument("--end-column", type=int, default=None)
    parser.add_argument("--max-columns", type=int, default=None)


def _plan_main(argv: Sequence[str] | None) -> int:
    parser = argparse.ArgumentParser(description="Plan c18 worker chunks")
    parser.add_argument("--prime", type=int, default=RANK7_G2_D1.primary_prime)
    _add_selection_args(parser)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="chunk")
    parser.add_argument("--output-suffix", default=".json.gz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = plan_c18_even_chunks(
        prime=args.prime,
        row_kind=args.row_kind,
        method=args.method,
        start_row=args.start_row,
        end_row=args.end_row,
        max_rows=args.max_rows,
        start_column=args.start_column,
        end_column=args.end_column,
        max_columns=args.max_columns,
        chunk_size=args.chunk_size,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        output_suffix=args.output_suffix,
        output_path=args.output,
    )
    print(json.dumps(_summary(payload), indent=2, sort_keys=True))
    return 0


def _run_main(argv: Sequence[str] | None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prime", type=int, default=RANK7_G2_D1.primary_prime)
    _add_selection_args(parser)
    parser.add_argument("--semantic-cache-max-size", type=int, default=None)
    parser.add_argument("--moment-cache-clear-size", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = run_c18_even_worker(
        prime=args.prime,
        row_kind=args.row_kind,
        method=args.method,
        start_row=args.start_row,
        end_row=args.end_row,
        max_rows=args.max_rows,
        start_column=args.start_column,
        end_column=args.end_column,
        max_columns=args.max_columns,
        semantic_cache_maxsize=args.semantic_cache_max_size,
        moment_cache_clear_size=args.moment_cache_clear_size,
        output_path=args.output,
        skip_existing=args.skip_existing,
    )
    print(json.dumps(_summary(payload), indent=2, sort_keys=True))
    return 0


def _run_manifest_main(argv: Sequence[str] | None) -> int:
    parser = argparse.ArgumentParser(description="Run one manifest chunk")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--semantic-cache-max-size", type=int, default=None)
    parser.add_argument("--moment-cache-clear-size", type=int, default=None)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args(argv)

    payload = run_manifest_chunk(
        args.manifest,
        args.task_id,
        semantic_cache_maxsize=args.semantic_cache_max_size,
        moment_cache_clear_size=args.moment_cache_clear_size,
        skip_existing=not args.recompute,
    )
    print(json.dumps(_summary(payload), indent=2, sort_keys=True))
    return 0


def _merge_main(argv: Sequence[str] | None) -> int:
    parser = argparse.ArgumentParser(description="Merge c18 worker chunks")
    parser.add_argument("chunks", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stop-rank", type=int, default=None)
    args = parser.parse_args(argv)

    payload = merge_c18_worker_outputs(
        args.chunks,
        output_path=args.output,
        stop_rank=args.stop_rank,
    )
    print(json.dumps(_summary(payload), indent=2, sort_keys=True))
    return 0


def _merge_manifest_main(argv: Sequence[str] | None) -> int:
    parser = argparse.ArgumentParser(description="Merge c18 worker manifest outputs")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stop-rank", type=int, default=None)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)

    payload = merge_c18_manifest_outputs(
        args.manifest,
        output_path=args.output,
        stop_rank=args.stop_rank,
        require_complete=not args.allow_missing,
    )
    print(json.dumps(_summary(payload), indent=2, sort_keys=True))
    return 0


def _summary(payload: dict[str, object]) -> dict[str, object]:
    keys = (
        "kind",
        "prime",
        "row_kind",
        "method",
        "normalized_method",
        "row_count",
        "column_count",
        "chunk_count",
        "processed_columns",
        "available_columns",
        "rank",
        "left_nullity",
        "manifest_chunk_count",
        "elapsed_seconds",
        "git_head",
        "git_dirty",
    )
    return {key: payload[key] for key in keys if key in payload}


if __name__ == "__main__":
    raise SystemExit(main())
