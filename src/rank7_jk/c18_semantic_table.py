"""Semantic value-table workflow for the c18 even/all-a calculation.

The all-a matrix has many repeated mathematical entries: each entry only
depends on the source-row defect and the sum of the source/test a-exponents.
This module computes those semantic values once, merges them into a table, and
then assembles streamed ranks by lookup.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence, Tuple

from .all_a_pairing import (
    all_a_cache_info,
    all_a_pairing_total_batched_mod,
    all_a_pairing_total_moment_mod,
)
from .c18_all_a_probe import _git_dirty, _git_head, _select_columns, _select_rows
from .c18_even_worker import read_json_maybe_gzip, write_json_maybe_gzip
from .config import FormulaConfig, RANK7_G2_D1
from .invariants import InvariantMonomial
from .mod_arith import require_prime
from .rank_stream import ColumnRankTracker, left_nullspace_mod

_SCHEMA_VERSION = 1


def semantic_key_id(defect: str, total_a_exp: Sequence[int]) -> str:
    """Return a stable compact id for one semantic all-a value."""

    return f"{defect}:{','.join(str(int(item)) for item in total_a_exp)}"


def parse_semantic_key_id(key: str) -> Tuple[str, Tuple[int, ...]]:
    """Parse a key produced by :func:`semantic_key_id`."""

    if ":" not in key:
        raise ValueError(f"semantic key {key!r} is missing ':'")
    defect, raw_exp = key.split(":", 1)
    if not defect:
        raise ValueError("semantic key defect is empty")
    if raw_exp == "":
        return defect, ()
    return defect, tuple(int(item) for item in raw_exp.split(","))


def enumerate_c18_semantic_keys(
    *,
    config: FormulaConfig = RANK7_G2_D1,
    row_kind: str = "even",
    start_row: int = 0,
    end_row: int | None = None,
    max_rows: int | None = None,
    start_column: int = 0,
    end_column: int | None = None,
    max_columns: int | None = None,
) -> dict[str, object]:
    """Enumerate unique semantic keys for a selected c18/all-a matrix."""

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
        raise ValueError("semantic key enumeration only accepts all-a test columns")

    records: dict[str, dict[str, object]] = {}
    for row in rows:
        row_a_exp = tuple(int(item) for item in row.monomial.a_exp)
        for column in columns:
            total_a_exp = tuple(
                row_a_exp[idx] + int(column.monomial.a_exp[idx])
                for idx in range(len(row_a_exp))
            )
            key = semantic_key_id(row.defect, total_a_exp)
            record = records.get(key)
            if record is None:
                records[key] = {
                    "key": key,
                    "defect": row.defect,
                    "total_a_exp": list(total_a_exp),
                    "use_count": 1,
                }
            else:
                record["use_count"] = int(record["use_count"]) + 1

    keys = sorted(
        records.values(),
        key=lambda item: _semantic_record_sort_key(config, item),
    )
    entry_count = len(rows) * len(columns)
    return {
        "row_count": len(rows),
        "column_count": len(columns),
        "entry_count": entry_count,
        "key_count": len(keys),
        "reuse_factor": 0.0 if not keys else entry_count / len(keys),
        "source_row_indices": list(source_row_indices),
        "source_row_names": [row.name for row in rows],
        "test_column_indices": list(test_column_indices),
        "test_column_names": [column.name for column in columns],
        "keys": keys,
    }


def plan_c18_semantic_keys(
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
    chunk_size: int = 100,
    output_dir: Path = Path("results/c18_semantic_values/chunks"),
    output_prefix: str = "key_chunk",
    output_suffix: str = ".json.gz",
    output_path: Path | None = None,
) -> dict[str, object]:
    """Create a deterministic manifest of unique semantic-value chunks."""

    p = require_prime(config.primary_prime if prime is None else prime)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    normalized_method = _normalize_method(method)
    enumeration = enumerate_c18_semantic_keys(
        config=config,
        row_kind=row_kind,
        start_row=start_row,
        end_row=end_row,
        max_rows=max_rows,
        start_column=start_column,
        end_column=end_column,
        max_columns=max_columns,
    )
    keys = list(enumeration["keys"])
    chunks = []
    for chunk_id, start_key in enumerate(range(0, len(keys), chunk_size)):
        end_key = min(start_key + chunk_size, len(keys))
        if start_key == end_key:
            continue
        output = output_dir / (
            f"{output_prefix}_{chunk_id:04d}_keys_{start_key:05d}_{end_key:05d}"
            f"{output_suffix}"
        )
        chunks.append(
            {
                "chunk_id": chunk_id,
                "start_key": start_key,
                "end_key": end_key,
                "key_count": end_key - start_key,
                "first_key": keys[start_key]["key"],
                "last_key": keys[end_key - 1]["key"],
                "output_path": str(output),
            }
        )

    payload = {
        "kind": "c18_semantic_key_manifest",
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
        "row_count": int(enumeration["row_count"]),
        "column_count": int(enumeration["column_count"]),
        "entry_count": int(enumeration["entry_count"]),
        "key_count": int(enumeration["key_count"]),
        "reuse_factor": float(enumeration["reuse_factor"]),
        "chunk_size": int(chunk_size),
        "chunk_count": len(chunks),
        "source_row_indices": list(enumeration["source_row_indices"]),
        "source_row_names": list(enumeration["source_row_names"]),
        "test_column_indices": list(enumeration["test_column_indices"]),
        "test_column_names": list(enumeration["test_column_names"]),
        "output_dir": str(output_dir),
        "output_prefix": output_prefix,
        "output_suffix": output_suffix,
        "keys": keys,
        "chunks": chunks,
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
    }
    if output_path is not None:
        write_json_maybe_gzip(output_path, payload)
    return payload


def run_key_manifest_chunk(
    manifest_path: Path,
    task_id: int,
    *,
    skip_existing: bool = True,
    beta_chunk_size: int = 2,
    max_chunk_terms: int = 200_000,
) -> dict[str, object]:
    """Evaluate one semantic-value chunk from a key manifest."""

    manifest = read_json_maybe_gzip(manifest_path)
    if manifest.get("kind") != "c18_semantic_key_manifest":
        raise ValueError("manifest kind must be c18_semantic_key_manifest")
    chunks = manifest.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError("manifest chunks must be a list")
    index = int(task_id)
    if index < 0 or index >= len(chunks):
        raise ValueError("task_id is outside the manifest chunk range")
    chunk = chunks[index]
    output_path = Path(str(chunk["output_path"]))
    if skip_existing and output_path.exists():
        payload = read_json_maybe_gzip(output_path)
        if payload.get("kind") == "c18_semantic_value_chunk" and payload.get("complete"):
            return payload

    keys = manifest.get("keys", [])
    if not isinstance(keys, list):
        raise ValueError("manifest keys must be a list")
    start_key = int(chunk["start_key"])
    end_key = int(chunk["end_key"])
    p = require_prime(int(manifest["prime"]))
    method = str(manifest["method"])
    normalized_method = _normalize_method(method)

    value_records = []
    value_seconds = []
    start = time.perf_counter()
    for record in keys[start_key:end_key]:
        key_start = time.perf_counter()
        defect = str(record["defect"])
        total_a_exp = tuple(int(item) for item in record["total_a_exp"])
        value = evaluate_semantic_key(
            defect,
            total_a_exp,
            prime=p,
            method=normalized_method,
            beta_chunk_size=beta_chunk_size,
            max_chunk_terms=max_chunk_terms,
        )
        elapsed = time.perf_counter() - key_start
        value_seconds.append(elapsed)
        value_records.append(
            {
                "key": str(record["key"]),
                "defect": defect,
                "total_a_exp": list(total_a_exp),
                "value": int(value) % p,
                "elapsed_seconds": elapsed,
            }
        )

    payload = {
        "kind": "c18_semantic_value_chunk",
        "schema_version": _SCHEMA_VERSION,
        "complete": True,
        "prime": p,
        "chunk_id": int(chunk["chunk_id"]),
        "manifest_path": str(manifest_path),
        "row_kind": str(manifest["row_kind"]),
        "method": method,
        "normalized_method": normalized_method,
        "start_key": start_key,
        "end_key": end_key,
        "key_count": len(value_records),
        "row_count": int(manifest["row_count"]),
        "column_count": int(manifest["column_count"]),
        "entry_count": int(manifest["entry_count"]),
        "source_row_indices": list(manifest["source_row_indices"]),
        "source_row_names": list(manifest["source_row_names"]),
        "test_column_indices": list(manifest["test_column_indices"]),
        "test_column_names": list(manifest["test_column_names"]),
        "values": value_records,
        "elapsed_seconds": time.perf_counter() - start,
        "value_seconds": value_seconds,
        "cache_info": all_a_cache_info(),
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
    }
    write_json_maybe_gzip(output_path, payload)
    return payload


def evaluate_semantic_key(
    defect: str,
    total_a_exp: Sequence[int],
    *,
    config: FormulaConfig = RANK7_G2_D1,
    prime: int | None = None,
    method: str = "batched",
    beta_chunk_size: int = 2,
    max_chunk_terms: int = 200_000,
) -> int:
    """Evaluate one semantic all-a value modulo ``prime``."""

    p = require_prime(config.primary_prime if prime is None else prime)
    normalized_method = _normalize_method(method)
    a_exp = tuple(int(item) for item in total_a_exp)
    if normalized_method == "synthetic":
        return _synthetic_semantic_value(config, defect, a_exp, p)

    total = _semantic_total_monomial(config, defect, a_exp)
    if normalized_method == "moment":
        return all_a_pairing_total_moment_mod(config, total, prime=p)
    if normalized_method == "batched":
        return all_a_pairing_total_batched_mod(
            config,
            total,
            prime=p,
            beta_chunk_size=beta_chunk_size,
            max_chunk_terms=max_chunk_terms,
        )
    raise AssertionError(f"unexpected normalized method {normalized_method!r}")


def merge_semantic_value_chunks(
    paths: Sequence[Path | str],
    *,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Merge semantic-value chunks into one lookup table."""

    if not paths:
        raise ValueError("at least one semantic value chunk is required")
    chunks = [read_json_maybe_gzip(Path(path)) for path in paths]
    _validate_value_chunks(chunks)

    first = chunks[0]
    p = require_prime(int(first["prime"]))
    value_by_key: dict[str, int] = {}
    record_by_key: dict[str, dict[str, object]] = {}
    duplicate_keys = set()
    for chunk in chunks:
        for record in chunk["values"]:
            key = str(record["key"])
            value = int(record["value"]) % p
            if key in value_by_key:
                duplicate_keys.add(key)
                if value_by_key[key] != value:
                    raise ValueError(f"semantic chunks disagree on key {key}")
                continue
            value_by_key[key] = value
            record_by_key[key] = {
                "key": key,
                "defect": str(record["defect"]),
                "total_a_exp": [int(item) for item in record["total_a_exp"]],
            }

    key_records = sorted(
        record_by_key.values(),
        key=lambda item: _semantic_record_sort_key(RANK7_G2_D1, item),
    )
    payload = {
        "kind": "c18_semantic_value_table",
        "schema_version": _SCHEMA_VERSION,
        "complete": True,
        "prime": p,
        "row_kind": first["row_kind"],
        "method": first.get("method"),
        "normalized_method": first.get("normalized_method"),
        "row_count": int(first["row_count"]),
        "column_count": int(first["column_count"]),
        "entry_count": int(first["entry_count"]),
        "key_count": len(value_by_key),
        "chunk_count": len(chunks),
        "input_paths": [str(path) for path in paths],
        "source_row_indices": list(first["source_row_indices"]),
        "source_row_names": list(first["source_row_names"]),
        "test_column_indices": list(first["test_column_indices"]),
        "test_column_names": list(first["test_column_names"]),
        "duplicate_keys": sorted(duplicate_keys),
        "key_records": key_records,
        "values": dict(sorted(value_by_key.items())),
        "chunk_elapsed_seconds": [
            float(chunk.get("elapsed_seconds", 0.0)) for chunk in chunks
        ],
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
    }
    if output_path is not None:
        write_json_maybe_gzip(output_path, payload)
    return payload


def merge_key_manifest_outputs(
    manifest_path: Path,
    *,
    output_path: Path | None = None,
    require_complete: bool = True,
) -> dict[str, object]:
    """Merge all semantic-value chunks listed in a key manifest."""

    manifest = read_json_maybe_gzip(manifest_path)
    if manifest.get("kind") != "c18_semantic_key_manifest":
        raise ValueError("manifest kind must be c18_semantic_key_manifest")
    chunk_paths = [Path(str(chunk["output_path"])) for chunk in manifest["chunks"]]
    missing = [str(path) for path in chunk_paths if not path.exists()]
    if missing and require_complete:
        raise ValueError(f"manifest has missing key chunk outputs: {missing[:5]}")
    existing = [path for path in chunk_paths if path.exists()]
    payload = merge_semantic_value_chunks(existing, output_path=output_path)
    payload["manifest_path"] = str(manifest_path)
    payload["manifest_chunk_count"] = int(manifest["chunk_count"])
    payload["manifest_key_count"] = int(manifest["key_count"])
    payload["missing_chunk_outputs"] = missing
    payload["complete"] = not missing
    if output_path is not None:
        write_json_maybe_gzip(output_path, payload)
    return payload


def assemble_rank_from_value_table(
    table_path: Path | str,
    *,
    config: FormulaConfig = RANK7_G2_D1,
    output_path: Path | None = None,
    stop_rank: int | None = None,
    compute_left_nullspace: bool = False,
    store_matrix: bool = False,
) -> dict[str, object]:
    """Assemble the selected all-a matrix by lookup and compute column rank."""

    table = read_json_maybe_gzip(Path(table_path))
    if table.get("kind") != "c18_semantic_value_table":
        raise ValueError("value table kind must be c18_semantic_value_table")
    if not table.get("complete", False):
        raise ValueError("value table is not complete")

    p = require_prime(int(table["prime"]))
    row_kind = str(table["row_kind"])
    rows, source_row_indices = _select_rows(
        config,
        row_kind,
        None,
        start_row=0,
        end_row=None,
        max_rows=None,
    )
    selected_source_indices = tuple(int(item) for item in table["source_row_indices"])
    row_by_index = {int(index): row for row, index in zip(rows, source_row_indices)}
    rows = tuple(row_by_index[index] for index in selected_source_indices)
    source_row_indices = tuple(selected_source_indices)

    all_columns, all_column_indices = _select_columns(
        config,
        None,
        start_column=0,
        end_column=None,
        max_columns=None,
    )
    selected_column_indices = tuple(int(item) for item in table["test_column_indices"])
    column_by_index = {
        int(index): column for column, index in zip(all_columns, all_column_indices)
    }
    columns = tuple(column_by_index[index] for index in selected_column_indices)
    if len(rows) != int(table["row_count"]):
        raise ValueError("value table row selection does not match current basis")
    if len(columns) != int(table["column_count"]):
        raise ValueError("value table column selection does not match current basis")

    values = table.get("values", {})
    if not isinstance(values, dict):
        raise ValueError("value table values must be a dict")
    value_by_key = {str(key): int(value) % p for key, value in values.items()}
    tracker = ColumnRankTracker(row_count=len(rows), prime=p)
    matrix_rows = [[] for _ in rows]
    column_seconds = []
    missing_keys = set()
    start = time.perf_counter()
    for column_index, column in zip(selected_column_indices, columns):
        if stop_rank is not None and tracker.rank >= stop_rank:
            break
        column_start = time.perf_counter()
        vector = []
        for row_index, row in enumerate(rows):
            key = _entry_semantic_key(row, column)
            value = value_by_key.get(key)
            if value is None:
                missing_keys.add(key)
                value = 0
            vector.append(value)
            matrix_rows[row_index].append(value)
        if missing_keys:
            raise ValueError(f"value table is missing semantic keys: {sorted(missing_keys)[:5]}")
        tracker.add_column(vector, index=column_index)
        column_seconds.append(time.perf_counter() - column_start)

    left_nullspace = None
    if compute_left_nullspace:
        left_nullspace = [
            list(vector) for vector in left_nullspace_mod(matrix_rows, p)
        ]

    selected_name_by_index = dict(
        zip(selected_column_indices, (column.name for column in columns))
    )
    payload = {
        "kind": "c18_semantic_table_rank",
        "schema_version": _SCHEMA_VERSION,
        "prime": p,
        "row_kind": row_kind,
        "method": table.get("method"),
        "normalized_method": table.get("normalized_method"),
        "value_table_path": str(table_path),
        "row_count": len(rows),
        "column_count": len(columns),
        "processed_columns": tracker.processed_columns,
        "source_row_indices": list(source_row_indices),
        "source_row_names": [row.name for row in rows],
        "test_column_indices": list(selected_column_indices),
        "test_column_names": [column.name for column in columns],
        "rank": tracker.rank,
        "left_nullity": tracker.nullity_left,
        "selected_column_indices": list(tracker.selected_indices),
        "selected_column_names": [
            selected_name_by_index[index] for index in tracker.selected_indices
        ],
        "elapsed_seconds": time.perf_counter() - start,
        "column_seconds": column_seconds,
        "left_nullspace": left_nullspace,
        "matrix": matrix_rows if store_matrix else None,
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
    }
    if output_path is not None:
        write_json_maybe_gzip(output_path, payload)
    return payload


def _entry_semantic_key(row, column) -> str:
    total_a_exp = tuple(
        int(row.monomial.a_exp[idx]) + int(column.monomial.a_exp[idx])
        for idx in range(len(row.monomial.a_exp))
    )
    return semantic_key_id(row.defect, total_a_exp)


def _semantic_total_monomial(
    config: FormulaConfig,
    defect: str,
    total_a_exp: Tuple[int, ...],
) -> InvariantMonomial:
    f_exp = [0 for _ in config.class_ranks]
    gamma_exp = [0 for _ in config.gamma_labels]
    matched = False
    for idx, r in enumerate(config.class_ranks):
        if defect == f"f{r}":
            f_exp[idx] = 1
            matched = True
            break
    if not matched:
        for idx, (r, s) in enumerate(config.gamma_labels):
            if defect == f"gamma{r}{s}":
                gamma_exp[idx] = 1
                matched = True
                break
    if not matched:
        raise ValueError(f"unknown semantic defect {defect!r}")
    return InvariantMonomial.from_exponents(
        config,
        a_exp=total_a_exp,
        f_exp=f_exp,
        gamma_exp=gamma_exp,
    )


def _synthetic_semantic_value(
    config: FormulaConfig,
    defect: str,
    total_a_exp: Tuple[int, ...],
    prime: int,
) -> int:
    defect_weight = _defect_sort_index(config, defect) + 1
    value = defect_weight * 17
    for idx, exp in enumerate(total_a_exp):
        value += (idx + 3) * (int(exp) + 1) * (int(exp) + 2)
    return value % prime


def _semantic_record_sort_key(
    config: FormulaConfig,
    item: dict[str, object],
) -> Tuple[int, Tuple[int, ...], str]:
    defect = str(item["defect"])
    return (
        _defect_sort_index(config, defect),
        tuple(int(value) for value in item["total_a_exp"]),
        str(item["key"]),
    )


def _defect_sort_index(config: FormulaConfig, defect: str) -> int:
    for idx, r in enumerate(config.class_ranks):
        if defect == f"f{r}":
            return idx
    offset = len(config.class_ranks)
    for idx, (r, s) in enumerate(config.gamma_labels):
        if defect == f"gamma{r}{s}":
            return offset + idx
    return offset + len(config.gamma_labels)


def _normalize_method(method: str) -> str:
    normalized = method.lower()
    if normalized == "semantic-batched":
        return "batched"
    if normalized in {"synthetic", "moment", "batched"}:
        return normalized
    raise ValueError("method must be synthetic, moment, batched, or semantic-batched")


def _validate_value_chunks(chunks: Sequence[dict[str, object]]) -> None:
    first = chunks[0]
    if first.get("kind") != "c18_semantic_value_chunk":
        raise ValueError("worker output kind must be c18_semantic_value_chunk")
    required = (
        "prime",
        "row_kind",
        "row_count",
        "column_count",
        "entry_count",
        "source_row_indices",
        "test_column_indices",
        "normalized_method",
    )
    for chunk in chunks[1:]:
        if chunk.get("kind") != "c18_semantic_value_chunk":
            raise ValueError("worker output kind must be c18_semantic_value_chunk")
        for key in required:
            if chunk.get(key) != first.get(key):
                raise ValueError(f"semantic value chunks disagree on {key}")


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] == "plan-keys":
        return _plan_keys_main(args_list[1:])
    if args_list and args_list[0] == "run-key-manifest":
        return _run_key_manifest_main(args_list[1:])
    if args_list and args_list[0] == "merge-key-values":
        return _merge_key_values_main(args_list[1:])
    if args_list and args_list[0] == "merge-key-manifest":
        return _merge_key_manifest_main(args_list[1:])
    if args_list and args_list[0] == "assemble-rank":
        return _assemble_rank_main(args_list[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.print_help()
    return 2


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--row-kind", choices=("all", "even", "gamma"), default="even")
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


def _plan_keys_main(argv: Sequence[str] | None) -> int:
    parser = argparse.ArgumentParser(description="Plan c18 semantic value chunks")
    parser.add_argument("--prime", type=int, default=RANK7_G2_D1.primary_prime)
    _add_selection_args(parser)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="key_chunk")
    parser.add_argument("--output-suffix", default=".json.gz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = plan_c18_semantic_keys(
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


def _run_key_manifest_main(argv: Sequence[str] | None) -> int:
    parser = argparse.ArgumentParser(description="Run one c18 semantic key chunk")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--beta-chunk-size", type=int, default=2)
    parser.add_argument("--max-chunk-terms", type=int, default=200_000)
    args = parser.parse_args(argv)

    task_id = _task_id(args.task_id)
    payload = run_key_manifest_chunk(
        args.manifest,
        task_id,
        skip_existing=not args.recompute,
        beta_chunk_size=args.beta_chunk_size,
        max_chunk_terms=args.max_chunk_terms,
    )
    print(json.dumps(_summary(payload), indent=2, sort_keys=True))
    return 0


def _merge_key_values_main(argv: Sequence[str] | None) -> int:
    parser = argparse.ArgumentParser(description="Merge c18 semantic value chunks")
    parser.add_argument("chunks", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = merge_semantic_value_chunks(args.chunks, output_path=args.output)
    print(json.dumps(_summary(payload), indent=2, sort_keys=True))
    return 0


def _merge_key_manifest_main(argv: Sequence[str] | None) -> int:
    parser = argparse.ArgumentParser(description="Merge chunks from a key manifest")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = merge_key_manifest_outputs(
        args.manifest,
        output_path=args.output,
        require_complete=not args.allow_missing,
    )
    print(json.dumps(_summary(payload), indent=2, sort_keys=True))
    return 0


def _assemble_rank_main(argv: Sequence[str] | None) -> int:
    parser = argparse.ArgumentParser(description="Assemble c18 rank from semantic table")
    parser.add_argument("table", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-rank", type=int, default=None)
    parser.add_argument("--left-nullspace", action="store_true")
    parser.add_argument("--store-matrix", action="store_true")
    args = parser.parse_args(argv)

    payload = assemble_rank_from_value_table(
        args.table,
        output_path=args.output,
        stop_rank=args.stop_rank,
        compute_left_nullspace=args.left_nullspace,
        store_matrix=args.store_matrix,
    )
    print(json.dumps(_summary(payload), indent=2, sort_keys=True))
    return 0


def _task_id(task_id: int | None) -> int:
    if task_id is not None:
        return int(task_id)
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw is None:
        raise ValueError("--task-id is required outside a Slurm array")
    return int(raw)


def _summary(payload: dict[str, object]) -> dict[str, object]:
    keys = (
        "kind",
        "complete",
        "prime",
        "row_kind",
        "method",
        "normalized_method",
        "row_count",
        "column_count",
        "entry_count",
        "key_count",
        "chunk_count",
        "chunk_id",
        "processed_columns",
        "rank",
        "left_nullity",
        "reuse_factor",
        "elapsed_seconds",
        "missing_chunk_outputs",
        "selected_column_indices",
    )
    return {key: payload[key] for key in keys if key in payload}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
