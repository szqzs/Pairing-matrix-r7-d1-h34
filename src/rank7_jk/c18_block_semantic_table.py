"""Semantic value-table workflow for supported c18 block probes.

This is the block-general version of the all-a semantic table path.  It is
intended for gamma-sensitive scouts where many row/column entries share the
same invariant shape and total exponents, but the useful column family is not
known in advance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence, Tuple

from .c18_all_a_probe import _git_dirty, _git_head, _select_rows
from .c18_basis import C18SourceRow, H62TestColumn
from .c18_block_probe import (
    UnsupportedBlockEntry,
    _entry_key_and_metadata,
    _evaluate_entry,
    _middle_out,
    _normalize_column_kind,
    _normalize_method,
    _ordered_columns,
    _ordered_rows,
    _select_columns,
)
from .c18_even_worker import read_json_maybe_gzip, write_json_maybe_gzip
from .config import FormulaConfig, RANK7_G2_D1
from .exterior import ExteriorAlgebra
from .mod_arith import require_prime
from .rank_stream import ColumnRankTracker, left_nullspace_mod

_SCHEMA_VERSION = 1


def enumerate_c18_block_semantic_keys(
    *,
    config: FormulaConfig = RANK7_G2_D1,
    row_kind: str = "gamma",
    column_kind: str = "f2-power",
    start_row: int = 0,
    end_row: int | None = None,
    max_rows: int | None = None,
    row_order: str = "defect-balanced",
    row_random_seed: int = 0,
    start_column: int = 0,
    end_column: int | None = None,
    max_columns: int | None = None,
    column_order: str = "balanced",
    column_random_seed: int = 0,
    unsupported: str = "error",
) -> dict[str, object]:
    """Enumerate unique semantic keys for a supported c18 block matrix."""

    unsupported_action = _normalize_table_unsupported(unsupported)
    normalized_column_kind = _normalize_column_kind(column_kind)
    rows, source_indices = _select_block_rows(
        config,
        row_kind,
        start_row=start_row,
        end_row=end_row,
        max_rows=max_rows,
        order=row_order,
        random_seed=row_random_seed,
    )
    columns, column_indices, available_column_count = _select_block_columns(
        config,
        normalized_column_kind,
        start_column=start_column,
        end_column=end_column,
        max_columns=max_columns,
        order=column_order,
        random_seed=column_random_seed,
    )

    exterior = ExteriorAlgebra(config)
    records: dict[str, dict[str, object]] = {}
    unsupported_entries = 0
    for row in rows:
        for column in columns:
            try:
                key, metadata = _entry_key_and_metadata(config, exterior, row, column)
            except UnsupportedBlockEntry:
                unsupported_entries += 1
                if unsupported_action == "zero":
                    continue
                raise
            record = records.get(key)
            if record is None:
                records[key] = _record_from_metadata(key, metadata, use_count=1)
            else:
                record["use_count"] = int(record["use_count"]) + 1

    keys = sorted(records.values(), key=_semantic_record_sort_key)
    entry_count = len(rows) * len(columns)
    return {
        "kind": "c18_block_semantic_key_enumeration",
        "schema_version": _SCHEMA_VERSION,
        "row_kind": row_kind,
        "column_kind": normalized_column_kind,
        "unsupported": unsupported_action,
        "row_order": row_order,
        "row_random_seed": int(row_random_seed),
        "column_order": column_order,
        "column_random_seed": int(column_random_seed),
        "start_row": int(start_row),
        "end_row": None if end_row is None else int(end_row),
        "max_rows": None if max_rows is None else int(max_rows),
        "start_column": int(start_column),
        "end_column": None if end_column is None else int(end_column),
        "max_columns": None if max_columns is None else int(max_columns),
        "row_count": len(rows),
        "available_column_count": int(available_column_count),
        "column_count": len(columns),
        "entry_count": entry_count,
        "unsupported_entries": int(unsupported_entries),
        "key_count": len(keys),
        "reuse_factor": 0.0 if not keys else entry_count / len(keys),
        "source_row_indices": list(source_indices),
        "source_row_names": [row.name for row in rows],
        "source_row_kinds": [row.kind for row in rows],
        "test_column_indices": list(column_indices),
        "test_column_names": [column.name for column in columns],
        "test_column_kinds": [column.kind for column in columns],
        "test_column_defects": [column.defect for column in columns],
        "keys": keys,
    }


def plan_c18_block_semantic_keys(
    *,
    config: FormulaConfig = RANK7_G2_D1,
    prime: int | None = None,
    row_kind: str = "gamma",
    column_kind: str = "f2-power",
    method: str = "semantic-batched",
    start_row: int = 0,
    end_row: int | None = None,
    max_rows: int | None = None,
    row_order: str = "defect-balanced",
    row_random_seed: int = 0,
    start_column: int = 0,
    end_column: int | None = None,
    max_columns: int | None = None,
    column_order: str = "balanced",
    column_random_seed: int = 0,
    unsupported: str = "error",
    chunk_size: int = 100,
    output_dir: Path = Path("results/c18_block_semantic_values/chunks"),
    output_prefix: str = "key_chunk",
    output_suffix: str = ".json.gz",
    output_path: Path | None = None,
) -> dict[str, object]:
    """Create a deterministic manifest of block semantic-value chunks."""

    p = require_prime(config.primary_prime if prime is None else prime)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    normalized_method = _normalize_method(method)
    enumeration = enumerate_c18_block_semantic_keys(
        config=config,
        row_kind=row_kind,
        column_kind=column_kind,
        start_row=start_row,
        end_row=end_row,
        max_rows=max_rows,
        row_order=row_order,
        row_random_seed=row_random_seed,
        start_column=start_column,
        end_column=end_column,
        max_columns=max_columns,
        column_order=column_order,
        column_random_seed=column_random_seed,
        unsupported=unsupported,
    )
    keys = list(enumeration["keys"])
    chunks = []
    for chunk_id, start_key in enumerate(range(0, len(keys), chunk_size)):
        end_key = min(start_key + chunk_size, len(keys))
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
        "kind": "c18_block_semantic_key_manifest",
        "schema_version": _SCHEMA_VERSION,
        "prime": p,
        "row_kind": str(enumeration["row_kind"]),
        "column_kind": str(enumeration["column_kind"]),
        "method": method,
        "normalized_method": normalized_method,
        "unsupported": str(enumeration["unsupported"]),
        "row_order": row_order,
        "row_random_seed": int(row_random_seed),
        "column_order": column_order,
        "column_random_seed": int(column_random_seed),
        "start_row": int(start_row),
        "end_row": None if end_row is None else int(end_row),
        "max_rows": None if max_rows is None else int(max_rows),
        "start_column": int(start_column),
        "end_column": None if end_column is None else int(end_column),
        "max_columns": None if max_columns is None else int(max_columns),
        "row_count": int(enumeration["row_count"]),
        "available_column_count": int(enumeration["available_column_count"]),
        "column_count": int(enumeration["column_count"]),
        "entry_count": int(enumeration["entry_count"]),
        "unsupported_entries": int(enumeration["unsupported_entries"]),
        "key_count": int(enumeration["key_count"]),
        "reuse_factor": float(enumeration["reuse_factor"]),
        "chunk_size": int(chunk_size),
        "chunk_count": len(chunks),
        "source_row_indices": list(enumeration["source_row_indices"]),
        "source_row_names": list(enumeration["source_row_names"]),
        "source_row_kinds": list(enumeration["source_row_kinds"]),
        "test_column_indices": list(enumeration["test_column_indices"]),
        "test_column_names": list(enumeration["test_column_names"]),
        "test_column_kinds": list(enumeration["test_column_kinds"]),
        "test_column_defects": list(enumeration["test_column_defects"]),
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


def run_block_key_manifest_chunk(
    manifest_path: Path,
    task_id: int,
    *,
    skip_existing: bool = True,
    beta_chunk_size: int = 2,
    max_chunk_terms: int = 200_000,
) -> dict[str, object]:
    """Evaluate one semantic-value chunk from a block key manifest."""

    manifest = read_json_maybe_gzip(manifest_path)
    if manifest.get("kind") != "c18_block_semantic_key_manifest":
        raise ValueError("manifest kind must be c18_block_semantic_key_manifest")
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
        if (
            payload.get("kind") == "c18_block_semantic_value_chunk"
            and payload.get("complete")
        ):
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
        metadata = _metadata_from_record(record)
        value = _evaluate_entry(
            RANK7_G2_D1,
            metadata,
            prime=p,
            method=normalized_method,
            beta_chunk_size=beta_chunk_size,
            max_chunk_terms=max_chunk_terms,
        )
        elapsed = time.perf_counter() - key_start
        value_seconds.append(elapsed)
        value_record = _record_from_metadata(str(record["key"]), metadata)
        value_record["value"] = int(value) % p
        value_record["elapsed_seconds"] = elapsed
        value_records.append(value_record)

    payload = {
        "kind": "c18_block_semantic_value_chunk",
        "schema_version": _SCHEMA_VERSION,
        "complete": True,
        "prime": p,
        "chunk_id": int(chunk["chunk_id"]),
        "manifest_path": str(manifest_path),
        "row_kind": str(manifest["row_kind"]),
        "column_kind": str(manifest["column_kind"]),
        "method": method,
        "normalized_method": normalized_method,
        "unsupported": str(manifest["unsupported"]),
        "row_order": str(manifest["row_order"]),
        "row_random_seed": int(manifest["row_random_seed"]),
        "column_order": str(manifest["column_order"]),
        "column_random_seed": int(manifest["column_random_seed"]),
        "start_key": start_key,
        "end_key": end_key,
        "key_count": len(value_records),
        "row_count": int(manifest["row_count"]),
        "available_column_count": int(manifest["available_column_count"]),
        "column_count": int(manifest["column_count"]),
        "entry_count": int(manifest["entry_count"]),
        "unsupported_entries": int(manifest["unsupported_entries"]),
        "source_row_indices": list(manifest["source_row_indices"]),
        "source_row_names": list(manifest["source_row_names"]),
        "source_row_kinds": list(manifest["source_row_kinds"]),
        "test_column_indices": list(manifest["test_column_indices"]),
        "test_column_names": list(manifest["test_column_names"]),
        "test_column_kinds": list(manifest["test_column_kinds"]),
        "test_column_defects": list(manifest["test_column_defects"]),
        "values": value_records,
        "elapsed_seconds": time.perf_counter() - start,
        "value_seconds": value_seconds,
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
    }
    write_json_maybe_gzip(output_path, payload)
    return payload


def merge_block_semantic_value_chunks(
    paths: Sequence[Path | str],
    *,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Merge block semantic-value chunks into one lookup table."""

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
            record_by_key[key] = _record_from_metadata(
                key,
                _metadata_from_record(record),
            )

    key_records = sorted(record_by_key.values(), key=_semantic_record_sort_key)
    payload = {
        "kind": "c18_block_semantic_value_table",
        "schema_version": _SCHEMA_VERSION,
        "complete": True,
        "prime": p,
        "row_kind": first["row_kind"],
        "column_kind": first["column_kind"],
        "method": first.get("method"),
        "normalized_method": first.get("normalized_method"),
        "unsupported": first.get("unsupported"),
        "row_order": first.get("row_order"),
        "row_random_seed": first.get("row_random_seed"),
        "column_order": first.get("column_order"),
        "column_random_seed": first.get("column_random_seed"),
        "row_count": int(first["row_count"]),
        "available_column_count": int(first["available_column_count"]),
        "column_count": int(first["column_count"]),
        "entry_count": int(first["entry_count"]),
        "unsupported_entries": int(first["unsupported_entries"]),
        "key_count": len(value_by_key),
        "chunk_count": len(chunks),
        "input_paths": [str(path) for path in paths],
        "source_row_indices": list(first["source_row_indices"]),
        "source_row_names": list(first["source_row_names"]),
        "source_row_kinds": list(first["source_row_kinds"]),
        "test_column_indices": list(first["test_column_indices"]),
        "test_column_names": list(first["test_column_names"]),
        "test_column_kinds": list(first["test_column_kinds"]),
        "test_column_defects": list(first["test_column_defects"]),
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


def merge_block_key_manifest_outputs(
    manifest_path: Path,
    *,
    output_path: Path | None = None,
    require_complete: bool = True,
) -> dict[str, object]:
    """Merge all block semantic-value chunks listed in a key manifest."""

    manifest = read_json_maybe_gzip(manifest_path)
    if manifest.get("kind") != "c18_block_semantic_key_manifest":
        raise ValueError("manifest kind must be c18_block_semantic_key_manifest")
    chunk_paths = [Path(str(chunk["output_path"])) for chunk in manifest["chunks"]]
    missing = [str(path) for path in chunk_paths if not path.exists()]
    if missing and require_complete:
        raise ValueError(f"manifest has missing key chunk outputs: {missing[:5]}")
    existing = [path for path in chunk_paths if path.exists()]
    if not existing and int(manifest["key_count"]) == 0:
        payload = _empty_value_table_from_manifest(manifest)
        if output_path is not None:
            write_json_maybe_gzip(output_path, payload)
    else:
        payload = merge_block_semantic_value_chunks(existing, output_path=output_path)
    payload["manifest_path"] = str(manifest_path)
    payload["manifest_chunk_count"] = int(manifest["chunk_count"])
    payload["manifest_key_count"] = int(manifest["key_count"])
    payload["missing_chunk_outputs"] = missing
    payload["complete"] = not missing
    if output_path is not None:
        write_json_maybe_gzip(output_path, payload)
    return payload


def _empty_value_table_from_manifest(manifest: dict[str, object]) -> dict[str, object]:
    return {
        "kind": "c18_block_semantic_value_table",
        "schema_version": _SCHEMA_VERSION,
        "complete": True,
        "prime": require_prime(int(manifest["prime"])),
        "row_kind": manifest["row_kind"],
        "column_kind": manifest["column_kind"],
        "method": manifest.get("method"),
        "normalized_method": manifest.get("normalized_method"),
        "unsupported": manifest.get("unsupported"),
        "row_order": manifest.get("row_order"),
        "row_random_seed": manifest.get("row_random_seed"),
        "column_order": manifest.get("column_order"),
        "column_random_seed": manifest.get("column_random_seed"),
        "row_count": int(manifest["row_count"]),
        "available_column_count": int(manifest["available_column_count"]),
        "column_count": int(manifest["column_count"]),
        "entry_count": int(manifest["entry_count"]),
        "unsupported_entries": int(manifest["unsupported_entries"]),
        "key_count": 0,
        "chunk_count": 0,
        "input_paths": [],
        "source_row_indices": list(manifest["source_row_indices"]),
        "source_row_names": list(manifest["source_row_names"]),
        "source_row_kinds": list(manifest["source_row_kinds"]),
        "test_column_indices": list(manifest["test_column_indices"]),
        "test_column_names": list(manifest["test_column_names"]),
        "test_column_kinds": list(manifest["test_column_kinds"]),
        "test_column_defects": list(manifest["test_column_defects"]),
        "duplicate_keys": [],
        "key_records": [],
        "values": {},
        "chunk_elapsed_seconds": [],
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
    }


def assemble_block_rank_from_value_table(
    table_path: Path | str,
    *,
    config: FormulaConfig = RANK7_G2_D1,
    output_path: Path | None = None,
    stop_rank: int | None = None,
    compute_left_nullspace: bool = False,
    store_matrix: bool = False,
) -> dict[str, object]:
    """Assemble the selected block matrix by lookup and compute column rank."""

    table = read_json_maybe_gzip(Path(table_path))
    if table.get("kind") != "c18_block_semantic_value_table":
        raise ValueError("value table kind must be c18_block_semantic_value_table")
    if not table.get("complete", False):
        raise ValueError("value table is not complete")

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

    values = table.get("values", {})
    if not isinstance(values, dict):
        raise ValueError("value table values must be a dict")
    value_by_key = {str(key): int(value) % p for key, value in values.items()}
    tracker = ColumnRankTracker(row_count=len(rows), prime=p)
    exterior = ExteriorAlgebra(config)
    matrix_rows = [[] for _ in rows] if compute_left_nullspace or store_matrix else None
    missing_keys = set()
    column_records = []
    start = time.perf_counter()
    for column_index, column in zip(selected_column_indices, columns):
        if stop_rank is not None and tracker.rank >= stop_rank:
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
        column_records.append(
            {
                "index": int(column_index),
                "name": column.name,
                "kind": column.kind,
                "defect": column.defect,
                "nonzero_count": int(nonzero_count),
                "unsupported_count": int(unsupported_count),
                "independent": bool(independent),
                "rank_after": int(tracker.rank),
                "left_nullity_after": int(tracker.nullity_left),
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
        "kind": "c18_block_semantic_table_rank",
        "schema_version": _SCHEMA_VERSION,
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
        "processed_columns": tracker.processed_columns,
        "source_row_indices": list(source_row_indices),
        "source_row_names": [row.name for row in rows],
        "source_row_kinds": [row.kind for row in rows],
        "test_column_indices": list(selected_column_indices),
        "test_column_names": [column.name for column in columns],
        "test_column_kinds": [column.kind for column in columns],
        "test_column_defects": [column.defect for column in columns],
        "rank": tracker.rank,
        "left_nullity": tracker.nullity_left,
        "selected_column_indices": list(tracker.selected_indices),
        "selected_column_names": [
            selected_name_by_index[index] for index in tracker.selected_indices
        ],
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


def _select_block_rows(
    config: FormulaConfig,
    row_kind: str,
    *,
    start_row: int,
    end_row: int | None,
    max_rows: int | None,
    order: str,
    random_seed: int,
) -> Tuple[Tuple[C18SourceRow, ...], Tuple[int, ...]]:
    rows, indices = _select_rows(
        config,
        row_kind,
        None,
        start_row=start_row,
        end_row=end_row,
        max_rows=None,
    )
    pairs = _ordered_rows(rows, indices, order=order, random_seed=random_seed)
    if max_rows is not None:
        if max_rows < 0:
            raise ValueError("max_rows must be nonnegative")
        pairs = pairs[:max_rows]
    return tuple(row for _idx, row in pairs), tuple(int(idx) for idx, _row in pairs)


def _select_block_columns(
    config: FormulaConfig,
    column_kind: str,
    *,
    start_column: int,
    end_column: int | None,
    max_columns: int | None,
    order: str,
    random_seed: int,
) -> tuple[Tuple[H62TestColumn, ...], Tuple[int, ...], int]:
    columns, indices = _select_columns(
        config,
        column_kind,
        start_column=start_column,
        end_column=end_column,
        max_columns=None,
    )
    pairs = _ordered_block_columns(
        config,
        columns,
        indices,
        column_kind=column_kind,
        order=order,
        random_seed=random_seed,
    )
    if max_columns is not None:
        if max_columns < 0:
            raise ValueError("max_columns must be nonnegative")
        pairs = pairs[:max_columns]
    return (
        tuple(column for _idx, column in pairs),
        tuple(int(idx) for idx, _column in pairs),
        len(columns),
    )


def _ordered_block_columns(
    config: FormulaConfig,
    columns: Sequence[H62TestColumn],
    indices: Sequence[int],
    *,
    column_kind: str,
    order: str,
    random_seed: int,
) -> list[tuple[int, H62TestColumn]]:
    if order in {"f2-power-balanced", "f2-balanced"}:
        if column_kind != "f2-power":
            raise ValueError("f2-power-balanced column order requires f2-power columns")
        return _f2_power_round_robin_order(indices, columns, reverse=False)
    if order in {"f2-power-desc-balanced", "f2-desc-balanced"}:
        if column_kind != "f2-power":
            raise ValueError(
                "f2-power-desc-balanced column order requires f2-power columns"
            )
        return _f2_power_round_robin_order(indices, columns, reverse=True)
    return _ordered_columns(
        config,
        columns,
        indices,
        order=order,
        random_seed=random_seed,
    )


def _f2_power_round_robin_order(
    indices: Sequence[int],
    columns: Sequence[H62TestColumn],
    *,
    reverse: bool,
) -> list[tuple[int, H62TestColumn]]:
    groups: dict[int, list[tuple[int, H62TestColumn]]] = {}
    for index, column in zip(indices, columns):
        groups.setdefault(_f2_power(column), []).append((int(index), column))
    for power, group in list(groups.items()):
        groups[power] = _middle_out(group)
    ordered: list[tuple[int, H62TestColumn]] = []
    powers = sorted(groups, reverse=reverse)
    depth = 0
    while True:
        added = False
        for power in powers:
            group = groups[power]
            if depth < len(group):
                ordered.append(group[depth])
                added = True
        if not added:
            break
        depth += 1
    return ordered


def _f2_power(column: H62TestColumn) -> int:
    return int(column.monomial.f_exp[0])


def _rows_by_stored_indices(
    config: FormulaConfig,
    row_kind: str,
    source_indices: Tuple[int, ...],
) -> tuple[Tuple[C18SourceRow, ...], Tuple[int, ...]]:
    all_rows, all_indices = _select_rows(
        config,
        row_kind,
        None,
        start_row=0,
        end_row=None,
        max_rows=None,
    )
    row_by_index = {int(index): row for row, index in zip(all_rows, all_indices)}
    return tuple(row_by_index[index] for index in source_indices), source_indices


def _columns_by_stored_indices(
    config: FormulaConfig,
    column_kind: str,
    column_indices: Tuple[int, ...],
) -> tuple[Tuple[H62TestColumn, ...], Tuple[int, ...]]:
    all_columns, all_indices = _select_columns(
        config,
        column_kind,
        start_column=0,
        end_column=None,
        max_columns=None,
    )
    column_by_index = {
        int(index): column for column, index in zip(all_columns, all_indices)
    }
    return tuple(column_by_index[index] for index in column_indices), column_indices


def _record_from_metadata(
    key: str,
    metadata: dict[str, object],
    *,
    use_count: int | None = None,
) -> dict[str, object]:
    record = {
        "key": key,
        "shape": str(metadata["shape"]),
        "total_a_exp": [int(item) for item in metadata.get("total_a_exp", ())],
        "total_f_exp": [int(item) for item in metadata.get("total_f_exp", ())],
    }
    if "total_gamma_exp" in metadata:
        record["total_gamma_exp"] = [
            int(item) for item in metadata.get("total_gamma_exp", ())
        ]
    if "b_mask" in metadata:
        record["b_mask"] = int(metadata["b_mask"])
    if "b_labels" in metadata:
        record["b_labels"] = [list(label) for label in metadata["b_labels"]]
    if use_count is not None:
        record["use_count"] = int(use_count)
    return record


def _metadata_from_record(record: dict[str, object]) -> dict[str, object]:
    metadata = {
        "shape": str(record["shape"]),
        "total_a_exp": [int(item) for item in record.get("total_a_exp", ())],
        "total_f_exp": [int(item) for item in record.get("total_f_exp", ())],
    }
    if "total_gamma_exp" in record:
        metadata["total_gamma_exp"] = [
            int(item) for item in record.get("total_gamma_exp", ())
        ]
    else:
        metadata["total_gamma_exp"] = []
    if "b_mask" in record:
        metadata["b_mask"] = int(record["b_mask"])
    if "b_labels" in record:
        metadata["b_labels"] = [list(label) for label in record["b_labels"]]
    return metadata


def _semantic_record_sort_key(item: dict[str, object]):
    shape_order = {
        "one-defect": 0,
        "f-only": 1,
        "f-gamma": 2,
        "f2-power": 3,
        "b-mask": 4,
    }
    return (
        shape_order.get(str(item.get("shape")), 99),
        tuple(int(value) for value in item.get("total_f_exp", ())),
        tuple(int(value) for value in item.get("total_gamma_exp", ())),
        int(item.get("b_mask", -1)),
        tuple(int(value) for value in item.get("total_a_exp", ())),
        str(item.get("key", "")),
    )


def _normalize_table_unsupported(unsupported: str) -> str:
    normalized = unsupported.lower().replace("_", "-")
    if normalized in {"error", "zero"}:
        return normalized
    raise ValueError("unsupported must be error or zero")


def _validate_value_chunks(chunks: Sequence[dict[str, object]]) -> None:
    first = chunks[0]
    if first.get("kind") != "c18_block_semantic_value_chunk":
        raise ValueError("worker output kind must be c18_block_semantic_value_chunk")
    required = (
        "prime",
        "row_kind",
        "column_kind",
        "row_count",
        "column_count",
        "entry_count",
        "unsupported_entries",
        "source_row_indices",
        "test_column_indices",
        "normalized_method",
        "unsupported",
        "row_order",
        "column_order",
    )
    for chunk in chunks[1:]:
        if chunk.get("kind") != "c18_block_semantic_value_chunk":
            raise ValueError("worker output kind must be c18_block_semantic_value_chunk")
        for key in required:
            if chunk.get(key) != first.get(key):
                raise ValueError(f"semantic value chunks disagree on {key}")


def _task_id(task_id: int | None) -> int:
    if task_id is not None:
        return int(task_id)
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw is None:
        raise ValueError("--task-id is required outside a Slurm array")
    return int(raw)


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
    parser.add_argument("--row-kind", choices=("all", "even", "gamma"), default="gamma")
    parser.add_argument(
        "--column-kind",
        choices=("all-a", "f2-power", "one-f", "one-gamma", "b-pair"),
        default="f2-power",
    )
    parser.add_argument(
        "--method",
        choices=("synthetic", "moment", "batched", "semantic-batched"),
        default="semantic-batched",
    )
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--row-order",
        choices=("sequential", "random", "defect-balanced"),
        default="defect-balanced",
    )
    parser.add_argument("--row-random-seed", type=int, default=0)
    parser.add_argument("--start-column", type=int, default=0)
    parser.add_argument("--end-column", type=int, default=None)
    parser.add_argument("--max-columns", type=int, default=None)
    parser.add_argument(
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
        default="balanced",
    )
    parser.add_argument("--column-random-seed", type=int, default=0)
    parser.add_argument("--unsupported", choices=("error", "zero"), default="error")


def _plan_keys_main(argv: Sequence[str] | None) -> int:
    parser = argparse.ArgumentParser(description="Plan c18 block semantic values")
    parser.add_argument("--prime", type=int, default=RANK7_G2_D1.primary_prime)
    _add_selection_args(parser)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="key_chunk")
    parser.add_argument("--output-suffix", default=".json.gz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = plan_c18_block_semantic_keys(
        prime=args.prime,
        row_kind=args.row_kind,
        column_kind=args.column_kind,
        method=args.method,
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
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        output_suffix=args.output_suffix,
        output_path=args.output,
    )
    print(json.dumps(_summary(payload), indent=2, sort_keys=True))
    return 0


def _run_key_manifest_main(argv: Sequence[str] | None) -> int:
    parser = argparse.ArgumentParser(description="Run one c18 block key chunk")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--beta-chunk-size", type=int, default=2)
    parser.add_argument("--max-chunk-terms", type=int, default=200_000)
    args = parser.parse_args(argv)

    payload = run_block_key_manifest_chunk(
        args.manifest,
        _task_id(args.task_id),
        skip_existing=not args.recompute,
        beta_chunk_size=args.beta_chunk_size,
        max_chunk_terms=args.max_chunk_terms,
    )
    print(json.dumps(_summary(payload), indent=2, sort_keys=True))
    return 0


def _merge_key_values_main(argv: Sequence[str] | None) -> int:
    parser = argparse.ArgumentParser(description="Merge c18 block semantic values")
    parser.add_argument("chunks", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = merge_block_semantic_value_chunks(args.chunks, output_path=args.output)
    print(json.dumps(_summary(payload), indent=2, sort_keys=True))
    return 0


def _merge_key_manifest_main(argv: Sequence[str] | None) -> int:
    parser = argparse.ArgumentParser(description="Merge chunks from a block key manifest")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = merge_block_key_manifest_outputs(
        args.manifest,
        output_path=args.output,
        require_complete=not args.allow_missing,
    )
    print(json.dumps(_summary(payload), indent=2, sort_keys=True))
    return 0


def _assemble_rank_main(argv: Sequence[str] | None) -> int:
    parser = argparse.ArgumentParser(description="Assemble c18 block rank")
    parser.add_argument("table", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-rank", type=int, default=None)
    parser.add_argument("--left-nullspace", action="store_true")
    parser.add_argument("--store-matrix", action="store_true")
    args = parser.parse_args(argv)

    payload = assemble_block_rank_from_value_table(
        args.table,
        output_path=args.output,
        stop_rank=args.stop_rank,
        compute_left_nullspace=args.left_nullspace,
        store_matrix=args.store_matrix,
    )
    print(json.dumps(_summary(payload), indent=2, sort_keys=True))
    return 0


def _summary(payload: dict[str, object]) -> dict[str, object]:
    keys = (
        "kind",
        "complete",
        "prime",
        "row_kind",
        "column_kind",
        "method",
        "normalized_method",
        "unsupported",
        "row_order",
        "column_order",
        "row_count",
        "available_column_count",
        "column_count",
        "entry_count",
        "unsupported_entries",
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
