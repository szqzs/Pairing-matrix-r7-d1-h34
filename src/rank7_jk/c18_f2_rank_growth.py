"""Rank-growth runner for the c18 high-f2 H62 test family."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from pathlib import Path
from typing import Sequence, Tuple

from .all_a_pairing import all_a_cache_info
from .c18_all_a_probe import _git_dirty, _git_head, _select_rows
from .c18_basis import C18SourceRow, H62TestColumn, h62_f2_power_test_columns
from .c18_block_probe import (
    UnsupportedBlockEntry,
    _entry_key_and_metadata,
    _evaluate_entry,
    _normalize_method,
    _ordered_columns,
    _ordered_rows,
)
from .c18_even_worker import read_json_maybe_gzip
from .config import FormulaConfig, RANK7_G2_D1
from .exterior import ExteriorAlgebra
from .mod_arith import mod_inv, require_prime
from .rank_stream import ColumnRankTracker, left_nullspace_mod, vec_mat_mul_mod

_SCHEMA_VERSION = 1


def run_c18_f2_rank_growth(
    *,
    config: FormulaConfig = RANK7_G2_D1,
    prime: int | None = None,
    method: str = "batched",
    row_kind: str = "all",
    start_row: int = 0,
    end_row: int | None = None,
    max_rows: int | None = None,
    row_order: str = "defect-balanced",
    row_random_seed: int = 0,
    start_column: int = 0,
    end_column: int | None = None,
    max_columns: int | None = None,
    column_order: str = "sequential",
    column_random_seed: int = 0,
    stop_rank: int | None = None,
    target_left_nullity: int | None = 1,
    max_semantic_keys: int | None = None,
    max_dependent_columns: int | None = None,
    beta_chunk_size: int = 2,
    max_chunk_terms: int = 200_000,
    store_selected_vectors: bool = True,
    store_nonzero_entries: bool = False,
    store_semantic_records: bool = False,
    store_left_nullspace: bool = False,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = 5,
    resume_from: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Stream high-f2 H62 columns and track rank growth.

    The default stop target is left-nullity one, i.e. rank ``row_count - 1``.
    This matches the c18-first goal of forcing a one-dimensional modular
    candidate relation while storing enough selected columns to recover it.
    """

    p = require_prime(config.primary_prime if prime is None else prime)
    normalized_method = _normalize_method(method)
    if checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be nonnegative")
    if max_dependent_columns is not None and max_dependent_columns < 0:
        raise ValueError("max_dependent_columns must be nonnegative")

    rows, source_indices = _select_rank_growth_rows(
        config,
        row_kind,
        start_row=start_row,
        end_row=end_row,
        max_rows=max_rows,
        order=row_order,
        random_seed=row_random_seed,
    )
    columns, column_indices = _select_f2_columns(
        config,
        start_column=start_column,
        end_column=end_column,
        max_columns=max_columns,
        order=column_order,
        random_seed=column_random_seed,
    )
    _validate_high_f2_prime(p, rows, columns)
    effective_stop_rank = _effective_stop_rank(
        len(rows),
        stop_rank=stop_rank,
        target_left_nullity=target_left_nullity,
    )

    tracker = ColumnRankTracker(row_count=len(rows), prime=p)
    selected_vectors: dict[int, Tuple[int, ...]] | None = (
        {} if store_selected_vectors else None
    )
    resume_state: dict[str, object] | None = None
    if resume_from is not None:
        resume_state = read_json_maybe_gzip(resume_from)
        _restore_from_checkpoint(
            resume_state,
            tracker=tracker,
            selected_vectors=selected_vectors,
            prime=p,
            method=method,
            row_kind=row_kind,
            source_indices=source_indices,
            column_indices=column_indices,
        )

    semantic_cache: dict[str, int] = {}
    semantic_records: dict[str, dict[str, object]] = {}
    cache_hits = (
        0 if resume_state is None else int(resume_state.get("semantic_cache_hits", 0))
    )
    cache_misses = (
        0 if resume_state is None else int(resume_state.get("semantic_cache_misses", 0))
    )
    attempted_entries = (
        0 if resume_state is None else int(resume_state.get("attempted_entries", 0))
    )
    nonzero_entries = (
        0 if resume_state is None else int(resume_state.get("nonzero_entries", 0))
    )
    nonzero_columns = (
        0 if resume_state is None else int(resume_state.get("nonzero_columns", 0))
    )
    unsupported_entries = (
        0 if resume_state is None else int(resume_state.get("unsupported_entries", 0))
    )
    column_records: list[dict[str, object]] = (
        [] if resume_state is None else list(resume_state.get("columns", []))
    )
    dependent_columns_since_rank_gain = (
        0
        if resume_state is None
        else int(
            resume_state.get(
                "dependent_columns_since_rank_gain",
                _trailing_dependent_columns(column_records),
            )
        )
    )
    nonzero_records: list[dict[str, object]] = (
        []
        if resume_state is None or not store_nonzero_entries
        else list(resume_state.get("nonzero_records", []))
    )
    exterior = ExteriorAlgebra(config)
    column_name_by_index = dict(zip(column_indices, (column.name for column in columns)))
    start = time.perf_counter()
    stop_reason = "exhausted_columns"
    next_column_pos = tracker.processed_columns
    last_checkpoint_processed = tracker.processed_columns

    for column_pos, (column_index, column) in enumerate(
        zip(column_indices, columns)
    ):
        if column_pos < next_column_pos:
            continue
        if effective_stop_rank is not None and tracker.rank >= effective_stop_rank:
            stop_reason = "stop_rank"
            break
        if max_semantic_keys is not None and cache_misses >= max_semantic_keys:
            stop_reason = "max_semantic_keys"
            break

        column_start = time.perf_counter()
        vector = []
        column_nonzero = 0
        for row_pos, row in enumerate(rows):
            try:
                key, metadata = _entry_key_and_metadata(config, exterior, row, column)
            except UnsupportedBlockEntry:
                unsupported_entries += 1
                raise

            if key in semantic_cache:
                cache_hits += 1
                value = semantic_cache[key]
            else:
                if max_semantic_keys is not None and cache_misses >= max_semantic_keys:
                    stop_reason = "max_semantic_keys"
                    break
                value = _evaluate_entry(
                    config,
                    metadata,
                    prime=p,
                    method=normalized_method,
                    beta_chunk_size=beta_chunk_size,
                    max_chunk_terms=max_chunk_terms,
                )
                semantic_cache[key] = int(value) % p
                if store_semantic_records:
                    semantic_records[key] = dict(metadata)
                    semantic_records[key]["key"] = key
                    semantic_records[key]["use_count"] = 0
                cache_misses += 1

            if store_semantic_records:
                semantic_records[key]["use_count"] = int(semantic_records[key]["use_count"]) + 1
            value %= p
            vector.append(value)
            attempted_entries += 1
            if value:
                column_nonzero += 1
                nonzero_entries += 1
                if store_nonzero_entries:
                    nonzero_records.append(
                        {
                            "row_position": row_pos,
                            "row_index": int(source_indices[row_pos]),
                            "row_name": row.name,
                            "row_kind": row.kind,
                            "column_position": column_pos,
                            "column_index": int(column_index),
                            "column_name": column.name,
                            "value": int(value),
                        }
                    )

        if len(vector) != len(rows):
            if stop_reason == "max_semantic_keys":
                break
            raise RuntimeError("incomplete column vector outside a known stop condition")

        vector_tuple = tuple(vector)
        independent = tracker.add_column(vector_tuple, index=column_index)
        if independent and selected_vectors is not None:
            selected_vectors[int(column_index)] = vector_tuple
        if independent:
            dependent_columns_since_rank_gain = 0
        else:
            dependent_columns_since_rank_gain += 1
        if column_nonzero:
            nonzero_columns += 1
        column_records.append(
            {
                "position": int(column_pos),
                "index": int(column_index),
                "name": column.name,
                "defect": column.defect,
                "f2_power": _f2_power(column),
                "nonzero_count": int(column_nonzero),
                "independent": bool(independent),
                "rank_after": int(tracker.rank),
                "left_nullity_after": int(tracker.nullity_left),
                "dependent_columns_since_rank_gain": int(
                    dependent_columns_since_rank_gain
                ),
                "elapsed_seconds": time.perf_counter() - column_start,
            }
        )

        if (
            max_dependent_columns is not None
            and dependent_columns_since_rank_gain >= max_dependent_columns
        ):
            stop_reason = "max_dependent_columns"
            break

        if (
            checkpoint_path is not None
            and checkpoint_interval > 0
            and tracker.processed_columns % checkpoint_interval == 0
        ):
            _write_checkpoint(
                checkpoint_path,
                payload=_payload(
                    kind="c18_f2_power_rank_growth_checkpoint",
                    schema_version=_SCHEMA_VERSION,
                    complete=False,
                    prime=p,
                    method=method,
                    normalized_method=normalized_method,
                    row_kind=row_kind,
                    row_order=row_order,
                    row_random_seed=row_random_seed,
                    column_order=column_order,
                    column_random_seed=column_random_seed,
                    start_row=start_row,
                    end_row=end_row,
                    max_rows=max_rows,
                    start_column=start_column,
                    end_column=end_column,
                    max_columns=max_columns,
                    row_count=len(rows),
                    source_indices=source_indices,
                    rows=rows,
                    available_column_count=len(h62_f2_power_test_columns(config)),
                    scheduled_column_count=len(columns),
                    column_indices=column_indices,
                    columns=columns,
                    tracker=tracker,
                    selected_vectors=selected_vectors,
                    column_records=column_records,
                    nonzero_records=nonzero_records,
                    attempted_entries=attempted_entries,
                    unsupported_entries=unsupported_entries,
                    cache_hits=cache_hits,
                    cache_misses=cache_misses,
                    semantic_records=semantic_records,
                    nonzero_entries=nonzero_entries,
                    nonzero_columns=nonzero_columns,
                    effective_stop_rank=effective_stop_rank,
                    target_left_nullity=target_left_nullity,
                    stop_reason="checkpoint",
                    elapsed_seconds=time.perf_counter() - start,
                    max_semantic_keys=max_semantic_keys,
                    max_dependent_columns=max_dependent_columns,
                    dependent_columns_since_rank_gain=dependent_columns_since_rank_gain,
                    beta_chunk_size=beta_chunk_size,
                    max_chunk_terms=max_chunk_terms,
                    store_semantic_records=store_semantic_records,
                    store_left_nullspace=store_left_nullspace,
                    resume_from=resume_from,
                    resume_state=resume_state,
                ),
            )
            last_checkpoint_processed = tracker.processed_columns
    else:
        stop_reason = "exhausted_columns"

    elapsed = time.perf_counter() - start
    payload = _payload(
        kind="c18_f2_power_rank_growth",
        schema_version=_SCHEMA_VERSION,
        complete=True,
        prime=p,
        method=method,
        normalized_method=normalized_method,
        row_kind=row_kind,
        row_order=row_order,
        row_random_seed=row_random_seed,
        column_order=column_order,
        column_random_seed=column_random_seed,
        start_row=start_row,
        end_row=end_row,
        max_rows=max_rows,
        start_column=start_column,
        end_column=end_column,
        max_columns=max_columns,
        row_count=len(rows),
        source_indices=source_indices,
        rows=rows,
        available_column_count=len(h62_f2_power_test_columns(config)),
        scheduled_column_count=len(columns),
        column_indices=column_indices,
        columns=columns,
        tracker=tracker,
        selected_vectors=selected_vectors,
        column_records=column_records,
        nonzero_records=nonzero_records,
        attempted_entries=attempted_entries,
        unsupported_entries=unsupported_entries,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        semantic_records=semantic_records,
        nonzero_entries=nonzero_entries,
        nonzero_columns=nonzero_columns,
        effective_stop_rank=effective_stop_rank,
        target_left_nullity=target_left_nullity,
        stop_reason=stop_reason,
        elapsed_seconds=elapsed,
        max_semantic_keys=max_semantic_keys,
        max_dependent_columns=max_dependent_columns,
        dependent_columns_since_rank_gain=dependent_columns_since_rank_gain,
        beta_chunk_size=beta_chunk_size,
        max_chunk_terms=max_chunk_terms,
        store_semantic_records=store_semantic_records,
        store_left_nullspace=store_left_nullspace,
        resume_from=resume_from,
        resume_state=resume_state,
    )
    if output_path is not None:
        _write_json_maybe_gzip_atomic(output_path, payload)
    if (
        checkpoint_path is not None
        and tracker.processed_columns != last_checkpoint_processed
    ):
        _write_checkpoint(checkpoint_path, payload=payload | {"complete": False})
    return payload


def _select_rank_growth_rows(
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


def _select_f2_columns(
    config: FormulaConfig,
    *,
    start_column: int,
    end_column: int | None,
    max_columns: int | None,
    order: str,
    random_seed: int,
) -> Tuple[Tuple[H62TestColumn, ...], Tuple[int, ...]]:
    if start_column < 0:
        raise ValueError("start_column must be nonnegative")
    if max_columns is not None and max_columns < 0:
        raise ValueError("max_columns must be nonnegative")
    all_columns = h62_f2_power_test_columns(config)
    stop = len(all_columns) if end_column is None else int(end_column)
    if stop < start_column:
        raise ValueError("end_column must be greater than or equal to start_column")
    selected = all_columns[start_column:stop]
    indices = tuple(range(start_column, start_column + len(selected)))
    pairs = _ordered_f2_columns(
        config,
        selected,
        indices,
        order=order,
        random_seed=random_seed,
    )
    if max_columns is not None:
        pairs = pairs[:max_columns]
    return tuple(column for _idx, column in pairs), tuple(int(idx) for idx, _column in pairs)


def _ordered_f2_columns(
    config: FormulaConfig,
    columns: Sequence[H62TestColumn],
    indices: Sequence[int],
    *,
    order: str,
    random_seed: int,
) -> list[tuple[int, H62TestColumn]]:
    if order in {"f2-power-balanced", "f2-balanced"}:
        return _f2_power_round_robin_order(indices, columns, reverse=False)
    if order in {"f2-power-desc-balanced", "f2-desc-balanced"}:
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


def _middle_out(items: Sequence[tuple[int, H62TestColumn]]) -> list[tuple[int, H62TestColumn]]:
    if not items:
        return []
    center = (len(items) - 1) // 2
    out = [items[center]]
    radius = 1
    while len(out) < len(items):
        right = center + radius
        left = center - radius
        if right < len(items):
            out.append(items[right])
        if left >= 0:
            out.append(items[left])
        radius += 1
    return out


def _f2_power(column: H62TestColumn) -> int:
    return int(column.monomial.f_exp[0])


def _validate_high_f2_prime(
    prime: int,
    rows: Sequence[C18SourceRow],
    columns: Sequence[H62TestColumn],
) -> None:
    if not rows or not columns:
        return
    max_row_f2 = max(int(row.monomial.f_exp[0]) for row in rows)
    max_column_f2 = max(int(column.monomial.f_exp[0]) for column in columns)
    max_total_f2 = max_row_f2 + max_column_f2
    if prime <= max_total_f2:
        raise ValueError(
            "high-f2 rank growth requires prime greater than the maximum total "
            f"f2 exponent {max_total_f2}; got p={prime}"
        )


def _effective_stop_rank(
    row_count: int,
    *,
    stop_rank: int | None,
    target_left_nullity: int | None,
) -> int | None:
    if stop_rank is not None:
        if stop_rank < 0:
            raise ValueError("stop_rank must be nonnegative")
        return min(int(stop_rank), int(row_count))
    if target_left_nullity is None:
        return None
    if target_left_nullity < 0:
        raise ValueError("target_left_nullity must be nonnegative")
    return max(0, int(row_count) - int(target_left_nullity))


def _payload(
    *,
    kind: str,
    schema_version: int,
    complete: bool,
    prime: int,
    method: str,
    normalized_method: str,
    row_kind: str,
    row_order: str,
    row_random_seed: int,
    column_order: str,
    column_random_seed: int,
    start_row: int,
    end_row: int | None,
    max_rows: int | None,
    start_column: int,
    end_column: int | None,
    max_columns: int | None,
    row_count: int,
    source_indices: Tuple[int, ...],
    rows: Tuple[C18SourceRow, ...],
    available_column_count: int,
    scheduled_column_count: int,
    column_indices: Tuple[int, ...],
    columns: Tuple[H62TestColumn, ...],
    tracker: ColumnRankTracker,
    selected_vectors: dict[int, Tuple[int, ...]] | None,
    column_records: Sequence[dict[str, object]],
    nonzero_records: Sequence[dict[str, object]],
    attempted_entries: int,
    unsupported_entries: int,
    cache_hits: int,
    cache_misses: int,
    semantic_records: dict[str, dict[str, object]],
    nonzero_entries: int,
    nonzero_columns: int,
    effective_stop_rank: int | None,
    target_left_nullity: int | None,
    stop_reason: str,
    elapsed_seconds: float,
    max_semantic_keys: int | None,
    max_dependent_columns: int | None,
    dependent_columns_since_rank_gain: int,
    beta_chunk_size: int,
    max_chunk_terms: int,
    store_semantic_records: bool,
    store_left_nullspace: bool,
    resume_from: Path | None,
    resume_state: dict[str, object] | None,
) -> dict[str, object]:
    selected_name_by_index = dict(zip(column_indices, (column.name for column in columns)))
    selected_indices = list(tracker.selected_indices)
    left_nullspace = _left_nullspace_payload(
        tracker,
        selected_vectors=selected_vectors,
        source_indices=source_indices,
        rows=rows,
        prime=prime,
        force=store_left_nullspace,
    )
    candidate = _candidate_left_null_vector(left_nullspace)
    return {
        "kind": kind,
        "schema_version": int(schema_version),
        "complete": bool(complete),
        "prime": int(prime),
        "method": method,
        "normalized_method": normalized_method,
        "column_kind": "f2-power",
        "row_kind": row_kind,
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
        "row_count": int(row_count),
        "available_column_count": int(available_column_count),
        "scheduled_column_count": int(scheduled_column_count),
        "processed_columns": int(tracker.processed_columns),
        "attempted_entries": int(attempted_entries),
        "unsupported_entries": int(unsupported_entries),
        "semantic_cache_hits": int(cache_hits),
        "semantic_cache_misses": int(cache_misses),
        "semantic_key_count": (
            len(semantic_records) if store_semantic_records else int(cache_misses)
        ),
        "nonzero_entries": int(nonzero_entries),
        "nonzero_columns": int(nonzero_columns),
        "rank": int(tracker.rank),
        "left_nullity": int(tracker.nullity_left),
        "target_left_nullity": (
            None if target_left_nullity is None else int(target_left_nullity)
        ),
        "stop_rank": None if effective_stop_rank is None else int(effective_stop_rank),
        "stop_reason": stop_reason,
        "source_row_indices": list(source_indices),
        "source_row_names": [row.name for row in rows],
        "source_row_kinds": [row.kind for row in rows],
        "test_column_indices": list(column_indices),
        "test_column_names": [column.name for column in columns],
        "selected_column_indices": selected_indices,
        "selected_column_names": [
            selected_name_by_index[index] for index in selected_indices
        ],
        "tracker_state": _tracker_state(tracker),
        "left_nullspace": left_nullspace,
        "candidate_left_null_vector": candidate,
        "selected_column_vectors": (
            None
            if selected_vectors is None
            else {
                str(index): list(vector)
                for index, vector in sorted(selected_vectors.items())
            }
        ),
        "columns": list(column_records),
        "nonzero_records": list(nonzero_records),
        "semantic_keys": list(semantic_records.values()) if store_semantic_records else [],
        "semantic_records_stored": bool(store_semantic_records),
        "elapsed_seconds": float(elapsed_seconds),
        "max_semantic_keys": (
            None if max_semantic_keys is None else int(max_semantic_keys)
        ),
        "max_dependent_columns": (
            None if max_dependent_columns is None else int(max_dependent_columns)
        ),
        "dependent_columns_since_rank_gain": int(dependent_columns_since_rank_gain),
        "beta_chunk_size": int(beta_chunk_size),
        "max_chunk_terms": int(max_chunk_terms),
        "resume_from": None if resume_from is None else str(resume_from),
        "resumed_processed_columns": (
            0
            if resume_state is None
            else int(resume_state.get("processed_columns", 0))
        ),
        "cache_info": all_a_cache_info(),
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
    }


def _tracker_state(tracker: ColumnRankTracker) -> dict[str, object]:
    return {
        "row_count": int(tracker.row_count),
        "prime": int(tracker.prime),
        "pivot_rows": [int(item) for item in tracker.pivot_rows],
        "basis_columns": [
            [int(value) for value in column] for column in tracker.basis_columns
        ],
        "selected_indices": [int(item) for item in tracker.selected_indices],
        "processed_columns": int(tracker.processed_columns),
    }


def _trailing_dependent_columns(column_records: Sequence[dict[str, object]]) -> int:
    total = 0
    for record in reversed(column_records):
        if record.get("independent"):
            break
        total += 1
    return total


def _left_nullspace_payload(
    tracker: ColumnRankTracker,
    *,
    selected_vectors: dict[int, Tuple[int, ...]] | None,
    source_indices: Tuple[int, ...],
    rows: Tuple[C18SourceRow, ...],
    prime: int,
    force: bool,
) -> dict[str, object]:
    if tracker.nullity_left > 1 and not force:
        return {
            "computed": False,
            "computed_from": None,
            "dimension": int(tracker.nullity_left),
            "vectors": [],
            "basis_column_dot_products": None,
            "basis_column_dot_products_zero": None,
            "selected_column_dot_products": None,
            "selected_column_dot_products_zero": None,
            "selected_vectors_complete": False,
            "reason": "left_nullity_exceeds_1",
        }

    matrix_rows = _matrix_rows_from_columns(tracker.basis_columns, tracker.row_count)
    if tracker.basis_columns:
        raw_vectors = left_nullspace_mod(matrix_rows, prime)
    else:
        raw_vectors = tuple(
            tuple(1 if idx == row_idx else 0 for idx in range(tracker.row_count))
            for row_idx in range(tracker.row_count)
        )
    vectors = tuple(_normalize_null_vector(vector, prime) for vector in raw_vectors)
    basis_dot_products = [
        list(vec_mat_mul_mod(vector, matrix_rows, prime)) for vector in vectors
    ]

    selected_dot_products = None
    selected_vectors_complete = False
    if selected_vectors is not None and all(
        int(index) in selected_vectors for index in tracker.selected_indices
    ):
        selected_columns = [selected_vectors[int(index)] for index in tracker.selected_indices]
        selected_matrix_rows = _matrix_rows_from_columns(
            selected_columns,
            tracker.row_count,
        )
        selected_dot_products = [
            list(vec_mat_mul_mod(vector, selected_matrix_rows, prime))
            for vector in vectors
        ]
        selected_vectors_complete = True

    return {
        "computed": True,
        "computed_from": "tracker_basis_columns",
        "dimension": len(vectors),
        "vectors": [
            _null_vector_record(vector, source_indices=source_indices, rows=rows)
            for vector in vectors
        ],
        "basis_column_dot_products": basis_dot_products,
        "basis_column_dot_products_zero": all(
            all(value % prime == 0 for value in dot_products)
            for dot_products in basis_dot_products
        ),
        "selected_column_dot_products": selected_dot_products,
        "selected_column_dot_products_zero": (
            None
            if selected_dot_products is None
            else all(
                all(value % prime == 0 for value in dot_products)
                for dot_products in selected_dot_products
            )
        ),
        "selected_vectors_complete": selected_vectors_complete,
    }


def _candidate_left_null_vector(
    left_nullspace: dict[str, object],
) -> dict[str, object] | None:
    if not left_nullspace.get("computed") or int(left_nullspace["dimension"]) != 1:
        return None
    vectors = left_nullspace.get("vectors")
    if not isinstance(vectors, list) or not vectors:
        return None
    candidate = dict(vectors[0])
    basis_verified = bool(left_nullspace.get("basis_column_dot_products_zero"))
    selected_verified = left_nullspace.get("selected_column_dot_products_zero")
    candidate["verified"] = basis_verified and (
        selected_verified is None or bool(selected_verified)
    )
    candidate["verification"] = {
        "basis_column_dot_products_zero": basis_verified,
        "selected_column_dot_products_zero": selected_verified,
        "selected_vectors_complete": bool(
            left_nullspace.get("selected_vectors_complete")
        ),
    }
    return candidate


def _matrix_rows_from_columns(
    columns: Sequence[Sequence[int]],
    row_count: int,
) -> Tuple[Tuple[int, ...], ...]:
    return tuple(
        tuple(int(column[row_idx]) for column in columns)
        for row_idx in range(row_count)
    )


def _normalize_null_vector(vector: Sequence[int], prime: int) -> Tuple[int, ...]:
    normalized = tuple(int(value) % prime for value in vector)
    pivot = next((value for value in normalized if value % prime), None)
    if pivot is None:
        return normalized
    scale = mod_inv(pivot, prime)
    return tuple(value * scale % prime for value in normalized)


def _null_vector_record(
    vector: Tuple[int, ...],
    *,
    source_indices: Tuple[int, ...],
    rows: Tuple[C18SourceRow, ...],
) -> dict[str, object]:
    entries = []
    for row_position, value in enumerate(vector):
        if value:
            row = rows[row_position]
            entries.append(
                {
                    "row_position": int(row_position),
                    "row_index": int(source_indices[row_position]),
                    "row_name": row.name,
                    "row_kind": row.kind,
                    "value": int(value),
                }
            )
    return {
        "values": [int(value) for value in vector],
        "support_size": len(entries),
        "entries": entries,
    }


def _restore_from_checkpoint(
    checkpoint: dict[str, object],
    *,
    tracker: ColumnRankTracker,
    selected_vectors: dict[int, Tuple[int, ...]] | None,
    prime: int,
    method: str,
    row_kind: str,
    source_indices: Tuple[int, ...],
    column_indices: Tuple[int, ...],
) -> None:
    if checkpoint.get("kind") not in {
        "c18_f2_power_rank_growth",
        "c18_f2_power_rank_growth_checkpoint",
    }:
        raise ValueError("resume checkpoint has the wrong kind")
    if int(checkpoint["prime"]) != int(prime):
        raise ValueError("resume checkpoint prime does not match")
    if str(checkpoint["method"]) != str(method):
        raise ValueError("resume checkpoint method does not match")
    if str(checkpoint["row_kind"]) != str(row_kind):
        raise ValueError("resume checkpoint row kind does not match")
    if tuple(int(item) for item in checkpoint["source_row_indices"]) != tuple(source_indices):
        raise ValueError("resume checkpoint row selection does not match")
    old_column_indices = tuple(int(item) for item in checkpoint["test_column_indices"])
    if tuple(column_indices[: len(old_column_indices)]) != old_column_indices:
        raise ValueError("resume checkpoint column selection is not a prefix")

    state = checkpoint.get("tracker_state")
    if not isinstance(state, dict):
        raise ValueError("resume checkpoint is missing tracker_state")
    tracker.pivot_rows = [int(item) for item in state["pivot_rows"]]
    tracker.basis_columns = [
        [int(value) % prime for value in column]
        for column in state["basis_columns"]
    ]
    tracker.selected_indices = [int(item) for item in state["selected_indices"]]
    tracker.processed_columns = int(state["processed_columns"])
    if tracker.row_count != int(state["row_count"]):
        raise ValueError("resume checkpoint tracker row count does not match")
    if tracker.prime != int(state["prime"]):
        raise ValueError("resume checkpoint tracker prime does not match")

    if selected_vectors is not None:
        selected_vectors.clear()
        raw_vectors = checkpoint.get("selected_column_vectors") or {}
        if not isinstance(raw_vectors, dict):
            raise ValueError("resume checkpoint selected_column_vectors must be an object")
        for index, vector in raw_vectors.items():
            selected_vectors[int(index)] = tuple(int(value) % prime for value in vector)


def _write_checkpoint(path: Path, *, payload: dict[str, object]) -> None:
    _write_json_maybe_gzip_atomic(path, _lean_checkpoint_payload(payload))


def _lean_checkpoint_payload(payload: dict[str, object]) -> dict[str, object]:
    lean = dict(payload)
    lean["semantic_keys"] = []
    lean["semantic_records_stored"] = False
    lean["semantic_records_omitted_from_checkpoint"] = True
    if lean.get("nonzero_records"):
        lean["nonzero_records"] = []
        lean["nonzero_records_omitted_from_checkpoint"] = True
    return lean


def _write_json_maybe_gzip_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if str(path).endswith(".gz"):
        data = gzip.compress(data)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp_path.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _summary(payload: dict[str, object]) -> dict[str, object]:
    keys = (
        "kind",
        "complete",
        "prime",
        "method",
        "normalized_method",
        "row_kind",
        "row_order",
        "column_kind",
        "column_order",
        "row_count",
        "available_column_count",
        "scheduled_column_count",
        "processed_columns",
        "attempted_entries",
        "nonzero_entries",
        "nonzero_columns",
        "rank",
        "left_nullity",
        "target_left_nullity",
        "stop_rank",
        "max_dependent_columns",
        "dependent_columns_since_rank_gain",
        "stop_reason",
        "elapsed_seconds",
        "git_head",
        "git_dirty",
    )
    return {key: payload[key] for key in keys if key in payload}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prime", type=int, default=RANK7_G2_D1.primary_prime)
    parser.add_argument(
        "--method",
        choices=("synthetic", "moment", "batched", "semantic-batched"),
        default="batched",
    )
    parser.add_argument("--row-kind", choices=("all", "even", "gamma"), default="all")
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
        default="sequential",
    )
    parser.add_argument("--column-random-seed", type=int, default=0)
    parser.add_argument("--stop-rank", type=int, default=None)
    parser.add_argument("--target-left-nullity", type=int, default=1)
    parser.add_argument("--no-target-left-nullity", action="store_true")
    parser.add_argument("--max-semantic-keys", type=int, default=None)
    parser.add_argument("--max-dependent-columns", type=int, default=None)
    parser.add_argument("--beta-chunk-size", type=int, default=2)
    parser.add_argument("--max-chunk-terms", type=int, default=200_000)
    parser.add_argument("--no-store-selected-vectors", action="store_true")
    parser.add_argument("--store-nonzero-entries", action="store_true")
    parser.add_argument("--store-semantic-records", action="store_true")
    parser.add_argument("--store-left-nullspace", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = run_c18_f2_rank_growth(
        prime=args.prime,
        method=args.method,
        row_kind=args.row_kind,
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
        stop_rank=args.stop_rank,
        target_left_nullity=(
            None if args.no_target_left_nullity else args.target_left_nullity
        ),
        max_semantic_keys=args.max_semantic_keys,
        max_dependent_columns=args.max_dependent_columns,
        beta_chunk_size=args.beta_chunk_size,
        max_chunk_terms=args.max_chunk_terms,
        store_selected_vectors=not args.no_store_selected_vectors,
        store_nonzero_entries=args.store_nonzero_entries,
        store_semantic_records=args.store_semantic_records,
        store_left_nullspace=args.store_left_nullspace,
        checkpoint_path=args.checkpoint,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=args.resume_from,
        output_path=args.output,
    )
    print(json.dumps(_summary(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
