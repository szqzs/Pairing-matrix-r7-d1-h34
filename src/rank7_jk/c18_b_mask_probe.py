"""Adaptive rank/diagnostic probe for direct b-mask c18 tests."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Sequence, Tuple

from .c18_all_a_probe import _git_dirty, _git_head, _select_rows
from .c18_b_mask_table import b_mask_key_id, evaluate_b_mask_key
from .c18_basis import H62TestColumn, h62_one_b_pair_test_columns
from .c18_even_worker import write_json_maybe_gzip
from .config import FormulaConfig, RANK7_G2_D1
from .exterior import ExteriorAlgebra
from .mod_arith import require_prime
from .rank_stream import ColumnRankTracker


def run_b_mask_adaptive_probe(
    *,
    config: FormulaConfig = RANK7_G2_D1,
    prime: int | None = None,
    method: str = "batched",
    start_row: int = 0,
    end_row: int | None = None,
    max_rows: int | None = None,
    row_order: str = "defect-balanced",
    row_random_seed: int = 0,
    start_column: int = 0,
    end_column: int | None = None,
    max_columns: int | None = 20,
    column_order: str = "mask-balanced",
    random_seed: int = 0,
    stop_rank: int | None = None,
    stop_on_nonzero: bool = False,
    max_semantic_keys: int | None = None,
    beta_chunk_size: int = 2,
    max_chunk_terms: int = 200_000,
    store_nonzero_entries: bool = True,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Stream selected direct b-mask columns against c18 even rows.

    The probe is intentionally diagnostic-first: it spreads columns across
    exterior masks, caches semantic values, and records whether any entries are
    nonzero before we commit to a large table build.
    """

    p = require_prime(config.primary_prime if prime is None else prime)
    normalized_method = _normalize_method(method)
    rows, source_row_indices = _select_rows(
        config,
        "even",
        None,
        start_row=start_row,
        end_row=end_row,
        max_rows=None,
    )
    ordered_rows = _ordered_rows(
        rows,
        source_row_indices,
        order=row_order,
        random_seed=row_random_seed,
    )
    if max_rows is not None:
        if max_rows < 0:
            raise ValueError("max_rows must be nonnegative")
        ordered_rows = ordered_rows[:max_rows]
    source_row_indices = tuple(index for index, _row in ordered_rows)
    rows = tuple(row for _index, row in ordered_rows)
    columns, test_column_indices = _select_b_mask_columns(
        config,
        start_column=start_column,
        end_column=end_column,
        max_columns=None,
    )
    ordered = _ordered_columns(
        config,
        columns,
        test_column_indices,
        order=column_order,
        random_seed=random_seed,
    )
    if max_columns is not None:
        if max_columns < 0:
            raise ValueError("max_columns must be nonnegative")
        ordered = ordered[:max_columns]

    tracker = ColumnRankTracker(row_count=len(rows), prime=p)
    semantic_cache: dict[str, int] = {}
    semantic_key_records: dict[str, dict[str, object]] = {}
    cache_hits = 0
    cache_misses = 0
    nonzero_entries = 0
    nonzero_columns = 0
    attempted_entries = 0
    column_records = []
    nonzero_records = []
    exterior = ExteriorAlgebra(config)
    start = time.perf_counter()
    stop_reason = "exhausted_columns"

    for column_index, column in ordered:
        if stop_rank is not None and tracker.rank >= stop_rank:
            stop_reason = "stop_rank"
            break
        if max_semantic_keys is not None and cache_misses >= max_semantic_keys:
            stop_reason = "max_semantic_keys"
            break

        column_start = time.perf_counter()
        b_mask = _column_b_mask(exterior, column)
        vector = []
        column_nonzero = 0
        for row_pos, row in enumerate(rows):
            total_a_exp = tuple(
                int(left) + int(right)
                for left, right in zip(row.monomial.a_exp, column.monomial.a_exp)
            )
            total_f_exp = tuple(int(item) for item in row.monomial.f_exp)
            key = b_mask_key_id(total_f_exp, b_mask, total_a_exp)
            if key in semantic_cache:
                cache_hits += 1
                value = semantic_cache[key]
            else:
                if (
                    max_semantic_keys is not None
                    and cache_misses >= max_semantic_keys
                ):
                    stop_reason = "max_semantic_keys"
                    break
                value = _evaluate_probe_key(
                    total_f_exp,
                    b_mask,
                    total_a_exp,
                    prime=p,
                    method=normalized_method,
                    config=config,
                    beta_chunk_size=beta_chunk_size,
                    max_chunk_terms=max_chunk_terms,
                )
                semantic_cache[key] = value
                semantic_key_records[key] = {
                    "key": key,
                    "total_f_exp": list(total_f_exp),
                    "b_mask": int(b_mask),
                    "b_labels": [list(label) for label in column.b_labels],
                    "total_a_exp": list(total_a_exp),
                    "use_count": 0,
                }
                cache_misses += 1
            semantic_key_records[key]["use_count"] = (
                int(semantic_key_records[key]["use_count"]) + 1
            )
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
                            "row_index": int(source_row_indices[row_pos]),
                            "row_name": row.name,
                            "column_index": int(column_index),
                            "column_name": column.name,
                            "key": key,
                            "value": int(value),
                        }
                    )
        if len(vector) != len(rows):
            break

        independent = tracker.add_column(vector, index=column_index)
        if column_nonzero:
            nonzero_columns += 1
        column_records.append(
            {
                "index": int(column_index),
                "name": column.name,
                "b_mask": int(b_mask),
                "b_labels": [list(label) for label in column.b_labels],
                "nonzero_count": int(column_nonzero),
                "independent": bool(independent),
                "elapsed_seconds": time.perf_counter() - column_start,
            }
        )
        if stop_on_nonzero and column_nonzero:
            stop_reason = "nonzero_entry"
            break
    else:
        stop_reason = "exhausted_columns"

    column_name_by_index = {
        int(record["index"]): str(record["name"]) for record in column_records
    }
    payload = {
        "kind": "c18_b_mask_adaptive_probe",
        "prime": p,
        "method": method,
        "normalized_method": normalized_method,
        "row_kind": "even",
        "row_count": len(rows),
        "available_column_count": len(columns),
        "scheduled_column_count": len(ordered),
        "processed_columns": tracker.processed_columns,
        "attempted_entries": attempted_entries,
        "semantic_cache_hits": cache_hits,
        "semantic_cache_misses": cache_misses,
        "semantic_key_count": len(semantic_cache),
        "nonzero_entries": nonzero_entries,
        "nonzero_columns": nonzero_columns,
        "rank": tracker.rank,
        "left_nullity": tracker.nullity_left,
        "selected_column_indices": list(tracker.selected_indices),
        "selected_column_names": [
            column_name_by_index[index] for index in tracker.selected_indices
        ],
        "source_row_indices": list(source_row_indices),
        "source_row_names": [row.name for row in rows],
        "column_order": column_order,
        "random_seed": int(random_seed),
        "row_order": row_order,
        "row_random_seed": int(row_random_seed),
        "stop_rank": None if stop_rank is None else int(stop_rank),
        "stop_on_nonzero": bool(stop_on_nonzero),
        "max_semantic_keys": (
            None if max_semantic_keys is None else int(max_semantic_keys)
        ),
        "stop_reason": stop_reason,
        "elapsed_seconds": time.perf_counter() - start,
        "columns": column_records,
        "nonzero_records": nonzero_records,
        "semantic_keys": list(semantic_key_records.values()),
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
    }
    if output_path is not None:
        write_json_maybe_gzip(output_path, payload)
    return payload


def _evaluate_probe_key(
    total_f_exp: Sequence[int],
    b_mask: int,
    total_a_exp: Sequence[int],
    *,
    prime: int,
    method: str,
    config: FormulaConfig,
    beta_chunk_size: int,
    max_chunk_terms: int,
) -> int:
    if method == "synthetic":
        return _synthetic_b_mask_value(total_f_exp, b_mask, total_a_exp, prime)
    return evaluate_b_mask_key(
        total_f_exp,
        b_mask,
        total_a_exp,
        config=config,
        prime=prime,
        method=method,
        beta_chunk_size=beta_chunk_size,
        max_chunk_terms=max_chunk_terms,
    )


def _synthetic_b_mask_value(
    total_f_exp: Sequence[int],
    b_mask: int,
    total_a_exp: Sequence[int],
    prime: int,
) -> int:
    value = int(b_mask).bit_count() * 19 + int(b_mask) % 97
    for idx, exp in enumerate(total_f_exp):
        value += (idx + 5) * int(exp)
    for idx, exp in enumerate(total_a_exp):
        value += (idx + 11) * (int(exp) + 1) * (int(exp) + 2)
    return value % prime


def _select_b_mask_columns(
    config: FormulaConfig,
    *,
    start_column: int,
    end_column: int | None,
    max_columns: int | None,
) -> Tuple[Tuple[H62TestColumn, ...], Tuple[int, ...]]:
    if start_column < 0:
        raise ValueError("start_column must be nonnegative")
    if max_columns is not None and max_columns < 0:
        raise ValueError("max_columns must be nonnegative")
    all_columns = h62_one_b_pair_test_columns(config)
    stop = len(all_columns) if end_column is None else int(end_column)
    if max_columns is not None:
        stop = min(stop, start_column + max_columns)
    if stop < start_column:
        raise ValueError("end_column must be greater than or equal to start_column")
    selected = all_columns[start_column:stop]
    return selected, tuple(range(start_column, start_column + len(selected)))


def _ordered_rows(
    rows,
    indices: Sequence[int],
    *,
    order: str,
    random_seed: int,
):
    pairs = list(zip((int(index) for index in indices), rows))
    if order == "sequential":
        return pairs
    if order == "random":
        rng = random.Random(int(random_seed))
        rng.shuffle(pairs)
        return pairs
    if order == "defect-balanced":
        groups = {}
        for index, row in pairs:
            groups.setdefault(row.defect, []).append((index, row))
        for defect, group in list(groups.items()):
            groups[defect] = _middle_out(group)
        ordered = []
        defects = sorted(groups, key=_defect_sort_key)
        depth = 0
        while True:
            added = False
            for defect in defects:
                group = groups[defect]
                if depth < len(group):
                    ordered.append(group[depth])
                    added = True
            if not added:
                break
            depth += 1
        return ordered
    raise ValueError("row_order must be sequential, random, or defect-balanced")


def _defect_sort_key(defect: str) -> tuple[int, int | str]:
    if defect.startswith("f"):
        try:
            return (0, int(defect[1:]))
        except ValueError:
            pass
    return (1, defect)


def _ordered_columns(
    config: FormulaConfig,
    columns: Sequence[H62TestColumn],
    indices: Sequence[int],
    *,
    order: str,
    random_seed: int,
) -> list[tuple[int, H62TestColumn]]:
    pairs = list(zip((int(index) for index in indices), columns))
    if order == "sequential":
        return pairs
    if order == "random":
        rng = random.Random(int(random_seed))
        rng.shuffle(pairs)
        return pairs
    if order == "mask-balanced":
        return _mask_balanced_order(config, pairs)
    raise ValueError("column_order must be sequential, random, or mask-balanced")


def _mask_balanced_order(
    config: FormulaConfig,
    pairs: Sequence[tuple[int, H62TestColumn]],
) -> list[tuple[int, H62TestColumn]]:
    exterior = ExteriorAlgebra(config)
    groups: dict[int, list[tuple[int, H62TestColumn]]] = {}
    for index, column in pairs:
        groups.setdefault(_column_b_mask(exterior, column), []).append((index, column))
    for b_mask, group in list(groups.items()):
        groups[b_mask] = _middle_out(group)

    ordered = []
    masks = sorted(groups)
    depth = 0
    while True:
        added = False
        for b_mask in masks:
            group = groups[b_mask]
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


def _column_b_mask(exterior: ExteriorAlgebra, column: H62TestColumn) -> int:
    if len(column.b_labels) != 2:
        raise ValueError("direct b-mask columns must have exactly two b labels")
    target = exterior.b_product_to_mask(column.b_labels)
    if target is None:
        raise ValueError("b labels wedge to zero")
    sign, b_mask = target
    if sign != 1:
        raise ValueError("b-pair columns must be stored in exterior order")
    return int(b_mask)


def _normalize_method(method: str) -> str:
    normalized = method.lower()
    if normalized == "semantic-batched":
        return "batched"
    if normalized in {"synthetic", "moment", "batched"}:
        return normalized
    raise ValueError("method must be synthetic, moment, batched, or semantic-batched")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prime", type=int, default=RANK7_G2_D1.primary_prime)
    parser.add_argument(
        "--method",
        choices=("synthetic", "moment", "batched", "semantic-batched"),
        default="batched",
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
    parser.add_argument("--max-columns", type=int, default=20)
    parser.add_argument(
        "--column-order",
        choices=("sequential", "random", "mask-balanced"),
        default="mask-balanced",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--stop-rank", type=int, default=None)
    parser.add_argument("--stop-on-nonzero", action="store_true")
    parser.add_argument("--max-semantic-keys", type=int, default=None)
    parser.add_argument("--beta-chunk-size", type=int, default=2)
    parser.add_argument("--max-chunk-terms", type=int, default=200_000)
    parser.add_argument("--no-store-nonzero-entries", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = run_b_mask_adaptive_probe(
        prime=args.prime,
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
        random_seed=args.random_seed,
        stop_rank=args.stop_rank,
        stop_on_nonzero=args.stop_on_nonzero,
        max_semantic_keys=args.max_semantic_keys,
        beta_chunk_size=args.beta_chunk_size,
        max_chunk_terms=args.max_chunk_terms,
        store_nonzero_entries=not args.no_store_nonzero_entries,
        output_path=args.output,
    )
    print(json.dumps(_summary(payload), indent=2, sort_keys=True))
    return 0


def _summary(payload: dict[str, object]) -> dict[str, object]:
    keys = (
        "kind",
        "prime",
        "method",
        "normalized_method",
        "row_count",
        "row_order",
        "available_column_count",
        "scheduled_column_count",
        "processed_columns",
        "attempted_entries",
        "semantic_cache_misses",
        "nonzero_entries",
        "nonzero_columns",
        "rank",
        "left_nullity",
        "stop_reason",
        "elapsed_seconds",
    )
    return {key: payload[key] for key in keys if key in payload}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
