"""Uniform block-rank diagnostics for c18 source/test slices."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Sequence, Tuple

from .all_a_pairing import (
    all_a_pairing_total_batched_mod,
    all_a_pairing_total_moment_mod,
    b_mask_pairing_total_batched_mod,
    b_mask_pairing_total_moment_mod,
    f2_power_pairing_total_batched_mod,
    f2_power_pairing_total_moment_mod,
    f_gamma_pairing_total_batched_mod,
    f_gamma_pairing_total_moment_mod,
    f_only_pairing_total_batched_mod,
    f_only_pairing_total_moment_mod,
)
from .c18_all_a_probe import _git_dirty, _git_head, _select_rows
from .c18_basis import (
    C18SourceRow,
    H62TestColumn,
    h62_all_a_test_columns,
    h62_f2_power_test_columns,
    h62_one_b_pair_test_columns,
    h62_one_f_test_columns,
    h62_one_gamma_test_columns,
)
from .c18_even_worker import write_json_maybe_gzip
from .config import FormulaConfig, RANK7_G2_D1
from .exterior import ExteriorAlgebra
from .invariants import InvariantMonomial
from .mod_arith import require_prime
from .rank_stream import ColumnRankTracker


def run_c18_block_probe(
    *,
    config: FormulaConfig = RANK7_G2_D1,
    prime: int | None = None,
    method: str = "batched",
    row_kind: str = "all",
    column_kind: str = "all-a",
    start_row: int = 0,
    end_row: int | None = None,
    max_rows: int | None = None,
    row_order: str = "defect-balanced",
    row_random_seed: int = 0,
    start_column: int = 0,
    end_column: int | None = None,
    max_columns: int | None = 20,
    column_order: str = "balanced",
    column_random_seed: int = 0,
    unsupported: str = "error",
    stop_rank: int | None = None,
    stop_on_nonzero: bool = False,
    max_semantic_keys: int | None = None,
    beta_chunk_size: int = 2,
    max_chunk_terms: int = 200_000,
    store_nonzero_entries: bool = True,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Stream a selected c18 source/test block into a rank tracker."""

    p = require_prime(config.primary_prime if prime is None else prime)
    normalized_method = _normalize_method(method)
    unsupported_action = _normalize_unsupported(unsupported)
    source_rows, source_indices = _select_rows(
        config,
        row_kind,
        None,
        start_row=start_row,
        end_row=end_row,
        max_rows=None,
    )
    row_pairs = _ordered_rows(
        source_rows,
        source_indices,
        order=row_order,
        random_seed=row_random_seed,
    )
    if max_rows is not None:
        if max_rows < 0:
            raise ValueError("max_rows must be nonnegative")
        row_pairs = row_pairs[:max_rows]
    rows = tuple(row for _idx, row in row_pairs)
    source_indices = tuple(idx for idx, _row in row_pairs)

    columns, column_indices = _select_columns(
        config,
        column_kind,
        start_column=start_column,
        end_column=end_column,
        max_columns=None,
    )
    column_pairs = _ordered_columns(
        config,
        columns,
        column_indices,
        order=column_order,
        random_seed=column_random_seed,
    )
    if max_columns is not None:
        if max_columns < 0:
            raise ValueError("max_columns must be nonnegative")
        column_pairs = column_pairs[:max_columns]

    tracker = ColumnRankTracker(row_count=len(rows), prime=p)
    semantic_cache: dict[str, int] = {}
    semantic_records: dict[str, dict[str, object]] = {}
    cache_hits = 0
    cache_misses = 0
    attempted_entries = 0
    unsupported_entries = 0
    skipped_columns = 0
    nonzero_entries = 0
    nonzero_columns = 0
    column_records = []
    nonzero_records = []
    exterior = ExteriorAlgebra(config)
    start = time.perf_counter()
    stop_reason = "exhausted_columns"

    for column_index, column in column_pairs:
        if stop_rank is not None and tracker.rank >= stop_rank:
            stop_reason = "stop_rank"
            break
        if max_semantic_keys is not None and cache_misses >= max_semantic_keys:
            stop_reason = "max_semantic_keys"
            break

        column_start = time.perf_counter()
        vector = []
        column_nonzero = 0
        column_unsupported = 0
        for row_pos, row in enumerate(rows):
            try:
                key, metadata = _entry_key_and_metadata(config, exterior, row, column)
            except UnsupportedBlockEntry:
                unsupported_entries += 1
                column_unsupported += 1
                if unsupported_action == "zero":
                    vector.append(0)
                    attempted_entries += 1
                    continue
                if unsupported_action == "skip-column":
                    break
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
                semantic_cache[key] = value
                semantic_records[key] = dict(metadata)
                semantic_records[key]["key"] = key
                semantic_records[key]["use_count"] = 0
                cache_misses += 1

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
                            "column_index": int(column_index),
                            "column_name": column.name,
                            "column_kind": column.kind,
                            "key": key,
                            "value": int(value),
                        }
                    )

        if len(vector) != len(rows):
            if stop_reason == "max_semantic_keys":
                break
            skipped_columns += 1
            if unsupported_action == "skip-column":
                continue
            break

        independent = tracker.add_column(vector, index=column_index)
        if column_nonzero:
            nonzero_columns += 1
        column_records.append(
            {
                "index": int(column_index),
                "name": column.name,
                "kind": column.kind,
                "defect": column.defect,
                "nonzero_count": int(column_nonzero),
                "unsupported_count": int(column_unsupported),
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
        "kind": "c18_block_probe",
        "prime": p,
        "method": method,
        "normalized_method": normalized_method,
        "row_kind": row_kind,
        "column_kind": column_kind,
        "unsupported": unsupported_action,
        "row_count": len(rows),
        "available_column_count": len(columns),
        "scheduled_column_count": len(column_pairs),
        "processed_columns": tracker.processed_columns,
        "skipped_columns": skipped_columns,
        "attempted_entries": attempted_entries,
        "unsupported_entries": unsupported_entries,
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
        "source_row_indices": list(source_indices),
        "source_row_names": [row.name for row in rows],
        "source_row_kinds": [row.kind for row in rows],
        "row_order": row_order,
        "row_random_seed": int(row_random_seed),
        "column_order": column_order,
        "column_random_seed": int(column_random_seed),
        "stop_rank": None if stop_rank is None else int(stop_rank),
        "stop_on_nonzero": bool(stop_on_nonzero),
        "max_semantic_keys": (
            None if max_semantic_keys is None else int(max_semantic_keys)
        ),
        "stop_reason": stop_reason,
        "elapsed_seconds": time.perf_counter() - start,
        "columns": column_records,
        "nonzero_records": nonzero_records,
        "semantic_keys": list(semantic_records.values()),
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
    }
    if output_path is not None:
        write_json_maybe_gzip(output_path, payload)
    return payload


class UnsupportedBlockEntry(NotImplementedError):
    """Raised when a source/test shape has no trusted evaluator yet."""


def _entry_key_and_metadata(
    config: FormulaConfig,
    exterior: ExteriorAlgebra,
    row: C18SourceRow,
    column: H62TestColumn,
) -> tuple[str, dict[str, object]]:
    total_a_exp = tuple(
        int(left) + int(right)
        for left, right in zip(row.monomial.a_exp, column.monomial.a_exp)
    )
    total_f_exp = tuple(
        int(left) + int(right)
        for left, right in zip(row.monomial.f_exp, column.monomial.f_exp)
    )
    total_gamma_exp = tuple(
        int(left) + int(right)
        for left, right in zip(row.monomial.gamma_exp, column.monomial.gamma_exp)
    )

    if column.kind in {"all_a", "one_f", "one_gamma", "f2_power"}:
        metadata = {
            "shape": _supported_invariant_shape(total_f_exp, total_gamma_exp),
            "total_a_exp": list(total_a_exp),
            "total_f_exp": list(total_f_exp),
            "total_gamma_exp": list(total_gamma_exp),
        }
        key = (
            f"shape={metadata['shape']};"
            f"f={','.join(str(item) for item in total_f_exp)};"
            f"g={','.join(str(item) for item in total_gamma_exp)};"
            f"a={','.join(str(item) for item in total_a_exp)}"
        )
        return key, metadata

    if column.kind == "one_b_pair":
        if row.kind != "even":
            raise UnsupportedBlockEntry("gamma source x b-pair tests are not supported")
        target = exterior.b_product_to_mask(column.b_labels)
        if target is None:
            raise UnsupportedBlockEntry("b labels wedge to zero")
        sign, b_mask = target
        if sign != 1:
            raise UnsupportedBlockEntry("b-pair columns must be in exterior order")
        if sum(total_f_exp) != 1 or any(total_gamma_exp):
            raise UnsupportedBlockEntry("direct b-mask evaluator expects one f and no gamma")
        metadata = {
            "shape": "b-mask",
            "total_a_exp": list(total_a_exp),
            "total_f_exp": list(total_f_exp),
            "b_mask": int(b_mask),
            "b_labels": [list(label) for label in column.b_labels],
        }
        key = (
            f"shape=b-mask;f={','.join(str(item) for item in total_f_exp)};"
            f"bmask={int(b_mask)};a={','.join(str(item) for item in total_a_exp)}"
        )
        return key, metadata

    raise UnsupportedBlockEntry(f"unsupported column kind {column.kind!r}")


def _supported_invariant_shape(
    total_f_exp: Tuple[int, ...],
    total_gamma_exp: Tuple[int, ...],
) -> str:
    f_count = sum(total_f_exp)
    gamma_count = sum(total_gamma_exp)
    delta_count = sum(total_f_exp[1:])
    if f_count + gamma_count == 1:
        return "one-defect"
    if gamma_count == 0 and 1 <= f_count <= 2 and delta_count <= 2:
        return "f-only"
    if f_count == 1 and gamma_count == 1 and delta_count <= 1:
        return "f-gamma"
    if total_f_exp and total_f_exp[0] >= 1 and gamma_count <= 1 and delta_count <= 1:
        return "f2-power"
    raise UnsupportedBlockEntry(
        f"unsupported invariant shape f={total_f_exp}, gamma={total_gamma_exp}"
    )


def _evaluate_entry(
    config: FormulaConfig,
    metadata: dict[str, object],
    *,
    prime: int,
    method: str,
    beta_chunk_size: int,
    max_chunk_terms: int,
) -> int:
    if method == "synthetic":
        return _synthetic_block_value(metadata, prime)

    shape = str(metadata["shape"])
    a_exp = tuple(int(item) for item in metadata["total_a_exp"])
    f_exp = tuple(int(item) for item in metadata.get("total_f_exp", ()))
    gamma_exp = tuple(int(item) for item in metadata.get("total_gamma_exp", ()))
    if shape == "one-defect":
        total = InvariantMonomial.from_exponents(
            config,
            a_exp=a_exp,
            f_exp=f_exp,
            gamma_exp=gamma_exp,
        )
        if method == "moment":
            return all_a_pairing_total_moment_mod(config, total, prime=prime)
        return all_a_pairing_total_batched_mod(
            config,
            total,
            prime=prime,
            beta_chunk_size=beta_chunk_size,
            max_chunk_terms=max_chunk_terms,
        )
    if shape == "f-only":
        total = InvariantMonomial.from_exponents(config, a_exp=a_exp, f_exp=f_exp)
        if method == "moment":
            return f_only_pairing_total_moment_mod(config, total, prime=prime)
        return f_only_pairing_total_batched_mod(
            config,
            total,
            prime=prime,
            beta_chunk_size=beta_chunk_size,
            max_chunk_terms=max_chunk_terms,
        )
    if shape == "f-gamma":
        total = InvariantMonomial.from_exponents(
            config,
            a_exp=a_exp,
            f_exp=f_exp,
            gamma_exp=gamma_exp,
        )
        if method == "moment":
            return f_gamma_pairing_total_moment_mod(config, total, prime=prime)
        return f_gamma_pairing_total_batched_mod(
            config,
            total,
            prime=prime,
            beta_chunk_size=beta_chunk_size,
            max_chunk_terms=max_chunk_terms,
        )
    if shape == "f2-power":
        total = InvariantMonomial.from_exponents(
            config,
            a_exp=a_exp,
            f_exp=f_exp,
            gamma_exp=gamma_exp,
        )
        if method == "moment":
            return f2_power_pairing_total_moment_mod(config, total, prime=prime)
        return f2_power_pairing_total_batched_mod(
            config,
            total,
            prime=prime,
            beta_chunk_size=beta_chunk_size,
            max_chunk_terms=max_chunk_terms,
        )
    if shape == "b-mask":
        b_mask = int(metadata["b_mask"])
        if method == "moment":
            return b_mask_pairing_total_moment_mod(
                config,
                a_exp=a_exp,
                f_exp=f_exp,
                b_mask=b_mask,
                prime=prime,
            )
        return b_mask_pairing_total_batched_mod(
            config,
            a_exp=a_exp,
            f_exp=f_exp,
            b_mask=b_mask,
            prime=prime,
            beta_chunk_size=beta_chunk_size,
            max_chunk_terms=max_chunk_terms,
        )
    raise UnsupportedBlockEntry(f"unsupported shape {shape!r}")


def _synthetic_block_value(metadata: dict[str, object], prime: int) -> int:
    value = sum(ord(ch) for ch in str(metadata["shape"]))
    for idx, exp in enumerate(metadata.get("total_a_exp", ())):
        value += (idx + 3) * (int(exp) + 1) * (int(exp) + 2)
    for idx, exp in enumerate(metadata.get("total_f_exp", ())):
        value += (idx + 11) * int(exp)
    for idx, exp in enumerate(metadata.get("total_gamma_exp", ())):
        value += (idx + 17) * int(exp)
    if "b_mask" in metadata:
        value += int(metadata["b_mask"]).bit_count() * 29 + int(metadata["b_mask"]) % 101
    return value % prime


def _select_columns(
    config: FormulaConfig,
    column_kind: str,
    *,
    start_column: int,
    end_column: int | None,
    max_columns: int | None,
) -> Tuple[Tuple[H62TestColumn, ...], Tuple[int, ...]]:
    if start_column < 0:
        raise ValueError("start_column must be nonnegative")
    if max_columns is not None and max_columns < 0:
        raise ValueError("max_columns must be nonnegative")
    normalized = _normalize_column_kind(column_kind)
    if normalized == "all-a":
        all_columns = h62_all_a_test_columns(config)
    elif normalized == "f2-power":
        all_columns = h62_f2_power_test_columns(config)
    elif normalized == "one-f":
        all_columns = h62_one_f_test_columns(config)
    elif normalized == "one-gamma":
        all_columns = h62_one_gamma_test_columns(config)
    elif normalized == "b-pair":
        all_columns = h62_one_b_pair_test_columns(config)
    else:
        raise AssertionError(f"unexpected column kind {normalized!r}")
    stop = len(all_columns) if end_column is None else int(end_column)
    if max_columns is not None:
        stop = min(stop, start_column + max_columns)
    if stop < start_column:
        raise ValueError("end_column must be greater than or equal to start_column")
    selected = all_columns[start_column:stop]
    return selected, tuple(range(start_column, start_column + len(selected)))


def _ordered_rows(
    rows: Sequence[C18SourceRow],
    indices: Sequence[int],
    *,
    order: str,
    random_seed: int,
) -> list[tuple[int, C18SourceRow]]:
    pairs = list(zip((int(index) for index in indices), rows))
    if order == "sequential":
        return pairs
    if order == "random":
        rng = random.Random(int(random_seed))
        rng.shuffle(pairs)
        return pairs
    if order == "defect-balanced":
        return _balanced_order(pairs, key_fn=lambda item: item[1].defect)
    raise ValueError("row_order must be sequential, random, or defect-balanced")


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
    if order == "balanced":
        exterior = ExteriorAlgebra(config)
        return _balanced_order(
            pairs,
            key_fn=lambda item: _column_balance_key(exterior, item[1]),
        )
    raise ValueError("column_order must be sequential, random, or balanced")


def _balanced_order(items, *, key_fn):
    groups = {}
    for item in items:
        groups.setdefault(key_fn(item), []).append(item)
    for key, group in list(groups.items()):
        groups[key] = _middle_out(group)
    ordered = []
    keys = sorted(groups, key=str)
    depth = 0
    while True:
        added = False
        for key in keys:
            group = groups[key]
            if depth < len(group):
                ordered.append(group[depth])
                added = True
        if not added:
            break
        depth += 1
    return ordered


def _column_balance_key(exterior: ExteriorAlgebra, column: H62TestColumn):
    if column.kind == "one_b_pair":
        target = exterior.b_product_to_mask(column.b_labels)
        return None if target is None else target[1]
    return column.defect or column.kind


def _middle_out(items):
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


def _normalize_method(method: str) -> str:
    normalized = method.lower()
    if normalized == "semantic-batched":
        return "batched"
    if normalized in {"synthetic", "moment", "batched"}:
        return normalized
    raise ValueError("method must be synthetic, moment, batched, or semantic-batched")


def _normalize_column_kind(column_kind: str) -> str:
    normalized = column_kind.lower().replace("_", "-")
    if normalized in {"all-a", "one-f", "one-gamma", "b-pair", "f2-power"}:
        return normalized
    if normalized in {"b-mask", "one-b-pair", "direct-b-pair"}:
        return "b-pair"
    raise ValueError("column_kind must be all-a, f2-power, one-f, one-gamma, or b-pair")


def _normalize_unsupported(unsupported: str) -> str:
    normalized = unsupported.lower().replace("_", "-")
    if normalized in {"error", "skip-column", "zero"}:
        return normalized
    raise ValueError("unsupported must be error, skip-column, or zero")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prime", type=int, default=RANK7_G2_D1.primary_prime)
    parser.add_argument(
        "--method",
        choices=("synthetic", "moment", "batched", "semantic-batched"),
        default="batched",
    )
    parser.add_argument("--row-kind", choices=("all", "even", "gamma"), default="all")
    parser.add_argument(
        "--column-kind",
        choices=("all-a", "f2-power", "one-f", "one-gamma", "b-pair"),
        default="all-a",
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
        choices=("sequential", "random", "balanced"),
        default="balanced",
    )
    parser.add_argument("--column-random-seed", type=int, default=0)
    parser.add_argument(
        "--unsupported",
        choices=("error", "skip-column", "zero"),
        default="error",
    )
    parser.add_argument("--stop-rank", type=int, default=None)
    parser.add_argument("--stop-on-nonzero", action="store_true")
    parser.add_argument("--max-semantic-keys", type=int, default=None)
    parser.add_argument("--beta-chunk-size", type=int, default=2)
    parser.add_argument("--max-chunk-terms", type=int, default=200_000)
    parser.add_argument("--no-store-nonzero-entries", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = run_c18_block_probe(
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
        "row_kind",
        "column_kind",
        "unsupported",
        "row_count",
        "available_column_count",
        "scheduled_column_count",
        "processed_columns",
        "skipped_columns",
        "attempted_entries",
        "unsupported_entries",
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
