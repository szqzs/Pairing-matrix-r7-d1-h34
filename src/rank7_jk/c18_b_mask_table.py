"""Direct b-mask test scaffold for c18 even rows."""

from __future__ import annotations

from typing import Sequence, Tuple

from .all_a_pairing import (
    b_mask_pairing_total_batched_mod,
    b_mask_pairing_total_moment_mod,
)
from .c18_all_a_probe import _select_rows
from .c18_basis import H62TestColumn, h62_one_b_pair_test_columns
from .config import FormulaConfig, RANK7_G2_D1
from .exterior import ExteriorAlgebra
from .mod_arith import require_prime


def b_mask_key_id(
    total_f_exp: Sequence[int],
    b_mask: int,
    total_a_exp: Sequence[int],
) -> str:
    f_part = ",".join(str(int(item)) for item in total_f_exp)
    a_part = ",".join(str(int(item)) for item in total_a_exp)
    return f"f={f_part};bmask={int(b_mask)};a={a_part}"


def enumerate_c18_even_b_mask_keys(
    *,
    config: FormulaConfig = RANK7_G2_D1,
    start_row: int = 0,
    end_row: int | None = None,
    max_rows: int | None = None,
    start_column: int = 0,
    end_column: int | None = None,
    max_columns: int | None = None,
) -> dict[str, object]:
    """Enumerate unique f/b-mask semantic keys for direct b-pair tests."""

    rows, source_row_indices = _select_rows(
        config,
        "even",
        None,
        start_row=start_row,
        end_row=end_row,
        max_rows=max_rows,
    )
    columns, test_column_indices = _select_b_mask_columns(
        config,
        start_column=start_column,
        end_column=end_column,
        max_columns=max_columns,
    )
    exterior = ExteriorAlgebra(config)

    records: dict[str, dict[str, object]] = {}
    for row in rows:
        for column in columns:
            if len(column.b_labels) != 2:
                raise ValueError("direct b-mask columns must have exactly two b labels")
            target = exterior.b_product_to_mask(column.b_labels)
            if target is None:
                continue
            sign, b_mask = target
            if sign != 1:
                raise ValueError("b-pair basis should be stored in exterior order")
            total_a_exp = tuple(
                int(left) + int(right)
                for left, right in zip(row.monomial.a_exp, column.monomial.a_exp)
            )
            total_f_exp = tuple(int(item) for item in row.monomial.f_exp)
            key = b_mask_key_id(total_f_exp, b_mask, total_a_exp)
            record = records.get(key)
            if record is None:
                records[key] = {
                    "key": key,
                    "total_f_exp": list(total_f_exp),
                    "b_mask": int(b_mask),
                    "b_labels": [list(label) for label in column.b_labels],
                    "total_a_exp": list(total_a_exp),
                    "use_count": 1,
                }
            else:
                record["use_count"] = int(record["use_count"]) + 1

    keys = sorted(
        records.values(),
        key=lambda item: (
            tuple(int(value) for value in item["total_f_exp"]),
            int(item["b_mask"]),
            tuple(int(value) for value in item["total_a_exp"]),
            str(item["key"]),
        ),
    )
    entry_count = len(rows) * len(columns)
    return {
        "kind": "c18_even_b_mask_semantic_key_enumeration",
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


def evaluate_b_mask_key(
    total_f_exp: Sequence[int],
    b_mask: int,
    total_a_exp: Sequence[int],
    *,
    config: FormulaConfig = RANK7_G2_D1,
    prime: int | None = None,
    method: str = "batched",
    beta_chunk_size: int = 2,
    max_chunk_terms: int = 200_000,
) -> int:
    """Evaluate one direct b-mask semantic key modulo a prime."""

    p = require_prime(config.primary_prime if prime is None else prime)
    normalized = method.lower()
    if normalized in {"batched", "semantic-batched"}:
        return b_mask_pairing_total_batched_mod(
            config,
            a_exp=total_a_exp,
            f_exp=total_f_exp,
            b_mask=int(b_mask),
            prime=p,
            beta_chunk_size=beta_chunk_size,
            max_chunk_terms=max_chunk_terms,
        )
    if normalized == "moment":
        return b_mask_pairing_total_moment_mod(
            config,
            a_exp=total_a_exp,
            f_exp=total_f_exp,
            b_mask=int(b_mask),
            prime=p,
        )
    raise ValueError("method must be moment, batched, or semantic-batched")


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
