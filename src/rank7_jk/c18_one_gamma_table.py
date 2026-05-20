"""Semantic-key scaffold for c18 even rows against one-gamma H62 tests."""

from __future__ import annotations

from typing import Sequence, Tuple

from .all_a_pairing import (
    f_gamma_pairing_total_batched_mod,
    f_gamma_pairing_total_moment_mod,
)
from .c18_all_a_probe import _select_rows
from .c18_basis import H62TestColumn, h62_one_gamma_test_columns
from .config import FormulaConfig, RANK7_G2_D1
from .invariants import InvariantMonomial
from .mod_arith import require_prime


def f_gamma_key_id(
    total_f_exp: Sequence[int],
    total_gamma_exp: Sequence[int],
    total_a_exp: Sequence[int],
) -> str:
    f_part = ",".join(str(int(item)) for item in total_f_exp)
    gamma_part = ",".join(str(int(item)) for item in total_gamma_exp)
    a_part = ",".join(str(int(item)) for item in total_a_exp)
    return f"f={f_part};g={gamma_part};a={a_part}"


def enumerate_c18_even_one_gamma_keys(
    *,
    config: FormulaConfig = RANK7_G2_D1,
    start_row: int = 0,
    end_row: int | None = None,
    max_rows: int | None = None,
    start_column: int = 0,
    end_column: int | None = None,
    max_columns: int | None = None,
) -> dict[str, object]:
    """Enumerate unique f+gamma semantic keys for the one-gamma test matrix."""

    rows, source_row_indices = _select_rows(
        config,
        "even",
        None,
        start_row=start_row,
        end_row=end_row,
        max_rows=max_rows,
    )
    columns, test_column_indices = _select_one_gamma_columns(
        config,
        start_column=start_column,
        end_column=end_column,
        max_columns=max_columns,
    )

    records: dict[str, dict[str, object]] = {}
    for row in rows:
        for column in columns:
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
            key = f_gamma_key_id(total_f_exp, total_gamma_exp, total_a_exp)
            record = records.get(key)
            if record is None:
                records[key] = {
                    "key": key,
                    "total_f_exp": list(total_f_exp),
                    "total_gamma_exp": list(total_gamma_exp),
                    "total_a_exp": list(total_a_exp),
                    "use_count": 1,
                }
            else:
                record["use_count"] = int(record["use_count"]) + 1

    keys = sorted(
        records.values(),
        key=lambda item: (
            tuple(int(value) for value in item["total_f_exp"]),
            tuple(int(value) for value in item["total_gamma_exp"]),
            tuple(int(value) for value in item["total_a_exp"]),
            str(item["key"]),
        ),
    )
    entry_count = len(rows) * len(columns)
    return {
        "kind": "c18_even_one_gamma_semantic_key_enumeration",
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


def evaluate_f_gamma_key(
    total_f_exp: Sequence[int],
    total_gamma_exp: Sequence[int],
    total_a_exp: Sequence[int],
    *,
    config: FormulaConfig = RANK7_G2_D1,
    prime: int | None = None,
    method: str = "batched",
) -> int:
    """Evaluate one f+gamma semantic key modulo a prime."""

    p = require_prime(config.primary_prime if prime is None else prime)
    total = InvariantMonomial.from_exponents(
        config,
        a_exp=tuple(int(item) for item in total_a_exp),
        f_exp=tuple(int(item) for item in total_f_exp),
        gamma_exp=tuple(int(item) for item in total_gamma_exp),
    )
    normalized = method.lower()
    if normalized in {"batched", "semantic-batched"}:
        return f_gamma_pairing_total_batched_mod(config, total, prime=p)
    if normalized == "moment":
        return f_gamma_pairing_total_moment_mod(config, total, prime=p)
    raise ValueError("method must be moment, batched, or semantic-batched")


def _select_one_gamma_columns(
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
    all_columns = h62_one_gamma_test_columns(config)
    stop = len(all_columns) if end_column is None else int(end_column)
    if max_columns is not None:
        stop = min(stop, start_column + max_columns)
    if stop < start_column:
        raise ValueError("end_column must be greater than or equal to start_column")
    selected = all_columns[start_column:stop]
    return selected, tuple(range(start_column, start_column + len(selected)))
