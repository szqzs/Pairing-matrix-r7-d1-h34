import json

import pytest

from rank7_jk.c18_all_a_probe import (
    actual_all_a_column,
    batched_semantic_all_a_column,
    benchmark_all_a_defects,
    evaluator_by_name,
    main,
    run_all_a_probe,
    semantic_actual_all_a_column,
    slow_actual_all_a_column,
    synthetic_all_a_column,
)
from rank7_jk.c18_basis import c18_source_rows, h62_all_a_test_columns


def test_synthetic_all_a_probe_exercises_rank_tracker_over_full_block():
    columns = h62_all_a_test_columns()
    result = run_all_a_probe(prime=101, evaluator=synthetic_all_a_column)

    assert result.row_count == 309
    assert result.column_count == 269
    assert result.processed_columns == 269
    assert result.rank == 269
    assert result.left_nullity == 40
    assert result.selected_column_indices == tuple(range(269))
    assert result.selected_column_names == tuple(column.name for column in columns)


def test_synthetic_all_a_probe_can_stop_at_target_rank():
    result = run_all_a_probe(prime=101, evaluator=synthetic_all_a_column, stop_rank=25)

    assert result.processed_columns == 25
    assert result.rank == 25
    assert result.left_nullity == 284
    assert result.selected_column_indices == tuple(range(25))


def test_synthetic_all_a_probe_supports_row_and_column_slices():
    columns = h62_all_a_test_columns()
    result = run_all_a_probe(
        prime=101,
        evaluator=synthetic_all_a_column,
        row_kind="even",
        start_row=5,
        max_rows=20,
        start_column=10,
        max_columns=5,
        store_selected_vectors=True,
    )

    assert result.row_count == 20
    assert result.column_count == 5
    assert result.processed_columns == 5
    assert result.source_row_indices == tuple(range(5, 25))
    assert result.test_column_indices == tuple(range(10, 15))
    assert result.rank == 5
    assert result.selected_column_indices == tuple(range(10, 15))
    assert result.selected_column_names == tuple(column.name for column in columns[10:15])
    assert result.selected_column_vectors is not None
    assert set(result.selected_column_vectors) == set(range(10, 15))


def test_synthetic_all_a_column_rejects_non_all_a_columns():
    rows = c18_source_rows()
    column = h62_all_a_test_columns()[0]
    bad_column = type(column)(kind="one_f", monomial=column.monomial, defect="f2")

    with pytest.raises(ValueError, match="all-a"):
        synthetic_all_a_column(0, bad_column, rows, 101)


def test_actual_all_a_backend_rejects_non_all_a_columns_before_formula_work():
    rows = c18_source_rows()
    column = h62_all_a_test_columns()[0]
    bad_column = type(column)(kind="one_f", monomial=column.monomial, defect="f2")

    with pytest.raises(ValueError, match="all-a"):
        actual_all_a_column(0, bad_column, rows, 101)


def test_evaluator_name_dispatch_is_explicit():
    assert evaluator_by_name("synthetic") is synthetic_all_a_column
    assert evaluator_by_name("actual") is actual_all_a_column
    assert evaluator_by_name("moment") is actual_all_a_column
    assert evaluator_by_name("semantic") is semantic_actual_all_a_column
    assert evaluator_by_name("semantic-batched") is batched_semantic_all_a_column
    assert evaluator_by_name("slow-actual") is slow_actual_all_a_column

    with pytest.raises(ValueError, match="unknown"):
        evaluator_by_name("pretend")


def test_cli_writes_probe_and_timing_json(tmp_path):
    output_path = tmp_path / "probe.json"
    timing_path = tmp_path / "timing.json"

    assert main(
        [
            "--prime",
            "101",
            "--max-columns",
            "2",
            "--output",
            str(output_path),
            "--timing-json",
            str(timing_path),
        ]
    ) == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    timing = json.loads(timing_path.read_text(encoding="utf-8"))

    assert payload["processed_columns"] == 2
    assert payload["rank"] == 2
    assert payload["test_column_indices"] == [0, 1]
    assert timing["processed_columns"] == 2
    assert "cache_info" in timing


def test_probe_checkpoint_records_partial_rank_state(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.json"

    result = run_all_a_probe(
        prime=101,
        evaluator=synthetic_all_a_column,
        max_columns=3,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=2,
    )

    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert result.processed_columns == 3
    assert payload["processed_columns"] == 3
    assert payload["rank"] == 3
    assert payload["selected_column_indices"] == [0, 1, 2]


def test_defect_benchmark_can_emit_empty_semantic_block():
    payload = benchmark_all_a_defects(
        prime=101,
        defects=("f2",),
        max_columns=0,
        method="moment",
    )

    assert payload["defects"] == ["f2"]
    assert payload["benchmarks"][0]["defect"] == "f2"
    assert payload["benchmarks"][0]["processed_columns"] == 0
