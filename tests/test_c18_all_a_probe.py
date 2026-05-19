import pytest

from rank7_jk.c18_all_a_probe import (
    actual_all_a_column,
    evaluator_by_name,
    run_all_a_probe,
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

    with pytest.raises(ValueError, match="unknown"):
        evaluator_by_name("pretend")
