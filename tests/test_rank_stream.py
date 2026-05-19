from rank7_jk.rank_stream import (
    ColumnRankTracker,
    left_nullspace_mod,
    mat_vec_mul_mod,
    matrix_rank_mod,
    right_nullspace_mod,
    vec_mat_mul_mod,
)


def test_rank_and_right_nullspace_for_known_matrix():
    p = 101
    matrix = [
        [1, 2, 3],
        [2, 4, 6],
        [1, 0, 1],
    ]

    assert matrix_rank_mod(matrix, p) == 2
    nullspace = right_nullspace_mod(matrix, p)
    assert nullspace == ((100, 100, 1),)
    assert mat_vec_mul_mod(matrix, nullspace[0], p) == (0, 0, 0)


def test_left_nullspace_for_known_row_relation():
    p = 101
    matrix = [
        [1, 0, 2],
        [2, 1, 5],
        [3, 1, 7],
    ]

    assert matrix_rank_mod(matrix, p) == 2
    nullspace = left_nullspace_mod(matrix, p)
    assert nullspace == ((100, 100, 1),)
    assert vec_mat_mul_mod(nullspace[0], matrix, p) == (0, 0, 0)


def test_column_rank_tracker_selects_independent_columns():
    p = 101
    columns = [
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
    ]
    tracker = ColumnRankTracker(row_count=4, prime=p)

    assert tracker.add_columns(columns) == 3
    assert tracker.rank == 3
    assert tracker.nullity_left == 1
    assert tracker.selected_indices == [0, 2, 4]


def test_column_rank_tracker_can_stop_at_target_rank():
    tracker = ColumnRankTracker(row_count=3, prime=101)
    columns = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]

    added = tracker.add_columns(columns, stop_rank=2)

    assert added == 2
    assert tracker.rank == 2
    assert tracker.processed_columns == 2


def test_column_rank_tracker_matches_dense_rank():
    p = 101
    rows_by_columns = [
        [1, 0, 2, 3],
        [0, 1, 4, 5],
        [0, 0, 0, 0],
        [2, 1, 8, 11],
    ]
    columns = [list(column) for column in zip(*rows_by_columns)]
    tracker = ColumnRankTracker(row_count=4, prime=p)
    tracker.add_columns(columns)

    assert tracker.rank == matrix_rank_mod(rows_by_columns, p)
    assert tracker.nullity_left == len(rows_by_columns) - matrix_rank_mod(rows_by_columns, p)
