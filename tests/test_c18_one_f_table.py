from rank7_jk.c18_one_f_table import (
    enumerate_c18_even_one_f_keys,
    evaluate_f_only_key,
    f_only_key_id,
)


def test_one_f_semantic_key_enumeration_dimensions():
    payload = enumerate_c18_even_one_f_keys()

    assert payload["row_count"] == 126
    assert payload["column_count"] == 1091
    assert payload["entry_count"] == 137_466
    assert payload["key_count"] == 16_033
    assert payload["reuse_factor"] > 8.5


def test_one_f_semantic_key_enumeration_slices():
    payload = enumerate_c18_even_one_f_keys(max_rows=3, max_columns=4)

    assert payload["entry_count"] == 12
    assert payload["source_row_indices"] == [0, 1, 2]
    assert payload["test_column_indices"] == [0, 1, 2, 3]
    assert payload["key_count"] <= 12


def test_f_only_key_id_is_stable():
    assert (
        f_only_key_id((1, 0, 1, 0, 0, 0), (2, 3, 0, 0, 1, 4))
        == "f=1,0,1,0,0,0;a=2,3,0,0,1,4"
    )


def test_evaluate_f_only_key_matches_moment_on_tiny_rank7_sample():
    payload = enumerate_c18_even_one_f_keys(max_rows=1, max_columns=1)
    key = payload["keys"][0]

    assert evaluate_f_only_key(
        key["total_f_exp"],
        key["total_a_exp"],
        prime=101,
        method="batched",
    ) == evaluate_f_only_key(
        key["total_f_exp"],
        key["total_a_exp"],
        prime=101,
        method="moment",
    )
