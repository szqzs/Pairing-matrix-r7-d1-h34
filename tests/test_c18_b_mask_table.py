from rank7_jk.c18_b_mask_table import (
    b_mask_key_id,
    enumerate_c18_even_b_mask_keys,
    evaluate_b_mask_key,
)


def test_b_mask_semantic_key_enumeration_slices():
    payload = enumerate_c18_even_b_mask_keys(max_rows=3, max_columns=4)

    assert payload["kind"] == "c18_even_b_mask_semantic_key_enumeration"
    assert payload["row_count"] == 3
    assert payload["column_count"] == 4
    assert payload["entry_count"] == 12
    assert payload["source_row_indices"] == [0, 1, 2]
    assert payload["test_column_indices"] == [0, 1, 2, 3]
    assert payload["key_count"] <= 12


def test_b_mask_key_id_is_stable():
    assert (
        b_mask_key_id((1, 0, 0, 0, 0, 0), 65, (2, 3, 0, 0, 1, 4))
        == "f=1,0,0,0,0,0;bmask=65;a=2,3,0,0,1,4"
    )


def test_evaluate_b_mask_key_matches_moment_on_tiny_rank7_sample():
    payload = enumerate_c18_even_b_mask_keys(max_rows=1, start_column=192, max_columns=1)
    key = payload["keys"][0]

    assert evaluate_b_mask_key(
        key["total_f_exp"],
        key["b_mask"],
        key["total_a_exp"],
        prime=101,
        method="batched",
    ) == evaluate_b_mask_key(
        key["total_f_exp"],
        key["b_mask"],
        key["total_a_exp"],
        prime=101,
        method="moment",
    )
