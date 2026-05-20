import json

from rank7_jk.c18_b_mask_probe import run_b_mask_adaptive_probe


def test_b_mask_adaptive_probe_synthetic_mask_balanced_order(tmp_path):
    output_path = tmp_path / "probe.json"

    payload = run_b_mask_adaptive_probe(
        prime=101,
        method="synthetic",
        max_rows=4,
        max_columns=5,
        column_order="mask-balanced",
        output_path=output_path,
    )

    assert payload["kind"] == "c18_b_mask_adaptive_probe"
    assert payload["processed_columns"] == 5
    assert payload["attempted_entries"] == 20
    assert payload["semantic_cache_misses"] == 20
    assert payload["nonzero_columns"] == 5
    assert payload["rank"] > 0
    assert [column["b_mask"] for column in payload["columns"]] == [3, 5, 6, 9, 10]
    assert json.loads(output_path.read_text(encoding="utf-8"))["processed_columns"] == 5


def test_b_mask_adaptive_probe_can_stop_on_first_synthetic_nonzero():
    payload = run_b_mask_adaptive_probe(
        prime=101,
        method="synthetic",
        max_rows=4,
        max_columns=5,
        column_order="sequential",
        stop_on_nonzero=True,
    )

    assert payload["processed_columns"] == 1
    assert payload["nonzero_entries"] == 4
    assert payload["stop_reason"] == "nonzero_entry"


def test_b_mask_adaptive_probe_balances_source_defects():
    payload = run_b_mask_adaptive_probe(
        prime=101,
        method="synthetic",
        max_rows=6,
        max_columns=1,
        row_order="defect-balanced",
        column_order="sequential",
    )

    assert [name.split()[-1] for name in payload["source_row_names"]] == [
        "f2",
        "f3",
        "f4",
        "f5",
        "f6",
        "f7",
    ]


def test_b_mask_adaptive_probe_honors_semantic_key_budget():
    payload = run_b_mask_adaptive_probe(
        prime=101,
        method="synthetic",
        max_rows=4,
        max_columns=5,
        column_order="sequential",
        max_semantic_keys=3,
    )

    assert payload["processed_columns"] == 0
    assert payload["attempted_entries"] == 3
    assert payload["semantic_cache_misses"] == 3
    assert payload["stop_reason"] == "max_semantic_keys"
