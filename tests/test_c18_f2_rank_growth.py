import json

import pytest

from rank7_jk.c18_f2_rank_growth import main, run_c18_f2_rank_growth


def test_f2_rank_growth_synthetic_streams_high_f2_columns(tmp_path):
    output_path = tmp_path / "growth.json"

    payload = run_c18_f2_rank_growth(
        prime=101,
        method="synthetic",
        max_rows=4,
        max_columns=4,
        target_left_nullity=None,
        output_path=output_path,
    )

    assert payload["kind"] == "c18_f2_power_rank_growth"
    assert payload["column_kind"] == "f2-power"
    assert payload["column_order"] == "sequential"
    assert payload["row_count"] == 4
    assert payload["processed_columns"] == 4
    assert payload["attempted_entries"] == 16
    assert payload["test_column_names"][:4] == [
        "f2^31",
        "a2 f2^29",
        "a3 f2^28",
        "a2^2 f2^27",
    ]
    assert payload["rank"] > 0
    assert payload["selected_column_vectors"] is not None
    assert json.loads(output_path.read_text(encoding="utf-8"))["processed_columns"] == 4


def test_f2_rank_growth_f2_power_balanced_samples_low_powers_early():
    payload = run_c18_f2_rank_growth(
        prime=101,
        method="synthetic",
        max_rows=2,
        max_columns=6,
        column_order="f2-power-balanced",
        target_left_nullity=None,
    )

    assert payload["column_order"] == "f2-power-balanced"
    assert [column["f2_power"] for column in payload["columns"]] == [1, 2, 3, 4, 5, 6]
    assert payload["test_column_indices"][0] > payload["test_column_indices"][-1]


def test_f2_rank_growth_max_columns_caps_after_ordering():
    payload = run_c18_f2_rank_growth(
        prime=101,
        method="synthetic",
        max_rows=1,
        max_columns=3,
        column_order="f2-power-desc-balanced",
        target_left_nullity=None,
    )

    assert [column["f2_power"] for column in payload["columns"]] == [31, 29, 28]


def test_f2_rank_growth_can_stop_on_dependent_plateau():
    payload = run_c18_f2_rank_growth(
        prime=101,
        method="synthetic",
        max_rows=2,
        max_columns=10,
        max_dependent_columns=1,
        target_left_nullity=None,
    )

    assert payload["stop_reason"] == "max_dependent_columns"
    assert payload["dependent_columns_since_rank_gain"] == 1
    assert payload["processed_columns"] < 10


def test_f2_rank_growth_emits_normalized_left_null_vector():
    payload = run_c18_f2_rank_growth(
        prime=101,
        method="synthetic",
        max_rows=4,
        max_columns=4,
        target_left_nullity=1,
    )

    assert payload["rank"] == 3
    assert payload["left_nullity"] == 1
    assert payload["candidate_left_null_vector"] is not None
    assert payload["left_nullspace"]["dimension"] == 1
    assert payload["left_nullspace"]["basis_column_dot_products_zero"] is True
    assert payload["left_nullspace"]["selected_column_dot_products_zero"] is True
    assert payload["candidate_left_null_vector"]["verified"] is True
    assert payload["candidate_left_null_vector"]["values"]
    first = next(
        value for value in payload["candidate_left_null_vector"]["values"] if value
    )
    assert first == 1


def test_f2_rank_growth_gates_large_left_nullspace_by_default():
    payload = run_c18_f2_rank_growth(
        prime=101,
        method="synthetic",
        max_rows=4,
        max_columns=1,
        target_left_nullity=None,
    )

    assert payload["left_nullity"] == 3
    assert payload["left_nullspace"]["computed"] is False
    assert payload["left_nullspace"]["dimension"] == 3
    assert payload["left_nullspace"]["vectors"] == []
    assert payload["candidate_left_null_vector"] is None

    forced = run_c18_f2_rank_growth(
        prime=101,
        method="synthetic",
        max_rows=4,
        max_columns=1,
        target_left_nullity=None,
        store_left_nullspace=True,
    )

    assert forced["left_nullspace"]["computed"] is True
    assert forced["left_nullspace"]["dimension"] == 3
    assert len(forced["left_nullspace"]["vectors"]) == 3


def test_f2_rank_growth_real_tiny_first_column_is_supported():
    payload = run_c18_f2_rank_growth(
        prime=101,
        method="batched",
        max_rows=2,
        max_columns=1,
        target_left_nullity=None,
    )

    assert payload["processed_columns"] == 1
    assert payload["attempted_entries"] == 2
    assert payload["unsupported_entries"] == 0
    assert payload["nonzero_entries"] > 0


def test_f2_rank_growth_can_resume_from_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.json"

    first = run_c18_f2_rank_growth(
        prime=101,
        method="synthetic",
        max_rows=4,
        max_columns=2,
        target_left_nullity=None,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=1,
    )
    resumed = run_c18_f2_rank_growth(
        prime=101,
        method="synthetic",
        max_rows=4,
        max_columns=4,
        target_left_nullity=None,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=1,
        resume_from=checkpoint_path,
    )

    assert first["processed_columns"] == 2
    assert resumed["resumed_processed_columns"] == 2
    assert resumed["processed_columns"] == 4
    assert len(resumed["columns"]) == 4
    assert resumed["attempted_entries"] == 16
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["semantic_keys"] == []
    assert checkpoint["semantic_records_omitted_from_checkpoint"] is True
    assert checkpoint["left_nullspace"]["basis_column_dot_products_zero"] is True


def test_f2_rank_growth_rejects_small_primes():
    with pytest.raises(ValueError, match="prime greater than"):
        run_c18_f2_rank_growth(
            prime=31,
            method="synthetic",
            max_rows=2,
            max_columns=1,
            target_left_nullity=None,
        )


def test_f2_rank_growth_cli_synthetic(tmp_path, capsys):
    output_path = tmp_path / "cli_growth.json"

    assert main(
        [
            "--prime",
            "101",
            "--method",
            "synthetic",
            "--max-rows",
            "3",
            "--max-columns",
            "2",
            "--no-target-left-nullity",
            "--output",
            str(output_path),
        ]
    ) == 0

    captured = capsys.readouterr()
    assert "c18_f2_power_rank_growth" in captured.out
    assert json.loads(output_path.read_text(encoding="utf-8"))["processed_columns"] == 2
