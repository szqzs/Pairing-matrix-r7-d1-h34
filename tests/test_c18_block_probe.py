import json

import pytest

from rank7_jk.c18_block_probe import UnsupportedBlockEntry, main, run_c18_block_probe


def test_block_probe_synthetic_all_a_all_rows(tmp_path):
    output_path = tmp_path / "block.json"

    payload = run_c18_block_probe(
        prime=101,
        method="synthetic",
        row_kind="all",
        column_kind="all-a",
        max_rows=6,
        max_columns=4,
        output_path=output_path,
    )

    assert payload["kind"] == "c18_block_probe"
    assert payload["row_count"] == 6
    assert payload["processed_columns"] == 4
    assert payload["attempted_entries"] == 24
    assert payload["unsupported_entries"] == 0
    assert payload["rank"] > 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["processed_columns"] == 4


def test_block_probe_real_tiny_gamma_all_a_path_runs():
    payload = run_c18_block_probe(
        prime=101,
        method="batched",
        row_kind="gamma",
        column_kind="all-a",
        max_rows=1,
        max_columns=1,
    )

    assert payload["row_count"] == 1
    assert payload["processed_columns"] == 1
    assert payload["attempted_entries"] == 1
    assert payload["unsupported_entries"] == 0


def test_block_probe_real_tiny_gamma_one_f_path_runs():
    payload = run_c18_block_probe(
        prime=101,
        method="batched",
        row_kind="gamma",
        column_kind="one-f",
        max_rows=1,
        max_columns=1,
    )

    assert payload["row_count"] == 1
    assert payload["processed_columns"] == 1
    assert payload["attempted_entries"] == 1
    assert payload["unsupported_entries"] == 0


def test_block_probe_real_tiny_f2_power_path_runs():
    payload = run_c18_block_probe(
        prime=101,
        method="batched",
        row_kind="all",
        column_kind="f2-power",
        max_rows=2,
        max_columns=1,
    )

    assert payload["row_count"] == 2
    assert payload["processed_columns"] == 1
    assert payload["attempted_entries"] == 2
    assert payload["unsupported_entries"] == 0
    assert payload["semantic_cache_misses"] == 2


def test_block_probe_reports_unsupported_gamma_one_gamma_by_default():
    with pytest.raises(UnsupportedBlockEntry):
        run_c18_block_probe(
            prime=101,
            method="batched",
            row_kind="gamma",
            column_kind="one-gamma",
            max_rows=1,
            max_columns=1,
        )


def test_block_probe_can_skip_unsupported_columns():
    payload = run_c18_block_probe(
        prime=101,
        method="batched",
        row_kind="gamma",
        column_kind="one-gamma",
        max_rows=1,
        max_columns=2,
        unsupported="skip-column",
    )

    assert payload["processed_columns"] == 0
    assert payload["skipped_columns"] == 2
    assert payload["unsupported_entries"] == 2


def test_block_probe_cli_synthetic(tmp_path, capsys):
    output_path = tmp_path / "cli_block.json"

    assert main(
        [
            "--prime",
            "101",
            "--method",
            "synthetic",
            "--row-kind",
            "even",
            "--column-kind",
            "f2-power",
            "--max-rows",
            "3",
            "--max-columns",
            "2",
            "--output",
            str(output_path),
        ]
    ) == 0

    captured = capsys.readouterr()
    assert "c18_block_probe" in captured.out
    assert json.loads(output_path.read_text(encoding="utf-8"))["processed_columns"] == 2
