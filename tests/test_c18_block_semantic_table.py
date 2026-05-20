import json
from pathlib import Path

import pytest

from rank7_jk.c18_block_probe import UnsupportedBlockEntry
from rank7_jk.c18_block_semantic_table import (
    assemble_combined_block_rank_from_value_tables,
    assemble_block_rank_from_value_table,
    enumerate_c18_block_semantic_keys,
    main,
    merge_block_key_manifest_outputs,
    plan_c18_block_semantic_keys,
    read_json_maybe_gzip,
    run_block_key_manifest_chunk,
)


def test_block_semantic_enumeration_f2_power_balanced_samples_low_powers():
    payload = enumerate_c18_block_semantic_keys(
        row_kind="gamma",
        column_kind="f2-power",
        max_rows=2,
        max_columns=6,
        column_order="f2-power-balanced",
    )

    assert payload["kind"] == "c18_block_semantic_key_enumeration"
    assert payload["row_count"] == 2
    assert payload["column_count"] == 6
    assert payload["entry_count"] == 12
    assert payload["unsupported_entries"] == 0
    assert payload["test_column_defects"] == [
        "f2^1",
        "f2^2",
        "f2^3",
        "f2^4",
        "f2^5",
        "f2^6",
    ]
    assert payload["key_count"] == 12


def test_block_semantic_manifest_run_merge_and_assemble_synthetic(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    output_dir = tmp_path / "chunks"
    table_path = tmp_path / "table.json.gz"
    rank_path = tmp_path / "rank.json"

    manifest = plan_c18_block_semantic_keys(
        prime=101,
        method="synthetic",
        row_kind="gamma",
        column_kind="one-f",
        max_rows=4,
        max_columns=3,
        chunk_size=2,
        output_dir=output_dir,
        output_path=manifest_path,
    )

    assert manifest["entry_count"] == 12
    assert manifest["key_count"] == 12
    assert manifest["chunk_count"] == 6

    for task_id in range(manifest["chunk_count"]):
        chunk = run_block_key_manifest_chunk(manifest_path, task_id)
        assert chunk["complete"] is True
        assert chunk["chunk_id"] == task_id

    table = merge_block_key_manifest_outputs(manifest_path, output_path=table_path)
    rank = assemble_block_rank_from_value_table(
        table_path,
        output_path=rank_path,
        compute_left_nullspace=True,
    )

    assert table["complete"] is True
    assert table["key_count"] == 12
    assert rank["rank"] == 3
    assert rank["left_nullity"] == 1
    assert len(rank["left_nullspace"]) == 1
    assert json.loads(rank_path.read_text(encoding="utf-8"))["rank"] == 3


def test_block_semantic_combined_rank_merges_tables_with_same_rows(tmp_path):
    table_paths = []
    for column_kind in ("f2-power", "one-f"):
        manifest_path = tmp_path / f"{column_kind}_manifest.json"
        table_path = tmp_path / f"{column_kind}_table.json.gz"
        manifest = plan_c18_block_semantic_keys(
            prime=101,
            method="synthetic",
            row_kind="gamma",
            column_kind=column_kind,
            max_rows=4,
            max_columns=3,
            chunk_size=20,
            output_dir=tmp_path / f"{column_kind}_chunks",
            output_path=manifest_path,
        )
        run_block_key_manifest_chunk(manifest_path, 0)
        merge_block_key_manifest_outputs(manifest_path, output_path=table_path)
        assert manifest["row_count"] == 4
        table_paths.append(table_path)

    combined_path = tmp_path / "combined_rank.json"
    combined = assemble_combined_block_rank_from_value_tables(
        table_paths,
        output_path=combined_path,
        compute_left_nullspace=True,
    )
    first = assemble_block_rank_from_value_table(table_paths[0])
    second = assemble_block_rank_from_value_table(table_paths[1])

    assert combined["kind"] == "c18_combined_block_semantic_table_rank"
    assert combined["table_count"] == 2
    assert combined["processed_columns"] == 6
    assert combined["rank"] >= max(first["rank"], second["rank"])
    assert combined["left_nullity"] == 4 - combined["rank"]
    assert len(combined["blocks"]) == 2
    assert json.loads(combined_path.read_text(encoding="utf-8"))["table_count"] == 2


def test_block_semantic_rejects_unsupported_gamma_gamma_by_default():
    with pytest.raises(UnsupportedBlockEntry):
        enumerate_c18_block_semantic_keys(
            row_kind="gamma",
            column_kind="one-gamma",
            max_rows=1,
            max_columns=1,
        )


def test_block_semantic_can_zero_unsupported_gamma_gamma(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    table_path = tmp_path / "table.json"

    manifest = plan_c18_block_semantic_keys(
        prime=101,
        method="synthetic",
        row_kind="gamma",
        column_kind="one-gamma",
        max_rows=2,
        max_columns=3,
        unsupported="zero",
        chunk_size=2,
        output_dir=tmp_path / "chunks",
        output_path=manifest_path,
    )
    table = merge_block_key_manifest_outputs(manifest_path, output_path=table_path)
    rank = assemble_block_rank_from_value_table(table_path)

    assert manifest["key_count"] == 0
    assert manifest["chunk_count"] == 0
    assert manifest["unsupported_entries"] == 6
    assert table["values"] == {}
    assert rank["processed_columns"] == 3
    assert rank["rank"] == 0
    assert rank["columns"][0]["unsupported_count"] == 2


def test_block_semantic_run_key_manifest_skips_existing_complete_output(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest = plan_c18_block_semantic_keys(
        prime=101,
        method="synthetic",
        row_kind="gamma",
        column_kind="one-f",
        max_rows=2,
        max_columns=2,
        chunk_size=20,
        output_dir=tmp_path / "chunks",
        output_path=manifest_path,
    )

    first = run_block_key_manifest_chunk(manifest_path, 0)
    second = run_block_key_manifest_chunk(manifest_path, 0)
    chunk_path = Path(manifest["chunks"][0]["output_path"])

    assert second["elapsed_seconds"] == first["elapsed_seconds"]
    assert read_json_maybe_gzip(chunk_path)["complete"] is True


def test_block_semantic_cli_plan_run_merge_and_assemble(tmp_path, capsys):
    manifest_path = tmp_path / "manifest.json"
    output_dir = tmp_path / "chunks"
    table_path = tmp_path / "table.json.gz"
    rank_path = tmp_path / "rank.json"

    assert main(
        [
            "plan-keys",
            "--prime",
            "101",
            "--method",
            "synthetic",
            "--row-kind",
            "gamma",
            "--column-kind",
            "one-f",
            "--max-rows",
            "3",
            "--max-columns",
            "2",
            "--chunk-size",
            "2",
            "--output-dir",
            str(output_dir),
            "--output",
            str(manifest_path),
        ]
    ) == 0
    assert main(["run-key-manifest", str(manifest_path), "--task-id", "0"]) == 0
    assert main(["run-key-manifest", str(manifest_path), "--task-id", "1"]) == 0
    assert main(["run-key-manifest", str(manifest_path), "--task-id", "2"]) == 0
    assert main(["merge-key-manifest", str(manifest_path), "--output", str(table_path)]) == 0
    assert main(["assemble-rank", str(table_path), "--output", str(rank_path)]) == 0

    captured = capsys.readouterr()
    assert "c18_block_semantic_key_manifest" in captured.out
    assert json.loads(rank_path.read_text(encoding="utf-8"))["processed_columns"] == 2


def test_block_semantic_bouchet_helper_scripts_exist_and_are_executable():
    root = Path(__file__).resolve().parents[1]
    scripts = [
        root / "scripts/bouchet/run_c18_block_semantic_key_chunk.sh",
        root / "scripts/bouchet/submit_c18_block_semantic_keys.sbatch",
        root / "scripts/bouchet/assemble_c18_block_semantic_table.sh",
    ]

    for script in scripts:
        assert script.exists()
        assert script.stat().st_mode & 0o111
