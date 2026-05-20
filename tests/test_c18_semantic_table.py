import json
from pathlib import Path

from rank7_jk.c18_semantic_table import (
    assemble_rank_from_value_table,
    enumerate_c18_semantic_keys,
    main,
    merge_key_manifest_outputs,
    plan_c18_semantic_keys,
    read_json_maybe_gzip,
    run_key_manifest_chunk,
    semantic_key_id,
)


def test_semantic_key_enumeration_has_expected_full_even_reuse():
    payload = enumerate_c18_semantic_keys(row_kind="even")

    assert payload["row_count"] == 126
    assert payload["column_count"] == 269
    assert payload["entry_count"] == 33_894
    assert payload["key_count"] == 6_086
    assert payload["reuse_factor"] > 5.5


def test_semantic_key_id_round_trip_format_is_stable():
    assert semantic_key_id("f5", (1, 0, 3, 0, 0, 2)) == "f5:1,0,3,0,0,2"


def test_semantic_manifest_plan_run_merge_and_assemble_synthetic(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    output_dir = tmp_path / "chunks"
    table_path = tmp_path / "table.json.gz"
    rank_path = tmp_path / "rank.json"

    manifest = plan_c18_semantic_keys(
        prime=101,
        method="synthetic",
        row_kind="even",
        max_rows=4,
        max_columns=3,
        chunk_size=2,
        output_dir=output_dir,
        output_path=manifest_path,
    )

    assert manifest["entry_count"] == 12
    assert manifest["key_count"] == 11
    assert manifest["chunk_count"] == 6

    for task_id in range(manifest["chunk_count"]):
        chunk = run_key_manifest_chunk(manifest_path, task_id)
        assert chunk["complete"] is True
        assert chunk["chunk_id"] == task_id

    table = merge_key_manifest_outputs(manifest_path, output_path=table_path)
    rank = assemble_rank_from_value_table(
        table_path,
        output_path=rank_path,
        compute_left_nullspace=True,
    )

    assert table["complete"] is True
    assert table["key_count"] == 11
    assert rank["rank"] == 3
    assert rank["left_nullity"] == 1
    assert len(rank["left_nullspace"]) == 1
    assert json.loads(rank_path.read_text(encoding="utf-8"))["rank"] == 3


def test_run_key_manifest_chunk_skips_existing_complete_output(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest = plan_c18_semantic_keys(
        prime=101,
        method="synthetic",
        row_kind="even",
        max_rows=3,
        max_columns=2,
        chunk_size=20,
        output_dir=tmp_path / "chunks",
        output_path=manifest_path,
    )

    first = run_key_manifest_chunk(manifest_path, 0)
    second = run_key_manifest_chunk(manifest_path, 0)
    chunk_path = Path(manifest["chunks"][0]["output_path"])

    assert second["elapsed_seconds"] == first["elapsed_seconds"]
    assert read_json_maybe_gzip(chunk_path)["complete"] is True


def test_semantic_cli_plan_run_merge_and_assemble(tmp_path, capsys):
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
            "--max-rows",
            "4",
            "--max-columns",
            "3",
            "--chunk-size",
            "5",
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
    assert main(
        [
            "assemble-rank",
            str(table_path),
            "--left-nullspace",
            "--output",
            str(rank_path),
        ]
    ) == 0

    captured = capsys.readouterr()
    assert "c18_semantic_key_manifest" in captured.out
    assert json.loads(rank_path.read_text(encoding="utf-8"))["rank"] == 3


def test_semantic_bouchet_helper_scripts_exist_and_are_executable():
    root = Path(__file__).resolve().parents[1]
    scripts = [
        root / "scripts/bouchet/run_c18_semantic_key_chunk.sh",
        root / "scripts/bouchet/submit_c18_semantic_keys.sbatch",
        root / "scripts/bouchet/assemble_c18_semantic_table.sh",
    ]

    for script in scripts:
        assert script.exists()
        assert script.stat().st_mode & 0o111
