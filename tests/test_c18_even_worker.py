import json
from pathlib import Path

from rank7_jk.c18_all_a_probe import run_all_a_probe, synthetic_all_a_column
from rank7_jk.c18_even_worker import (
    main,
    merge_c18_manifest_outputs,
    merge_c18_worker_outputs,
    plan_c18_even_chunks,
    read_json_maybe_gzip,
    run_c18_even_worker,
    run_manifest_chunk,
)


def test_worker_writes_compressed_synthetic_chunk_and_merge(tmp_path):
    chunk_path = tmp_path / "chunk_000.json.gz"

    payload = run_c18_even_worker(
        prime=101,
        method="synthetic",
        row_kind="even",
        max_rows=8,
        max_columns=3,
        output_path=chunk_path,
    )
    loaded = read_json_maybe_gzip(chunk_path)
    merged = merge_c18_worker_outputs([chunk_path])

    assert payload["kind"] == "c18_all_a_chunk"
    assert loaded["processed_columns"] == 3
    assert loaded["source_row_indices"] == list(range(8))
    assert [column["index"] for column in loaded["columns"]] == [0, 1, 2]
    assert merged["rank"] == 3
    assert merged["left_nullity"] == 5
    assert merged["selected_column_indices"] == [0, 1, 2]


def test_worker_merge_matches_one_shot_synthetic_probe(tmp_path):
    chunk_a = tmp_path / "chunk_a.json.gz"
    chunk_b = tmp_path / "chunk_b.json.gz"
    run_c18_even_worker(
        prime=101,
        method="synthetic",
        row_kind="even",
        max_rows=8,
        start_column=0,
        max_columns=2,
        output_path=chunk_a,
    )
    run_c18_even_worker(
        prime=101,
        method="synthetic",
        row_kind="even",
        max_rows=8,
        start_column=2,
        max_columns=3,
        output_path=chunk_b,
    )

    merged = merge_c18_worker_outputs([chunk_b, chunk_a])
    one_shot = run_all_a_probe(
        prime=101,
        evaluator=synthetic_all_a_column,
        row_kind="even",
        max_rows=8,
        max_columns=5,
    )

    assert merged["processed_columns"] == one_shot.processed_columns
    assert merged["rank"] == one_shot.rank
    assert merged["left_nullity"] == one_shot.left_nullity
    assert merged["selected_column_indices"] == list(one_shot.selected_column_indices)


def test_worker_merge_reports_duplicate_column_indices(tmp_path):
    chunk_a = tmp_path / "chunk_a.json.gz"
    chunk_b = tmp_path / "chunk_b.json.gz"
    for path in (chunk_a, chunk_b):
        run_c18_even_worker(
            prime=101,
            method="synthetic",
            row_kind="even",
            max_rows=8,
            start_column=1,
            max_columns=2,
            output_path=path,
        )

    merged = merge_c18_worker_outputs([chunk_a, chunk_b])

    assert merged["duplicate_column_indices"] == [1, 2]


def test_worker_cli_run_and_merge(tmp_path, capsys):
    chunk_path = tmp_path / "cli_chunk.json.gz"
    merge_path = tmp_path / "merge.json"

    assert main(
        [
            "run",
            "--prime",
            "101",
            "--method",
            "synthetic",
            "--max-rows",
            "8",
            "--max-columns",
            "3",
            "--output",
            str(chunk_path),
        ]
    ) == 0
    assert main(["merge", str(chunk_path), "--output", str(merge_path)]) == 0

    captured = capsys.readouterr()
    assert "c18_all_a_chunk" in captured.out
    payload = json.loads(merge_path.read_text(encoding="utf-8"))
    assert payload["rank"] == 3


def test_worker_manifest_plan_run_and_merge(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    output_dir = tmp_path / "chunks"

    manifest = plan_c18_even_chunks(
        prime=101,
        method="synthetic",
        row_kind="even",
        max_rows=8,
        max_columns=5,
        chunk_size=2,
        output_dir=output_dir,
        output_path=manifest_path,
    )

    assert manifest["chunk_count"] == 3
    assert [chunk["start_column"] for chunk in manifest["chunks"]] == [0, 2, 4]
    assert [chunk["end_column"] for chunk in manifest["chunks"]] == [2, 4, 5]

    for task_id in range(3):
        payload = run_manifest_chunk(manifest_path, task_id)
        assert payload["chunk_id"] == task_id
        assert payload["complete"] is True

    merged = merge_c18_manifest_outputs(manifest_path)

    assert merged["processed_columns"] == 5
    assert merged["rank"] == 5
    assert merged["left_nullity"] == 3
    assert merged["missing_chunk_outputs"] == []


def test_run_manifest_chunk_skips_existing_complete_output(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    output_dir = tmp_path / "chunks"
    manifest = plan_c18_even_chunks(
        prime=101,
        method="synthetic",
        row_kind="even",
        max_rows=8,
        max_columns=2,
        chunk_size=2,
        output_dir=output_dir,
        output_path=manifest_path,
    )

    first = run_manifest_chunk(manifest_path, 0)
    chunk_path = manifest["chunks"][0]["output_path"]
    second = run_manifest_chunk(manifest_path, 0)

    assert second["elapsed_seconds"] == first["elapsed_seconds"]
    assert read_json_maybe_gzip(chunk_path)["complete"] is True


def test_worker_cli_plan_run_manifest_and_merge_manifest(tmp_path, capsys):
    manifest_path = tmp_path / "manifest.json"
    merge_path = tmp_path / "merged.json"
    output_dir = tmp_path / "chunks"

    assert main(
        [
            "plan",
            "--prime",
            "101",
            "--method",
            "synthetic",
            "--max-rows",
            "8",
            "--max-columns",
            "3",
            "--chunk-size",
            "2",
            "--output-dir",
            str(output_dir),
            "--output",
            str(manifest_path),
        ]
    ) == 0
    assert main(["run-manifest", str(manifest_path), "--task-id", "0"]) == 0
    assert main(["run-manifest", str(manifest_path), "--task-id", "1"]) == 0
    assert main(["merge-manifest", str(manifest_path), "--output", str(merge_path)]) == 0

    captured = capsys.readouterr()
    assert "c18_all_a_chunk_manifest" in captured.out
    payload = json.loads(merge_path.read_text(encoding="utf-8"))
    assert payload["rank"] == 3


def test_bouchet_helper_scripts_exist_and_are_executable():
    root = Path(__file__).resolve().parents[1]
    scripts = [
        root / "scripts/bouchet/run_c18_even_chunk.sh",
        root / "scripts/bouchet/submit_c18_even_array.sbatch",
        root / "scripts/bouchet/merge_c18_even_manifest.sh",
    ]

    for script in scripts:
        assert script.exists()
        assert script.stat().st_mode & 0o111
