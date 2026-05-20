import json
from pathlib import Path

from rank7_jk.c18_block_semantic_table import (
    merge_block_key_manifest_outputs,
    run_block_key_manifest_chunk,
)
from rank7_jk_fast.semantic_pipeline import (
    assemble_rank_with_plateau,
    main,
    plan_fast_manifest,
    plan_scout_suite,
)


def _synthetic_one_f_table(tmp_path, *, max_rows=4, max_columns=8):
    manifest_path = tmp_path / "manifest.json"
    table_path = tmp_path / "table.json"
    manifest = plan_fast_manifest(
        output_path=manifest_path,
        output_dir=tmp_path / "chunks",
        prime=101,
        method="synthetic",
        row_kind="gamma",
        column_kind="one-f",
        max_rows=max_rows,
        max_columns=max_columns,
        row_order="sequential",
        column_order="balanced",
        chunk_size=100,
    )
    assert manifest["chunk_count"] == 1
    run_block_key_manifest_chunk(manifest_path, 0)
    merge_block_key_manifest_outputs(manifest_path, output_path=table_path)
    return table_path


def test_fast_rank_assembler_can_stop_after_dependent_plateau(tmp_path):
    table_path = _synthetic_one_f_table(tmp_path)

    full = assemble_rank_with_plateau(table_path, max_dependent_columns=None)
    plateau = assemble_rank_with_plateau(table_path, max_dependent_columns=1)

    assert full["stop_reason"] == "exhausted_columns"
    assert full["processed_columns"] == 8
    assert full["rank"] == 4
    assert plateau["stop_reason"] == "max_dependent_columns"
    assert plateau["rank"] == full["rank"]
    assert plateau["processed_columns"] == 5
    assert plateau["dependent_columns_since_rank_gain"] == 1


def test_fast_rank_assembler_can_store_left_nullspace_only_when_requested(tmp_path):
    table_path = _synthetic_one_f_table(tmp_path, max_rows=4, max_columns=4)

    without = assemble_rank_with_plateau(table_path)
    with_nullspace = assemble_rank_with_plateau(
        table_path,
        compute_left_nullspace=True,
    )

    assert without["rank"] == 4
    assert without["left_nullity"] == 0
    assert without["left_nullspace"] is None
    assert with_nullspace["left_nullspace"] == []


def test_fast_cli_run_chunk_reports_manifest_output_path(tmp_path, capsys):
    manifest_path = tmp_path / "manifest.json"
    output_dir = tmp_path / "chunks"

    assert main(
        [
            "plan",
            "--prime",
            "101",
            "--method",
            "synthetic",
            "--row-kind",
            "gamma",
            "--column-kind",
            "one-f",
            "--max-rows",
            "2",
            "--max-columns",
            "2",
            "--chunk-size",
            "10",
            "--output-dir",
            str(output_dir),
            "--output",
            str(manifest_path),
        ]
    ) == 0
    plan_summary = json.loads(capsys.readouterr().out)
    assert plan_summary["column_order"] == "balanced"

    assert main(["run-chunk", str(manifest_path), "--task-id", "0"]) == 0
    summary = json.loads(capsys.readouterr().out)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert summary["kind"] == "c18_block_semantic_value_chunk"
    assert summary["complete"] is True
    assert summary["output_path"] == manifest["chunks"][0]["output_path"]


def test_fast_plan_scout_suite_keeps_default_blocks_explicit(tmp_path):
    payload = plan_scout_suite(
        output_root=tmp_path / "scouts",
        prime=101,
        method="synthetic",
        chunk_size=100,
    )

    names = [item["path"].split("/")[-2] for item in payload["manifests"]]
    assert payload["kind"] == "fast_c18_scout_suite"
    assert names == [
        "all_f2_balanced_128",
        "gamma_f2_balanced_256",
        "even_one_gamma_128",
        "gamma_one_f_128",
        "even_b_pair_64",
    ]
    assert all(item["kind"] == "c18_block_semantic_key_manifest" for item in payload["manifests"])


def test_fast_console_script_is_registered():
    root = Path(__file__).resolve().parents[1]
    pyproject_text = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert (
        'rank7-c18-fast-semantic = "rank7_jk_fast.semantic_pipeline:main"'
        in pyproject_text
    )


def test_fast_bouchet_helper_scripts_exist_and_are_executable():
    root = Path(__file__).resolve().parents[1]
    scripts = [
        root / "scripts/bouchet/plan_c18_fast_scout_suite.sh",
        root / "scripts/bouchet/submit_c18_fast_scout.sbatch",
        root / "scripts/bouchet/assemble_c18_fast_scout.sh",
    ]

    for script in scripts:
        assert script.exists()
        assert script.stat().st_mode & 0o111
