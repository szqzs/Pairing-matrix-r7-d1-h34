import json
from pathlib import Path

from jsonschema import Draft202012Validator

from rank7_jk import repro


def test_tree_hash_accepts_project_relative_paths():
    relative = repro.tree_sha256([Path("src/rank7_jk/config.py")])
    absolute = repro.tree_sha256([repro.PROJECT_ROOT / "src/rank7_jk/config.py"])

    assert relative == absolute


def test_gate_b_artifact_validates_against_schema(tmp_path):
    schema_path = repro.PROJECT_ROOT / "schemas/math_gate.schema.json"
    artifact_path = tmp_path / "gate_B_rank5_regression.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    repro.write_json(artifact_path, repro.gate_b_payload())
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(artifact)
    assert artifact["schema_sha256"] == repro.file_sha256(schema_path)
    assert artifact["source_tree_sha256"] == repro.default_source_tree_hash()
    assert artifact["source_state"]["git_head"]
    assert "schemas/math_gate.schema.json" in _tree_members_for_default_hash()
    assert len(artifact["rank5_regression"]["scalar_results"]) == 5
    assert len(artifact["rank5_regression"]["public_minor_summaries"]) == 11
    assert len(artifact["rank5_regression"]["executable_minor_results"]) == 4
    assert all(item["passed"] for item in artifact["rank5_regression"]["scalar_results"])
    assert len(artifact["old_rank5_repo_reference"]["certificate_sha256"]) == 11
    assert len(artifact["old_rank5_repo_reference"]["computed_columns_sha256"]) == 11
    assert artifact["old_rank5_repo_reference"]["summary_sha256"]


def test_gate_c_artifact_validates_against_schema(tmp_path):
    schema_path = repro.PROJECT_ROOT / "schemas/math_gate.schema.json"
    artifact_path = tmp_path / "gate_C_rank7_smoke.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    repro.write_json(artifact_path, repro.gate_c_payload())
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(artifact)
    assert artifact["schema_sha256"] == repro.file_sha256(schema_path)
    assert artifact["source_tree_sha256"] == repro.default_source_tree_hash()
    assert artifact["source_state"]["git_head"]
    assert artifact["github_repository"] == "https://github.com/szqzs/Pairing-matrix-r7-d1-h34"
    assert len(artifact["residue_transition_smoke"]) == 5
    assert all(item["passed"] for item in artifact["residue_transition_smoke"])
    for item in artifact["residue_transition_smoke"]:
        assert set(item["observed_mod"]) == {"1000033", "2305843009213693951"}


def test_command_transcript_capture_records_exit_code_and_hashes():
    transcripts = repro.capture_command_transcripts(
        ["python -c 'print(123)'"],
        capture=True,
    )

    assert len(transcripts) == 1
    assert transcripts[0]["exit_code"] == 0
    assert transcripts[0]["stdout_bytes"] == 4
    assert len(transcripts[0]["stdout_sha256"]) == 64
    assert len(transcripts[0]["stderr_sha256"]) == 64


def _tree_members_for_default_hash():
    members = set()
    for base in [
        repro.PROJECT_ROOT / "src",
        repro.PROJECT_ROOT / "tests",
        repro.PROJECT_ROOT / "schemas",
    ]:
        for item in base.rglob("*"):
            if item.is_file() and "__pycache__" not in item.parts and item.name != ".DS_Store":
                members.add(item.relative_to(repro.PROJECT_ROOT).as_posix())
    return members
