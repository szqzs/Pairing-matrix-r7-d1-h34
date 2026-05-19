import json
from pathlib import Path

from jsonschema import Draft202012Validator

from rank7_jk import repro


def test_tree_hash_accepts_project_relative_paths():
    relative = repro.tree_sha256([Path("src/rank7_jk/config.py")])
    absolute = repro.tree_sha256([repro.PROJECT_ROOT / "src/rank7_jk/config.py"])

    assert relative == absolute


def test_gate_b_artifact_validates_against_schema():
    schema_path = repro.PROJECT_ROOT / "schemas/math_gate.schema.json"
    artifact_path = repro.PROJECT_ROOT / "artifacts/math_gates/gate_B_rank5_regression.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
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
    assert artifact["command_transcripts"]
    assert all(item["exit_code"] == 0 for item in artifact["command_transcripts"])


def test_gate_c_artifact_validates_against_schema():
    schema_path = repro.PROJECT_ROOT / "schemas/math_gate.schema.json"
    artifact_path = repro.PROJECT_ROOT / "artifacts/math_gates/gate_C_rank7_smoke.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(artifact)
    assert artifact["schema_sha256"] == repro.file_sha256(schema_path)
    assert artifact["source_tree_sha256"] == repro.default_source_tree_hash()
    assert artifact["source_state"]["git_head"]
    assert artifact["github_repository"] == "https://github.com/szqzs/Pairing-matrix-r7-d1-h34"
    assert artifact["github_remote_head"]
    assert len(artifact["residue_transition_smoke"]) == 5
    assert all(item["passed"] for item in artifact["residue_transition_smoke"])
    for item in artifact["residue_transition_smoke"]:
        assert set(item["observed_mod"]) == {"1000033", "2305843009213693951"}
    assert artifact["command_transcripts"]
    assert all(item["exit_code"] == 0 for item in artifact["command_transcripts"])


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
