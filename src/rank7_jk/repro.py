"""Reproducibility helpers for math gates and future run manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Sequence

import sympy as sp

from .checks import run_structural_checks
from .config import RANK7_H34_H62
from .rank5_regression import (
    RANK5_C20_MINOR_FIXTURE,
    RANK5_FORMULA,
    RANK5_PUBLIC_MINOR_SUMMARIES,
    RANK5_PUBLIC_SCALAR_FIXTURES,
    RANK5_SMALL_PUBLIC_MINOR_FIXTURES,
)
from .rank7_smoke import run_residue_smoke_cases
from . import slow_evaluator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "schemas/math_gate.schema.json"
OLD_RANK5_REPO = Path("/tmp/codex-repo-study-pairing-matrix")
RANK7_GITHUB_URL = "https://github.com/szqzs/Pairing-matrix-r7-d1-h34"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _resolve_tree_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _tree_item_label(item: Path) -> str:
    try:
        return item.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return item.as_posix()


def tree_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(_resolve_tree_path(path) for path in paths):
        if path.is_dir():
            files = sorted(item for item in path.rglob("*") if item.is_file())
        elif path.is_file():
            files = [path]
        else:
            continue
        for item in files:
            if "__pycache__" in item.parts or item.name == ".DS_Store":
                continue
            rel = _tree_item_label(item)
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            digest.update(file_sha256(item).encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


def _git_head(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return None
    if result.returncode:
        return None
    return result.stdout.strip() or None


def _git_dirty(path: Path) -> bool | None:
    if not path.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--short"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return None
    if result.returncode:
        return None
    return bool(result.stdout.strip())


def _git_remote_head(url: str, ref: str = "refs/heads/main") -> str | None:
    try:
        result = subprocess.run(
            ["git", "ls-remote", url, ref],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    pieces = result.stdout.split()
    return pieces[0] if pieces else None


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment_payload() -> dict[str, Any]:
    return {
        "executable": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "sympy": sp.__version__,
        "pytest": _package_version("pytest"),
        "jsonschema": _package_version("jsonschema"),
    }


def old_rank5_certificate_hashes() -> dict[str, str]:
    if not OLD_RANK5_REPO.exists():
        return {}
    out: dict[str, str] = {}
    for item in sorted(OLD_RANK5_REPO.glob("c*/certificate.json")):
        out[item.relative_to(OLD_RANK5_REPO).as_posix()] = file_sha256(item)
    return out


def old_rank5_computed_column_hashes() -> dict[str, str]:
    if not OLD_RANK5_REPO.exists():
        return {}
    out: dict[str, str] = {}
    for item in sorted(OLD_RANK5_REPO.glob("c*/computed_columns_mod_p.json.gz")):
        out[item.relative_to(OLD_RANK5_REPO).as_posix()] = file_sha256(item)
    return out


def old_rank5_summary_hash() -> str | None:
    summary = OLD_RANK5_REPO / "summary.json"
    return file_sha256(summary) if summary.exists() else None


def default_source_tree_hash() -> str:
    return tree_sha256(
        [
            PROJECT_ROOT / "src",
            PROJECT_ROOT / "tests",
            PROJECT_ROOT / "schemas",
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "pyproject.toml",
            PROJECT_ROOT / ".gitignore",
            PROJECT_ROOT / "OVERALL_PLAN.txt",
            PROJECT_ROOT / "STEP1_MATH_VERIFICATION_PLAN.txt",
        ]
    )


def capture_command_transcripts(
    commands: Iterable[str],
    *,
    capture: bool,
) -> list[dict[str, Any]]:
    if not capture:
        return []
    transcripts = []
    for command in commands:
        start = time.perf_counter()
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        duration = time.perf_counter() - start
        transcripts.append(
            {
                "command": command,
                "exit_code": result.returncode,
                "stdout_sha256": bytes_sha256(result.stdout),
                "stderr_sha256": bytes_sha256(result.stderr),
                "stdout_bytes": len(result.stdout),
                "stderr_bytes": len(result.stderr),
                "duration_seconds": round(duration, 3),
            }
        )
    return transcripts


def transcripts_passed(transcripts: Sequence[dict[str, Any]]) -> bool:
    return all(int(item["exit_code"]) == 0 for item in transcripts)


def gate_b_payload(*, capture_transcripts: bool = False) -> dict[str, Any]:
    scalar_results = []
    for fixture in RANK5_PUBLIC_SCALAR_FIXTURES:
        observed = slow_evaluator.pairing_mod_prime(
            RANK5_FORMULA,
            fixture.left,
            fixture.right,
            prime=fixture.prime,
        )
        scalar_results.append(
            {
                "name": fixture.name,
                "left": fixture.left_name,
                "right": fixture.right_name,
                "prime": fixture.prime,
                "expected_mod": fixture.expected_mod,
                "observed_mod": observed,
                "passed": observed == fixture.expected_mod,
            }
        )

    minor_results = []
    for minor in RANK5_SMALL_PUBLIC_MINOR_FIXTURES:
        matrix = slow_evaluator.pairing_matrix_mod_prime(
            RANK5_FORMULA,
            minor.rows,
            minor.columns,
            prime=minor.prime,
        )
        observed_det = slow_evaluator.determinant_mod(matrix, minor.prime)
        minor_results.append(
            {
                "name": minor.name,
                "chern_degree": minor.chern_degree,
                "rows": list(minor.row_names),
                "columns": list(minor.column_names),
                "prime": minor.prime,
                "expected_det_mod": minor.expected_det_mod,
                "observed_det_mod": observed_det,
                "passed": observed_det == minor.expected_det_mod,
                "matrix_mod_prime": [list(row) for row in matrix],
            }
        )

    structural_checks = run_structural_checks(problem=RANK7_H34_H62)
    source_tree_hash = default_source_tree_hash()
    commands = [
        "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q "
        "-p no:cacheprovider tests/test_rank5_regression_scaffold.py "
        "tests/test_residue_oracle.py tests/test_formula_identities.py",
        "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m rank7_jk.checks",
    ]
    command_transcripts = capture_command_transcripts(
        commands,
        capture=capture_transcripts,
    )
    all_passed = (
        all(item["passed"] for item in scalar_results)
        and all(item["passed"] for item in minor_results)
        and structural_checks["status"] == "passed"
        and transcripts_passed(command_transcripts)
    )

    return {
        "schema_version": 1,
        "gate": "Gate B: rank-5 public regression",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all_passed else "failed",
        "project_root": str(PROJECT_ROOT),
        "source_tree_sha256": source_tree_hash,
        "source_state": {
            "git_head": _git_head(PROJECT_ROOT),
            "git_dirty": _git_dirty(PROJECT_ROOT),
        },
        "schema_sha256": file_sha256(SCHEMA_PATH) if SCHEMA_PATH.exists() else None,
        "environment": environment_payload(),
        "commands": commands,
        "command_transcripts": command_transcripts,
        "rank7_problem": {
            "rank": RANK7_H34_H62.formula.rank,
            "genus": RANK7_H34_H62.formula.genus,
            "determinant_degree": RANK7_H34_H62.formula.determinant_degree,
            "source_degree": RANK7_H34_H62.source_degree,
            "test_degree": RANK7_H34_H62.test_degree,
            "expected_relation_chern_degree": RANK7_H34_H62.expected_relation_chern_degree,
            "top_degree": RANK7_H34_H62.formula.top_degree,
        },
        "rank5_regression": {
            "rank": RANK5_FORMULA.rank,
            "genus": RANK5_FORMULA.genus,
            "top_degree": RANK5_FORMULA.top_degree,
            "scalar_results": scalar_results,
            "public_minor_summaries": [
                {
                    "chern_degree": item.chern_degree,
                    "source_dimension": item.source_dimension,
                    "rank": item.rank,
                    "prime": item.prime,
                    "expected_det_mod": item.expected_det_mod,
                }
                for item in RANK5_PUBLIC_MINOR_SUMMARIES
            ],
            "executable_minor_results": minor_results,
            "c20_minor": next(
                item for item in minor_results if item["name"] == RANK5_C20_MINOR_FIXTURE.name
            ),
        },
        "structural_checks": structural_checks,
        "old_rank5_repo_reference": {
            "path": str(OLD_RANK5_REPO),
            "git_head": _git_head(OLD_RANK5_REPO),
            "git_dirty": _git_dirty(OLD_RANK5_REPO),
            "usage": "read-only convention study; no import dependency",
            "certificate_sha256": old_rank5_certificate_hashes(),
            "computed_columns_sha256": old_rank5_computed_column_hashes(),
            "summary_sha256": old_rank5_summary_hash(),
        },
    }


def gate_c_payload(*, capture_transcripts: bool = False) -> dict[str, Any]:
    source_tree_hash = default_source_tree_hash()
    smoke_results = run_residue_smoke_cases()
    commands = [
        "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q "
        "-p no:cacheprovider tests/test_gate_c_residue_transition.py",
    ]
    command_transcripts = capture_command_transcripts(
        commands,
        capture=capture_transcripts,
    )
    all_passed = all(bool(item["passed"]) for item in smoke_results) and transcripts_passed(
        command_transcripts
    )
    return {
        "schema_version": 1,
        "gate": "Gate C: rank-7 residue-transition smoke",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all_passed else "failed",
        "project_root": str(PROJECT_ROOT),
        "github_repository": RANK7_GITHUB_URL,
        "github_remote_head": _git_remote_head(RANK7_GITHUB_URL + ".git"),
        "source_tree_sha256": source_tree_hash,
        "source_state": {
            "git_head": _git_head(PROJECT_ROOT),
            "git_dirty": _git_dirty(PROJECT_ROOT),
        },
        "schema_sha256": file_sha256(SCHEMA_PATH) if SCHEMA_PATH.exists() else None,
        "environment": environment_payload(),
        "commands": commands,
        "command_transcripts": command_transcripts,
        "rank7_problem": {
            "rank": RANK7_H34_H62.formula.rank,
            "genus": RANK7_H34_H62.formula.genus,
            "determinant_degree": RANK7_H34_H62.formula.determinant_degree,
            "source_degree": RANK7_H34_H62.source_degree,
            "test_degree": RANK7_H34_H62.test_degree,
            "expected_relation_chern_degree": RANK7_H34_H62.expected_relation_chern_degree,
            "top_degree": RANK7_H34_H62.formula.top_degree,
        },
        "residue_transition_smoke": smoke_results,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gate_b = sub.add_parser("gate-b", help="write the Gate B math-gate artifact")
    gate_b.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/math_gates/gate_B_rank5_regression.json",
    )

    gate_c = sub.add_parser("gate-c", help="write the Gate C rank-7 smoke artifact")
    gate_c.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/math_gates/gate_C_rank7_smoke.json",
    )

    tree_hash = sub.add_parser("tree-hash", help="print the current source tree hash")
    tree_hash.add_argument("paths", nargs="*", type=Path)

    args = parser.parse_args(argv)
    if args.command == "gate-b":
        payload = gate_b_payload(capture_transcripts=True)
        write_json(args.output, payload)
        print(json.dumps({"output": str(args.output), "status": payload["status"]}))
    elif args.command == "gate-c":
        payload = gate_c_payload(capture_transcripts=True)
        write_json(args.output, payload)
        print(json.dumps({"output": str(args.output), "status": payload["status"]}))
    elif args.command == "tree-hash":
        paths = args.paths or [
            PROJECT_ROOT / "src",
            PROJECT_ROOT / "tests",
            PROJECT_ROOT / "schemas",
        ]
        print(tree_sha256(paths))


if __name__ == "__main__":
    main()
