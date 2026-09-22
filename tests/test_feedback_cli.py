import json
import subprocess
import sys
from pathlib import Path


def run_cli(root: Path, database: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "jevbro.feedback_cli",
            "--db",
            str(database),
            *args,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_feedback_cli_full_lifecycle(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    database = tmp_path / "feedback.sqlite3"

    started = run_cli(
        root,
        database,
        "start",
        "--agent",
        "codex",
        "--task",
        "Implement feedback CLI",
        "--repo",
        "jev-my-bro",
    )
    session = json.loads(started.stdout)
    feedback_id = session["feedback_id"]
    assert session["status"] == "active"

    listed = json.loads(run_cli(root, database, "list", "--status", "active").stdout)
    assert listed[0]["feedback_id"] == feedback_id

    shown = json.loads(run_cli(root, database, "show", feedback_id).stdout)
    assert shown["task"] == "Implement feedback CLI"

    completed = json.loads(
        run_cli(
            root,
            database,
            "complete",
            feedback_id,
            "--choice",
            "implement",
            "--tests",
            "passed",
            "--task-success",
            "success",
            "--test-command",
            "pytest -q",
            "--test-summary",
            "tests passed",
            "--summary",
            "Implemented lifecycle CLI.",
            "--changed-file",
            "jevbro/feedback_cli.py",
            "--commit",
            "abc123",
        ).stdout
    )
    assert completed["status"] == "completed"
    assert completed["tests_passed"] is True
    assert completed["task_success"] is True

    stats = json.loads(run_cli(root, database, "stats").stdout)
    assert stats["total_sessions"] == 1
    assert stats["completed_sessions"] == 1
    assert stats["session_test_pass_rate"] == 1.0

    report = tmp_path / "evaluation.json"
    run_cli(root, database, "eval", "--report", str(report))
    assert json.loads(report.read_text(encoding="utf-8"))["total_sessions"] == 1

    export = tmp_path / "feedback.jsonl"
    run_cli(root, database, "export", "--output", str(export))
    row = json.loads(export.read_text(encoding="utf-8").strip())
    assert row["feedback_id"] == feedback_id
    assert row["final_choice"] == "implement"


def test_root_windows_wrapper_invokes_cli(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    database = tmp_path / "wrapper.sqlite3"
    result = subprocess.run(
        [
            "cmd",
            "/c",
            str(root / "jev-feedback.cmd"),
            "--db",
            str(database),
            "status",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    stats = json.loads(result.stdout)
    assert stats["total_sessions"] == 0
    assert stats["total_decisions"] == 0


def test_legacy_scripts_evaluate_and_session_export(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    database = tmp_path / "feedback.sqlite3"
    report = tmp_path / "evaluation.json"
    export = tmp_path / "feedback.jsonl"

    start = json.loads(
        run_cli(
            root,
            database,
            "start",
            "--agent",
            "claude_code",
            "--task",
            "Verify scripts",
        ).stdout
    )
    run_cli(
        root,
        database,
        "complete",
        start["feedback_id"],
        "--choice",
        "done",
        "--tests",
        "passed",
        "--task-success",
        "success",
    )

    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "evaluate_feedback.py"),
            "--db",
            str(database),
            "--report",
            str(report),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "export_feedback.py"),
            "--db",
            str(database),
            "--output",
            str(export),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(report.read_text(encoding="utf-8"))
    assert metrics["total_sessions"] == 1
    assert metrics["session_test_pass_rate"] == 1.0
    rows = export.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["feedback_id"] == start["feedback_id"]
