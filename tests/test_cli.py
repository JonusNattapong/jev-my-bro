import json
from pathlib import Path

from jevbro import cli
from jevbro.feedback import FeedbackStore


def test_cli_task_start_routes_to_mcp(monkeypatch, capsys) -> None:
    calls = []

    def fake_remote(url, tool, arguments=None):
        calls.append((url, tool, arguments or {}))
        return {"task_id": "jevtask-test", "status": "running"}

    monkeypatch.setattr(cli, "_remote", fake_remote)
    cli.main(
        [
            "--url",
            "http://127.0.0.1:9999/mcp",
            "task",
            "start",
            "Fix a regression",
            "--agent",
            "codex",
            "--repo",
            "demo",
            "--agent-model",
            "gpt-test",
        ]
    )

    assert calls == [
        (
            "http://127.0.0.1:9999/mcp",
            "jev_task_start",
            {
                "context": "Fix a regression",
                "source_agent": "codex",
                "language": None,
                "abstain_threshold": 0.6,
                "session_id": None,
                "repo": "demo",
                "agent_model": "gpt-test",
                "notes": None,
            },
        )
    ]
    assert json.loads(capsys.readouterr().out)["task_id"] == "jevtask-test"


def test_cli_task_complete_and_fail_preserve_verification(monkeypatch, capsys) -> None:
    calls = []

    def fake_remote(url, tool, arguments=None):
        calls.append((tool, arguments or {}))
        return {"task_id": arguments["task_id"], "status": "completed" if tool.endswith("complete") else "failed"}

    monkeypatch.setattr(cli, "_remote", fake_remote)

    cli.main(
        [
            "task",
            "complete",
            "jevtask-1",
            "--choice",
            "minimal_patch",
            "--tests-passed",
            "--test-command",
            "pytest -q",
            "--test-exit-code",
            "0",
            "--changed-file",
            "jevbro/core.py",
            "--duration-ms",
            "123",
        ]
    )
    capsys.readouterr()

    cli.main(
        [
            "task",
            "fail",
            "jevtask-2",
            "--reason",
            "dependency unavailable",
            "--tests-failed",
            "--test-exit-code",
            "1",
        ]
    )
    capsys.readouterr()

    complete_tool, complete_args = calls[0]
    assert complete_tool == "jev_task_complete"
    assert complete_args["tests_passed"] is True
    assert complete_args["test_exit_code"] == 0
    assert complete_args["changed_files"] == ["jevbro/core.py"]
    assert complete_args["duration_ms"] == 123

    fail_tool, fail_args = calls[1]
    assert fail_tool == "jev_task_fail"
    assert fail_args["failure_reason"] == "dependency unavailable"
    assert fail_args["tests_passed"] is False
    assert fail_args["test_exit_code"] == 1


def test_cli_eval_and_export_use_local_store(tmp_path: Path, capsys) -> None:
    database = tmp_path / "feedback.sqlite3"
    report = tmp_path / "evaluation.json"
    export = tmp_path / "feedback.jsonl"

    store = FeedbackStore(database)
    session = store.start_session(
        source_agent="codex",
        task="Completed task",
        feedback_id="jevtask-test",
        task_id="jevtask-test",
    )
    store.complete_session(session["feedback_id"], final_choice="execute", task_success=True)

    cli.main(["eval", "--db", str(database), "--report", str(report)])
    assert report.exists()
    capsys.readouterr()

    cli.main(["export", "--db", str(database), "--output", str(export)])
    output = json.loads(capsys.readouterr().out)
    assert output["records"] == 1
    assert export.exists()
