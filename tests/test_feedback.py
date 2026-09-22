import json
import sqlite3
from pathlib import Path

import pytest

from jevbro.feedback import FeedbackStore, SCHEMA_VERSION, redact_sensitive_text


def decision_result(*, confidence: float = 0.8, abstain: bool = False) -> dict:
    return {
        "engine": "jev-my-bro",
        "decision": None if abstain else "ask_user",
        "raw_decision": "ask_user",
        "confidence": confidence,
        "abstain": abstain,
        "abstain_threshold": 0.6,
        "needs_review": 0.9,
        "prohibited": 0.1,
        "risk": 3.0,
        "answers": {"action": {"choice": "ask_user", "confidence": confidence}},
        "usage": {"input_tokens": 10},
        "advisory": True,
    }


def test_session_lifecycle_links_decisions_and_exports(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    session = store.start_session(
        source_agent="codex",
        task="Fix a local regression",
        repo="jev-my-bro",
        task_id="task-1",
    )
    feedback_id = session["feedback_id"]
    assert feedback_id.startswith("jevfb-")
    assert session["status"] == "active"

    store.record_decision(
        decision_id="jev-test-1",
        feedback_id=feedback_id,
        source_agent="codex",
        context="Should this bounded fix execute now?",
        language="en",
        model_name="test-model",
        abstain_threshold=0.6,
        result=decision_result(),
    )

    completed = store.complete_session(
        feedback_id,
        final_choice="minimal_patch",
        tests_passed=True,
        task_success=True,
        test_command="pytest -q",
        test_summary="17 passed",
        implementation_summary="Applied bounded patch.",
        changed_files=["jevbro/feedback.py"],
        commit_sha="abc123",
    )
    assert completed["status"] == "completed"
    assert completed["tests_passed"] is True
    assert completed["task_success"] is True
    assert completed["changed_files"] == ["jevbro/feedback.py"]
    assert len(completed["decisions"]) == 1
    assert completed["decisions"][0]["decision_id"] == "jev-test-1"

    output = tmp_path / "sessions.jsonl"
    assert store.export_sessions_jsonl(output) == 1
    exported = json.loads(output.read_text(encoding="utf-8").strip())
    assert exported["feedback_id"] == feedback_id
    assert exported["final_choice"] == "minimal_patch"
    assert exported["decisions"][0]["raw_decision"] == "ask_user"


def test_stats_separate_session_and_legacy_decision_outcomes(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    session = store.start_session(
        source_agent="claude_code",
        task="Implement feedback lifecycle",
    )
    store.record_decision(
        decision_id="jev-session-decision",
        feedback_id=session["feedback_id"],
        source_agent="claude_code",
        context="Proceed with implementation?",
        language="en",
        model_name="test-model",
        abstain_threshold=0.6,
        result=decision_result(confidence=0.5, abstain=True),
    )
    store.complete_session(
        session["feedback_id"],
        final_choice="implement",
        tests_passed=True,
        task_success=True,
    )

    store.record_decision(
        decision_id="jev-legacy",
        source_agent="codex",
        context="Deploy production",
        language="en",
        model_name="test-model",
        abstain_threshold=0.6,
        result=decision_result(),
    )
    store.record_outcome(
        "jev-legacy",
        agent_choice="ask_user",
        tests_passed=False,
        task_success=False,
    )

    stats = store.stats()
    assert stats["total_sessions"] == 1
    assert stats["completed_sessions"] == 1
    assert stats["session_test_pass_rate"] == 1.0
    assert stats["total_decisions"] == 2
    assert stats["completed_decision_outcomes"] == 1
    assert stats["decision_test_pass_rate"] == 0.0
    assert stats["agreement_rate"] == 1.0
    assert stats["by_agent"]["claude_code"]["completed_sessions"] == 1


def test_active_session_rejects_mismatched_agent_and_completed_session_decision(
    tmp_path: Path,
) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    session = store.start_session(source_agent="opencode", task="Task")

    with pytest.raises(ValueError):
        store.record_decision(
            decision_id="jev-wrong-agent",
            feedback_id=session["feedback_id"],
            source_agent="codex",
            context="Question",
            language="en",
            model_name="test-model",
            abstain_threshold=0.6,
            result=decision_result(),
        )

    store.complete_session(session["feedback_id"], final_choice="done")

    with pytest.raises(ValueError):
        store.record_decision(
            decision_id="jev-after-complete",
            feedback_id=session["feedback_id"],
            source_agent="opencode",
            context="Question",
            language="en",
            model_name="test-model",
            abstain_threshold=0.6,
            result=decision_result(),
        )


def test_feedback_redacts_common_secret_shapes_in_sessions_and_decisions(
    tmp_path: Path,
) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    session = store.start_session(
        source_agent="opencode",
        task="api_key=abc123456789 token: supersecret123",
        repo="hf_abcdefghijklmnop",
    )
    stored_session = store.get_session(session["feedback_id"], include_decisions=False)
    assert "abc123456789" not in stored_session["task"]
    assert "supersecret123" not in stored_session["task"]
    assert "hf_abcdefghijklmnop" not in (stored_session["repo"] or "")

    store.record_decision(
        decision_id="jev-secret",
        feedback_id=session["feedback_id"],
        source_agent="opencode",
        context="password=verysecret token: anothersecret",
        language="en",
        model_name="test-model",
        abstain_threshold=0.6,
        result=decision_result(),
    )
    context = store.get("jev-secret")["context"]
    assert "verysecret" not in context
    assert "anothersecret" not in context
    assert "<redacted>" in context


def test_unknown_ids_and_retried_terminal_updates_are_safe(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    with pytest.raises(KeyError):
        store.get_session("missing")
    with pytest.raises(KeyError):
        store.record_outcome("missing", agent_choice="execute")

    session = store.start_session(source_agent="codex", task="Task")
    first = store.complete_session(session["feedback_id"], final_choice="done")
    retried = store.complete_session(session["feedback_id"], final_choice="again")
    assert first["status"] == "completed"
    assert retried["status"] == "completed"
    assert retried["final_choice"] == "done"

    failed = store.start_session(source_agent="codex", task="Blocked")
    first_failure = store.fail_session(
        failed["feedback_id"],
        failure_reason="dependency unavailable",
        test_exit_code=1,
        duration_ms=123,
    )
    retried_failure = store.fail_session(
        failed["feedback_id"],
        failure_reason="different retry message",
    )
    assert first_failure["status"] == "failed"
    assert retried_failure["status"] == "failed"
    assert retried_failure["failure_reason"] == "dependency unavailable"
    assert retried_failure["test_exit_code"] == 1
    assert retried_failure["duration_ms"] == 123


def test_v1_database_migrates_feedback_id_column_and_sessions_table(tmp_path: Path) -> None:
    database = tmp_path / "feedback.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE feedback_records (
                decision_id TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                source_agent TEXT NOT NULL,
                session_id TEXT,
                task_id TEXT,
                context TEXT NOT NULL,
                language TEXT,
                model_name TEXT NOT NULL,
                abstain_threshold REAL NOT NULL,
                promoted_decision TEXT,
                raw_decision TEXT NOT NULL,
                confidence REAL NOT NULL,
                abstain INTEGER NOT NULL,
                needs_review REAL,
                prohibited REAL,
                risk REAL,
                response_json TEXT NOT NULL,
                outcome_recorded_at TEXT,
                agent_choice TEXT,
                tests_passed INTEGER,
                task_success INTEGER,
                test_command TEXT,
                test_summary TEXT,
                implementation_summary TEXT,
                notes TEXT
            )
            """
        )

    store = FeedbackStore(database)
    with sqlite3.connect(database) as connection:
        record_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(feedback_records)").fetchall()
        }
        session_table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='feedback_sessions'"
        ).fetchone()
    assert "feedback_id" in record_columns
    assert session_table is not None
    with sqlite3.connect(database) as connection:
        session_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(feedback_sessions)").fetchall()
        }
    assert {"agent_model", "test_exit_code", "duration_ms", "failure_reason"} <= session_columns

    session = store.start_session(
        source_agent="codex",
        task="Migrated",
        agent_model="gpt-test",
    )
    assert session["schema_version"] == SCHEMA_VERSION
    assert session["agent_model"] == "gpt-test"


def test_redact_sensitive_text_handles_none() -> None:
    assert redact_sensitive_text(None) is None
