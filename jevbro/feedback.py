"""Durable feedback storage and evaluation helpers for Jev coding-agent use."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

DEFAULT_FEEDBACK_DB = Path("artifacts/feedback/jev_feedback.sqlite3")
SCHEMA_VERSION = 2

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|secret)\b"
    r"(\s*[:=]\s*)([\"']?)([^\s\"']+)([\"']?)"
)
_SECRET_TOKEN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{12,}|hf_[A-Za-z0-9]{12,}|gh[pousr]_[A-Za-z0-9]{12,})\b"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_sensitive_text(text: str | None) -> str | None:
    """Redact common credential shapes while preserving useful task context."""

    if text is None:
        return None
    redacted = _SECRET_ASSIGNMENT.sub(
        lambda match: f"{match.group(1)}{match.group(2)}<redacted>",
        text,
    )
    return _SECRET_TOKEN.sub("<redacted>", redacted)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parse_json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    return json.loads(value)


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


class FeedbackStore:
    """SQLite-backed Jev session, decision, and outcome store."""

    def __init__(self, path: str | Path = DEFAULT_FEEDBACK_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS feedback_sessions (
                    feedback_id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    source_agent TEXT NOT NULL,
                    task TEXT NOT NULL,
                    repo TEXT,
                    external_session_id TEXT,
                    task_id TEXT,
                    status TEXT NOT NULL,
                    final_choice TEXT,
                    tests_passed INTEGER,
                    task_success INTEGER,
                    test_command TEXT,
                    test_summary TEXT,
                    implementation_summary TEXT,
                    changed_files_json TEXT,
                    commit_sha TEXT,
                    notes TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_feedback_sessions_status
                    ON feedback_sessions(status);
                CREATE INDEX IF NOT EXISTS idx_feedback_sessions_agent
                    ON feedback_sessions(source_agent);
                CREATE INDEX IF NOT EXISTS idx_feedback_sessions_created
                    ON feedback_sessions(created_at);

                CREATE TABLE IF NOT EXISTS feedback_records (
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
                );

                CREATE INDEX IF NOT EXISTS idx_feedback_source_agent
                    ON feedback_records(source_agent);
                CREATE INDEX IF NOT EXISTS idx_feedback_created_at
                    ON feedback_records(created_at);
                CREATE INDEX IF NOT EXISTS idx_feedback_outcome
                    ON feedback_records(outcome_recorded_at);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(feedback_records)").fetchall()
            }
            if "feedback_id" not in columns:
                connection.execute("ALTER TABLE feedback_records ADD COLUMN feedback_id TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_record_session "
                "ON feedback_records(feedback_id)"
            )
            session_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(feedback_sessions)").fetchall()
            }
            session_migrations = {
                "agent_model": "TEXT",
                "test_exit_code": "INTEGER",
                "duration_ms": "INTEGER",
                "failure_reason": "TEXT",
            }
            for name, sql_type in session_migrations.items():
                if name not in session_columns:
                    connection.execute(
                        f"ALTER TABLE feedback_sessions ADD COLUMN {name} {sql_type}"
                    )

    def start_session(
        self,
        *,
        source_agent: str,
        task: str,
        repo: str | None = None,
        external_session_id: str | None = None,
        task_id: str | None = None,
        feedback_id: str | None = None,
        agent_model: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        identifier = feedback_id or f"jevfb-{uuid4().hex}"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO feedback_sessions (
                    feedback_id, schema_version, created_at, source_agent,
                    task, repo, external_session_id, task_id, status,
                    agent_model, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                """,
                (
                    identifier,
                    SCHEMA_VERSION,
                    utc_now(),
                    source_agent,
                    redact_sensitive_text(task) or "",
                    redact_sensitive_text(repo),
                    redact_sensitive_text(external_session_id),
                    redact_sensitive_text(task_id),
                    redact_sensitive_text(agent_model),
                    redact_sensitive_text(notes),
                ),
            )
        return self.get_session(identifier)

    def complete_session(
        self,
        feedback_id: str,
        *,
        final_choice: str,
        tests_passed: bool | None = None,
        task_success: bool | None = None,
        test_command: str | None = None,
        test_exit_code: int | None = None,
        test_summary: str | None = None,
        implementation_summary: str | None = None,
        changed_files: list[str] | None = None,
        commit_sha: str | None = None,
        duration_ms: int | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        existing = self.get_session(feedback_id, include_decisions=False)
        if existing["status"] != "active":
            if existing["status"] == "completed":
                return self.get_session(feedback_id)
            raise ValueError(
                f"feedback session {feedback_id} is already {existing['status']}"
            )
        safe_changed_files = [
            redact_sensitive_text(path) or ""
            for path in (changed_files or [])
        ]
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE feedback_sessions
                SET completed_at = ?,
                    status = 'completed',
                    final_choice = ?,
                    tests_passed = ?,
                    task_success = ?,
                    test_command = ?,
                    test_exit_code = ?,
                    test_summary = ?,
                    implementation_summary = ?,
                    changed_files_json = ?,
                    commit_sha = ?,
                    duration_ms = ?,
                    failure_reason = NULL,
                    notes = ?
                WHERE feedback_id = ?
                """,
                (
                    utc_now(),
                    redact_sensitive_text(final_choice) or "",
                    None if tests_passed is None else int(tests_passed),
                    None if task_success is None else int(task_success),
                    redact_sensitive_text(test_command),
                    test_exit_code,
                    redact_sensitive_text(test_summary),
                    redact_sensitive_text(implementation_summary),
                    _json(safe_changed_files),
                    redact_sensitive_text(commit_sha),
                    duration_ms,
                    redact_sensitive_text(notes),
                    feedback_id,
                ),
            )
        return self.get_session(feedback_id)

    def fail_session(
        self,
        feedback_id: str,
        *,
        failure_reason: str,
        final_choice: str = "failed",
        tests_passed: bool | None = None,
        test_command: str | None = None,
        test_exit_code: int | None = None,
        test_summary: str | None = None,
        implementation_summary: str | None = None,
        changed_files: list[str] | None = None,
        commit_sha: str | None = None,
        duration_ms: int | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        existing = self.get_session(feedback_id, include_decisions=False)
        if existing["status"] != "active":
            if existing["status"] == "failed":
                return self.get_session(feedback_id)
            raise ValueError(
                f"feedback session {feedback_id} is already {existing['status']}"
            )
        safe_changed_files = [
            redact_sensitive_text(path) or ""
            for path in (changed_files or [])
        ]
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE feedback_sessions
                SET completed_at = ?,
                    status = 'failed',
                    final_choice = ?,
                    tests_passed = ?,
                    task_success = 0,
                    test_command = ?,
                    test_exit_code = ?,
                    test_summary = ?,
                    implementation_summary = ?,
                    changed_files_json = ?,
                    commit_sha = ?,
                    duration_ms = ?,
                    failure_reason = ?,
                    notes = ?
                WHERE feedback_id = ?
                """,
                (
                    utc_now(),
                    redact_sensitive_text(final_choice) or "failed",
                    None if tests_passed is None else int(tests_passed),
                    redact_sensitive_text(test_command),
                    test_exit_code,
                    redact_sensitive_text(test_summary),
                    redact_sensitive_text(implementation_summary),
                    _json(safe_changed_files),
                    redact_sensitive_text(commit_sha),
                    duration_ms,
                    redact_sensitive_text(failure_reason) or "unspecified failure",
                    redact_sensitive_text(notes),
                    feedback_id,
                ),
            )
        return self.get_session(feedback_id)

    def get_session(
        self,
        feedback_id: str,
        *,
        include_decisions: bool = True,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM feedback_sessions WHERE feedback_id = ?",
                (feedback_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown feedback_id: {feedback_id}")
        session = self._session_row_to_dict(row)
        if include_decisions:
            session["decisions"] = self.decisions_for_session(feedback_id)
        return session

    def sessions(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
        include_decisions: bool = False,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM feedback_sessions"
        parameters: list[Any] = []
        if status:
            query += " WHERE status = ?"
            parameters.append(status)
        query += " ORDER BY created_at DESC, feedback_id DESC"
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(max(0, int(limit)))
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        sessions = [self._session_row_to_dict(row) for row in rows]
        if include_decisions:
            for session in sessions:
                session["decisions"] = self.decisions_for_session(session["feedback_id"])
        return sessions

    def record_decision(
        self,
        *,
        decision_id: str,
        source_agent: str,
        context: str,
        language: str | None,
        model_name: str,
        abstain_threshold: float,
        result: dict[str, Any],
        feedback_id: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> None:
        if feedback_id is not None:
            session = self.get_session(feedback_id, include_decisions=False)
            if session["status"] != "active":
                raise ValueError(f"feedback session {feedback_id} is not active")
            if source_agent == "unknown":
                source_agent = session["source_agent"]
            elif source_agent != session["source_agent"]:
                raise ValueError(
                    f"source_agent {source_agent!r} does not match feedback session "
                    f"{session['source_agent']!r}"
                )
            if task_id is None:
                task_id = session.get("task_id")
            if session_id is None:
                session_id = session.get("external_session_id")

        safe_context = redact_sensitive_text(context) or ""
        safe_session = redact_sensitive_text(session_id)
        safe_task = redact_sensitive_text(task_id)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO feedback_records (
                    decision_id, schema_version, created_at, source_agent,
                    session_id, task_id, context, language, model_name,
                    abstain_threshold, promoted_decision, raw_decision,
                    confidence, abstain, needs_review, prohibited, risk,
                    response_json, feedback_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    SCHEMA_VERSION,
                    utc_now(),
                    source_agent,
                    safe_session,
                    safe_task,
                    safe_context,
                    language,
                    model_name,
                    float(abstain_threshold),
                    result.get("decision"),
                    result["raw_decision"],
                    float(result["confidence"]),
                    int(bool(result["abstain"])),
                    result.get("needs_review"),
                    result.get("prohibited"),
                    result.get("risk"),
                    _json(result),
                    feedback_id,
                ),
            )

    def record_outcome(
        self,
        decision_id: str,
        *,
        agent_choice: str,
        tests_passed: bool | None = None,
        task_success: bool | None = None,
        test_command: str | None = None,
        test_summary: str | None = None,
        implementation_summary: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Legacy per-decision outcome API retained for backward compatibility."""

        fields = {
            "agent_choice": redact_sensitive_text(agent_choice) or "",
            "tests_passed": None if tests_passed is None else int(tests_passed),
            "task_success": None if task_success is None else int(task_success),
            "test_command": redact_sensitive_text(test_command),
            "test_summary": redact_sensitive_text(test_summary),
            "implementation_summary": redact_sensitive_text(implementation_summary),
            "notes": redact_sensitive_text(notes),
        }
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE feedback_records
                SET outcome_recorded_at = ?,
                    agent_choice = ?,
                    tests_passed = ?,
                    task_success = ?,
                    test_command = ?,
                    test_summary = ?,
                    implementation_summary = ?,
                    notes = ?
                WHERE decision_id = ?
                """,
                (
                    utc_now(),
                    fields["agent_choice"],
                    fields["tests_passed"],
                    fields["task_success"],
                    fields["test_command"],
                    fields["test_summary"],
                    fields["implementation_summary"],
                    fields["notes"],
                    decision_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown decision_id: {decision_id}")
        return self.get(decision_id)

    def get(self, decision_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM feedback_records WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown decision_id: {decision_id}")
        return self._row_to_dict(row)

    def decisions_for_session(self, feedback_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM feedback_records
                WHERE feedback_id = ?
                ORDER BY created_at, decision_id
                """,
                (feedback_id,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def records(self, *, include_pending: bool = True) -> list[dict[str, Any]]:
        query = "SELECT * FROM feedback_records"
        if not include_pending:
            query += " WHERE outcome_recorded_at IS NOT NULL"
        query += " ORDER BY created_at, decision_id"
        with self._connect() as connection:
            rows = connection.execute(query).fetchall()
        return [self._row_to_dict(row) for row in rows]

    @staticmethod
    def _rate(items: Iterable[dict[str, Any]], predicate) -> float | None:
        materialized = list(items)
        if not materialized:
            return None
        return round(
            sum(1 for item in materialized if predicate(item)) / len(materialized),
            4,
        )

    def stats(self) -> dict[str, Any]:
        sessions = self.sessions(include_decisions=False)
        active_sessions = [row for row in sessions if row["status"] == "active"]
        completed_sessions = [row for row in sessions if row["status"] == "completed"]
        failed_sessions = [row for row in sessions if row["status"] == "failed"]
        finished_sessions = completed_sessions + failed_sessions
        tested_sessions = [
            row for row in finished_sessions if row["tests_passed"] is not None
        ]
        successful_sessions = [
            row for row in finished_sessions if row["task_success"] is not None
        ]

        records = self.records(include_pending=True)
        completed_records = [
            row for row in records if row["outcome_recorded_at"] is not None
        ]
        tested_records = [
            row for row in completed_records if row["tests_passed"] is not None
        ]
        comparable_labels = {"execute", "ask_user", "reject"}
        comparable = [
            row
            for row in completed_records
            if row["agent_choice"] in comparable_labels and row["raw_decision"]
        ]
        agreed = [row for row in comparable if row["agent_choice"] == row["raw_decision"]]
        disagreed = [
            row for row in comparable if row["agent_choice"] != row["raw_decision"]
        ]
        abstained = [row for row in records if row["abstain"]]
        promoted = [row for row in records if not row["abstain"]]

        agents = sorted(
            {row["source_agent"] for row in sessions}
            | {row["source_agent"] for row in records}
        )
        by_agent: dict[str, dict[str, Any]] = {}
        for source_agent in agents:
            agent_sessions = [
                row for row in sessions if row["source_agent"] == source_agent
            ]
            agent_completed = [
                row for row in agent_sessions if row["status"] == "completed"
            ]
            agent_failed = [
                row for row in agent_sessions if row["status"] == "failed"
            ]
            agent_finished = agent_completed + agent_failed
            agent_tested = [
                row for row in agent_finished if row["tests_passed"] is not None
            ]
            agent_decisions = [
                row for row in records if row["source_agent"] == source_agent
            ]
            by_agent[source_agent] = {
                "sessions": len(agent_sessions),
                "active_sessions": len(
                    [row for row in agent_sessions if row["status"] == "active"]
                ),
                "completed_sessions": len(agent_completed),
                "failed_sessions": len(agent_failed),
                "finished_sessions": len(agent_finished),
                "session_completion_rate": round(
                    len(agent_finished) / len(agent_sessions),
                    4,
                )
                if agent_sessions
                else None,
                "session_test_pass_rate": self._rate(
                    agent_tested,
                    lambda row: row["tests_passed"] is True,
                ),
                "decisions": len(agent_decisions),
            }

        total_sessions = len(sessions)
        total_decisions = len(records)
        return {
            "schema_version": SCHEMA_VERSION,
            "database": str(self.path),
            "total_sessions": total_sessions,
            "active_sessions": len(active_sessions),
            "completed_sessions": len(completed_sessions),
            "failed_sessions": len(failed_sessions),
            "finished_sessions": len(finished_sessions),
            "session_completion_rate": round(
                len(finished_sessions) / total_sessions,
                4,
            )
            if total_sessions
            else None,
            "session_test_pass_rate": self._rate(
                tested_sessions,
                lambda row: row["tests_passed"] is True,
            ),
            "session_task_success_rate": self._rate(
                successful_sessions,
                lambda row: row["task_success"] is True,
            ),
            "total_decisions": total_decisions,
            "abstention_rate": self._rate(records, lambda row: row["abstain"]),
            "average_confidence": round(
                sum(row["confidence"] for row in records) / total_decisions,
                4,
            )
            if total_decisions
            else None,
            "completed_decision_outcomes": len(completed_records),
            "decision_test_pass_rate": self._rate(
                tested_records,
                lambda row: row["tests_passed"] is True,
            ),
            "decision_task_success_rate": self._rate(
                [row for row in completed_records if row["task_success"] is not None],
                lambda row: row["task_success"] is True,
            ),
            # Compatibility aliases for pre-session clients.
            "completed_outcomes": len(completed_records),
            "completion_rate": round(
                len(completed_records) / total_decisions,
                4,
            )
            if total_decisions
            else None,
            "tested_outcomes": len(tested_records),
            "test_pass_rate": self._rate(
                tested_records,
                lambda row: row["tests_passed"] is True,
            ),
            "task_success_rate": self._rate(
                [row for row in completed_records if row["task_success"] is not None],
                lambda row: row["task_success"] is True,
            ),
            "agreement_evaluable_count": len(comparable),
            "agreement_count": len(agreed),
            "agreement_rate": self._rate(
                comparable,
                lambda row: row["agent_choice"] == row["raw_decision"],
            ),
            "agreement_test_pass_rate": self._rate(
                [row for row in agreed if row["tests_passed"] is not None],
                lambda row: row["tests_passed"] is True,
            ),
            "disagreement_test_pass_rate": self._rate(
                [row for row in disagreed if row["tests_passed"] is not None],
                lambda row: row["tests_passed"] is True,
            ),
            "abstained_decisions": len(abstained),
            "promoted_decisions": len(promoted),
            "by_agent": by_agent,
        }

    def export_sessions_jsonl(
        self,
        output: str | Path,
        *,
        include_active: bool = False,
    ) -> int:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.sessions(status=None, include_decisions=True)
        if not include_active:
            rows = [row for row in rows if row["status"] != "active"]
        with output_path.open("w", encoding="utf-8") as handle:
            for row in reversed(rows):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        return len(rows)

    def export_jsonl(
        self,
        output: str | Path,
        *,
        include_pending: bool = False,
    ) -> int:
        """Legacy per-decision export retained for compatibility."""

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        records = self.records(include_pending=include_pending)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        return len(records)

    @staticmethod
    def _session_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        session = dict(row)
        session["tests_passed"] = _bool_or_none(session["tests_passed"])
        session["task_success"] = _bool_or_none(session["task_success"])
        session["changed_files"] = _parse_json(
            session.pop("changed_files_json"),
            [],
        )
        return session

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["abstain"] = bool(record["abstain"])
        record["tests_passed"] = _bool_or_none(record["tests_passed"])
        record["task_success"] = _bool_or_none(record["task_success"])
        record["response"] = _parse_json(record.pop("response_json"), {})
        return record
