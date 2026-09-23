"""Transport-independent Jev decision and coding-task lifecycle core."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from jevbro.feedback import FeedbackStore
from jevbro.questions import detect_question_language
from jevbro.router import DecideRequest, decision_response

SourceAgent = Literal["claude_code", "codex", "opencode", "unknown"]


def validate_threshold(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError("abstain_threshold must be between 0 and 1")
    return value


class JevCore:
    """Shared Jev business logic used by MCP and CLI-facing workflows."""

    def __init__(self, agent: Any, feedback_store: FeedbackStore, *, model_name: str) -> None:
        self.agent = agent
        self.store = feedback_store
        self.model_name = model_name

    def _evaluate(
        self,
        context: str,
        *,
        language: Literal["en", "th"] | None,
        abstain_threshold: float,
        source_agent: SourceAgent,
    ) -> tuple[dict[str, Any], str, float]:
        threshold = validate_threshold(abstain_threshold)
        resolved_language = language or detect_question_language(context)
        base = decision_response(
            self.agent,
            DecideRequest(context=context, language=resolved_language),
        )
        raw_decision = base["decision"]
        confidence = float(base["confidence"])
        abstain = confidence < threshold
        return (
            {
                **base,
                "source_agent": source_agent,
                "decision": None if abstain else raw_decision,
                "raw_decision": raw_decision,
                "abstain": abstain,
                "abstain_threshold": threshold,
                "advisory": True,
            },
            resolved_language,
            threshold,
        )

    def health(self) -> dict[str, Any]:
        stats = self.store.stats()
        return {
            "ok": True,
            "engine": "jev-my-bro",
            "model": self.model_name,
            "runtime": "native-laya",
            "advisory": True,
            "feedback": {
                "enabled": True,
                "database": str(self.store.path),
                "total_tasks": stats.get("total_sessions", 0),
                "running_tasks": stats.get("active_sessions", 0),
                "completed_tasks": stats.get("completed_sessions", 0),
                "failed_tasks": stats.get("failed_sessions", 0),
                "total_sessions": stats.get("total_sessions", 0),
                "active_sessions": stats.get("active_sessions", 0),
                "completed_sessions": stats.get("completed_sessions", 0),
                "failed_sessions": stats.get("failed_sessions", 0),
                "total_decisions": stats.get("total_decisions", 0),
            },
        }

    def model_info(self) -> dict[str, Any]:
        return {
            "engine": "jev-my-bro",
            "model": self.model_name,
            "primitives": ["choice", "noul", "score"],
            "task_lifecycle": ["start", "complete", "fail", "status", "list"],
            "advisory": True,
            "authorization_control": False,
        }

    def decide(
        self,
        context: str,
        *,
        language: Literal["en", "th"] | None = None,
        abstain_threshold: float = 0.60,
        source_agent: SourceAgent = "unknown",
        session_id: str | None = None,
        task_id: str | None = None,
        feedback_id: str | None = None,
    ) -> dict[str, Any]:
        correlation_id = feedback_id or task_id
        if correlation_id is not None:
            session = self.store.get_session(correlation_id, include_decisions=False)
            if source_agent == "unknown":
                source_agent = session["source_agent"]
            if session_id is None:
                session_id = session.get("external_session_id")
            if task_id is None:
                task_id = session.get("task_id")

        result, resolved_language, threshold = self._evaluate(
            context,
            language=language,
            abstain_threshold=abstain_threshold,
            source_agent=source_agent,
        )
        decision_id = f"jev-{uuid4().hex}"
        result.update(
            {
                "decision_id": decision_id,
                "task_id": task_id,
                "feedback_id": correlation_id,
            }
        )
        self.store.record_decision(
            decision_id=decision_id,
            feedback_id=correlation_id,
            source_agent=source_agent,
            context=context,
            language=resolved_language,
            model_name=self.model_name,
            abstain_threshold=threshold,
            result=result,
            session_id=session_id,
            task_id=task_id,
        )
        return result

    def task_start(
        self,
        context: str,
        *,
        source_agent: SourceAgent = "unknown",
        language: Literal["en", "th"] | None = None,
        abstain_threshold: float = 0.60,
        session_id: str | None = None,
        repo: str | None = None,
        agent_model: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        result, resolved_language, threshold = self._evaluate(
            context,
            language=language,
            abstain_threshold=abstain_threshold,
            source_agent=source_agent,
        )
        task_id = f"jevtask-{uuid4().hex}"
        decision_id = f"jev-{uuid4().hex}"
        result.update(
            {
                "decision_id": decision_id,
                "task_id": task_id,
                "feedback_id": task_id,
            }
        )
        session = self.store.start_session(
            source_agent=source_agent,
            task=context,
            repo=repo,
            external_session_id=session_id,
            task_id=task_id,
            feedback_id=task_id,
            agent_model=agent_model,
            notes=notes,
        )
        self.store.record_decision(
            decision_id=decision_id,
            feedback_id=task_id,
            source_agent=source_agent,
            context=context,
            language=resolved_language,
            model_name=self.model_name,
            abstain_threshold=threshold,
            result=result,
            session_id=session_id,
            task_id=task_id,
        )
        return {
            "task_id": task_id,
            "status": "running",
            "decision_id": decision_id,
            "source_agent": source_agent,
            "repo": session.get("repo"),
            "agent_model": session.get("agent_model"),
            "decision": result["decision"],
            "raw_decision": result["raw_decision"],
            "gated_decision": result["gated_decision"],
            "confidence": result["confidence"],
            "abstain": result["abstain"],
            "needs_review": result.get("needs_review"),
            "prohibited": result.get("prohibited"),
            "risk": result.get("risk"),
            "advisory": True,
        }

    def task_complete(
        self,
        task_id: str,
        *,
        agent_choice: str,
        tests_passed: bool | None = None,
        test_command: str | None = None,
        test_exit_code: int | None = None,
        test_summary: str | None = None,
        implementation_summary: str | None = None,
        changed_files: list[str] | None = None,
        git_commit: str | None = None,
        duration_ms: int | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        session = self.store.complete_session(
            task_id,
            final_choice=agent_choice,
            tests_passed=tests_passed,
            task_success=True,
            test_command=test_command,
            test_exit_code=test_exit_code,
            test_summary=test_summary,
            implementation_summary=implementation_summary,
            changed_files=changed_files,
            commit_sha=git_commit,
            duration_ms=duration_ms,
            notes=notes,
        )
        return self._task_view(session)

    def task_fail(
        self,
        task_id: str,
        *,
        failure_reason: str,
        agent_choice: str = "failed",
        tests_passed: bool | None = None,
        test_command: str | None = None,
        test_exit_code: int | None = None,
        test_summary: str | None = None,
        implementation_summary: str | None = None,
        changed_files: list[str] | None = None,
        git_commit: str | None = None,
        duration_ms: int | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        session = self.store.fail_session(
            task_id,
            failure_reason=failure_reason,
            final_choice=agent_choice,
            tests_passed=tests_passed,
            test_command=test_command,
            test_exit_code=test_exit_code,
            test_summary=test_summary,
            implementation_summary=implementation_summary,
            changed_files=changed_files,
            commit_sha=git_commit,
            duration_ms=duration_ms,
            notes=notes,
        )
        return self._task_view(session)

    def task_status(self, task_id: str) -> dict[str, Any]:
        return self._task_view(self.store.get_session(task_id))

    def task_list(
        self,
        *,
        status: Literal["running", "completed", "failed"] | None = None,
        source_agent: SourceAgent | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        storage_status = "active" if status == "running" else status
        sessions = self.store.sessions(
            status=storage_status,
            limit=max(1, min(limit, 500)),
            include_decisions=False,
        )
        if source_agent is not None:
            sessions = [row for row in sessions if row["source_agent"] == source_agent]
        return [self._task_view(row) for row in sessions]

    def feedback_start(
        self,
        *,
        task: str,
        source_agent: SourceAgent,
        repo: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        return self.store.start_session(
            source_agent=source_agent,
            task=task,
            repo=repo,
            external_session_id=session_id,
            task_id=task_id,
        )

    def feedback_complete(self, feedback_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.store.complete_session(feedback_id, **kwargs)

    def feedback_get(self, feedback_id: str) -> dict[str, Any]:
        return self.store.get_session(feedback_id)

    def record_outcome(self, decision_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.store.record_outcome(decision_id, **kwargs)

    def feedback_stats(self) -> dict[str, Any]:
        return self.store.stats()

    def predict(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        return self.agent.predict(state, questions)

    @staticmethod
    def _task_view(session: dict[str, Any]) -> dict[str, Any]:
        status = "running" if session["status"] == "active" else session["status"]
        return {
            **session,
            "feedback_id": session["feedback_id"],
            "task_id": session.get("task_id") or session["feedback_id"],
            "status": status,
        }
