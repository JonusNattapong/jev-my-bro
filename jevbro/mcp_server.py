"""MCP adapter for Jev decision support and coding-task lifecycle."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import laya
from mcp.server import MCPServer

from jevbro.core import JevCore, SourceAgent
from jevbro.feedback import DEFAULT_FEEDBACK_DB, FeedbackStore

SERVER_INSTRUCTIONS = """Jev is an advisory decision model and coding-task telemetry layer.
For non-trivial coding work, call jev_task_start before implementation and keep
its task_id. Perform repository work and verification with the coding agent's
normal tools. Before the final user response, call jev_task_complete with actual
verification evidence, or jev_task_fail when the task is blocked/failed.
Use jev_decide for additional bounded decisions inside a task. Jev never edits
files, runs shell commands, or replaces repository policy or human approvals.
The older jev_feedback_* tools remain available for compatibility."""


def create_mcp_server(
    agent: Any,
    *,
    model_name: str = "jev-my-bro-v0.2",
    feedback_store: FeedbackStore | None = None,
) -> MCPServer:
    store = feedback_store or FeedbackStore(DEFAULT_FEEDBACK_DB)
    core = JevCore(agent, store, model_name=model_name)
    mcp = MCPServer("jev-my-bro", instructions=SERVER_INSTRUCTIONS)

    @mcp.tool()
    def jev_health() -> dict[str, Any]:
        """Check model, MCP adapter, feedback store, and task lifecycle readiness."""
        return core.health()

    @mcp.tool()
    def jev_model_info() -> dict[str, Any]:
        """Describe Jev's decision and task-lifecycle capabilities."""
        return core.model_info()

    @mcp.tool()
    def jev_task_start(
        context: str,
        source_agent: str = "unknown",
        language: Literal["en", "th"] | None = None,
        abstain_threshold: float = 0.60,
        prohibited_threshold: float | None = None,
        review_threshold: float | None = None,
        session_id: str | None = None,
        repo: str | None = None,
        agent_model: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Start a tracked coding task and return task_id plus Jev's initial decision."""
        return core.task_start(
            context,
            source_agent=source_agent,
            language=language,
            abstain_threshold=abstain_threshold,
            prohibited_threshold=prohibited_threshold,
            review_threshold=review_threshold,
            session_id=session_id,
            repo=repo,
            agent_model=agent_model,
            notes=notes,
        )

    @mcp.tool()
    def jev_task_complete(
        task_id: str,
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
        """Complete a tracked task with observed verification evidence."""
        return core.task_complete(
            task_id,
            agent_choice=agent_choice,
            tests_passed=tests_passed,
            test_command=test_command,
            test_exit_code=test_exit_code,
            test_summary=test_summary,
            implementation_summary=implementation_summary,
            changed_files=changed_files,
            git_commit=git_commit,
            duration_ms=duration_ms,
            notes=notes,
        )

    @mcp.tool()
    def jev_task_fail(
        task_id: str,
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
        """Fail/block a tracked task while preserving failure and test evidence."""
        return core.task_fail(
            task_id,
            failure_reason=failure_reason,
            agent_choice=agent_choice,
            tests_passed=tests_passed,
            test_command=test_command,
            test_exit_code=test_exit_code,
            test_summary=test_summary,
            implementation_summary=implementation_summary,
            changed_files=changed_files,
            git_commit=git_commit,
            duration_ms=duration_ms,
            notes=notes,
        )

    @mcp.tool()
    def jev_task_status(task_id: str) -> dict[str, Any]:
        """Inspect one tracked task and its attached Jev decisions."""
        return core.task_status(task_id)

    @mcp.tool()
    def jev_task_list(
        status: Literal["running", "completed", "failed"] | None = None,
        source_agent: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List tracked tasks, optionally filtered by lifecycle state or agent."""
        items = core.task_list(status=status, source_agent=source_agent, limit=limit)
        return {"count": len(items), "items": items}

    @mcp.tool()
    def jev_decide(
        context: str,
        task_id: str | None = None,
        feedback_id: str | None = None,
        language: Literal["en", "th"] | None = None,
        abstain_threshold: float = 0.60,
        prohibited_threshold: float | None = None,
        review_threshold: float | None = None,
        source_agent: str = "unknown",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist one advisory decision, optionally attached to a tracked task."""
        return core.decide(
            context,
            task_id=task_id,
            feedback_id=feedback_id,
            language=language,
            abstain_threshold=abstain_threshold,
            prohibited_threshold=prohibited_threshold,
            review_threshold=review_threshold,
            source_agent=source_agent,
            session_id=session_id,
        )

    @mcp.tool()
    def jev_feedback_start(
        task: str,
        source_agent: str = "unknown",
        repo: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Compatibility: start the older feedback-session lifecycle."""
        session = core.feedback_start(
            task=task,
            source_agent=source_agent,
            repo=repo,
            session_id=session_id,
            task_id=task_id,
        )
        return {
            "ok": True,
            "feedback_id": session["feedback_id"],
            "status": session["status"],
            "source_agent": session["source_agent"],
            "created_at": session["created_at"],
        }

    @mcp.tool()
    def jev_feedback_complete(
        feedback_id: str,
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
        """Compatibility: complete the older feedback-session lifecycle."""
        session = core.feedback_complete(
            feedback_id,
            final_choice=final_choice,
            tests_passed=tests_passed,
            task_success=task_success,
            test_command=test_command,
            test_exit_code=test_exit_code,
            test_summary=test_summary,
            implementation_summary=implementation_summary,
            changed_files=changed_files,
            commit_sha=commit_sha,
            duration_ms=duration_ms,
            notes=notes,
        )
        return {
            "ok": True,
            "feedback_id": feedback_id,
            "status": session["status"],
            "source_agent": session["source_agent"],
            "final_choice": session["final_choice"],
            "tests_passed": session["tests_passed"],
            "task_success": session["task_success"],
            "decision_count": len(session.get("decisions", [])),
            "completed_at": session["completed_at"],
        }

    @mcp.tool()
    def jev_feedback_get(feedback_id: str) -> dict[str, Any]:
        """Compatibility: inspect one older feedback session."""
        return core.feedback_get(feedback_id)

    @mcp.tool()
    def jev_record_outcome(
        decision_id: str,
        agent_choice: str,
        tests_passed: bool | None = None,
        task_success: bool | None = None,
        test_command: str | None = None,
        test_summary: str | None = None,
        implementation_summary: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Compatibility: attach an outcome to a standalone jev_decide call."""
        record = core.record_outcome(
            decision_id,
            agent_choice=agent_choice,
            tests_passed=tests_passed,
            task_success=task_success,
            test_command=test_command,
            test_summary=test_summary,
            implementation_summary=implementation_summary,
            notes=notes,
        )
        return {
            "ok": True,
            "decision_id": decision_id,
            "source_agent": record["source_agent"],
            "raw_decision": record["raw_decision"],
            "agent_choice": record["agent_choice"],
            "tests_passed": record["tests_passed"],
            "task_success": record["task_success"],
            "outcome_recorded_at": record["outcome_recorded_at"],
        }

    @mcp.tool()
    def jev_feedback_stats() -> dict[str, Any]:
        """Return aggregate real-use decision and task metrics."""
        return core.feedback_stats()

    @mcp.tool()
    def jev_predict(
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Run lower-level typed Laya questions without lifecycle semantics."""
        return core.predict(state, questions)

    return mcp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve jev-my-bro over MCP")
    parser.add_argument("--model", default="JonusNattapong/jev-my-bro-th1200")
    parser.add_argument("--device")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="streamable-http",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--path", default="/mcp")
    parser.add_argument("--feedback-db", default=str(DEFAULT_FEEDBACK_DB))
    parser.add_argument("--quantize", action="store_true", help="Enable INT8 quantization on CPU")
    parser.add_argument("--rules-config", help="Path to rules.yaml or rules.json config file")
    return parser.parse_args()


def run_server(
    *,
    model: str = "JonusNattapong/jev-my-bro-th1200",
    device: str | None = None,
    transport: str = "streamable-http",
    host: str = "127.0.0.1",
    port: int = 8787,
    path: str = "/mcp",
    feedback_db: str | Path = DEFAULT_FEEDBACK_DB,
    quantize: bool = False,
    rules_config: str | Path | None = None,
) -> None:
    from jevbro.rules import load_rules_config

    # Auto-load rules config if provided or present in working directory
    load_rules_config(rules_config)

    agent = laya.Agent(model, device=device)

    # Optional dynamic INT8 quantization for CPU acceleration
    if quantize:
        try:
            import torch

            if hasattr(agent, "model") and hasattr(agent.model, "encoder"):
                agent.model.encoder = torch.quantization.quantize_dynamic(
                    agent.model.encoder, {torch.nn.Linear}, dtype=torch.qint8
                )
        except Exception:
            pass  # Gracefully fall back to standard model

    store = FeedbackStore(feedback_db)
    mcp = create_mcp_server(agent, model_name=model, feedback_store=store)
    if transport == "stdio":
        mcp.run(transport="stdio")
        return
    mcp.run(
        transport="streamable-http",
        host=host,
        port=port,
        streamable_http_path=path,
        stateless_http=True,
        json_response=True,
    )


def main() -> None:
    args = parse_args()
    run_server(
        model=args.model,
        device=args.device,
        transport=args.transport,
        host=args.host,
        port=args.port,
        path=args.path,
        feedback_db=args.feedback_db,
        quantize=args.quantize,
        rules_config=args.rules_config,
    )


if __name__ == "__main__":
    main()
