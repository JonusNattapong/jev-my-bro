import asyncio
from pathlib import Path

from mcp import Client

from jevbro.feedback import FeedbackStore
from jevbro.mcp_server import create_mcp_server


class FakeAgent:
    def predict(self, state, questions):
        return {
            "answers": {
                "action": {"choice": "ask_user", "confidence": 0.8},
                "needs_review": {"noul": 0.9},
                "prohibited": {"noul": 0.1},
                "risk": {"score": 3.0},
            },
            "usage": {"input_tokens": 10},
        }


def make_server(tmp_path: Path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    return create_mcp_server(FakeAgent(), feedback_store=store), store


def test_mcp_lists_task_and_compatibility_tools(tmp_path: Path) -> None:
    async def run() -> None:
        server, _ = make_server(tmp_path)
        async with Client(server, raise_exceptions=True) as client:
            result = await client.list_tools()
            names = {tool.name for tool in result.tools}
            assert names == {
                "jev_decide",
                "jev_feedback_complete",
                "jev_feedback_get",
                "jev_feedback_start",
                "jev_feedback_stats",
                "jev_health",
                "jev_model_info",
                "jev_predict",
                "jev_record_outcome",
                "jev_task_complete",
                "jev_task_fail",
                "jev_task_list",
                "jev_task_start",
                "jev_task_status",
            }

    asyncio.run(run())


def test_task_lifecycle_start_decide_complete_and_list(tmp_path: Path) -> None:
    async def run() -> None:
        server, store = make_server(tmp_path)
        async with Client(server, raise_exceptions=True) as client:
            started = await client.call_tool(
                "jev_task_start",
                {
                    "context": "Patch and verify a local regression",
                    "source_agent": "claude_code",
                    "repo": "jev-my-bro",
                    "agent_model": "claude-test",
                },
            )
            task = started.structured_content
            task_id = task["task_id"]
            assert task_id.startswith("jevtask-")
            assert task["status"] == "running"
            assert task["decision_id"].startswith("jev-")
            assert task["source_agent"] == "claude_code"

            extra = await client.call_tool(
                "jev_decide",
                {
                    "task_id": task_id,
                    "context": "Should this bounded patch execute now?",
                },
            )
            assert extra.structured_content["task_id"] == task_id
            assert extra.structured_content["source_agent"] == "claude_code"

            completed = await client.call_tool(
                "jev_task_complete",
                {
                    "task_id": task_id,
                    "agent_choice": "minimal_patch",
                    "tests_passed": True,
                    "test_command": "pytest -q",
                    "test_exit_code": 0,
                    "test_summary": "all passed",
                    "implementation_summary": "Applied bounded patch.",
                    "changed_files": ["jevbro/core.py"],
                    "git_commit": "abc123",
                    "duration_ms": 1250,
                },
            )
            result = completed.structured_content
            assert result["status"] == "completed"
            assert result["tests_passed"] is True
            assert result["task_success"] is True
            assert result["test_exit_code"] == 0
            assert result["duration_ms"] == 1250
            assert len(result["decisions"]) == 2

            retried = await client.call_tool(
                "jev_task_complete",
                {"task_id": task_id, "agent_choice": "different-retry"},
            )
            assert retried.structured_content["final_choice"] == "minimal_patch"

            fetched = await client.call_tool(
                "jev_task_status",
                {"task_id": task_id},
            )
            assert fetched.structured_content["status"] == "completed"
            assert fetched.structured_content["agent_model"] == "claude-test"

            listing = await client.call_tool(
                "jev_task_list",
                {"status": "completed", "source_agent": "claude_code"},
            )
            assert listing.structured_content["count"] == 1
            assert listing.structured_content["items"][0]["task_id"] == task_id

            stored = store.get_session(task_id)
            assert stored["feedback_id"] == task_id
            assert stored["task_id"] == task_id

    asyncio.run(run())


def test_task_fail_is_persisted_and_retry_safe(tmp_path: Path) -> None:
    async def run() -> None:
        server, _ = make_server(tmp_path)
        async with Client(server, raise_exceptions=True) as client:
            started = await client.call_tool(
                "jev_task_start",
                {
                    "context": "Run a build that depends on a missing SDK",
                    "source_agent": "codex",
                },
            )
            task_id = started.structured_content["task_id"]

            failed = await client.call_tool(
                "jev_task_fail",
                {
                    "task_id": task_id,
                    "failure_reason": "SDK unavailable",
                    "tests_passed": False,
                    "test_command": "build.cmd",
                    "test_exit_code": 1,
                    "duration_ms": 500,
                },
            )
            result = failed.structured_content
            assert result["status"] == "failed"
            assert result["task_success"] is False
            assert result["failure_reason"] == "SDK unavailable"
            assert result["test_exit_code"] == 1

            retried = await client.call_tool(
                "jev_task_fail",
                {
                    "task_id": task_id,
                    "failure_reason": "retry should not overwrite",
                },
            )
            assert retried.structured_content["failure_reason"] == "SDK unavailable"

            listing = await client.call_tool(
                "jev_task_list",
                {"status": "failed"},
            )
            assert listing.structured_content["count"] == 1
            assert listing.structured_content["items"][0]["task_id"] == task_id

    asyncio.run(run())


def test_full_legacy_feedback_lifecycle_remains_compatible(tmp_path: Path) -> None:
    async def run() -> None:
        server, store = make_server(tmp_path)
        async with Client(server, raise_exceptions=True) as client:
            started = await client.call_tool(
                "jev_feedback_start",
                {
                    "task": "Patch and verify a local regression",
                    "source_agent": "opencode",
                    "repo": "jev-my-bro",
                },
            )
            feedback_id = started.structured_content["feedback_id"]

            decision = await client.call_tool(
                "jev_decide",
                {
                    "feedback_id": feedback_id,
                    "context": "Should this bounded patch execute now?",
                    "abstain_threshold": 0.6,
                },
            )
            assert decision.structured_content["feedback_id"] == feedback_id
            assert decision.structured_content["source_agent"] == "opencode"

            completed = await client.call_tool(
                "jev_feedback_complete",
                {
                    "feedback_id": feedback_id,
                    "final_choice": "minimal_patch",
                    "tests_passed": True,
                    "task_success": True,
                },
            )
            assert completed.structured_content["status"] == "completed"
            assert completed.structured_content["decision_count"] == 1

            fetched = await client.call_tool(
                "jev_feedback_get",
                {"feedback_id": feedback_id},
            )
            assert fetched.structured_content["final_choice"] == "minimal_patch"
            assert len(fetched.structured_content["decisions"]) == 1

            stored = store.get_session(feedback_id)
            assert stored["source_agent"] == "opencode"

    asyncio.run(run())


def test_standalone_decide_and_legacy_outcome_remain_available(tmp_path: Path) -> None:
    async def run() -> None:
        server, _ = make_server(tmp_path)
        async with Client(server, raise_exceptions=True) as client:
            decision = await client.call_tool(
                "jev_decide",
                {
                    "context": "Deploy the service to production",
                    "source_agent": "codex",
                },
            )
            content = decision.structured_content
            assert content["feedback_id"] is None
            assert content["decision_id"].startswith("jev-")

            outcome = await client.call_tool(
                "jev_record_outcome",
                {
                    "decision_id": content["decision_id"],
                    "agent_choice": "ask_user",
                    "tests_passed": True,
                    "task_success": True,
                },
            )
            assert outcome.structured_content["tests_passed"] is True

    asyncio.run(run())
