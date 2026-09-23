"""CLI for Jev task lifecycle, MCP access, evaluation, and export."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_MCP_URL = "http://127.0.0.1:8787/mcp"


def _print(value: Any) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


async def _call_remote(url: str, tool: str, arguments: dict[str, Any]) -> Any:
    from mcp import Client

    async with Client(url) as client:
        result = await client.call_tool(tool, arguments)
        if result.is_error:
            message = "MCP tool call failed"
            if result.content:
                texts = [getattr(item, "text", "") for item in result.content]
                message = " ".join(text for text in texts if text).strip() or message
            raise RuntimeError(message)
        return result.structured_content


def _remote(url: str, tool: str, arguments: dict[str, Any] | None = None) -> Any:
    return asyncio.run(_call_remote(url, tool, arguments or {}))


def _tests_value(args: argparse.Namespace) -> bool | None:
    if getattr(args, "tests_passed", False):
        return True
    if getattr(args, "tests_failed", False):
        return False
    return None


def _add_verification_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tests-passed", action="store_true")
    group.add_argument("--tests-failed", action="store_true")
    parser.add_argument("--test-command")
    parser.add_argument("--test-exit-code", type=int)
    parser.add_argument("--test-summary")
    parser.add_argument("--implementation-summary")
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--git-commit")
    parser.add_argument("--duration-ms", type=int)
    parser.add_argument("--notes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jev", description="Jev task lifecycle CLI")
    parser.add_argument("--url", default=DEFAULT_MCP_URL, help="Jev MCP URL")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("health", help="Check the running Jev MCP service")
    sub.add_parser("stats", help="Show feedback/task metrics")

    decide = sub.add_parser("decide", help="Record a standalone Jev decision")
    decide.add_argument("context")
    decide.add_argument(
        "--agent",
        default="unknown",
        help="Source agent identifier (e.g. claude_code, antigravity, cursor, codex, opencode)",
    )
    decide.add_argument("--language", choices=["en", "th"])
    decide.add_argument("--threshold", type=float, default=0.60)
    decide.add_argument("--prohibited-threshold", type=float)
    decide.add_argument("--review-threshold", type=float)
    decide.add_argument("--session-id")
    decide.add_argument("--task-id")

    task = sub.add_parser("task", help="Tracked coding-task lifecycle")
    task_sub = task.add_subparsers(dest="task_command", required=True)

    start = task_sub.add_parser("start", help="Start a tracked coding task")
    start.add_argument("context")
    start.add_argument(
        "--agent",
        default="unknown",
        help="Source agent identifier (e.g. claude_code, antigravity, cursor, codex, opencode)",
    )
    start.add_argument("--language", choices=["en", "th"])
    start.add_argument("--threshold", type=float, default=0.60)
    start.add_argument("--prohibited-threshold", type=float)
    start.add_argument("--review-threshold", type=float)
    start.add_argument("--session-id")
    start.add_argument("--repo")
    start.add_argument("--agent-model")
    start.add_argument("--notes")

    complete = task_sub.add_parser("complete", help="Complete a tracked coding task")
    complete.add_argument("task_id")
    complete.add_argument("--choice", required=True)
    _add_verification_args(complete)

    fail = task_sub.add_parser("fail", help="Fail/block a tracked coding task")
    fail.add_argument("task_id")
    fail.add_argument("--reason", required=True)
    fail.add_argument("--choice", default="failed")
    _add_verification_args(fail)

    status = task_sub.add_parser("status", help="Show one tracked task")
    status.add_argument("task_id")

    listing = task_sub.add_parser("list", help="List recent tracked tasks")
    listing.add_argument("--status", choices=["running", "completed", "failed"])
    listing.add_argument(
        "--agent",
        help="Source agent identifier filter",
    )
    listing.add_argument("--limit", type=int, default=50)

    serve = sub.add_parser("serve", help="Run the Jev MCP server")
    serve.add_argument("--model", default="JonusNattapong/jev-my-bro-th1200")
    serve.add_argument("--device")
    serve.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="streamable-http",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--path", default="/mcp")
    serve.add_argument("--feedback-db", default="artifacts/feedback/jev_feedback.sqlite3")
    serve.add_argument("--quantize", action="store_true", help="Enable dynamic INT8 quantization on CPU")
    serve.add_argument("--rules-config", help="Path to rules.yaml or rules.json config file")

    evaluate = sub.add_parser("eval", help="Write aggregate feedback metrics")
    evaluate.add_argument("--db", default="artifacts/feedback/jev_feedback.sqlite3")
    evaluate.add_argument("--report", default="artifacts/feedback/evaluation.json")

    export = sub.add_parser("export", help="Export task feedback JSONL")
    export.add_argument("--db", default="artifacts/feedback/jev_feedback.sqlite3")
    export.add_argument("--output", default="artifacts/feedback/v0.3-feedback.jsonl")
    export.add_argument("--include-running", action="store_true")

    harvest = sub.add_parser("harvest", help="Extract hard negatives and discrepancies for active learning")
    harvest.add_argument("--db", default="artifacts/feedback/jev_feedback.sqlite3")
    harvest.add_argument("--output", default="data/active_learning/candidates.csv")
    harvest.add_argument("--report", default="artifacts/feedback/active_learning_report.json")
    harvest.add_argument("--no-dedupe", action="store_true")
    return parser


def _verification_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "tests_passed": _tests_value(args),
        "test_command": args.test_command,
        "test_exit_code": args.test_exit_code,
        "test_summary": args.test_summary,
        "implementation_summary": args.implementation_summary,
        "changed_files": args.changed_file or None,
        "git_commit": args.git_commit,
        "duration_ms": args.duration_ms,
        "notes": args.notes,
    }


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        from jevbro.mcp_server import run_server

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
        return

    if args.command == "eval":
        from jevbro.feedback import FeedbackStore

        store = FeedbackStore(args.db)
        report = store.stats()
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _print(report)
        return

    if args.command == "export":
        from jevbro.feedback import FeedbackStore

        store = FeedbackStore(args.db)
        count = store.export_sessions_jsonl(
            args.output,
            include_active=args.include_running,
        )
        _print({"ok": True, "records": count, "output": args.output})
        return

    if args.command == "harvest":
        from jevbro.active_learning import ActiveLearningHarvester, find_existing_requests
        from jevbro.feedback import FeedbackStore

        store = FeedbackStore(args.db)
        harvester = ActiveLearningHarvester(store)
        existing = set() if args.no_dedupe else find_existing_requests(["data"])
        candidates = harvester.harvest(existing_requests=existing)
        count = harvester.export_csv(candidates, args.output)
        _print({
            "ok": True,
            "candidates_harvested": len(candidates),
            "records_exported": count,
            "output": args.output,
        })
        return

    if args.command == "health":
        _print(_remote(args.url, "jev_health"))
        return

    if args.command == "stats":
        _print(_remote(args.url, "jev_feedback_stats"))
        return

    if args.command == "decide":
        payload = {
            "context": args.context,
            "source_agent": args.agent,
            "language": args.language,
            "abstain_threshold": args.threshold,
            "session_id": args.session_id,
            "task_id": args.task_id,
        }
        if args.prohibited_threshold is not None:
            payload["prohibited_threshold"] = args.prohibited_threshold
        if args.review_threshold is not None:
            payload["review_threshold"] = args.review_threshold
        _print(_remote(args.url, "jev_decide", payload))
        return

    if args.command == "task" and args.task_command == "start":
        payload = {
            "context": args.context,
            "source_agent": args.agent,
            "language": args.language,
            "abstain_threshold": args.threshold,
            "session_id": args.session_id,
            "repo": args.repo,
            "agent_model": args.agent_model,
            "notes": args.notes,
        }
        if args.prohibited_threshold is not None:
            payload["prohibited_threshold"] = args.prohibited_threshold
        if args.review_threshold is not None:
            payload["review_threshold"] = args.review_threshold
        _print(_remote(args.url, "jev_task_start", payload))
        return

    if args.command == "task" and args.task_command == "complete":
        payload = _verification_payload(args)
        payload.update({"task_id": args.task_id, "agent_choice": args.choice})
        _print(_remote(args.url, "jev_task_complete", payload))
        return

    if args.command == "task" and args.task_command == "fail":
        payload = _verification_payload(args)
        payload.update(
            {
                "task_id": args.task_id,
                "failure_reason": args.reason,
                "agent_choice": args.choice,
            }
        )
        _print(_remote(args.url, "jev_task_fail", payload))
        return

    if args.command == "task" and args.task_command == "status":
        _print(_remote(args.url, "jev_task_status", {"task_id": args.task_id}))
        return

    if args.command == "task" and args.task_command == "list":
        _print(
            _remote(
                args.url,
                "jev_task_list",
                {
                    "status": args.status,
                    "source_agent": args.agent,
                    "limit": args.limit,
                },
            )
        )
        return

    parser.error("unsupported command")


if __name__ == "__main__":
    main()
