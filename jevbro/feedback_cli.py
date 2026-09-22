from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevbro.feedback import DEFAULT_FEEDBACK_DB, FeedbackStore


def _parse_tristate(value: str) -> bool | None:
    normalized = value.lower()
    if normalized in {"true", "pass", "passed", "success", "yes"}:
        return True
    if normalized in {"false", "fail", "failed", "error", "no"}:
        return False
    if normalized in {"null", "none", "unknown", "not-run"}:
        return None
    raise argparse.ArgumentTypeError(f"invalid tri-state value: {value}")


def _print_json(value) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jev-feedback")
    parser.add_argument("--db", default=str(DEFAULT_FEEDBACK_DB))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status")
    sub.add_parser("stats")

    list_parser = sub.add_parser("list")
    list_parser.add_argument("--status", choices=("active", "completed"))
    list_parser.add_argument("--limit", type=int, default=20)

    show_parser = sub.add_parser("show")
    show_parser.add_argument("feedback_id")

    start = sub.add_parser("start")
    start.add_argument("--agent", required=True, choices=("claude_code", "codex", "opencode", "unknown"))
    start.add_argument("--task", required=True)
    start.add_argument("--repo")
    start.add_argument("--session-id")
    start.add_argument("--task-id")

    complete = sub.add_parser("complete")
    complete.add_argument("feedback_id")
    complete.add_argument("--choice", required=True)
    complete.add_argument("--tests", type=_parse_tristate, default=None)
    complete.add_argument("--task-success", type=_parse_tristate, default=None)
    complete.add_argument("--test-command")
    complete.add_argument("--test-summary")
    complete.add_argument("--summary")
    complete.add_argument("--changed-file", action="append", default=[])
    complete.add_argument("--commit")
    complete.add_argument("--notes")

    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--report", default="artifacts/feedback/evaluation.json")

    export = sub.add_parser("export")
    export.add_argument("--output", default="artifacts/feedback/v0.3-feedback.jsonl")
    export.add_argument("--include-active", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    store = FeedbackStore(args.db)

    if args.command in {"status", "stats"}:
        _print_json(store.stats())
        return
    if args.command == "list":
        _print_json(store.sessions(status=args.status, limit=args.limit, include_decisions=False))
        return
    if args.command == "show":
        _print_json(store.get_session(args.feedback_id))
        return
    if args.command == "start":
        _print_json(store.start_session(
            source_agent=args.agent,
            task=args.task,
            repo=args.repo,
            external_session_id=args.session_id,
            task_id=args.task_id,
        ))
        return
    if args.command == "complete":
        _print_json(store.complete_session(
            args.feedback_id,
            final_choice=args.choice,
            tests_passed=args.tests,
            task_success=args.task_success,
            test_command=args.test_command,
            test_summary=args.test_summary,
            implementation_summary=args.summary,
            changed_files=args.changed_file,
            commit_sha=args.commit,
            notes=args.notes,
        ))
        return
    if args.command == "eval":
        report = store.stats()
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        _print_json(report)
        return
    if args.command == "export":
        count = store.export_sessions_jsonl(args.output, include_active=args.include_active)
        print(f"exported {count} feedback sessions -> {args.output}")
        return


if __name__ == "__main__":
    main()
