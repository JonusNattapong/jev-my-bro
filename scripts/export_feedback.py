from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevbro.feedback import DEFAULT_FEEDBACK_DB, FeedbackStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Jev feedback sessions as JSONL for review and v0.3 research."
    )
    parser.add_argument("--db", default=str(DEFAULT_FEEDBACK_DB))
    parser.add_argument(
        "--output",
        default="artifacts/feedback/v0.3-feedback.jsonl",
    )
    parser.add_argument(
        "--include-active",
        action="store_true",
        help="Include active sessions that have not been completed yet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = FeedbackStore(args.db)
    count = store.export_sessions_jsonl(args.output, include_active=args.include_active)
    print(f"exported {count} feedback sessions -> {args.output}")


if __name__ == "__main__":
    main()
