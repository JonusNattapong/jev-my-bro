from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevbro.feedback import DEFAULT_FEEDBACK_DB, FeedbackStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate real coding-agent feedback collected by Jev."
    )
    parser.add_argument("--db", default=str(DEFAULT_FEEDBACK_DB))
    parser.add_argument("--report", default="artifacts/feedback/evaluation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = FeedbackStore(args.db)
    report = store.stats()
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
