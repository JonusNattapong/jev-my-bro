"""Extract hard negatives and discrepancies from Jev feedback store into active learning dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from jevbro.active_learning import ActiveLearningHarvester, find_existing_requests
from jevbro.feedback import FeedbackStore


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Extract hard negatives and discrepancy cases from Jev feedback store."
    )
    parser.add_argument(
        "--db",
        default="artifacts/feedback/jev_feedback.sqlite3",
        help="Path to SQLite feedback database",
    )
    parser.add_argument(
        "--output",
        default="data/active_learning/th_active_learning_candidates.csv",
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--report",
        default="artifacts/feedback/active_learning_report.json",
        help="Path to output JSON summary report",
    )
    parser.add_argument(
        "--dedupe-root",
        action="append",
        default=["data"],
        help="Dataset roots to scan for duplicate requests",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable deduplication against existing datasets",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="Starting number for generated case IDs",
    )
    parser.add_argument(
        "--prefix",
        default="th-act-",
        help="Case ID prefix",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Target split for generated cases (train/validation/test)",
    )

    args = parser.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    store = FeedbackStore(db_path)
    harvester = ActiveLearningHarvester(store)

    existing: set[str] = set()
    if not args.no_dedupe:
        print("Scanning existing datasets for deduplication...")
        existing = find_existing_requests(args.dedupe_root)
        print(f"Found {len(existing)} existing request strings.")

    candidates = harvester.harvest(existing_requests=existing)
    print(f"Harvested {len(candidates)} candidate cases from feedback database.")

    type_counts = Counter(c.discrepancy_type for c in candidates)
    domain_counts = Counter(c.domain for c in candidates)
    action_counts = Counter(c.suggested_action for c in candidates)
    risk_counts = Counter(c.suggested_risk for c in candidates)

    print("\nSummary by discrepancy type:")
    for dtype, count in type_counts.items():
        print(f"  - {dtype}: {count}")

    print("\nSummary by suggested action:")
    for action, count in action_counts.items():
        print(f"  - {action}: {count}")

    count_exported = harvester.export_csv(
        candidates,
        args.output,
        start_id=args.start_id,
        prefix=args.prefix,
        split=args.split,
    )
    print(f"\nSaved {count_exported} cases to: {args.output}")

    if args.report:
        report_data = {
            "total_candidates": len(candidates),
            "by_discrepancy_type": dict(type_counts),
            "by_domain": dict(domain_counts),
            "by_suggested_action": dict(action_counts),
            "by_suggested_risk": dict(risk_counts),
            "output_csv": args.output,
            "candidates": [
                {
                    "task_id": c.task_id,
                    "request": c.request,
                    "domain": c.domain,
                    "discrepancy_type": c.discrepancy_type,
                    "model_gated": c.model_gated_decision,
                    "model_prohibited": c.model_prohibited,
                    "suggested_action": c.suggested_action,
                    "suggested_risk": c.suggested_risk,
                    "reason": c.reason,
                }
                for c in candidates
            ],
        }
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved evaluation report to: {args.report}")


if __name__ == "__main__":
    main()
