"""Analyze Fast-Path Rule Engine hit rate against real-world SQLite feedback data."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from jevbro.rules import evaluate_rules, load_rules_config


def analyze_hit_rate(
    db_path: str | Path,
    rules_config: str | Path | None = None,
) -> dict[str, Any]:
    if rules_config:
        load_rules_config(rules_config)
    else:
        load_rules_config()

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT decision_id, context, source_agent, language, raw_decision FROM feedback_records")
    rows = cursor.fetchall()
    conn.close()

    total = len(rows)
    if total == 0:
        return {
            "total_decisions": 0,
            "fast_path_hits": 0,
            "fast_path_hit_rate": 0.0,
            "fall_through": 0,
            "fall_through_rate": 0.0,
            "rule_breakdown": {},
            "top_fall_through_contexts": [],
        }

    fast_path_count = 0
    rule_counts: Counter[str] = Counter()
    fall_through_contexts: list[str] = []

    for row in rows:
        ctx = row["context"] or ""
        match = evaluate_rules(ctx)
        if match is not None:
            fast_path_count += 1
            rule_counts[match.rule_id] += 1
        else:
            fall_through_contexts.append(ctx.strip())

    fall_through_count = total - fast_path_count
    hit_rate = fast_path_count / total
    fall_rate = fall_through_count / total

    top_fall = [
        {"context": ctx, "count": count}
        for ctx, count in Counter(fall_through_contexts).most_common(10)
    ]

    return {
        "total_decisions": total,
        "fast_path_hits": fast_path_count,
        "fast_path_hit_rate": round(hit_rate, 4),
        "fall_through": fall_through_count,
        "fall_through_rate": round(fall_rate, 4),
        "rule_breakdown": dict(rule_counts.most_common()),
        "top_fall_through_contexts": top_fall,
    }


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Analyze Fast-Path Rule hit rate from telemetry")
    parser.add_argument("--db", default="artifacts/feedback/jev_feedback.sqlite3")
    parser.add_argument("--rules-config", help="Optional rules.yaml to test with")
    parser.add_argument("--output", help="Optional path to write JSON report")
    args = parser.parse_args()

    report = analyze_hit_rate(args.db, rules_config=args.rules_config)

    print("=" * 60)
    print(" [REPORT] JEV FAST-PATH RULE ENGINE HIT RATE ANALYSIS")
    print("=" * 60)
    print(f"Total Decisions Analyzed: {report['total_decisions']}")
    print(f"Layer 1 Fast-Path Hits:   {report['fast_path_hits']} ({report['fast_path_hit_rate']*100:.1f}%)")
    print(f"Layer 2 Model Fallback:   {report['fall_through']} ({report['fall_through_rate']*100:.1f}%)")
    print("-" * 60)
    print("Rules Triggered:")
    for rule_id, count in report["rule_breakdown"].items():
        pct = (count / report["total_decisions"]) * 100
        print(f"  - {rule_id:<32} {count:>4} hits ({pct:>5.1f}%)")
    if not report["rule_breakdown"]:
        print("  (No rules matched)")
    print("-" * 60)
    print("Top Recurring Fall-Through Contexts (Candidates for new rules):")
    for item in report["top_fall_through_contexts"][:5]:
        preview = item["context"].replace("\n", " ")[:60]
        print(f"  [{item['count']}x] {preview}")
    print("=" * 60)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nReport exported to: {args.output}")


if __name__ == "__main__":
    main()
