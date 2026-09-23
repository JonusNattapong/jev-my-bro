"""Build the Thai 1,200-case set with long coding-agent task-context contrasts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from build_th_curated_500 import read_rows
from csv_to_dataset import build_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/th_curated_1200"))
    args = parser.parse_args()

    sources = [
        *sorted(Path("data/custom_v1").glob("th_*.csv")),
        Path("data/th_curated_500/th_06_new_handwritten.csv"),
        Path("data/th_curated_560/th_07_ordinal_balance.csv"),
        Path("data/th_curated_620/th_08_boundary_cases.csv"),
        Path("data/th_curated_700/th_09_score_disambiguation.csv"),
        Path("data/th_curated_760/th_10_risk13_train.csv"),
        Path("data/th_curated_760/th_11_validation_expansion.csv"),
        Path("data/th_curated_760/th_12_calibration_expansion.csv"),
        Path("data/th_curated_760/th_13_test_expansion.csv"),
        Path("data/th_curated_880/th_14_paired_boundaries_train.csv"),
        Path("data/th_curated_880/th_15_paired_12_23_train.csv"),
        Path("data/th_curated_960/th_16_jevbench_hard_cases_train.csv"),
        Path("data/th_curated_1200/th_17_agent_task_context_train.csv"),
    ]
    rows = [row for source in sources for row in read_rows(source)]
    if len(rows) != 1500:
        raise SystemExit(f"expected 1500 authored rows, got {len(rows)}")

    splits = {"train": [], "validation": [], "calibration": [], "test": []}
    seen_ids: set[str] = set()
    seen_requests: set[str] = set()
    source_prefixes = {
        "th-new-": "jev-my-bro-handwritten-th-v2",
        "th-ord-": "jev-my-bro-handwritten-th-ordinal-balance",
        "th-bnd-": "jev-my-bro-handwritten-th-boundary-contrast",
        "th-scr-": "jev-my-bro-handwritten-th-score-disambiguation",
        "th-r13-": "jev-my-bro-handwritten-th-risk13-contrast",
        "th-val-": "jev-my-bro-handwritten-th-validation-expansion",
        "th-cal-": "jev-my-bro-handwritten-th-calibration-expansion",
        "th-tst-": "jev-my-bro-handwritten-th-test-expansion",
        "th-pair-": "jev-my-bro-handwritten-th-paired-boundary-contrast",
        "th-amb-": "jev-my-bro-handwritten-th-ambiguous-contrast",
        "th-route-": "jev-my-bro-handwritten-th-routing-hard-contrast",
        "th-trap-": "jev-my-bro-handwritten-th-trap-adversarial-contrast",
        "th-judge-": "jev-my-bro-handwritten-th-judge-hard-contrast",
        "th-agt-": "jev-my-bro-authored-th-agent-task-context",
    }
    for row in rows:
        case = build_case(row, "th", "handwritten")
        for prefix, source in source_prefixes.items():
            if case["id"].startswith(prefix):
                case["state"]["source"] = source
                break
        if case["id"] in seen_ids or case["state"]["request"] in seen_requests:
            raise SystemExit(f"duplicate authored case: {case['id']}")
        seen_ids.add(case["id"])
        seen_requests.add(case["state"]["request"])
        splits[row["split"]].append(case)

    expected = {"train": 1200, "validation": 100, "calibration": 100, "test": 100}
    if {name: len(values) for name, values in splits.items()} != expected:
        raise SystemExit({name: len(values) for name, values in splits.items()})

    expected_risks = Counter({str(level): 20 for level in range(5)})
    for name in ("validation", "calibration", "test"):
        actual = Counter(case["gold"]["risk"]["label"] for case in splits[name])
        if actual != expected_risks:
            raise SystemExit(f"{name}: expected balanced risks, got {actual}")

    args.output.mkdir(parents=True, exist_ok=True)
    for name, cases in splits.items():
        (args.output / f"{name}.jsonl").write_text(
            "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
            encoding="utf-8",
        )
        print(name, len(cases), "risk=", Counter(case["gold"]["risk"]["label"] for case in cases))


if __name__ == "__main__":
    main()
