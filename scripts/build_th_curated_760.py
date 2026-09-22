"""Build the Thai 760-case set with expanded balanced holdouts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from build_th_curated_500 import read_rows
from csv_to_dataset import build_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/th_curated_760"))
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
    ]
    rows = [row for source in sources for row in read_rows(source)]
    if len(rows) != 1060:
        raise SystemExit(f"expected 1060 authored rows, got {len(rows)}")

    splits = {"train": [], "validation": [], "calibration": [], "test": []}
    seen_ids: set[str] = set()
    seen_requests: set[str] = set()
    for row in rows:
        case = build_case(row, "th", "handwritten")
        if case["id"].startswith("th-new-"):
            case["state"]["source"] = "jev-my-bro-handwritten-th-v2"
        if case["id"].startswith("th-ord-"):
            case["state"]["source"] = "jev-my-bro-handwritten-th-ordinal-balance"
        if case["id"].startswith("th-bnd-"):
            case["state"]["source"] = "jev-my-bro-handwritten-th-boundary-contrast"
        if case["id"].startswith("th-scr-"):
            case["state"]["source"] = "jev-my-bro-handwritten-th-score-disambiguation"
        if case["id"].startswith("th-r13-"):
            case["state"]["source"] = "jev-my-bro-handwritten-th-risk13-contrast"
        if case["id"].startswith("th-val-"):
            case["state"]["source"] = "jev-my-bro-handwritten-th-validation-expansion"
        if case["id"].startswith("th-cal-"):
            case["state"]["source"] = "jev-my-bro-handwritten-th-calibration-expansion"
        if case["id"].startswith("th-tst-"):
            case["state"]["source"] = "jev-my-bro-handwritten-th-test-expansion"
        if case["id"] in seen_ids or case["state"]["request"] in seen_requests:
            raise SystemExit(f"duplicate authored case: {case['id']}")
        seen_ids.add(case["id"])
        seen_requests.add(case["state"]["request"])
        splits[row["split"]].append(case)

    expected = {"train": 760, "validation": 100, "calibration": 100, "test": 100}
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
