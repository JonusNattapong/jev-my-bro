"""Build the Thai 500-case training set from handwritten CSV sources."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from csv_to_dataset import build_case


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/th_curated_500"))
    args = parser.parse_args()
    source = Path("data/custom_v1")
    new_source = source.parent / "th_curated_500" / "th_06_new_handwritten.csv"
    rows = [row for path in sorted(source.glob("th_*.csv")) for row in read_rows(path)]
    rows.extend(read_rows(new_source))
    if len(rows) != 620:
        raise SystemExit(f"expected 620 authored rows, got {len(rows)}")

    splits = {"train": [], "validation": [], "calibration": [], "test": []}
    seen_ids: set[str] = set()
    seen_requests: set[str] = set()
    for row in rows:
        case = build_case(row, "th", "handwritten")
        if case["id"].startswith("th-new-"):
            case["state"]["source"] = "jev-my-bro-handwritten-th-v2"
        if case["id"] in seen_ids or case["state"]["request"] in seen_requests:
            raise SystemExit(f"duplicate authored case: {case['id']}")
        seen_ids.add(case["id"])
        seen_requests.add(case["state"]["request"])
        splits[row["split"]].append(case)

    # The original 40/40/40 holdout rows remain untouched and independent;
    # the 120 new rows are all assigned to train, producing 500 train cases.
    if len(splits["train"]) != 500 or any(len(splits[name]) != 40 for name in splits if name != "train"):
        raise SystemExit({name: len(values) for name, values in splits.items()})
    args.output.mkdir(parents=True, exist_ok=True)
    for name, cases in splits.items():
        (args.output / f"{name}.jsonl").write_text(
            "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
            encoding="utf-8",
        )
        print(name, len(cases), Counter(case["gold"]["action"]["label"] for case in cases))


if __name__ == "__main__":
    main()
