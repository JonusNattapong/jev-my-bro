"""Build the Thai 620-case set with boundary and authorization contrasts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from build_th_curated_500 import read_rows
from csv_to_dataset import build_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/th_curated_620"))
    args = parser.parse_args()

    sources = [
        *sorted(Path("data/custom_v1").glob("th_*.csv")),
        Path("data/th_curated_500/th_06_new_handwritten.csv"),
        Path("data/th_curated_560/th_07_ordinal_balance.csv"),
        Path("data/th_curated_620/th_08_boundary_cases.csv"),
    ]
    rows = [row for source in sources for row in read_rows(source)]
    if len(rows) != 740:
        raise SystemExit(f"expected 740 authored rows, got {len(rows)}")

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
        if case["id"] in seen_ids or case["state"]["request"] in seen_requests:
            raise SystemExit(f"duplicate authored case: {case['id']}")
        seen_ids.add(case["id"])
        seen_requests.add(case["state"]["request"])
        splits[row["split"]].append(case)

    if len(splits["train"]) != 620 or any(
        len(splits[name]) != 40 for name in splits if name != "train"
    ):
        raise SystemExit({name: len(values) for name, values in splits.items()})

    args.output.mkdir(parents=True, exist_ok=True)
    for name, cases in splits.items():
        (args.output / f"{name}.jsonl").write_text(
            "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
            encoding="utf-8",
        )
        print(name, len(cases), "risk=", Counter(case["gold"]["risk"]["label"] for case in cases))


if __name__ == "__main__":
    main()
