from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
DATASET = ROOT / "data" / "th_curated_880"


def read_jsonl(name: str) -> list[dict]:
    with (DATASET / name).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_th_curated_880_has_expected_independent_splits() -> None:
    splits = {name: read_jsonl(f"{name}.jsonl") for name in ("train", "validation", "calibration", "test")}
    assert {name: len(rows) for name, rows in splits.items()} == {
        "train": 880,
        "validation": 100,
        "calibration": 100,
        "test": 100,
    }

    all_rows = [row for rows in splits.values() for row in rows]
    assert len({row["id"] for row in all_rows}) == 1180
    assert len({row["state"]["request"] for row in all_rows}) == 1180
    assert {row["language"] for row in all_rows} == {"th"}
    for name in ("validation", "calibration", "test"):
        assert Counter(row["gold"]["risk"]["label"] for row in splits[name]) == Counter(
            {str(level): 20 for level in range(5)}
        )


def test_th_curated_880_contains_120_paired_boundary_cases() -> None:
    train = read_jsonl("train.jsonl")
    paired = [row for row in train if row["id"].startswith("th-pair-")]
    assert len(paired) == 120
    assert Counter(row["gold"]["risk"]["label"] for row in paired) == Counter(
        {"1": 20, "2": 40, "3": 40, "4": 20}
    )

    for row in paired:
        action = row["gold"]["action"]["label"]
        needs_review = row["gold"]["needs_review"]["label"]
        prohibited = row["gold"]["prohibited"]["label"]
        if action == "execute":
            assert needs_review == "false"
            assert prohibited == "false"
        elif action == "ask_user":
            assert needs_review == "true"
            assert prohibited == "false"
        else:
            assert action == "reject"
            assert needs_review == "true"
            assert prohibited == "true"
