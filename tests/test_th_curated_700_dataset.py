from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
DATASET = ROOT / "data" / "th_curated_700"


def read_jsonl(name: str) -> list[dict]:
    with (DATASET / name).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_th_curated_700_has_independent_expected_splits() -> None:
    splits = {name: read_jsonl(f"{name}.jsonl") for name in ("train", "validation", "calibration", "test")}
    assert {name: len(rows) for name, rows in splits.items()} == {
        "train": 700,
        "validation": 40,
        "calibration": 40,
        "test": 40,
    }

    all_rows = [row for rows in splits.values() for row in rows]
    assert len({row["id"] for row in all_rows}) == 820
    assert len({row["state"]["request"] for row in all_rows}) == 820
    assert {row["language"] for row in all_rows} == {"th"}


def test_th_curated_700_adds_score_focused_cases_with_valid_labels() -> None:
    train = read_jsonl("train.jsonl")
    added = [row for row in train if row["id"].startswith("th-scr-")]
    assert len(added) == 80
    assert Counter(row["gold"]["risk"]["label"] for row in added) == Counter(
        {"0": 10, "1": 15, "2": 30, "3": 10, "4": 15}
    )

    for row in added:
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
