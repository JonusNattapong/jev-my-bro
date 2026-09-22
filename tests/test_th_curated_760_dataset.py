from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
DATASET = ROOT / "data" / "th_curated_760"


def read_jsonl(name: str) -> list[dict]:
    with (DATASET / name).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_th_curated_760_has_100_case_balanced_holdouts() -> None:
    splits = {name: read_jsonl(f"{name}.jsonl") for name in ("train", "validation", "calibration", "test")}
    assert {name: len(rows) for name, rows in splits.items()} == {
        "train": 760,
        "validation": 100,
        "calibration": 100,
        "test": 100,
    }

    all_rows = [row for rows in splits.values() for row in rows]
    assert len({row["id"] for row in all_rows}) == 1060
    assert len({row["state"]["request"] for row in all_rows}) == 1060
    assert {row["language"] for row in all_rows} == {"th"}
    for name in ("validation", "calibration", "test"):
        assert Counter(row["gold"]["risk"]["label"] for row in splits[name]) == Counter(
            {str(level): 20 for level in range(5)}
        )


def test_th_curated_760_adds_explicit_risk_1_and_3_training_contrasts() -> None:
    train = read_jsonl("train.jsonl")
    added = [row for row in train if row["id"].startswith("th-r13-")]
    assert len(added) == 60
    assert Counter(row["gold"]["risk"]["label"] for row in added) == Counter({"1": 30, "3": 30})

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
