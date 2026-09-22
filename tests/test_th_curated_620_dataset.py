from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
DATASET = ROOT / "data" / "th_curated_620"


def read_jsonl(name: str) -> list[dict]:
    with (DATASET / name).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_th_curated_620_has_expected_split_sizes_and_unique_requests() -> None:
    splits = {name: read_jsonl(f"{name}.jsonl") for name in ("train", "validation", "calibration", "test")}

    assert {name: len(rows) for name, rows in splits.items()} == {
        "train": 620,
        "validation": 40,
        "calibration": 40,
        "test": 40,
    }

    all_rows = [row for rows in splits.values() for row in rows]
    assert len({row["id"] for row in all_rows}) == 740
    assert len({row["state"]["request"] for row in all_rows}) == 740
    assert {row["language"] for row in all_rows} == {"th"}


def test_th_curated_620_preserves_boundary_cases_and_action_contract() -> None:
    train = read_jsonl("train.jsonl")
    boundary = [row for row in train if row["id"].startswith("th-bnd-")]

    assert len(boundary) == 60
    assert Counter(row["gold"]["risk"]["label"] for row in boundary) == Counter(
        {"1": 20, "2": 20, "3": 10, "4": 10}
    )

    for row in train:
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
