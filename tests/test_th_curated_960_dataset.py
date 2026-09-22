from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
DATASET = ROOT / "data" / "th_curated_960"


def read_jsonl(name: str) -> list[dict]:
    with (DATASET / name).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_th_curated_960_has_independent_balanced_holdouts() -> None:
    splits = {name: read_jsonl(f"{name}.jsonl") for name in ("train", "validation", "calibration", "test")}
    assert {name: len(rows) for name, rows in splits.items()} == {
        "train": 960,
        "validation": 100,
        "calibration": 100,
        "test": 100,
    }
    all_rows = [row for rows in splits.values() for row in rows]
    assert len({row["id"] for row in all_rows}) == 1260
    assert len({row["state"]["request"] for row in all_rows}) == 1260
    assert {row["language"] for row in all_rows} == {"th"}
    for name in ("validation", "calibration", "test"):
        assert Counter(row["gold"]["risk"]["label"] for row in splits[name]) == Counter(
            {str(level): 20 for level in range(5)}
        )


def test_th_curated_960_adds_diverse_12_and_23_pairs() -> None:
    train = read_jsonl("train.jsonl")
    paired = [row for row in train if row["id"].startswith("th-pair-")]
    assert len(paired) == 200
    assert Counter(row["gold"]["risk"]["label"] for row in paired) == Counter(
        {"1": 40, "2": 80, "3": 60, "4": 20}
    )
    assert len({row["state"]["domain"] for row in paired}) >= 15
