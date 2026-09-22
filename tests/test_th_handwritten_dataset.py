from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
DATASET = ROOT / "data" / "th_handwritten_v2"


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATASET / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_th_handwritten_splits_are_independent_and_balanced() -> None:
    train = read_csv("train.csv")
    holdout = read_csv("holdout.csv")

    assert len(train) == 500
    assert len(holdout) == 120
    assert len({row["id"] for row in train + holdout}) == 620
    assert not ({row["request"] for row in train} & {row["request"] for row in holdout})

    split_counts = Counter(row["split"] for row in holdout)
    assert split_counts == Counter({"validation": 40, "calibration": 40, "test": 40})

    for split in ("validation", "calibration", "test"):
        rows = [row for row in holdout if row["split"] == split]
        assert Counter(row["risk"] for row in rows) == Counter({str(level): 8 for level in range(5)})
        assert len({row["request"] for row in rows}) == 40

        for row in rows:
            if row["action"] == "execute":
                assert row["needs_review"] == "false"
                assert row["prohibited"] == "false"
            elif row["action"] == "ask_user":
                assert row["needs_review"] == "true"
                assert row["prohibited"] == "false"
            else:
                assert row["action"] == "reject"
                assert row["needs_review"] == "true"
                assert row["prohibited"] == "true"
