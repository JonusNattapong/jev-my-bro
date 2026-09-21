from collections import Counter
from pathlib import Path

import pytest

from training import LABELS
from training.common import build_text, read_jsonl


@pytest.mark.parametrize("name", ["train.jsonl", "validation.jsonl", "calibration.jsonl", "test.jsonl"])
def test_dataset_contract(name: str) -> None:
    rows = read_jsonl(Path("data") / name)
    counts = Counter(row["label"] for row in rows)
    assert set(counts) == set(LABELS)
    assert min(counts.values()) >= 10

    seen = set()
    for row in rows:
        assert row["context"] not in seen
        seen.add(row["context"])
        assert row["options"] == list(LABELS)
        assert "Context:" in build_text(row)


def test_splits_do_not_overlap() -> None:
    splits = {}
    for name in ["train", "validation", "calibration", "test"]:
        rows = read_jsonl(Path("data") / f"{name}.jsonl")
        splits[name] = {row["context"] for row in rows}

    names = list(splits)
    for idx, left in enumerate(names):
        for right in names[idx + 1:]:
            assert splits[left].isdisjoint(splits[right]), f"{left} overlaps {right}"
