from pathlib import Path

from jevbro.schema import EXPECTED_IDS, read_cases, summarize


def test_all_splits_are_valid_and_balanced() -> None:
    for split in ("train", "validation", "calibration", "test"):
        cases = read_cases(Path("data") / f"{split}.jsonl")
        summary = summarize(cases)
        assert summary["cases"] > 0
        assert set(summary["languages"]) == {"en", "th"}
        actions = summary["actions"]
        assert actions["execute"] == actions["ask_user"] == actions["reject"]
        assert all(set(case["questions"]) == EXPECTED_IDS for case in cases)


def test_splits_do_not_share_requests() -> None:
    values = {}
    for split in ("train", "validation", "calibration", "test"):
        cases = read_cases(Path("data") / f"{split}.jsonl")
        values[split] = {case["state"]["request"] for case in cases}

    names = list(values)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert values[left].isdisjoint(values[right])
