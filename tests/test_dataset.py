from pathlib import Path

import importlib.util

from jevbro.schema import EXPECTED_IDS, read_cases, summarize


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_dataset", Path(__file__).parents[1] / "scripts" / "validate_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validator_flags_degenerate_needs_review() -> None:
    """needs_review equal to (action != execute) makes the ask_user gate rule inverted."""
    module = _load_validator()

    def case(action: str, review: bool) -> dict:
        return {
            "gold": {
                "action": {"label": action},
                "needs_review": {"label": "true" if review else "false"},
            }
        }

    degenerate = [case("execute", False), case("ask_user", True), case("reject", True)]
    assert module.needs_review_degeneracy(degenerate) is not None

    mixed = degenerate + [case("execute", True), case("reject", False)]
    assert module.needs_review_degeneracy(mixed) is None


def test_all_splits_are_valid_and_balanced() -> None:
    for split in ("train", "validation", "calibration", "test"):
        cases = read_cases(Path("data") / f"{split}.jsonl")
        summary = summarize(cases)
        assert summary["cases"] > 0
        assert set(summary["languages"]) == {"en", "th"}
        actions = summary["actions"]
        assert actions["execute"] == actions["ask_user"] == actions["reject"]
        assert all(set(case["questions"]) == EXPECTED_IDS for case in cases)
        risk_levels = {str(case["gold"]["risk"]["label"]) for case in cases}
        assert risk_levels == {"0", "1", "2", "3", "4"}


def test_splits_do_not_share_requests() -> None:
    values = {}
    for split in ("train", "validation", "calibration", "test"):
        cases = read_cases(Path("data") / f"{split}.jsonl")
        values[split] = {case["state"]["request"] for case in cases}

    names = list(values)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert values[left].isdisjoint(values[right])
