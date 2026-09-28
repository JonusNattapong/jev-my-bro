from __future__ import annotations

from pathlib import Path

import pytest

from jevbro.evaluate import macro_recall


def test_locked_evaluation_accepts_multi_corpus_calibration_provenance() -> None:
    """calibrate records a list of paths when the model serves several distributions."""
    from jevbro.evaluate import validate_locked_decoder

    config = {
        "calibration": {
            "decoder_selection": "calibration_only",
            "dataset": [
                "data/th_curated_1200/calibration.jsonl",
                "data/tool_call_400/calibration.jsonl",
            ],
        },
        "score_decoder": "threshold",
        "temperature": [1.0, 1.0, 1.0],
        "score_thresholds": [0.8, 1.4, 2.3, 3.0],
    }
    validate_locked_decoder(config, Path("data/tool_call_400/test.jsonl"))


def test_locked_evaluation_rejects_a_test_split_named_in_multi_corpus_provenance() -> None:
    from jevbro.evaluate import validate_locked_decoder

    config = {
        "calibration": {
            "decoder_selection": "calibration_only",
            "dataset": [
                "data/th_curated_1200/calibration.jsonl",
                "data/tool_call_400/test.jsonl",
            ],
        },
        "score_decoder": "threshold",
        "temperature": [1.0, 1.0, 1.0],
        "score_thresholds": [0.8, 1.4, 2.3, 3.0],
    }
    with pytest.raises(ValueError):
        validate_locked_decoder(config, Path("data/th_curated_1200/test.jsonl"))

    assert macro_recall(
        {
            "0": {"support": 2, "recall": 1.0},
            "1": {"support": 2, "recall": 0.5},
            "2": {"support": 0, "recall": None},
        }
    ) == pytest.approx(0.75)


def test_selected_and_argmax_reports_can_be_compared_independently() -> None:
    selected = {
        "0": {"support": 2, "recall": 1.0},
        "1": {"support": 2, "recall": 0.0},
    }
    argmax = {
        "0": {"support": 2, "recall": 0.5},
        "1": {"support": 2, "recall": 1.0},
    }
    assert macro_recall(selected) == pytest.approx(0.5)
    assert macro_recall(argmax) == pytest.approx(0.75)
