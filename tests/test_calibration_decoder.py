from pathlib import Path

import pytest

from jevbro.calibrate import select_score_decoder
from jevbro.calibrate import validate_calibration_data
from jevbro.evaluate import validate_locked_decoder


def test_collapsed_score_thresholds_use_argmax_decoder() -> None:
    assert select_score_decoder([1.0, 2.0, 2.0, 2.000001]) == "argmax"


def test_separated_score_thresholds_use_threshold_decoder() -> None:
    assert select_score_decoder([0.8, 1.6, 2.4, 3.2]) == "threshold"


def test_calibration_rejects_test_split() -> None:
    with pytest.raises(ValueError, match="test split"):
        validate_calibration_data(Path("data/th_curated_960/test.jsonl"))


def test_locked_decoder_requires_calibration_provenance(tmp_path: Path) -> None:
    config = {
        "score_decoder": "threshold",
        "temperature": [1.2, 1.4, 1.1],
        "score_thresholds": [0.8, 1.6, 2.4, 3.2],
    }
    with pytest.raises(ValueError, match="calibration provenance"):
        validate_locked_decoder(config, tmp_path / "test.jsonl")


def test_locked_decoder_accepts_calibrated_model(tmp_path: Path) -> None:
    config = {
        "score_decoder": "threshold",
        "temperature": [1.2, 1.4, 1.1],
        "score_thresholds": [0.8, 1.6, 2.4, 3.2],
        "calibration": {
            "dataset": str(tmp_path / "calibration.jsonl"),
            "decoder_selection": "calibration_only",
        },
    }
    validate_locked_decoder(config, tmp_path / "test.jsonl")
