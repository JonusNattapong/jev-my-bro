from jevbro.calibrate import select_score_decoder


def test_collapsed_score_thresholds_use_argmax_decoder() -> None:
    assert select_score_decoder([1.0, 2.0, 2.0, 2.000001]) == "argmax"


def test_separated_score_thresholds_use_threshold_decoder() -> None:
    assert select_score_decoder([0.8, 1.6, 2.4, 3.2]) == "threshold"
