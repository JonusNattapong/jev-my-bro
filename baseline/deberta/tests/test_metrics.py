import numpy as np

from training.common import expected_calibration_error, multiclass_brier, negative_log_likelihood, softmax_np


def test_softmax_rows_sum_to_one() -> None:
    probs = softmax_np(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_confident_correct_predictions_have_good_metrics() -> None:
    probs = np.array([
        [0.98, 0.01, 0.01],
        [0.01, 0.98, 0.01],
        [0.01, 0.01, 0.98],
    ])
    labels = np.array([0, 1, 2])
    assert negative_log_likelihood(probs, labels) < 0.03
    assert multiclass_brier(probs, labels) < 0.01
    assert expected_calibration_error(probs, labels) < 0.03
