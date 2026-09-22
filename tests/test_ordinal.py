import math

import pytest
import torch

from jevbro.ordinal import (
    coral_boundary_logits,
    coral_class_probabilities,
    cumulative_ordinal_loss,
    effective_number_weights,
    expected_level,
    hard_level_from_expected,
    hard_level_from_thresholds,
    ordinal_soft_target,
    quadratic_weighted_kappa,
    ranked_probability_loss,
    ranked_probability_score,
)


def test_coral_projection_exposes_four_monotone_boundaries() -> None:
    logits = torch.tensor([[4.0, 2.0, 0.0, -2.0, -4.0]])
    boundaries = coral_boundary_logits(logits)
    probabilities = coral_class_probabilities(boundaries)

    assert boundaries.shape == (1, 4)
    assert torch.all(boundaries[:, :-1] > boundaries[:, 1:])
    assert torch.allclose(probabilities.sum(-1), torch.ones(1), atol=1e-6)
    assert torch.all(probabilities >= 0)


def test_ordinal_soft_target_is_unimodal_and_distance_aware() -> None:
    target = ordinal_soft_target(2, levels=5, sigma=0.75)
    assert sum(target) == pytest.approx(1.0)
    assert target[2] > target[1] > target[0]
    assert target[2] > target[3] > target[4]
    assert target[1] == pytest.approx(target[3])
    assert expected_level(target) == pytest.approx(2.0)


def test_ranked_probability_loss_penalizes_far_errors_more() -> None:
    target = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
    mask = torch.ones_like(target, dtype=torch.bool)
    near = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]])
    far = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    assert ranked_probability_loss(near, target, mask).item() < ranked_probability_loss(far, target, mask).item()


def test_cumulative_ordinal_loss_penalizes_far_errors_more() -> None:
    target = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
    mask = torch.ones_like(target, dtype=torch.bool)
    near = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]])
    far = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    assert cumulative_ordinal_loss(near, target, mask).item() < cumulative_ordinal_loss(far, target, mask).item()


def test_effective_number_weights_upweight_rare_levels() -> None:
    weights = effective_number_weights([0] * 100 + [1] * 10 + [2] * 50, beta=0.99, levels=5)
    assert weights[1] > weights[2] > weights[0]
    assert weights[3] == 0.0
    assert weights[4] == 0.0


def test_numpy_rps_and_qwk() -> None:
    target = [0.0, 0.0, 1.0, 0.0, 0.0]
    near = [0.0, 0.0, 0.0, 1.0, 0.0]
    far = [1.0, 0.0, 0.0, 0.0, 0.0]
    assert ranked_probability_score(near, target) < ranked_probability_score(far, target)
    assert quadratic_weighted_kappa([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]) == pytest.approx(1.0)


def test_expected_level_decoder_uses_nearest_ordinal_level() -> None:
    assert hard_level_from_expected(0.49) == 0
    assert hard_level_from_expected(0.50) == 1
    assert hard_level_from_expected(1.49) == 1
    assert hard_level_from_expected(1.50) == 2
    assert hard_level_from_expected(4.8) == 4


def test_threshold_decoder_uses_calibrated_boundaries() -> None:
    thresholds = [0.8, 1.4, 2.6, 3.5]
    assert hard_level_from_thresholds(0.79, thresholds) == 0
    assert hard_level_from_thresholds(0.80, thresholds) == 1
    assert hard_level_from_thresholds(1.80, thresholds) == 2
    assert hard_level_from_thresholds(3.60, thresholds) == 4


def test_validation_checkpoint_selection_prefers_qwk_then_lower_rps() -> None:
    from jevbro.train import is_better_score_checkpoint

    assert is_better_score_checkpoint({"qwk": 0.80, "rps": 0.10}, {"qwk": 0.79, "rps": 0.05})
    assert is_better_score_checkpoint({"qwk": 0.80, "rps": 0.04}, {"qwk": 0.80, "rps": 0.05})
    assert not is_better_score_checkpoint({"qwk": 0.79, "rps": 0.01}, {"qwk": 0.80, "rps": 0.20})


def test_checkpoint_selection_balances_accuracy_and_score_metrics() -> None:
    from jevbro.train import checkpoint_selection_score, is_better_checkpoint

    general_model = {
        "selection_score": checkpoint_selection_score(
            {
                "accuracy": 0.80,
                "score": {"macro_recall": 0.60, "within_one_accuracy": 0.90},
            }
        ),
        "qwk": 0.80,
        "rps": 0.10,
    }
    ordinal_model = {
        "selection_score": checkpoint_selection_score(
            {
                "accuracy": 0.77,
                "score": {"macro_recall": 0.80, "within_one_accuracy": 0.95},
            }
        ),
        "qwk": 0.80,
        "rps": 0.10,
    }

    assert general_model["selection_score"] == pytest.approx(0.75)
    assert ordinal_model["selection_score"] == pytest.approx(0.8165)
    assert is_better_checkpoint(ordinal_model, general_model)
