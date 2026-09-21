import math

import pytest
import torch

from jevbro.ordinal import (
    effective_number_weights,
    expected_level,
    ordinal_soft_target,
    quadratic_weighted_kappa,
    ranked_probability_loss,
    ranked_probability_score,
)


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
