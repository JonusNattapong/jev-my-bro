from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np
import torch


def ordinal_soft_target(label: int, levels: int = 5, sigma: float = 0.75) -> list[float]:
    if not 0 <= label < levels:
        raise ValueError("label out of range")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    weights = [math.exp(-((idx - label) ** 2) / (2.0 * sigma**2)) for idx in range(levels)]
    total = sum(weights)
    return [value / total for value in weights]


def expected_level(probabilities: Sequence[float]) -> float:
    return float(sum(index * float(value) for index, value in enumerate(probabilities)))


def hard_level_from_expected(value: float, levels: int = 5) -> int:
    """Decode an ordinal score by nearest expected level, avoiding argmax collapse."""
    if levels < 2:
        raise ValueError("levels must be at least 2")
    return max(0, min(levels - 1, int(math.floor(float(value) + 0.5))))


def ranked_probability_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    probs = probabilities * mask
    target = target * mask
    cdf_p = torch.cumsum(probs, dim=-1)
    cdf_t = torch.cumsum(target, dim=-1)
    levels = mask.sum(dim=-1).clamp(min=2).to(probabilities.dtype)
    return ((((cdf_p - cdf_t) ** 2) * mask).sum(dim=-1) / (levels - 1.0))


def ranked_probability_score(pred: Sequence[float], target: Sequence[float]) -> float:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.shape != t.shape or p.ndim != 1 or len(p) < 2:
        raise ValueError("pred and target must be same-length 1D distributions")
    return float(np.square(np.cumsum(p) - np.cumsum(t)).sum() / (len(p) - 1))


def quadratic_weighted_kappa(y_true: Sequence[int], y_pred: Sequence[int], levels: int = 5) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred lengths differ")
    if not y_true:
        return float("nan")
    observed = np.zeros((levels, levels), dtype=np.float64)
    for truth, pred in zip(y_true, y_pred):
        observed[int(truth), int(pred)] += 1.0
    hist_true = observed.sum(axis=1)
    hist_pred = observed.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / observed.sum()
    denom = float((levels - 1) ** 2)
    weights = np.fromfunction(lambda i, j: ((i - j) ** 2) / denom, (levels, levels), dtype=float)
    expected_cost = float((weights * expected).sum())
    if expected_cost <= 1e-12:
        return 1.0
    return 1.0 - float((weights * observed).sum()) / expected_cost


def effective_number_weights(labels: Sequence[int], beta: float, levels: int = 5) -> list[float]:
    if beta <= 0.0:
        return [1.0] * levels
    if beta >= 1.0:
        raise ValueError("beta must be in [0, 1)")
    counts = Counter(int(label) for label in labels)
    raw = []
    for level in range(levels):
        count = counts.get(level, 0)
        raw.append(0.0 if count == 0 else (1.0 - beta) / (1.0 - beta**count))
    represented = [weight for weight in raw if weight > 0.0]
    scale = sum(represented) / len(represented) if represented else 1.0
    return [weight / scale if weight > 0.0 else 0.0 for weight in raw]
