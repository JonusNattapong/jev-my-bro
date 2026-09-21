from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset

from training import LABEL2ID


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_text(row: dict) -> str:
    options = " | ".join(row.get("options", ["execute", "ask_user", "reject"]))
    question = row.get("question", "What should the agent do?")
    return (
        f"Context:\n{row['context']}\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options}"
    )


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            label = row.get("label")
            if label not in LABEL2ID:
                raise ValueError(f"{path}:{line_no}: invalid label {label!r}")
            if not isinstance(row.get("context"), str) or not row["context"].strip():
                raise ValueError(f"{path}:{line_no}: context must be non-empty text")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: dataset is empty")
    return rows


def to_hf_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_list(
        [{"text": build_text(row), "label": LABEL2ID[row["label"]]} for row in rows]
    )


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=-1, keepdims=True)


def multiclass_brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray) -> float:
    chosen = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(chosen, 1e-12, 1.0)).mean())


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(boundaries[:-1], boundaries[1:]):
        mask = ((confidences >= low) & (confidences <= high)) if high == 1.0 else ((confidences >= low) & (confidences < high))
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def summarize_probabilities(probabilities: np.ndarray, labels: np.ndarray, *, bins: int = 15) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "nll": negative_log_likelihood(probabilities, labels),
        "brier": multiclass_brier(probabilities, labels),
        "ece": expected_calibration_error(probabilities, labels, bins=bins),
    }
