from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.common import softmax_np, summarize_probabilities
from training.logits import collect_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_t = torch.tensor(logits, dtype=torch.float64)
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
        loss = F.cross_entropy(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).detach().clamp(0.05, 20.0))


def main() -> None:
    args = parse_args()
    logits, labels = collect_logits(
        args.model,
        args.data,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    before = softmax_np(logits)
    temperature = fit_temperature(logits, labels)
    after = softmax_np(logits / temperature)
    payload = {
        "method": "temperature_scaling",
        "temperature": temperature,
        "dataset": str(args.data),
        "metrics_before": summarize_probabilities(before, labels),
        "metrics_after": summarize_probabilities(after, labels),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
