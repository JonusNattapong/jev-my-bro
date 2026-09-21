from __future__ import annotations

import argparse
import json

from training.common import softmax_np, summarize_probabilities
from training.logits import collect_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logits, labels = collect_logits(
        args.model,
        args.data,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    temperature = 1.0
    if args.calibration:
        with open(args.calibration, "r", encoding="utf-8") as handle:
            temperature = float(json.load(handle)["temperature"])
    result = {
        "examples": int(len(labels)),
        "temperature": temperature,
        "raw": summarize_probabilities(softmax_np(logits), labels),
        "calibrated": summarize_probabilities(softmax_np(logits / temperature), labels),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
