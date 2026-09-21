from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import laya
import numpy as np

from jevbro.schema import read_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/laya-model")
    parser.add_argument("--data", default="data/test.jsonl")
    parser.add_argument("--device")
    parser.add_argument("--report", default="artifacts/test-report.json")
    return parser.parse_args()


def ece(confidence: list[float], correct: list[float], bins: int = 15) -> float:
    if not confidence:
        return float("nan")
    conf = np.asarray(confidence, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (conf > low) & (conf <= high)
        if mask.any():
            value += mask.mean() * abs(conf[mask].mean() - corr[mask].mean())
    return float(value)


def distribution(question: dict, answer: dict) -> np.ndarray:
    qtype = question["type"]
    if qtype == "choice":
        return np.asarray([answer["probabilities"][key] for key in question["criteria"]], dtype=np.float64)
    if qtype == "noul":
        p = float(answer["noul"])
        return np.asarray([1.0 - p, p], dtype=np.float64)
    return np.asarray(
        [answer["probabilities"][str(index)] for index in range(len(question["criteria"]))],
        dtype=np.float64,
    )


def gold_distribution(question: dict, gold: dict) -> np.ndarray:
    probs = gold["probabilities"]
    if question["type"] == "choice":
        return np.asarray([probs[key] for key in question["criteria"]], dtype=np.float64)
    if question["type"] == "noul":
        return np.asarray([probs["false"], probs["true"]], dtype=np.float64)
    return np.asarray([probs[str(index)] for index in range(len(question["criteria"]))], dtype=np.float64)


def main() -> None:
    args = parse_args()
    cases = read_cases(args.data)
    agent = laya.Agent(args.model, device=args.device)

    primitive = defaultdict(lambda: {"n": 0, "correct": 0.0, "brier": 0.0, "soft_accuracy": 0.0})
    language = defaultdict(lambda: {"n": 0, "correct": 0.0})
    confidences: list[float] = []
    correctness: list[float] = []
    score_errors: list[float] = []

    for index, case in enumerate(cases, 1):
        result = agent.predict(case["state"], case["questions"])
        for qid, question in case["questions"].items():
            answer = result["answers"][qid]
            gold = case["gold"][qid]
            pred = distribution(question, answer)
            target = gold_distribution(question, gold)
            pred = pred / pred.sum()
            target = target / target.sum()
            qtype = question["type"]

            if qtype == "choice":
                correct = float(answer["choice"] == gold["label"])
            elif qtype == "noul":
                predicted_label = "true" if float(answer["noul"]) >= 0.5 else "false"
                correct = float(predicted_label == str(gold["label"]).lower())
            else:
                predicted_level = int(np.argmax(pred))
                correct = float(predicted_level == int(gold["label"]))
                score_errors.append(abs(float(answer["score"]) - float(gold["score"])))

            primitive[qtype]["n"] += 1
            primitive[qtype]["correct"] += correct
            primitive[qtype]["brier"] += float(np.square(pred - target).sum())
            primitive[qtype]["soft_accuracy"] += float((pred * target).sum())
            language[case["language"]]["n"] += 1
            language[case["language"]]["correct"] += correct
            confidences.append(float(pred.max()))
            correctness.append(correct)

        if index % 25 == 0:
            print(f"[eval] {index}/{len(cases)} cases")

    primitive_report = {}
    total_n = 0
    total_correct = 0.0
    for qtype, values in primitive.items():
        count = values["n"]
        total_n += count
        total_correct += values["correct"]
        primitive_report[qtype] = {
            "n": count,
            "accuracy": values["correct"] / count,
            "brier": values["brier"] / count,
            "soft_accuracy": values["soft_accuracy"] / count,
        }

    report = {
        "model": args.model,
        "dataset": args.data,
        "cases": len(cases),
        "decisions": total_n,
        "accuracy": total_correct / max(1, total_n),
        "ece": ece(confidences, correctness),
        "score_mae": float(np.mean(score_errors)) if score_errors else None,
        "by_primitive": primitive_report,
        "by_language": {
            lang: {"n": values["n"], "accuracy": values["correct"] / values["n"]}
            for lang, values in language.items()
        },
    }

    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
