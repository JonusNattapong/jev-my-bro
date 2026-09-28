from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from laya.common import confidence_from_probs

from jevbro.batching import collate_items
from jevbro.checkpoint import load_trainable
from jevbro.data import load_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit one temperature per Laya question type")
    parser.add_argument("--model", default="artifacts/laya-model")
    parser.add_argument("--data", default="data/calibration.jsonl")
    parser.add_argument("--report", default="artifacts/calibration-report.json")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def validate_calibration_data(path: Path) -> None:
    """Reject the test split so calibration cannot silently leak test labels."""
    lowered = {part.lower() for part in path.parts}
    if path.name.lower() == "test.jsonl" or "test" in lowered:
        raise ValueError(
            f"refusing to fit calibration on a test split: {path}; "
            "use calibration.jsonl"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(model, tokenizer, items: list[dict], device: torch.device, batch_size: int):
    model.eval()
    values: list[tuple[int, list[float], list[float]]] = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            batch = collate_items(chunk, tokenizer.pad_token_id)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            marker_pos = batch["marker_pos"].to(device)
            marker_mask = batch["marker_mask"].to(device)
            qtype = batch["qtype"].to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits, _ = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
            logits = logits.float().cpu()
            for index, item in enumerate(chunk):
                count = len(item["markers"])
                values.append((item["qtype"], logits[index, :count].tolist(), list(item["target"])))
    return values


def fit_temperature(rows: list[tuple[list[float], list[float]]]) -> float:
    if len(rows) < 10:
        return 1.0
    max_options = max(len(logits) for logits, _ in rows)
    logits = torch.full((len(rows), max_options), -1e4, dtype=torch.float64)
    target = torch.zeros((len(rows), max_options), dtype=torch.float64)
    for index, (row_logits, row_target) in enumerate(rows):
        logits[index, : len(row_logits)] = torch.tensor(row_logits, dtype=torch.float64)
        target[index, : len(row_target)] = torch.tensor(row_target, dtype=torch.float64)

    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.1, 10.0)
        loss = -(target * torch.log_softmax(logits / temperature, -1)).sum(-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().detach().clamp(0.1, 10.0))


def soft_nll(rows: list[tuple[list[float], list[float]]], temperature: float) -> float:
    total = 0.0
    for logits, target in rows:
        logit_tensor = torch.tensor(logits, dtype=torch.float64) / temperature
        target_tensor = torch.tensor(target, dtype=torch.float64)
        total += float(-(target_tensor * torch.log_softmax(logit_tensor, -1)).sum().item())
    return total / max(1, len(rows))


def fit_abstain_threshold(
    rows: list[tuple[list[float], list[float]]],
    temperature: float,
    minimum_active_accuracy: float = 0.90,
) -> tuple[float, dict]:
    """Pick the lowest confidence threshold that still answers accurately.

    Confidence here is Laya's normalized Shannon entropy, `1 - H(p)/log(k)`, not the top
    probability. For a three-way choice it stays far below 1.0 unless the model is extremely
    peaked, so the hardcoded 0.6 default abstains on essentially every request and nulls every
    `decision`. Measured on the calibration split this checkpoint's choice confidence tops out
    near 0.44.

    Lowering it is only safe if accuracy on the answered subset is verified, so the threshold is
    the one that minimises the abstain rate while keeping accuracy when answering at or above the
    target. Taking the largest threshold instead would maximise abstention and make the flag
    meaningless.
    """
    scored: list[tuple[float, int, int]] = []
    for logits, target in rows:
        probs = torch.softmax(torch.tensor(logits, dtype=torch.float64) / temperature, -1)
        values = [float(value) for value in probs]
        confidence = confidence_from_probs(np.asarray(values, dtype=np.float64), len(values))
        gold = max(range(len(target)), key=target.__getitem__)
        predicted = int(max(range(len(values)), key=values.__getitem__))
        scored.append((confidence, predicted, gold))
    if not scored:
        return 0.6, {"items": 0, "note": "no rows; kept the default threshold"}

    candidates = sorted({round(value, 4) for value, _, _ in scored})
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        active = [row for row in scored if row[0] >= threshold]
        if not active:
            continue
        accuracy = sum(1 for _, predicted, gold in active if predicted == gold) / len(active)
        if accuracy < minimum_active_accuracy:
            continue
        abstain_rate = 1.0 - len(active) / len(scored)
        if best is None or abstain_rate < best[1]:
            best = (threshold, abstain_rate, accuracy)
    if best is None:
        return 1.0, {
            "items": len(scored),
            "note": "no threshold met the accuracy target; the flag abstains on everything",
        }
    return best[0], {
        "items": len(scored),
        "abstain_rate": round(best[1], 4),
        "accuracy_when_answering": round(best[2], 4),
        "minimum_active_accuracy": minimum_active_accuracy,
    }


def fit_score_thresholds(
    rows: list[tuple[list[float], list[float]]],
    temperature: float,
    levels: int = 5,
) -> tuple[list[float], list[float]]:
    values: list[float] = []
    labels: list[int] = []
    for logits, target in rows:
        probs = torch.softmax(torch.tensor(logits, dtype=torch.float64) / temperature, -1)
        values.append(float((probs * torch.arange(len(probs), dtype=torch.float64)).sum().item()))
        labels.append(max(range(len(target)), key=target.__getitem__))

    unique = sorted(set(values))
    candidates = [unique[0] - 1e-6]
    candidates.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    candidates.append(unique[-1] + 1e-6)
    thresholds: list[float] = []
    scores: list[float] = []
    for boundary in range(levels - 1):
        negatives = sum(label <= boundary for label in labels)
        positives = len(labels) - negatives
        best_threshold = candidates[0]
        best_score = -1.0
        for threshold in candidates:
            true_negative = sum(label <= boundary and value <= threshold for value, label in zip(values, labels))
            true_positive = sum(label > boundary and value > threshold for value, label in zip(values, labels))
            balanced = 0.5 * (
                true_negative / max(1, negatives) + true_positive / max(1, positives)
            )
            if balanced > best_score:
                best_score = balanced
                best_threshold = threshold
        if thresholds and best_threshold <= thresholds[-1]:
            best_threshold = thresholds[-1] + 1e-6
        thresholds.append(float(best_threshold))
        scores.append(float(best_score))
    return thresholds, scores


def select_score_decoder(thresholds: list[float], minimum_gap: float = 1e-3) -> str:
    """Avoid threshold decoding when calibration collapses adjacent boundaries."""
    if len(thresholds) != 4:
        return "argmax"
    if any(right - left <= minimum_gap for left, right in zip(thresholds, thresholds[1:])):
        return "argmax"
    return "threshold"


def main() -> None:
    args = parse_args()
    calibration_path = Path(args.data)
    validate_calibration_data(calibration_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, cfg, _ = load_trainable(args.model, device)
    items = load_items(tokenizer, cfg, calibration_path)
    raw = collect(model, tokenizer, items, device, args.batch_size)

    temperatures = [1.0, 1.0, 1.0]
    abstain_thresholds = [0.6, 0.6, 0.6]
    report = {"dataset": args.data, "question_types": {}}
    names = {0: "choice", 1: "score", 2: "noul"}
    for qtype in range(3):
        selected = [(logits, target) for row_type, logits, target in raw if row_type == qtype]
        temperature = fit_temperature(selected)
        temperatures[qtype] = temperature
        abstain_threshold, abstain_report = fit_abstain_threshold(selected, temperature)
        abstain_thresholds[qtype] = abstain_threshold
        report["question_types"][names[qtype]] = {
            "items": len(selected),
            "temperature": temperature,
            "soft_nll_before": soft_nll(selected, 1.0),
            "soft_nll_after": soft_nll(selected, temperature),
            "abstain_threshold": abstain_threshold,
            "abstain_fit": abstain_report,
        }

    score_rows = [(logits, target) for row_type, logits, target in raw if row_type == 1]
    score_thresholds, score_boundary_balanced_accuracy = fit_score_thresholds(
        score_rows,
        temperatures[1],
    )
    score_decoder = select_score_decoder(score_thresholds)

    calibration_provenance = {
        "dataset": str(calibration_path),
        "dataset_sha256": sha256_file(calibration_path),
        "decoder_selection": "calibration_only",
        "report": str(Path(args.report)),
    }
    cfg["temperature"] = temperatures
    cfg["abstain_threshold_by_qtype"] = {
        names[index]: value for index, value in enumerate(abstain_thresholds)
    }
    cfg["score_thresholds"] = score_thresholds
    cfg["score_decoder"] = score_decoder
    cfg["calibration"] = calibration_provenance
    Path(args.model, "rl_agent_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report["temperature"] = temperatures
    report["score_thresholds"] = score_thresholds
    report["score_decoder"] = score_decoder
    report["decoder_selection"] = "calibration_only"
    report["calibration_dataset_sha256"] = calibration_provenance["dataset_sha256"]
    report["score_boundary_balanced_accuracy"] = score_boundary_balanced_accuracy
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
