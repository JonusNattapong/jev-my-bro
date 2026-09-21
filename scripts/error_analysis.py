from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import laya
import numpy as np

from jevbro.evaluate import distribution
from jevbro.schema import read_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Produce per-decision Laya error analysis")
    parser.add_argument("--model", default="artifacts/laya-model")
    parser.add_argument("--data", default="data/test.jsonl")
    parser.add_argument("--device")
    parser.add_argument("--report", default="artifacts/error-analysis.json")
    return parser.parse_args()


def request_text(state: object) -> str:
    if isinstance(state, dict):
        return " ".join(str(value) for value in state.values())
    return str(state)


def wording_flags(text: str) -> list[str]:
    lowered = text.lower()
    flags: list[str] = []
    if any(token in lowered for token in ("approve", "approval", "อนุมัติ", "อนุญาต")):
        flags.append("explicit_approval_wording")
    if any(token in lowered for token in ("revoke", "revoked", "ถอนสิทธิ์", "ยกเลิกการอนุมัติ")):
        flags.append("revoked_approval_wording")
    if any(token in lowered for token in ("conflict", "contradict", "แต่", "however", "แม้ว่า")):
        flags.append("conflicting_context_wording")
    if any(token in lowered for token in ("```", "--", "{", "curl ", "python ", "git ", "kubectl ", "tool")):
        flags.append("code_or_tool_payload")
    if re.search(r"[\u0e00-\u0e7f]", text) and re.search(r"[A-Za-z]", text):
        flags.append("mixed_language")
    if len(text.split()) >= 60:
        flags.append("long_context")
    return flags or ["plain_context"]


def predicted_label(question: dict, answer: dict, probabilities: np.ndarray) -> str:
    if question["type"] == "choice":
        return str(answer["choice"])
    if question["type"] == "noul":
        return "true" if float(answer["noul"]) >= 0.5 else "false"
    return str(int(np.argmax(probabilities)))


def increment(bucket: dict, key: str, correct: bool) -> None:
    bucket[key]["total"] += 1
    bucket[key]["correct"] += int(correct)


def finalize(bucket: dict) -> dict:
    return {
        key: {
            **values,
            "errors": values["total"] - values["correct"],
            "accuracy": values["correct"] / max(1, values["total"]),
            "error_rate": (values["total"] - values["correct"]) / max(1, values["total"]),
        }
        for key, values in sorted(bucket.items())
    }


def main() -> None:
    args = parse_args()
    cases = read_cases(args.data)
    agent = laya.Agent(args.model, device=args.device)
    by_primitive = defaultdict(lambda: {"total": 0, "correct": 0})
    by_language = defaultdict(lambda: {"total": 0, "correct": 0})
    by_domain = defaultdict(lambda: {"total": 0, "correct": 0})
    by_difficulty = defaultdict(lambda: {"total": 0, "correct": 0})
    by_flag = defaultdict(lambda: {"total": 0, "correct": 0})
    confusion = defaultdict(lambda: {"total": 0, "correct": 0})
    errors: list[dict] = []

    for case in cases:
        result = agent.predict(case["state"], case["questions"])
        state_text = request_text(case["state"])
        flags = wording_flags(state_text)
        domain = str(case.get("state", {}).get("domain", "unknown")) if isinstance(case["state"], dict) else "unknown"
        difficulty = str(case.get("state", {}).get("difficulty", "unknown")) if isinstance(case["state"], dict) else "unknown"
        for qid, question in case["questions"].items():
            answer = result["answers"][qid]
            gold = case["gold"][qid]
            predicted = predicted_label(question, answer, distribution(question, answer))
            expected = str(gold["label"]).lower()
            correct = predicted == expected
            qtype = question["type"]
            for bucket, key in (
                (by_primitive, qtype),
                (by_language, case["language"]),
                (by_domain, domain),
                (by_difficulty, difficulty),
                (confusion, f"{expected}->{predicted}"),
            ):
                increment(bucket, key, correct)
            for flag in flags:
                increment(by_flag, flag, correct)
            if not correct:
                errors.append(
                    {
                        "case_id": case["id"],
                        "question_id": qid,
                        "primitive": qtype,
                        "language": case["language"],
                        "domain": domain,
                        "difficulty": difficulty,
                        "expected": expected,
                        "predicted": predicted,
                        "flags": flags,
                        "request": state_text,
                    }
                )

    report = {
        "model": args.model,
        "dataset": args.data,
        "device": args.device or "auto",
        "cases": len(cases),
        "decisions": sum(len(case["questions"]) for case in cases),
        "errors": len(errors),
        "error_rate": len(errors) / max(1, sum(len(case["questions"]) for case in cases)),
        "by_primitive": finalize(by_primitive),
        "by_language": finalize(by_language),
        "by_domain": finalize(by_domain),
        "by_difficulty": finalize(by_difficulty),
        "by_wording_flag": finalize(by_flag),
        "confusion": finalize(confusion),
        "error_examples": errors,
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("cases", "decisions", "errors", "error_rate")}, indent=2))


if __name__ == "__main__":
    main()
