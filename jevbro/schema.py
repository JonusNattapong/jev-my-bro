from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

QTYPES = {"choice", "score", "noul"}
EXPECTED_IDS = {"action", "needs_review", "prohibited", "risk"}


class DatasetError(ValueError):
    pass


def read_cases(path: str | Path) -> list[dict]:
    path = Path(path)
    cases: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            validate_case(case, source=f"{path}:{line_no}")
            cases.append(case)
    if not cases:
        raise DatasetError(f"{path}: empty dataset")
    return cases


def _assert_distribution(values: dict, expected_keys: Iterable[str], source: str) -> None:
    expected = list(expected_keys)
    if list(values.keys()) != expected:
        raise DatasetError(f"{source}: probability keys {list(values)} != {expected}")
    probs = [float(values[key]) for key in expected]
    if any(value < 0.0 or value > 1.0 for value in probs):
        raise DatasetError(f"{source}: probabilities must be in [0, 1]")
    if abs(sum(probs) - 1.0) > 1e-5:
        raise DatasetError(f"{source}: probabilities sum to {sum(probs):.8f}, not 1")


def validate_case(case: dict, source: str = "case") -> None:
    if not isinstance(case.get("id"), str) or not case["id"]:
        raise DatasetError(f"{source}: missing id")
    if case.get("language") not in {"en", "th"}:
        raise DatasetError(f"{source}: language must be en or th")
    if not isinstance(case.get("state"), (str, dict, list)):
        raise DatasetError(f"{source}: state must be text, object, or list")

    questions = case.get("questions")
    gold = case.get("gold")
    if not isinstance(questions, dict) or set(questions) != EXPECTED_IDS:
        raise DatasetError(f"{source}: expected question ids {sorted(EXPECTED_IDS)}")
    if not isinstance(gold, dict) or set(gold) != set(questions):
        raise DatasetError(f"{source}: gold must match question ids")

    for qid, question in questions.items():
        qtype = question.get("type")
        if qtype not in QTYPES:
            raise DatasetError(f"{source}/{qid}: invalid type {qtype!r}")
        if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
            raise DatasetError(f"{source}/{qid}: missing instructions")

        answer = gold[qid]
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict):
            raise DatasetError(f"{source}/{qid}: missing probability target")

        if qtype == "choice":
            criteria = question.get("criteria")
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise DatasetError(f"{source}/{qid}: choice requires >=2 criteria")
            _assert_distribution(probabilities, criteria.keys(), f"{source}/{qid}")
            if answer.get("label") not in criteria:
                raise DatasetError(f"{source}/{qid}: label not in criteria")
        elif qtype == "noul":
            _assert_distribution(probabilities, ("false", "true"), f"{source}/{qid}")
            if str(answer.get("label")).lower() not in {"false", "true"}:
                raise DatasetError(f"{source}/{qid}: noul label must be true/false")
            expected = float(probabilities["true"])
            if abs(float(answer.get("noul", expected)) - expected) > 1e-5:
                raise DatasetError(f"{source}/{qid}: noul target disagrees with probabilities")
        else:
            criteria = question.get("criteria")
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise DatasetError(f"{source}/{qid}: score requires >=2 levels")
            keys = [str(index) for index in range(len(criteria))]
            _assert_distribution(probabilities, keys, f"{source}/{qid}")
            if str(answer.get("label")) not in keys:
                raise DatasetError(f"{source}/{qid}: score label out of range")


def summarize(cases: list[dict]) -> dict:
    languages = Counter(case["language"] for case in cases)
    actions = Counter(case["gold"]["action"]["label"] for case in cases)
    qtypes = Counter()
    for case in cases:
        qtypes.update(question["type"] for question in case["questions"].values())
    return {
        "cases": len(cases),
        "decisions": sum(len(case["questions"]) for case in cases),
        "languages": dict(languages),
        "actions": dict(actions),
        "qtypes": dict(qtypes),
    }
