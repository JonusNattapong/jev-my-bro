from __future__ import annotations

from typing import Iterable

from laya.common import QTYPES, build_sequence, render_options

from jevbro.schema import read_cases


def target_distribution(question: dict, gold: dict) -> list[float]:
    qtype = question["type"]
    probabilities = gold["probabilities"]
    if qtype == "choice":
        values = [float(probabilities[key]) for key in question["criteria"]]
    elif qtype == "noul":
        values = [float(probabilities["false"]), float(probabilities["true"])]
    else:
        values = [float(probabilities[str(index)]) for index in range(len(question["criteria"]))]
    total = sum(values)
    if total <= 0:
        raise ValueError("target distribution has zero mass")
    return [value / total for value in values]


def build_item(tokenizer, cfg: dict, case: dict, qid: str) -> dict | None:
    question = case["questions"][qid]
    gold = case["gold"][qid]
    qtype = question["type"]
    criteria = question.get("criteria")
    internal = {"t": qtype, "ins": question["instructions"], "crit": criteria}

    sequence, markers = build_sequence(
        tokenizer,
        case["state"],
        internal,
        int(cfg.get("max_len", 1024)),
        int(cfg.get("head_max_len", 256)),
    )
    expected_options = len(render_options(internal))
    if len(markers) != expected_options:
        return None

    target = target_distribution(question, gold)
    return {
        "ids": sequence,
        "markers": markers,
        "qtype": QTYPES[qtype],
        "target": target,
        "label": max(range(len(target)), key=target.__getitem__),
        "case_id": case["id"],
        "qid": qid,
        "language": case["language"],
    }


def build_items(tokenizer, cfg: dict, cases: Iterable[dict]) -> list[dict]:
    items: list[dict] = []
    skipped = 0
    for case in cases:
        for qid in case["questions"]:
            item = build_item(tokenizer, cfg, case, qid)
            if item is None:
                skipped += 1
            else:
                items.append(item)
    if skipped:
        print(f"[data] skipped {skipped} questions because options exceeded head_max_len")
    if not items:
        raise ValueError("no training items were produced")
    return items


def load_items(tokenizer, cfg: dict, path: str) -> list[dict]:
    return build_items(tokenizer, cfg, read_cases(path))
