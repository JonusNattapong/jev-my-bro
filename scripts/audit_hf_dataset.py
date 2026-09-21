"""Audit every case in the provenance-aware expanded dataset.

The audit is intentionally conservative: it flags questionable cases instead
of silently changing human or source labels. Rule-based variants may be fixed
with ``--fix-generated`` when their own invariant is violated.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevbro.schema import read_cases


SPLITS = ("train", "validation", "calibration", "test")
VARIANTS = {"base", "approval_context", "revoked_approval", "conflict_context", "ood_mixed_indirect"}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def has_thai(text: str) -> bool:
    return bool(re.search(r"[\u0e00-\u0e7f]", text))


def has_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit every expanded jev-my-bro case")
    parser.add_argument("--root", default="data/hf_expanded")
    parser.add_argument("--report", default="artifacts/hf-dataset-audit.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    by_split: dict[str, list[dict]] = {split: read_cases(Path(args.root) / f"{split}.jsonl") for split in SPLITS}
    all_cases = [case for cases in by_split.values() for case in cases]
    flags: list[dict] = []
    requests: dict[str, list[str]] = defaultdict(list)
    families: dict[str, set[str]] = defaultdict(set)
    source_counts = Counter()
    variant_counts = Counter()
    action_counts = Counter()
    language_counts = Counter()

    for split, cases in by_split.items():
        for case in cases:
            state = case["state"]
            text = str(state["request"])
            variant = state.get("variant_type", "unknown")
            source = state.get("source_dataset", "unknown")
            family = state.get("scenario_family")
            requests[normalize(text)].append(case["id"])
            if family:
                families[family].add(split)
            source_counts[source] += 1
            variant_counts[variant] += 1
            action_counts[case["gold"]["action"]["label"]] += 1
            language_counts[case["language"]] += 1

            def flag(code: str, detail: str) -> None:
                flags.append({"case_id": case["id"], "split": split, "code": code, "detail": detail})

            required = ("source_dataset", "source_license", "attribution", "scenario_family", "variant_type", "label_status")
            for key in required:
                if not state.get(key):
                    flag("missing_provenance", key)
            # Thai commands often contain only a few whitespace-separated
            # segments. Keep short commands as useful hard cases; only an
            # actually empty/near-empty request is invalid.
            if len(text.strip()) < 4:
                flag("too_short", "request has fewer than four characters")
            if variant not in VARIANTS and state.get("label_status") == "rule_reviewed":
                flag("unknown_variant", variant)

            action = case["gold"]["action"]["label"]
            review = case["gold"]["needs_review"]["label"] == "true"
            prohibited = case["gold"]["prohibited"]["label"] == "true"
            if prohibited and action != "reject":
                flag("prohibited_action_mismatch", f"prohibited=true but action={action}")
            if variant in {"revoked_approval", "conflict_context"} and not (action == "ask_user" and review):
                flag("approval_invariant", f"variant requires ask_user/review, got {action}/{review}")
            if variant == "approval_context" and not prohibited and action != "execute":
                flag("approval_invariant", f"explicit approval should execute when not prohibited, got {action}")
            if variant == "ood_mixed_indirect" and not (has_thai(text) and has_latin(text)):
                flag("mixed_language_missing", "OOD mixed variant lacks both Thai and Latin text")
            if variant == "mixed_language" and not (has_thai(text) and has_latin(text)):
                flag("mixed_language_missing", "mixed variant lacks both Thai and Latin text")
            if case["language"] == "th" and not has_thai(text):
                flag("language_mismatch", "Thai case has no Thai characters")
            if case["language"] == "en" and not has_latin(text):
                flag("language_mismatch", "English case has no Latin characters")

    for request, ids in requests.items():
        if len(ids) > 1:
            for case_id in ids:
                flags.append({"case_id": case_id, "code": "duplicate_request", "detail": f"{len(ids)} normalized matches"})
    for family, split_set in families.items():
        if len(split_set) > 1:
            flags.append({"family": family, "code": "family_split_leakage", "detail": sorted(split_set)})

    report = {
        "root": args.root,
        "cases": len(all_cases),
        "decisions": len(all_cases) * 4,
        "source_counts": dict(source_counts),
        "variant_counts": dict(variant_counts),
        "action_counts": dict(action_counts),
        "language_counts": dict(language_counts),
        "scenario_families": len(families),
        "flags": flags,
        "flag_count": len(flags),
        "label_status_note": "rule_reviewed is conservative automated rubric output, not human annotation",
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("cases", "decisions", "scenario_families", "flag_count")}, indent=2))
    if flags:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
