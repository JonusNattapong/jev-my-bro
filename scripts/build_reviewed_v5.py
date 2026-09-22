"""Build the evidence-first v5 dataset from curated rows and reviewed variants.

This is a deterministic curation step, not a label generator. The original
JSONL files remain untouched. HF base rows are excluded because their imported
labels are heavily skewed; only rows with explicit approval/revocation/conflict
or OOD context are retained from the expanded source.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "validation", "calibration", "test")
REVIEWED_VARIANTS = {
    "approval_context",
    "revoked_approval",
    "conflict_context",
    "ood_mixed_indirect",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def has_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def add_evidence_fields(row: dict[str, Any]) -> dict[str, Any]:
    state = dict(row["state"])
    request = str(state.get("request", ""))
    state["evidence"] = {
        "authorization_signal": (
            "revoked"
            if has_any(request, (r"revok", r"ยกเลิกสิทธิ์", r"ยกเลิกการอนุมัติ"))
            else "conflict"
            if has_any(request, (r"conflict", r"ขัดแย้ง", r"หยุดและยืนยัน"))
            else "explicit"
            if has_any(request, (r"approved", r"approval", r"authorized", r"อนุมัติ", r"ได้รับอนุญาต"))
            else "unknown"
        ),
        "environment_signal": (
            "production"
            if has_any(request, (r"\bprod(?:uction)?\b", r"production", r"production"))
            else "staging"
            if has_any(request, (r"\bstaging\b", r"ทดสอบ"))
            else "local"
            if has_any(request, (r"\blocal\b", r"ในเครื่อง", r"working tree"))
            else "unknown"
        ),
        "scope_signal": (
            "broad"
            if has_any(request, (r"public", r"shared", r"all users", r"ทุกคน", r"วงกว้าง"))
            else "bounded"
            if has_any(request, (r"single", r"one ", r"เฉพาะ", r"ขอบเขตแคบ"))
            else "unknown"
        ),
        "reversibility_signal": (
            "reversible"
            if has_any(request, (r"rollback", r"revers", r"backup", r"recover", r"ย้อนกลับ", r"กู้คืน"))
            else "destructive"
            if has_any(request, (r"delete", r"destroy", r"wipe", r"ทำลาย", r"ลบข้อมูล"))
            else "unknown"
        ),
    }
    enriched = dict(row)
    enriched["state"] = state
    return enriched


def family_id(row: dict[str, Any]) -> str:
    state = row.get("state", {})
    return str(state.get("scenario_family") or f"static:{row['id']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("data"))
    parser.add_argument("--expanded", type=Path, default=Path("data/hf_expanded"))
    parser.add_argument("--output", type=Path, default=Path("data/reviewed_v5"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"selected_variants": sorted(REVIEWED_VARIANTS), "splits": {}}
    all_families: dict[str, str] = {}

    for split in SPLITS:
        base_rows = load_jsonl(args.base / f"{split}.jsonl")
        expanded_rows = load_jsonl(args.expanded / f"{split}.jsonl")
        selected = [
            row
            for row in expanded_rows
            if row.get("state", {}).get("variant_type") in REVIEWED_VARIANTS
        ]

        rows = [add_evidence_fields(row) for row in base_rows + selected]
        seen_ids: set[str] = set()
        for row in rows:
            if row["id"] in seen_ids:
                raise ValueError(f"duplicate id in {split}: {row['id']}")
            seen_ids.add(row["id"])
            family = family_id(row)
            previous = all_families.get(family)
            if previous is not None and previous != split:
                raise ValueError(f"scenario family leaks across splits: {family} ({previous}, {split})")
            all_families[family] = split

        (args.output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        report["splits"][split] = {
            "cases": len(rows),
            "curated_cases": len(base_rows),
            "selected_expanded_cases": len(selected),
            "actions": dict(Counter(row["gold"]["action"]["label"] for row in rows)),
            "risks": dict(Counter(row["gold"]["risk"]["label"] for row in rows)),
        }

    report["scenario_families"] = len(all_families)
    (args.output / "CURATION_REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
