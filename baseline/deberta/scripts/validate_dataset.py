from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

LABELS = ("execute", "ask_user", "reject")
SPLITS = ("train", "validation", "calibration", "test")


def main() -> None:
    contexts_by_split: dict[str, set[str]] = {}
    failed = False

    for split in SPLITS:
        path = Path("data") / f"{split}.jsonl"
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                if row.get("label") not in LABELS:
                    raise SystemExit(f"{path}:{line_no}: invalid label")
                if row.get("options") != list(LABELS):
                    raise SystemExit(f"{path}:{line_no}: invalid options order")
                context = row.get("context")
                if not isinstance(context, str) or not context.strip():
                    raise SystemExit(f"{path}:{line_no}: invalid context")
                rows.append(row)

        contexts = [row["context"] for row in rows]
        if len(contexts) != len(set(contexts)):
            raise SystemExit(f"{path}: duplicate context inside split")
        contexts_by_split[split] = set(contexts)
        counts = Counter(row["label"] for row in rows)
        print(f"{split:11} total={len(rows):3} " + " ".join(f"{label}={counts[label]:3}" for label in LABELS))
        if set(counts) != set(LABELS):
            failed = True

    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            overlap = contexts_by_split[left] & contexts_by_split[right]
            if overlap:
                raise SystemExit(f"{left}/{right}: {len(overlap)} overlapping contexts")

    if failed:
        raise SystemExit("dataset is missing at least one label")
    print("dataset validation: OK")


if __name__ == "__main__":
    main()
