from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevbro.schema import read_cases, summarize

SPLITS = ("train", "validation", "calibration", "test")


def main() -> None:
    contexts: dict[str, set[str]] = {}
    ids: set[str] = set()
    for split in SPLITS:
        path = Path("data") / f"{split}.jsonl"
        cases = read_cases(path)
        summary = summarize(cases)
        print(
            f"{split:11} cases={summary['cases']:4} decisions={summary['decisions']:4} "
            f"languages={summary['languages']} actions={summary['actions']}"
        )
        local_contexts = set()
        for case in cases:
            if case["id"] in ids:
                raise SystemExit(f"duplicate case id across splits: {case['id']}")
            ids.add(case["id"])
            request = case["state"]["request"] if isinstance(case["state"], dict) else str(case["state"])
            if request in local_contexts:
                raise SystemExit(f"{split}: duplicate request text")
            local_contexts.add(request)
        contexts[split] = local_contexts

    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = contexts[left] & contexts[right]
            if overlap:
                raise SystemExit(f"{left}/{right}: {len(overlap)} overlapping requests")

    print("dataset validation: OK")


if __name__ == "__main__":
    main()
