from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevbro.schema import read_cases, summarize

SPLITS = ("train", "validation", "calibration", "test")

ACTIONS = ("execute", "ask_user", "reject")


def needs_review_degeneracy(cases: list[dict]) -> str | None:
    """Report when needs_review is fully determined by the action label.

    If needs_review == (action != execute) in every case, the question carries no information
    the action head does not already carry. The gate turns needs_review >= 0.5 into ask_user, so
    a degenerate corpus makes that rule mean "ask when the model is comfortable", which is
    backwards. Both shipped corpora are in this state, which is why the measured separation on
    the tool-call split is negative.
    """
    table: dict[tuple[str, str], int] = defaultdict(int)
    for case in cases:
        table[
            (case["gold"]["action"]["label"], str(case["gold"]["needs_review"]["label"]).lower())
        ] += 1
    exceptions = 0
    for action in ACTIONS:
        if action == "execute":
            exceptions += table.get((action, "true"), 0)
        else:
            exceptions += table.get((action, "false"), 0)
    if exceptions:
        return None
    detail = ", ".join(
        f"{action}:{table.get((action, 'true'), 0)}/{table.get((action, 'false'), 0)}"
        for action in ACTIONS
    )
    return (
        "needs_review is a deterministic function of the action label "
        f"(true/false per action -> {detail}). The noul needs_review head cannot carry an "
        "independent signal, and the ask_user gate rule becomes inverted."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument(
        "--allow-degenerate-needs-review",
        action="store_true",
        help="report the needs_review degeneracy as a warning instead of failing",
    )
    args = parser.parse_args()
    root = Path(args.root)
    contexts: dict[str, set[str]] = {}
    families: dict[str, set[str]] = {}
    ids: set[str] = set()
    degenerate_splits: list[str] = []
    for split in SPLITS:
        path = root / f"{split}.jsonl"
        cases = read_cases(path)
        summary = summarize(cases)
        print(
            f"{split:11} cases={summary['cases']:4} decisions={summary['decisions']:4} "
            f"languages={summary['languages']} actions={summary['actions']}"
        )
        finding = needs_review_degeneracy(cases)
        if finding:
            degenerate_splits.append(split)
            message = f"{split}: {finding}"
            if args.allow_degenerate_needs_review:
                print(f"WARNING {message}")
            else:
                raise SystemExit(
                    message
                    + " Relabel so needs_review varies independently, or pass "
                    "--allow-degenerate-needs-review to acknowledge it."
                )
        local_contexts = set()
        local_families = set()
        for case in cases:
            if case["id"] in ids:
                raise SystemExit(f"duplicate case id across splits: {case['id']}")
            ids.add(case["id"])
            request = case["state"]["request"] if isinstance(case["state"], dict) else str(case["state"])
            if request in local_contexts:
                raise SystemExit(f"{split}: duplicate request text")
            local_contexts.add(request)
            family = case.get("state", {}).get("scenario_family") if isinstance(case.get("state"), dict) else None
            if family:
                local_families.add(family)
        contexts[split] = local_contexts
        families[split] = local_families

    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = contexts[left] & contexts[right]
            if overlap:
                raise SystemExit(f"{left}/{right}: {len(overlap)} overlapping requests")
            family_overlap = families[left] & families[right]
            if family_overlap:
                raise SystemExit(f"{left}/{right}: {len(family_overlap)} overlapping scenario families")

    print("dataset validation: OK")


if __name__ == "__main__":
    main()
