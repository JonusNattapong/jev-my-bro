"""Verify that writer/scenario groups never cross dataset splits."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    groups: dict[str, set[str]] = defaultdict(set)
    for path in sorted(args.root.glob("*.jsonl")):
        split = path.stem
        for line in path.read_text(encoding="utf-8").splitlines():
            case = json.loads(line)
            state = case.get("state", {})
            group = state.get("writer_id") or state.get("scenario_family")
            if not group:
                raise SystemExit(f"{path}: missing writer_id/scenario_family for {case.get('id')}")
            groups[str(group)].add(split)

    leaked = {group: sorted(splits) for group, splits in groups.items() if len(splits) > 1}
    if leaked:
        raise SystemExit(f"groups cross splits: {leaked}")
    print(f"split groups: OK ({len(groups)} groups, no cross-split leakage)")


if __name__ == "__main__":
    main()
