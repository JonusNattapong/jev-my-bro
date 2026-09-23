# Documentation

This directory is the documentation index for `jev-my-bro`.

The repository includes the published v0.2 path plus the latest Thai
`laya-th960` training/evaluation experiment. Current development work includes
the Score v4 ordinal-risk redesign, split-safe calibration, and the shared
MCP/task-feedback workflow.

## Canonical paths

| Purpose | Path |
| --- | --- |
| Active provenance-aware dataset | `../data/hf_expanded/` |
| Bootstrap dataset | `../data/{train,validation,calibration,test}.jsonl` |
| Dataset card | `../data/hf_expanded/README.md` |
| Dataset provenance manifest | `../data/hf_expanded/SOURCE_MANIFEST.json` |
| Local serving checkpoint | `../artifacts/laya-model/` |
| Latest Score v4 reports | `../artifacts/v41/` |
| Latest Thai dataset | `../data/th_curated_1200/` |
| Latest Thai model card/results | [`../README.md`](../README.md) |
| Hugging Face Thai model card | `model-cards/jev-my-bro-th1200.md` |
| Thai th960 model card | `model-cards/jev-my-bro-th960.md` |
| Thai result visualization | `assets/laya-th960-results.svg` |
| Claude Code hook setup | [`CLAUDE_HOOK_SETUP.md`](CLAUDE_HOOK_SETUP.md) |
| CLI executable after editable install | `../.venv/Scripts/jev.exe` |
| MCP endpoint | `http://127.0.0.1:8787/mcp` |
| Feedback database | `../artifacts/feedback/jev_feedback.sqlite3` |

`artifacts/v41/` contains evaluation reports, not a model checkpoint. Use
`JonusNattapong/jev-my-bro-th1200` or `artifacts/laya-model/` with `--model`.

## Document map

| Document | Purpose |
| --- | --- |
| [Claude Hook Setup](CLAUDE_HOOK_SETUP.md) | Automated PreToolUse governance hook for Claude Code, Antigravity, and Cursor |
| [MCP integration](MCP_INTEGRATION.md) | Start the shared Jev MCP server and connect Codex, Claude Code, and OpenCode |
| [Feedback loop](FEEDBACK_LOOP.md) | Task lifecycle, SQLite feedback, active learning harvest, evaluation, and export |
| [Dataset expansion plan](DATASET_EXPANSION_PLAN.md) | Current 8,508-case dataset status, coverage rules, split isolation, and review gates |
| [Dataset review rubric](DATASET_REVIEW_RUBRIC.md) | Human-review rules for action/review/prohibited/risk labels |
| [Score v4](SCORE_V4.md) | Ordinal risk semantics, targets, loss, metrics, and current v4 experiment status |
| [Third-party foundations](THIRD_PARTY.md) | Laya dependency plus dataset source licenses and attribution |

Project overview and public-facing usage live in
[`../README.md`](../README.md). Agent-specific repository rules live in
[`../AGENTS.md`](../AGENTS.md).

## Current dataset status

The active dataset is `../data/hf_expanded/`:

| Split | Cases |
| --- | ---: |
| Train | 5,731 |
| Validation | 879 |
| Calibration | 944 |
| Test | 954 |
| **Total** | **8,508** |

The dataset combines the 1,008 project-authored bootstrap cases with 7,500
selected/transformed cases from MASSIVE Thai, BANKING77, and Hermes function
calling. Imported governance labels are `rule_reviewed`, not human-reviewed.

Current audit status:

```text
cases              8508
decisions          34032
scenario_families  2508
audit flags        0
```

The Hugging Face dataset is
`JonusNattapong/jev-my-bro-dataset`.

The latest Thai model and dataset:
`JonusNattapong/jev-my-bro-th1200` (1,200 curated Thai cases including agent contexts,
86.75% accuracy, 100% L4 risk recall) and
`JonusNattapong/jev-my-bro-th960` with its benchmark visualization in
[`assets/laya-th960-results.svg`](assets/laya-th960-results.svg).

## Local MCP quick start

From the repository root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps

.\.venv\Scripts\jev.exe serve --model JonusNattapong/jev-my-bro-th1200 --host 127.0.0.1 --port 8787 --quantize
```

Then connect clients to:

```text
http://127.0.0.1:8787/mcp
```

See [MCP_INTEGRATION.md](MCP_INTEGRATION.md) for client-specific setup.

## Verification

For documentation/data changes:

```powershell
python scripts/validate_dataset.py --root data/hf_expanded
python scripts/audit_hf_dataset.py --root data/hf_expanded
```

For code changes, also run the relevant Python tests/compile checks and Go tests
when `server/go` changes, following [`../AGENTS.md`](../AGENTS.md).

## Documentation maintenance

- Prefer repository-relative paths in Markdown.
- On Windows command examples, use `.\.venv\Scripts\jev.exe` as the canonical
  CLI path.
- Keep `THIRD_PARTY.md` in this `docs/` directory and update it when upstream
  dependencies or dataset sources change.
- Dataset licensing is documented by `../LICENSE`, `THIRD_PARTY.md`, the
  dataset card, and `SOURCE_MANIFEST.json`.
- Do not describe report-only directories such as `artifacts/v41/` as model
  checkpoints.
- Do not publish new model-quality metrics unless the referenced training and
  evaluation run actually exists.
