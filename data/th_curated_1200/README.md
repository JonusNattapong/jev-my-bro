---
pretty_name: Jev Thai curated 1,200
language:
- th
license: apache-2.0
task_categories:
- text-classification
tags:
- text
- thai
- governance
- safety
- risk-classification
size_categories:
- 1K<n<10K
---

# Jev Thai curated 1,200

Thai-language data for the next Jev training run. This release keeps the
1,080-case set unchanged and adds 120 training cases written in the style of
coding-agent task contexts: the long, multi-clause descriptions that agents
send to `jev_task_start` and `jev_decide`, not short one-line requests.

## Why

Every earlier training request is a single sentence (65 characters on
average, 114 at most). Real agent contexts are 120-460 characters and contain
meta phrases such as "ผู้ใช้สั่งให้ทำแล้ว" or "ไม่มีการ push". On those inputs
`jev-my-bro-th960` scores `prohibited` high for ordinary work. Probing the
nine distinct contexts recorded in the local feedback store gave:

| Context | `prohibited` |
|---|---:|
| Eight routine coding/training tasks | 0.43-0.75 (mean 0.62) |
| Force-push over shared `main` and delete a release branch | 0.78 |

At a 0.5 gate almost every routine task is rejected, and the prohibited head
barely separates routine work from a destructive request.

## Added cases

All 120 cases are in the train split, with IDs `th-agt-001` to `th-agt-120`,
in `th_17_agent_task_context_train.csv`.

| Action | Cases | Risk distribution | What they cover |
|---|---:|---|---|
| execute | 60 | 0: 20, 1: 25, 2: 15 | read-only inspection, local edits, tests, own-branch commits, pushes the user asked for to repos they own |
| ask_user | 36 | 2: 16, 3: 16, 4: 4 | scope beyond the instruction, shared branches, production, publishing, user work that is not committed |
| reject | 24 | 3: 10, 4: 14 | secret exfiltration, faked test results, bypassing hooks or branch protection, prompt injection from files |

The ask_user and reject cases use the same agent-context style as the execute
cases on purpose. The model should learn that the harm described in the
context decides the label; the long agent-style phrasing on its own should
not push the label either way.

## Provenance

The 120 added cases were written by Claude Code (`writer_id=author_claude`),
not by a human annotator, and have not yet been reviewed by one. Review them
against `docs/DATASET_REVIEW_RUBRIC.md` before training on this set.

The nine contexts in the table above were kept out of the training data so
they remain usable as a probe.

## Dataset summary

| Split | Cases | Decisions | Risk distribution |
|---|---:|---:|---|
| train | 1,200 | 4,800 | 0: 132, 1: 243, 2: 277, 3: 331, 4: 217 |
| validation | 100 | 400 | 20 at every level 0-4 |
| calibration | 100 | 400 | 20 at every level 0-4 |
| test | 100 | 400 | 20 at every level 0-4 |

Validation, calibration, and test are byte-identical to the 1,080-case
release, so headline metrics stay comparable. They contain no long
agent-style contexts, so they cannot measure this fix. Re-probe the feedback
store contexts after training to see whether routine-task `prohibited` scores
drop while the force-push case stays high.

## Build and validate

Run from the repository root:

```bash
python scripts/build_th_curated_1200.py
python scripts/validate_dataset.py --root data/th_curated_1200
```

The builder depends on the 1,080-case source files, including
`data/th_curated_960/th_16_jevbench_hard_cases_train.csv`.

## Limitations

These are development cases, not a production authorization policy. Keep
model predictions separate from the final authorization and policy gate.
