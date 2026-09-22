# AGENTS.md

## Mission

jev-my-bro is a self-hosted typed decision model specialized for agent/tool governance.

## Current architecture

- Foundation: Laya 0.3.4
- Base checkpoint: convaiinnovations/laya-multilingual
- Languages in bootstrap training data: English and Thai
- Primitives: choice, noul, score
- Training: RLCD plus ordinal soft-target cross entropy and direct RPS for score
- Calibration: dedicated calibration split, one temperature per question type
- Runtime: native Laya/PyTorch model served through FastAPI
- Agent integration: shared MCP v2 Streamable HTTP adapter in `jevbro/mcp_server.py`
- Optional integration gateway: Go reverse proxy
- Legacy baseline: baseline/deberta

## Coding-agent use of Jev

- Use `jev_decide` for bounded operational/governance choices where an independent learned signal is useful.
- Jev is advisory and must not override repository policy, explicit approvals, tests, or safety boundaries.
- If Jev returns `abstain: true`, rely on stronger evidence, policy, or human review rather than its raw decision.
- Do not use Jev as a code generator; the coding agent remains responsible for implementation and verification.
- For every non-trivial coding task, call `jev_task_start` before implementation and preserve its `task_id`.
- Pass that `task_id` to any additional `jev_decide` calls made during the task; one task may have multiple Jev decisions.
- Implement and run tests with the coding agent's normal tools. Jev lifecycle tools never execute shell commands.
- Before finishing successfully, call `jev_task_complete` with the final choice, changed files, actual test evidence, exit code, duration, and commit SHA when available.
- If the task is blocked or fails, call `jev_task_fail` with the reason and any verification evidence collected so far.
- Set `tests_passed=true` only after relevant tests actually pass, `false` only after an observed failure, and leave it null when tests were not run.
- `jev_feedback_*` and `jev_record_outcome` remain compatibility APIs; new tasks use the task lifecycle.

## Definition of done for model changes

1. Preserve split independence: train, validation, calibration, test.
2. Never fit calibration on validation or test.
3. Keep probability targets normalized.
4. Run `python scripts/validate_dataset.py --root data/hf_expanded` for the active dataset.
5. Run pytest for repository tests.
6. Compile all changed Python modules.
7. Run Go tests for server/go when the gateway changes.
8. Do not claim trained-model metrics unless training/evaluation actually ran.
9. Do not commit model weights, tokens, secrets, or Colab credentials.

## Data policy

The checked-in JSONL files are the source of truth. Do not add a dataset generator unless explicitly requested. Bootstrap labels are development data, not a production authorization policy.

## Upstream

Laya is an Apache-2.0 dependency. Keep `docs/THIRD_PARTY.md` current when upstream usage changes.
