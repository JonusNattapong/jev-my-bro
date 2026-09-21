# AGENTS.md

## Mission

jev-my-bro is a self-hosted typed decision model specialized for agent/tool governance.

## Current architecture

- Foundation: Laya 0.3.4
- Base checkpoint: convaiinnovations/laya-multilingual
- Languages in bootstrap training data: English and Thai
- Primitives: choice, noul, score
- Training: RLCD plus soft-target cross entropy
- Calibration: dedicated calibration split, one temperature per question type
- Runtime: native Laya/PyTorch model served through FastAPI
- Optional integration gateway: Go reverse proxy
- Legacy baseline: baseline/deberta

## Definition of done for model changes

1. Preserve split independence: train, validation, calibration, test.
2. Never fit calibration on validation or test.
3. Keep probability targets normalized.
4. Run scripts/validate_dataset.py.
5. Run pytest for repository tests.
6. Compile all changed Python modules.
7. Run Go tests for server/go when the gateway changes.
8. Do not claim trained-model metrics unless training/evaluation actually ran.
9. Do not commit model weights, tokens, secrets, or Colab credentials.

## Data policy

The checked-in JSONL files are the source of truth. Do not add a dataset generator unless explicitly requested. Bootstrap labels are development data, not a production authorization policy.

## Upstream

Laya is an Apache-2.0 dependency. Keep THIRD_PARTY.md current when upstream usage changes.
