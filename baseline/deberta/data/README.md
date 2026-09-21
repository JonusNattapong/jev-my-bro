# Dataset

Static bootstrap dataset for jev-my-bro v0.1. There is no dataset generator in the repository.

## Decision semantics

- **execute**: the action is local/read-only/reversible, explicitly scoped and approved, or otherwise has no meaningful external side effect.
- **ask_user**: the action can be legitimate but causes an external, shared, destructive, financial, publication, access-control, or production side effect and the context does not establish sufficient approval.
- **reject**: the action is explicitly unauthorized, deceptive, privacy-invasive, malicious, evasive, fraudulent, or intended to bypass security/audit controls.

## Splits

- train: 384 (128 execute / 128 ask_user / 128 reject)
- validation: 96 (32 execute / 32 ask_user / 32 reject)
- calibration: 96 (32 execute / 32 ask_user / 32 reject)
- test: 96 (32 execute / 32 ask_user / 32 reject)

The calibration split is reserved for fitting temperature scaling only. The test split must remain untouched until final evaluation.

## Notes

The dataset intentionally contains hard negatives and lexical traps, such as safe read-only contexts containing words like `delete`, `API_KEY`, `authentication`, and `backdoor`. This reduces reliance on single-keyword shortcuts.

This is bootstrap data, not a production authorization policy. Real deployment requires reviewed operational examples, out-of-distribution evaluation, and feedback from actual outcomes.
