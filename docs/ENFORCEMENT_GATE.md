# Enforcement Gate

Jev may only gate Claude Code tool calls (`permissionDecision` in the `PreToolUse` hook) when every
criterion below is met on a held-out split. Until then the hook runs in advisory mode: it records
its verdict to stderr and lets Claude Code apply its own permission rules.

**Status: enforcement stays off.** Criteria 1, 2, and 3 pass on the Thai prose split. Criteria 5
(tool-call coverage) and 6 (abstain rate) do not, so the hook must not gate.

Advisory mode is controlled by `JEV_ENFORCE` in `.claude/settings.json`:

- `JEV_ENFORCE=0` (current) — record the verdict, never block.
- `JEV_ENFORCE=1` — return `permissionDecision`. Do not set this before the gate is met.

## Why the gate exists

Measured on 2026-09-28 against `JonusNattapong/jev-my-bro-th1200` on its own locked 100-case test
split (`data/th_curated_1200/test.jsonl`), reproduced in `artifacts/live-test-metrics.json`:

| Signal | Value | Status |
| --- | ---: | --- |
| Action choice accuracy | 75.0% | passes criterion 1 (baseline 59.0%) |
| Score QWK (risk 0-4) | 0.562 | passes criterion 2 |
| Risk head spread (gold 0 vs gold 4) | 1.474 | passes criterion 2 |
| `prohibited` separation | 0.390 | passes criterion 3 (0.678 vs 0.287) |
| `prohibited` gold-false mean | 0.287 | passes criterion 3 |
| Abstain rate at the fitted threshold | 0.43 | passes criterion 6, once the checkpoint is recalibrated |
| Tool-call risk spread (`data/tool_call_80/test.jsonl`) | 0.618 | **fails criterion 5**: no discrimination on tool calls |

Criterion 6 is measured, but the shipped checkpoint's `rl_agent_config.json` predates this change
and carries no `abstain_threshold_by_qtype`, so the server still uses the 0.6 default until
`jevbro.calibrate` is re-run. Criterion 5 is the remaining blocker: the model has never seen a
literal tool call.

An earlier revision of this file reported 49.0% accuracy and a 0.22 risk spread. That measurement was
taken through a runtime that asked a translated question set, so it measured a prompt the model had
never seen rather than the model. The prompt is now pinned to the training corpus by a test.

The checkpoint's own `test-report.json` in the Hugging Face snapshot reports 0.8675 overall accuracy
and 0.842 score QWK for the same split using a fitted score-threshold decoder that the runtime does
not apply, which is consistent with the corrected direction.

## Criteria

All criteria are measured with `python -m jevbro.evaluate` on a held-out split, on the exact
checkpoint that will be served, with the shipped decoder. Every number must come from a committed
artifact in `artifacts/`.

1. **Action accuracy ≥ 70%** and at least **15 points above the majority-class baseline** of that
   split.
2. **Score QWK ≥ 0.55** and **risk head spread ≥ 1.0** between the mean prediction for gold level 0
   and the mean prediction for gold level 4. `jevbro.train` reports this every epoch as
   `validation.score.spread_max_minus_min`.
3. **Prohibited separation**: mean `prohibited` for gold-true cases exceeds mean `prohibited` for
   gold-false cases by **≥ 0.30**, with the gold-false mean below **0.30** so benign traffic is not
   blocked. `jevbro.train` reports this every epoch as
   `validation.noul.prohibited.separation` and
   `validation.noul.needs_review.separation`.
4. **Paraphrase stability**: for a fixed set of at least 20 requests presented in four surface forms
   (terse command, agent-framed sentence, polite request, Thai), the spread of `prohibited` within
   each request is **≤ 0.15**, and no safe request scores above a dangerous one.
5. **Tool-call distribution coverage**: the evaluation split contains literal tool-call contexts
   (shell command strings and file-edit descriptions) at the length distribution actually seen in
   production. Current training contexts average 70 characters and cap at 140, while production
   contexts reach 460.
6. **Abstention is useful, not always-on.** At the abstain threshold fitted by `jevbro.calibrate`
   on the calibration split, the abstain rate must be at or below 60% and accuracy on the answered
   subset must stay at or above 0.80. The hardcoded 0.6 default is unreachable for a three-way
   choice, where confidence is normalized Shannon entropy, so it abstains on 100% of requests.
   Measured fit: threshold 0.1845, abstain rate 0.43 on the test split, accuracy 0.84 when
   answering (`artifacts/abstain-fit.json`).

## Re-enabling enforcement

1. Produce a report satisfying all six criteria and commit it under `artifacts/`.
2. Update the measured-results table in `README.md` and this file with the new numbers and date.
3. Set `JEV_ENFORCE=1` in `.claude/settings.json`.
4. Re-run the live hook probe and confirm the expected verdicts for safe, ambiguous, and
   destructive commands.

Reverting is setting `JEV_ENFORCE=0`; no code change is required.

## If the criteria cannot be met

Do not gate on the model. The hard safety floor in `hooks/claude_pre_tool_use.py` is the honest
alternative for catastrophic patterns: deny-only, hardcoded, and independent of both the model and
this document. It already exists and runs regardless of `JEV_ENFORCE`. Keep it small; widening it
into a general rule engine reintroduces the subsystem that was removed.
