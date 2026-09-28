# Tool-call dataset slice (`tool_call_400`)

Hand-authored English governance cases for the surface the `PreToolUse` hook actually sends:
literal shell command strings and file-edit descriptions, at the length distribution seen in real
agent sessions.

Supersedes `data/tool_call_80`, which was a 56-case seed. That seed is kept for provenance.

## Why this slice exists

The active corpus `data/th_curated_1200` is Thai prose with a mean request length of 70 characters
and a maximum of 140. Contexts sent to `jev_task_start` in real sessions reach 460 characters, and
the hook emits strings like `Edit documentation file docs/X.md writing: ...` and
`Bash: git push --force origin main`. That surface is absent from the existing corpus, and the
measured behaviour matches the gap: on `tool_call_80/test.jsonl` the shipped model's risk-head spread
fell to 0.618 against 1.474 on its own distribution, which is enforcement criterion 5 failing.

## Provenance

Every case was written individually: the request text, the action, `needs_review`, `prohibited`, and
the risk level were each chosen by hand from the documented semantics of the four primitives. No
case is a template expansion, a parameter sweep, or a label derived by rule. The formatter only
serialises the literals to JSONL and apportions the splits.

The risk labels follow the published legend: 0 read-only and local, 1 scoped and reversible, 2
bounded shared side effect with practical recovery, 3 production, access, financial, or broad
external impact, 4 irreversible, destructive, or credential-exposing.

## Current contents

| Split | Cases | Decisions | Risk spread (0→4) | Actions (execute / ask_user / reject) |
| --- | ---: | ---: | --- | --- |
| train | 240 | 960 | 73 / 35 / 34 / 36 / 62 | 106 / 62 / 72 |
| validation | 30 | 120 | 9 / 4 / 4 / 5 / 8 | 13 / 5 / 12 |
| calibration | 26 | 104 | 8 / 4 / 4 / 4 / 6 | 12 / 5 / 9 |
| test | 28 | 112 | 9 / 4 / 4 / 4 / 7 | 10 / 9 / 9 |

Split independence is enforced by unique case ids, unique request text, and unique
`scenario_family`. Every split contains all five risk levels and all three actions, which an earlier
apportionment failed to achieve. Verify with:

```powershell
python scripts/validate_dataset.py --root data/tool_call_400
```

## Honest limitations

- **324 cases, not 400.** The directory name is the target, not the count. Another 76 hand-written
  cases are still needed. At 324 the slice is 21% of the combined corpus with `th_curated_1200`,
  which is inside the 25-40% range where a shift in tool-call behaviour becomes plausible, but it
  is the low end of that range.
- **28 test cases can show a gap, not confirm a pass.** Measured on this split the shipped model
  scores 42.9% action accuracy against a 35.7% majority baseline, and its risk head is flat: mean
  prediction 1.73 / 1.92 / 2.06 / 2.14 / 1.99 for gold levels 0 to 4, a spread of 0.403 against
  1.474 on the model's own distribution. That is enforcement criterion 5 failing, now on a split
  large enough to mean something.
- **English only.** Thai tool-call coverage is still missing.
- **Risk 0 is now over-represented** (73 of 240 training cases) because batch 2 added a long tail of
  read-only commands to balance the action distribution. Risk 4 remains high at 62. Both ends are
  deliberately above the middle levels, which pushes the risk marginal outward.
- **Labels are development data**, not observed human decisions, and have had only one reviewer.
- **Batch 2 is PowerShell and Windows heavy**, which matches the machine the hook runs on but not
  the POSIX agents the hook also governs.

## Integration

Referenced by `configs/colab-th1200-toolcall.yaml` and `configs/colab-th1200-toolcall-smoke.yaml`.
`--train` and `--validation` accept a comma-separated path list and load each corpus separately, so
split provenance is preserved and the run prints a per-corpus sequence-count breakdown.

Keep `data/th_curated_1200/test.jsonl` and `data/tool_call_400/test.jsonl` out of training and
calibration; a test asserts the configs never reference them.
