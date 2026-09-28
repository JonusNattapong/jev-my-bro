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

431 cases (409 English, 22 Thai). The directory name reflects the 400-case target plus the
over-apportioned remainder; it is not a claim of 400.

| Split | Cases | Decisions | Risk spread (0→4) | Actions (execute / ask_user / reject) |
| --- | ---: | ---: | --- | --- |
| train | 321 | 1284 | 94 / 42 / 55 / 52 / 78 | 134 / 88 / 99 |
| validation | 40 | 160 | 12 / 5 / 7 / 6 / 10 | 14 / 14 / 12 |
| calibration | 34 | 136 | 10 / 4 / 6 / 6 / 8 | 13 / 12 / 9 |
| test | 36 | 144 | 10 / 5 / 6 / 6 / 9 | 14 / 8 / 14 |

Split independence is enforced by unique case ids, unique request text, and unique
`scenario_family`. Every split contains all five risk levels and all three actions, which an earlier
apportionment failed to achieve. Verify with:

```powershell
python scripts/validate_dataset.py --root data/tool_call_400
```

## Honest limitations

- **The test split now settles the question: the model fails on tool calls.** Measured on the
  36-case test split, action accuracy is 36.1% against a 38.9% majority baseline, i.e. below
  guessing, and the risk head is flat at 1.54 / 1.69 / 1.93 / 2.01 / 2.10 across gold levels 0
  to 4, a spread of 0.565 against 1.474 on the model's own distribution. Enforcement criterion 5
  is confirmed failing, and this is the first split large enough to say so.
- **431 cases is 22% of the combined corpus** with `th_curated_1200`, the low end of the range
  where a shift becomes plausible. The corpus is now large enough to train on, not large enough to
  assume it will work.
- **Thai coverage is 22 cases**, which is enough to stop the gap being total and not enough to
  learn from. Real sessions are Thai-speaking and the model was trained on Thai prose, so a Thai
  request carrying a literal command is a real production surface.
- **Batch 2 is PowerShell and Windows heavy** and batch 3 adds POSIX depth, but the two are not
  balanced across the corpus, and the hook's hard safety floor is still POSIX-shaped.
- **Labels are development data**, not observed human decisions, and have had only one reviewer.
  The user asked for a second reviewer before this reaches a training mix; that has not happened.


## Integration

Referenced by `configs/colab-th1200-toolcall.yaml` and `configs/colab-th1200-toolcall-smoke.yaml`.
`--train` and `--validation` accept a comma-separated path list and load each corpus separately, so
split provenance is preserved and the run prints a per-corpus sequence-count breakdown.

Keep `data/th_curated_1200/test.jsonl` and `data/tool_call_400/test.jsonl` out of training and
calibration; a test asserts the configs never reference them.
