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
| train | 101 | 404 | 16 / 14 / 18 / 17 / 36 | 30 / 33 / 38 |
| validation | 17 | 68 | 3 / 2 / 3 / 3 / 6 | 4 / 5 / 8 |
| calibration | 15 | 60 | 2 / 2 / 3 / 3 / 5 | 4 / 4 / 7 |
| test | 13 | 52 | 2 / 2 / 2 / 2 / 5 | 4 / 4 / 5 |

Split independence is enforced by unique case ids, unique request text, and unique
`scenario_family`. Every split contains all five risk levels and all three actions, which an earlier
apportionment failed to achieve. Verify with:

```powershell
python scripts/validate_dataset.py --root data/tool_call_400
```

## Honest limitations

- **146 cases, not 400.** The directory name is the target, not the count. Another 254 hand-written
  cases are still needed before the slice can carry the weight of a training mix; at 146 it is
  7.6% of the combined corpus with `th_curated_1200`, against the 25–40% that would plausibly move
  the model's behaviour on tool calls.
- **13 test cases cannot decide criterion 5.** The split is balanced but tiny. It can show a
  regression, not confirm a pass.
- **English only.** Thai tool-call coverage is still missing.
- **Risk 4 is over-represented** (36 of 101 training cases) because catastrophic operations have
  many distinct spellings while safe reads do not. This is deliberate but it biases the action
  marginal toward `reject`.
- **Labels are development data**, not observed human decisions, and have not had a second reviewer.

## Integration

Referenced by `configs/colab-th1200-toolcall.yaml` and `configs/colab-th1200-toolcall-smoke.yaml`.
`--train` and `--validation` accept a comma-separated path list and load each corpus separately, so
split provenance is preserved and the run prints a per-corpus sequence-count breakdown.

Keep `data/th_curated_1200/test.jsonl` and `data/tool_call_400/test.jsonl` out of training and
calibration; a test asserts the configs never reference them.
