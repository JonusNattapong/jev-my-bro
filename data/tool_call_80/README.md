# Tool-call dataset slice (`tool_call_80`)

English tool-call governance cases: literal shell command strings and file-edit descriptions of the
shape the `PreToolUse` hook actually sends, at the length distribution seen in real agent sessions.

## Why this slice exists

The active corpus `data/th_curated_1200` is Thai prose with a mean request length of 70 characters
and a maximum of 140. Contexts sent to `jev_task_start` in real sessions reach 460 characters, and
the hook emits strings like `Edit documentation file docs/X.md writing: ...`. That surface is
completely absent from the existing corpus, and the measured behaviour matches the gap: the same
request scored `prohibited` 0.238 to 0.736 across four paraphrases, and a read-only `git status`
scored higher than `rm -rf /`.

## Contents

| Split | Cases | Decisions |
| --- | ---: | ---: |
| train | 56 | 224 |
| validation | 8 | 32 |
| calibration | 8 | 32 |
| test | 8 | 32 |

Cases are round-robin interleaved across risk levels before splitting, so every split has the same
risk profile. Split independence is enforced by unique case ids, unique request text, and unique
`scenario_family`; verify with:

```powershell
python scripts/validate_dataset.py --root data/tool_call_80
```

## Honest limitations

- **80 cases is a seed, not a training set.** The model needs far more than this to learn the
  tool-call surface, and 8 test cases cannot support the statistical claim required by
  `docs/ENFORCEMENT_GATE.md`. Expand before evaluating.
- **English only.** Thai tool-call coverage is still missing.
- **Authored, not derived from real telemetry.** Labels reflect the documented semantics of the four
  primitives, not observed human decisions. The bootstrap labels are development data.
- Risk and action labels were assigned by hand; each split's `reject` share is small, so per-class
  recall on the current test split is not meaningful.

## Integration

This slice is not referenced by any training config yet. To include it, extend the `train` and
`validation` paths in a config (for example `configs/colab-th1200.yaml`) rather than merging files,
so split provenance stays auditable. Keep `data/th_curated_1200/test.jsonl` untouched: it is the
locked split used for the existing reported metrics.
