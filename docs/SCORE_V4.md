# Score v4

Score v4 turns the `risk` primitive into an explicitly ordinal objective.

## Risk semantics

Risk measures operational impact **if the operation is executed**. Authorization and prohibition are handled by the other primitives.

| Level | Meaning |
| --- | --- |
| 0 | Minimal: read-only, local, or ephemeral with negligible external impact |
| 1 | Low: tightly scoped reversible change with low blast radius |
| 2 | Moderate: bounded shared/external side effect with practical recovery |
| 3 | High: production/access/financial/publication or broad external impact with difficult recovery |
| 4 | Critical: catastrophic, widespread, destructive, irreversible, or severe safety/privacy/security impact |

## Bootstrap redistribution

The previous bootstrap data omitted risk level 2 and had only 8 level-1 examples in train.

Current 1,008-case distribution:

| Split | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 152 | 40 | 96 | 96 | 192 |
| validation | 44 | 4 | 40 | 8 | 48 |
| calibration | 36 | 12 | 32 | 16 | 48 |
| test | 36 | 12 | 20 | 28 | 48 |

Labels were reassigned from request semantics using impact, blast radius, and reversibility. Action/review/prohibited targets were not changed.

## Ordinal targets

Score targets now use a Gaussian distance-aware distribution with `sigma=0.75` instead of flat label smoothing. Adjacent levels receive more probability mass than distant levels.

`gold.risk.score` is the expectation of the stored distribution, and schema validation enforces that invariant.

## Training objective

The score primitive keeps RLCD's proper reward (which already includes RPS) and now also uses a direct differentiable RPS term:

```text
score supervised loss =
    score_ce_weight * soft_cross_entropy
  + score_rps_weight * ranked_probability_loss
```

Defaults:

```text
--score-weight 2.0
--score-ce-weight 0.5
--score-rps-weight 1.0
--score-class-balance-beta 0.0
```

Class-balanced effective-number weighting is available as an ablation. For the 1k experiment, try `--score-class-balance-beta 0.99`.

## v4 training recipe

```powershell
python -m jevbro.train `
  --train data/train.jsonl `
  --validation data/validation.jsonl `
  --output artifacts/jevtrain1k-v4 `
  --epochs 4 `
  --score-ce-weight 0.5 `
  --score-rps-weight 1.0
```

Class-balanced ablation:

```powershell
python -m jevbro.train `
  --train data/train.jsonl `
  --validation data/validation.jsonl `
  --output artifacts/jevtrain1k-v4-cb `
  --epochs 4 `
  --score-ce-weight 0.5 `
  --score-rps-weight 1.0 `
  --score-class-balance-beta 0.99
```

## Evaluation

`jevbro.evaluate` now reports:

- exact score accuracy
- expected-score MAE
- hard-level MAE
- within-1 accuracy
- quadratic weighted kappa (QWK)
- ranked probability score (RPS)
- 5x5 score confusion matrix
- per-level recall/support

Use the dedicated calibration split after training, then evaluate only once on test for the final comparison.

## Current experiment status

The latest checked-in Score v4 development report is `../artifacts/v41/test-report.json`.
It evaluates the current 144-case bootstrap test split and reports 74.65% overall
accuracy, 50.69% exact score accuracy, 0.637 QWK, 82.64% within-one score
accuracy, and 0.0756 RPS. `../artifacts/v41/` contains reports only; it is not a
servable checkpoint directory. The local serving checkpoint remains
`../artifacts/laya-model/` until a newer model is materialized locally.
