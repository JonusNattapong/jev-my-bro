---
pretty_name: Jev Thai curated 1,080
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

# Jev Thai curated 1,080

Thai-language manually authored data for the next Jev training run. This
release preserves the 960-case baseline and adds 120 new training cases
targeting the weakest public JevBench families: ambiguous authorization,
hard routing, trap/adversarial requests, and hard judgment boundaries.

The added cases cover unclear scope or authority, ask-user versus reject
decisions, conflicting instructions, policy-bypass language, multi-condition
routing, and near-boundary risk judgments. They are all assigned to the train
split. Validation, calibration, and test remain independent 100-case splits
from the 960-case baseline.

## Dataset summary

| Split | Cases | Decisions | Risk distribution |
|---|---:|---:|---|
| train | 1,080 | 4,320 | 0: 112, 1: 218, 2: 246, 3: 305, 4: 199 |
| validation | 100 | 400 | 20 at every level 0-4 |
| calibration | 100 | 400 | 20 at every level 0-4 |
| test | 100 | 400 | 20 at every level 0-4 |

The 120 new training cases are distributed as follows:

| Focus | Cases |
|---|---:|
| ambiguous | 30 |
| routing_hard | 30 |
| trap/adversarial | 40 |
| judge_hard | 20 |
| Total | 120 |

The additions contain exactly 24 cases at each risk level. The handwritten
source is `th_16_jevbench_hard_cases_train.csv` in the 960-case source area;
the 960-case release itself is not overwritten.

## Build and validate

Run from the repository root:

```bash
python scripts/build_th_curated_1080.py
python scripts/validate_dataset.py --root data/th_curated_1080
```

Expected output is `dataset validation: OK` with split sizes
1,080/100/100/100. The builder rejects duplicate IDs and request text and
keeps validation, calibration, and test risk-balanced.

## Training order

Train on `train.jsonl`, select the checkpoint from validation, calibrate only
on `calibration.jsonl`, then run the locked final evaluation on `test.jsonl`.
Do not use the test split to select a decoder or tune thresholds.

## Limitations

These are manually authored development cases and do not represent every
Thai production request. The JevBench-targeted additions are intended to
address observed failure families, not to replace an independent production
holdout. Keep model predictions separate from the final authorization and
policy gate.
