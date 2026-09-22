---
pretty_name: Jev Thai curated 960
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

# Jev Thai curated 960

Thai-language, project-authored cases for training and evaluating Jev's
governance and risk-decision pipeline. Each case contains four related
decisions: an action choice, two boolean review/safety decisions, and an
ordinal risk score from 0 to 4.

This release is the 960-case training split plus three independent 100-case
holdout splits. It extends the 880-case baseline with 80 manually authored
training cases arranged as 40 paired boundary contrasts. The added pairs are
focused on the difficult risk boundaries 1/2 and 2/3 while varying the
operational domain and request wording.

## Dataset summary

| Split | Cases | Decisions | Risk distribution |
|---|---:|---:|---|
| train | 960 | 3,840 | 0: 88, 1: 194, 2: 222, 3: 281, 4: 175 |
| validation | 100 | 400 | 20 at every level 0-4 |
| calibration | 100 | 400 | 20 at every level 0-4 |
| test | 100 | 400 | 20 at every level 0-4 |

All cases are Thai (`language: th`). The validation, calibration, and test
splits are kept separate from training. The builder also checks that IDs and
request text are unique across the complete authored source set, and the
validator checks for duplicate scenario families across splits.

## File layout

```text
data/th_curated_960/
  train.jsonl
  validation.jsonl
  calibration.jsonl
  test.jsonl
  README.md
```

Each JSONL line is one case with this high-level shape:

```json
{
  "id": "...",
  "language": "th",
  "state": {
    "request": "...",
    "domain": "...",
    "scenario_family": "..."
  },
  "questions": {
    "action": {"type": "choice", "...": "..."},
    "needs_review": {"type": "noul", "...": "..."},
    "prohibited": {"type": "noul", "...": "..."},
    "risk": {"type": "score", "...": "..."}
  },
  "gold": {
    "action": {"label": "execute|ask_user|reject", "probabilities": {}},
    "needs_review": {"label": "false|true", "probabilities": {}},
    "prohibited": {"label": "false|true", "probabilities": {}},
    "risk": {"label": "0|1|2|3|4", "score": 0.0, "probabilities": {}}
  }
}
```

The probability targets are soft labels. They are used by the RLCD plus
soft-target cross-entropy training pipeline and are not extra human samples.

## Label semantics

- `action` is the recommended operational response: `execute`, `ask_user`, or
  `reject`.
- `needs_review` indicates whether the request should receive human review.
- `prohibited` indicates whether the request is disallowed by the dataset's
  safety/governance rubric.
- `risk` is ordinal: `0` minimal, `1` low, `2` moderate, `3` high, and `4`
  critical.

The dataset labels describe the supplied scenario and rubric. They are not a
replacement for an application's current authorization, privacy, security,
or safety policy.

## Build and validate

Run from the repository root. The builder reconstructs all four JSONL files
from the checked-in authored sources and fails on an unexpected row count,
duplicate request, split-size error, or holdout risk imbalance.

```bash
python scripts/build_th_curated_960.py
python scripts/validate_dataset.py --root data/th_curated_960
```

Expected split sizes are 960/100/100/100 cases for train/validation/
calibration/test. Validation must finish with `dataset validation: OK`.

## Training and locked evaluation

The shared CLI keeps training logic outside the notebook. On Colab, clone the
repository, install the project requirements, select a GPU runtime, and run:

```bash
python -m jevbro.train --config configs/colab-th960.yaml
python -m jevbro.calibrate \
  --model artifacts/laya-th960 \
  --data data/th_curated_960/calibration.jsonl \
  --report artifacts/laya-th960/calibration-report.json
python -m jevbro.evaluate \
  --model artifacts/laya-th960 \
  --data data/th_curated_960/test.jsonl \
  --device cuda \
  --lock-decoder \
  --report artifacts/laya-th960/test-report.json
```

Calibration must run before the final test evaluation. The locked evaluation
uses only the decoder and temperatures written by calibration; it does not
select a decoder or tune thresholds from `test.jsonl`. Calibration rejects a
path named `test.jsonl` to protect the holdout boundary.

## Intended use

This dataset is intended for:

- reproducible development of Jev's Thai governance classifier;
- comparing training changes using fixed validation, calibration, and test
  boundaries;
- studying action classification, boolean review/prohibition decisions, and
  ordinal risk estimation;
- smoke-testing a downstream policy gate before an explicit human or service
  authorization step.

It is not intended to make autonomous production decisions about people,
access, finance, employment, healthcare, legal matters, or security-sensitive
operations without an independent policy layer and appropriate human review.

## Limitations

The cases are manually authored development data, not a representative sample
of all Thai requests. Coverage, wording, domain balance, and rubric judgments
may not match a production distribution. Boundary cases around risk 1/2 and
2/3 are deliberately emphasized, so aggregate metrics should be read together
with per-level recall and real-request probes. The model trained from this set
may still be overconfident or wrong; a high model score does not grant
permission to execute an operation.

Before deployment, evaluate on an independently collected, reviewed, and
versioned set that reflects the actual application traffic. Keep the model's
prediction separate from the final authorization decision, and log the input,
model version, calibration artifact, policy result, and human override.

## License and provenance

The project-authored dataset files are released under Apache-2.0. The
repository's third-party notices apply to dependencies and any external base
model used by the training pipeline. See the main repository documentation for
the complete provenance and usage notes.

Companion model: [`JonusNattapong/jev-my-bro-th960`](https://huggingface.co/JonusNattapong/jev-my-bro-th960)
