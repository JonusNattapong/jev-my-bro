---
language:
- th
license: apache-2.0
library_name: laya
pipeline_tag: text-classification
base_model:
- convaiinnovations/laya-multilingual
datasets:
- JonusNattapong/jev-my-bro-dataset-th960
tags:
- thai
- governance
- safety
- risk-classification
- rlcd
model-index:
- name: Jev laya-th960
  results:
  - task:
      type: text-classification
      name: Typed governance decision
    dataset:
      name: Jev Thai curated 960
      type: jev-my-bro-dataset-th960
      split: test
    metrics:
    - type: accuracy
      value: 0.88
      name: Overall accuracy
    - type: expected_calibration_error
      value: 0.13914140560873656
      name: ECE
    - type: negative_log_likelihood
      value: 0.6540229828235362
      name: NLL
    - type: brier_score
      value: 0.11594753390737665
      name: Brier score
    - type: score_expected_mae
      value: 0.351666
      name: Score expected MAE
    - type: score_qwk
      value: 0.915
      name: Score QWK
    - type: score_within_one_accuracy
      value: 0.96
      name: Score within-one accuracy
    - type: score_macro_recall
      value: 0.78
      name: Score macro recall
---

# Jev laya-th960

Jev laya-th960 is a Thai typed decision model for agent and tool governance.
Given an operational request, it returns four signals: an action choice,
whether human review is needed, whether the operation is prohibited, and an
ordinal operational-risk score from 0 to 4.

This is a research and engineering checkpoint. It is an advisory signal and
must not replace explicit authorization, repository policy, human review, audit
logs, or an independent safety layer.

## Model details

- Base model: `convaiinnovations/laya-multilingual`
- Foundation: Laya 0.3.4
- Language: Thai (`th`)
- Decision primitives: `choice`, `noul`, and ordinal `score`
- Training: RLCD, soft-target cross-entropy, ordinal cumulative loss, and RPS
- Selected checkpoint: epoch 5, selected from validation metrics
- Calibration: one temperature per primitive plus score thresholds
- Decoder: `threshold`, selected from the calibration split only
- Repository: `JonusNattapong/jev-my-bro-th960`

## Evaluation

The final locked evaluation used `data/th_curated_960/test.jsonl` with 100
cases and 400 typed decisions. Each risk level has 20 cases. The test split
was not used for training, temperature fitting, threshold fitting, or decoder
selection.

| Metric | Value |
| --- | ---: |
| Overall accuracy | **88.00%** |
| ECE | 0.1391 |
| NLL | 0.6540 |
| Brier score | 0.1159 |
| Score expected MAE | 0.3517 |
| Score hard MAE | 0.2600 |
| Score within-one accuracy | **96.00%** |
| Score QWK | **0.9150** |
| Score RPS | 0.0433 |
| Score macro recall | **0.7800** |

### Accuracy by primitive

| Primitive | Accuracy | Brier |
| --- | ---: | ---: |
| `choice` | 89.00% | 0.1167 |
| `noul` | 92.50% | 0.0572 |
| `score` | 78.00% | 0.2327 |

### Score recall by risk level

| Risk | Support | Recall |
| ---: | ---: | ---: |
| 0 | 20 | 85% |
| 1 | 20 | 60% |
| 2 | 20 | 90% |
| 3 | 20 | 65% |
| 4 | 20 | 90% |

Risk 1 and risk 3 are the weakest per-level slices. The ECE of 0.1391 also
shows that confidence is not perfectly calibrated; consumers should use the
probabilities for escalation and monitoring, not as guarantees.

## Usage

The repository is private unless the owner changes its visibility. Authenticate
with Hugging Face without putting a token in source code:

```python
from huggingface_hub import notebook_login
notebook_login()
```

Then load the model with the Jev question schema:

```python
import json
import torch
from laya import Agent
from jevbro.questions import default_questions

agent = Agent(
    "JonusNattapong/jev-my-bro-th960",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

result = agent.predict(
    {
        "request": "เปลี่ยนค่า firewall ของ production ให้เปิดจากอินเทอร์เน็ต",
        "domain": "network",
    },
    default_questions("th"),
)

print(json.dumps(result["answers"], ensure_ascii=False, indent=2))
```

Keep the policy gate outside the model. A prohibited signal has priority over
the raw action choice:

```python
answers = result["answers"]
if answers["prohibited"]["noul"] >= 0.5 or answers["action"]["choice"] == "reject":
    decision = "reject"
elif answers["needs_review"]["noul"] >= 0.5 or answers["action"]["choice"] == "ask_user":
    decision = "human_review"
else:
    decision = "execute"
```

## Qualitative smoke example

For the request `เปลี่ยนค่า firewall ของ production ให้เปิดจากอินเทอร์เน็ต`,
the raw model returned `action=ask_user`, `needs_review=0.809`,
`prohibited=0.605`, and `risk_score=2.877`. The external policy gate therefore
resolved the request to `reject`. This illustrates why the model output must be
combined with an explicit policy gate.

## Intended use and limitations

Intended uses include Thai-language governance experiments, routing requests to
human review, risk triage, and testing typed decision interfaces. The model is
not intended to autonomously authorize production changes, handle secrets,
make legal or compliance decisions, or serve as the only control for destructive
operations.

The evaluation set is manually curated development data. It does not prove
performance on unseen organizations, languages, domains, or production traffic.
Validate the model with reviewed operational data and real outcome feedback
before using it in a consequential workflow.

## Dataset and reproducibility

The matching dataset is
[`JonusNattapong/jev-my-bro-dataset-th960`](https://huggingface.co/datasets/JonusNattapong/jev-my-bro-dataset-th960).
The repository contains the source rows, split builder, T4 configuration, and
locked calibration/evaluation workflow:

```bash
python scripts/build_th_curated_960.py
python scripts/validate_dataset.py --root data/th_curated_960
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

## License

The project code and project-authored data are licensed under Apache-2.0. Laya
is also Apache-2.0. Review the repository's
[`docs/THIRD_PARTY.md`](https://github.com/JonusNattapong/jev-my-bro/blob/main/docs/THIRD_PARTY.md)
before redistributing derived artifacts.
