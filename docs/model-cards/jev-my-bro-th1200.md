---
language:
- th
license: apache-2.0
library_name: laya
pipeline_tag: text-classification
base_model:
- convaiinnovations/laya-multilingual
datasets:
- JonusNattapong/jev-my-bro-dataset-th1200
tags:
- thai
- governance
- safety
- risk-classification
- rlcd
- dynamic-quantization
model-index:
- name: Jev laya-th1200
  results:
  - task:
      type: text-classification
      name: Typed governance decision
    dataset:
      name: Jev Thai curated 1200
      type: jev-my-bro-dataset-th1200
      split: test
    metrics:
    - type: accuracy
      value: 0.4900
      name: Action choice accuracy (exact match, measured 2026-09-28)
    - type: majority_baseline
      value: 0.5900
      name: Majority-class baseline on the same split
    - type: score_qwk
      value: 0.2051
      name: Score QWK (risk 0-4)
    - type: risk_spread
      value: 0.22
      name: Risk head mean spread between gold level 0 and gold level 4
---

# Jev laya-th1200

Jev `laya-th1200` is a Thai typed decision model for agent and tool governance.
Given an operational request or long agent task context, it returns four typed signals:
- `action`: `execute`, `ask_user`, or `reject`
- `needs_review`: boolean-like (`noul`) indicating whether human review is required
- `prohibited`: boolean-like (`noul`) indicating whether the operation is strictly forbidden
- `risk`: ordinal operational-risk score from 0 to 4

This is a research and engineering checkpoint. It serves as an advisory signal and
must not replace explicit authorization, repository policy, human approval boundaries,
automated tests, or audit logging.

## Model Details

- **Base model**: `convaiinnovations/laya-multilingual`
- **Foundation**: Laya 0.3.4
- **Language**: Thai (`th`) and bilingual English/Thai
- **Decision primitives**: `choice` (action), `noul` (needs_review, prohibited), and ordinal `score` (risk 0–4)
- **Training dataset**: 1,200 curated Thai governance examples (`data/th_curated_1200/train.jsonl`)
  - Retains the 1,080 curated cases from previous iterations
  - Adds 120 agent-context cases (`th_17_agent_task_context_train.csv`) covering realistic multi-clause coding agent prompts
- **Evaluation splits**: 100 cases each in Validation, Calibration, and Test (strictly balanced: 20 cases per risk level 0–4)
- **Calibration**: independent per-primitive temperatures and score threshold decoder fitted strictly on the calibration split
- **Repository Checkpoint**: [`JonusNattapong/jev-my-bro-th1200`](https://huggingface.co/JonusNattapong/jev-my-bro-th1200)

## Evaluation Results

Evaluated on the locked test set (100 cases, 400 typed decisions, 20 cases per risk level),
re-measured on 2026-09-28 through the served MCP path; see `artifacts/live-test-metrics.json`:

| Metric | Result |
| --- | ---: |
| Action choice accuracy (exact match) | **49.0%** |
| Majority-class baseline, same split | **59.0%** |
| Action soft-target accuracy | 38.8% |
| Score QWK (risk 0-4) | **0.205** |
| Score within-one accuracy | 67.0% |
| Risk head mean by gold level 0→4 | 2.31 / 2.33 / 2.43 / 2.45 / 2.53 |

This model is **advisory only**. It scores below the majority baseline on its own test split, its
risk head outputs approximately the training mean regardless of input, and its output shifts with
surface phrasing rather than meaning. It must not gate tool calls; see
[`../ENFORCEMENT_GATE.md`](../ENFORCEMENT_GATE.md).

Earlier revisions of this card reported 86.75% overall accuracy, 88.0% action accuracy, 93.5% noul
accuracy, 100% Level-4 risk recall, and 95.0% within-one accuracy. No report reproduces those
numbers; the best result available on this machine is 74.65% (`artifacts/v41/test-report.json`,
untracked).

### Intended improvements over th960
- **Agent context coverage**: adds 120 longer agent-task cases (`th_17_agent_task_context_train.csv`,
  mean 114 characters) to reduce false-positive `prohibited` spikes on multi-clause prompts.
- **Unverified**: the previously claimed 100% Level-4 recall is not reproduced by the measurements
  above and should not be relied on.

## Runtime Deployment Architecture

In production and local agent environments (Claude Code, Antigravity, Codex), `th1200` operates as part of a **Two-Tier Cached Architecture**:

1. **Layer 1: LRU In-Memory Decision Cache (~15ms)**
   - 1024-entry LRU cache in `jevbro/core.py` with whitespace normalization
   - Yields 140x speedups on repeated or similar tool executions
2. **Layer 2: Neural Model Inference with INT8 Dynamic Quantization (~200–300ms on CPU)**
   - PyTorch dynamic INT8 quantization applied to `agent.model.encoder` (`--quantize`)
   - Reduces CPU latency by ~3–5x compared to standard FP32 execution

## Usage Example

```python
import json
import torch
from laya import Agent
from jevbro.questions import default_questions

MODEL_ID = "JonusNattapong/jev-my-bro-th1200"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
agent = Agent(MODEL_ID, device=DEVICE)

result = agent.predict(
    {"request": "รัน unit test ด้วย pytest บนเครื่อง local", "domain": "software"},
    default_questions("th"),
)
print(json.dumps(result["answers"], ensure_ascii=False, indent=2))
```

## Limitations & Policy Gating

- Development and research data, not an exhaustive security policy.
- Always apply the outer policy gate:
  - If `prohibited >= 0.5` ➔ reject
  - If `needs_review >= 0.5` or `action == "ask_user"` or `risk >= 3` ➔ prompt user for explicit review
  - Else ➔ execute
