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

| Metric | Runtime path | Checkpoint report |
| --- | ---: | ---: |
| Action choice accuracy (exact match) | 75.0% | 86.75% |
| Score QWK (risk 0-4) | 0.562 | 0.842 |
| Score within-one accuracy | 87.0% | 94.0% |
| Risk head mean by gold level 0→4 | 1.38 / 1.76 / 2.18 / 2.42 / 2.86 | monotone |
| Level 4 risk recall | — | 100% |

The checkpoint's `test-report.json` uses a fitted score-threshold decoder and per-question-type
temperatures that the runtime does not apply, which accounts for the gap between the two columns.

**The question text is part of the model input.** All 1,400 corpus cases carry one identical
question block, and the checkpoints were trained with it. Asking a translated question set
collapsed the same weights from 75.0% to 26.7% action accuracy and from a 1.474 risk spread to
0.181. `jevbro.questions` now returns the trained prompt verbatim and `tests/test_questions.py`
pins it against the data.

Known gaps, measured: the model abstains on 100% of requests (mean confidence 0.211 against a 0.6
threshold), and it has no discrimination on tool-call inputs (risk spread 0.618 on
`data/tool_call_80/test.jsonl`). It is not fit to gate tool calls; see
[`../ENFORCEMENT_GATE.md`](../ENFORCEMENT_GATE.md).

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
