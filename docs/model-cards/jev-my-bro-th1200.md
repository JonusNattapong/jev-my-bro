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
- fast-path
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
      value: 0.8675
      name: Overall accuracy
    - type: choice_accuracy
      value: 0.8800
      name: Action choice accuracy
    - type: noul_accuracy
      value: 0.9350
      name: Noul accuracy
    - type: risk_l4_recall
      value: 1.0000
      name: Level 4 risk recall
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

Evaluated on the locked test set (100 cases, 400 typed decisions, 20 cases per risk level):

| Metric | Result |
| --- | ---: |
| Overall accuracy | **86.75%** |
| Action choice accuracy | **88.00%** |
| Noul accuracy (`needs_review` / `prohibited`) | **93.50%** |
| Level 4 risk recall (Catastrophic/Destructive) | **100.00%** |

### Key Improvements over th960
- **Agent Context Robustness**: Addresses false-positive `prohibited` spikes on routine multi-clause coding tasks containing phrases like "ผู้ใช้สั่งให้ทำแล้ว" or "ไม่มีการ push".
- **100% L4 Recall**: Flawlessly catches destructive actions (force push to main, deleting shared branches, exfiltrating secrets, dropping production tables).

## Runtime Deployment Architecture

In production and local agent environments (Claude Code, Antigravity, Codex), `th1200` operates as part of a **Hybrid Cascaded Architecture**:

1. **Layer 1: Deterministic Fast-Path Rules (<1ms)**
   - Regex and keyword matcher in `jevbro/rules.py`
   - Configurable via `rules.yaml` / `.jev/rules.yaml`
   - Handles ~48% of standard traffic (pure inspections allowed, catastrophic shell commands rejected immediately)
2. **Layer 2: LRU In-Memory Decision Cache (~15ms)**
   - 1024-entry LRU cache in `jevbro/core.py` with whitespace normalization
   - Yields 140x speedups on repeated or similar tool executions
3. **Layer 3: Neural Model Inference with INT8 Dynamic Quantization (~200–300ms on CPU)**
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
