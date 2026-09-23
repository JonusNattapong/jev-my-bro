# jev-my-bro

Jev is a self-hosted typed decision model for agent and tool governance. It
turns a request into four signals:

- `action`: `execute`, `ask_user`, or `reject`
- `needs_review`: whether a human must review before execution
- `prohibited`: whether the operation must be blocked
- `risk`: an ordinal operational-impact score from 0 to 4

Jev is an advisory component. It must not replace repository policy, explicit
approval, tests, audit logs, or a human safety boundary.

## Latest Thai model

The latest manually curated Thai model is `laya-th1200`:

- Model: [JonusNattapong/jev-my-bro-th1200](https://huggingface.co/JonusNattapong/jev-my-bro-th1200)
- Model Card: [`docs/model-cards/jev-my-bro-th1200.md`](docs/model-cards/jev-my-bro-th1200.md)
- Training data: 1,200 Thai cases (`data/th_curated_1200/train.jsonl`)
- Validation, calibration, and test: 100 cases each
- Test: 100 cases, 400 typed decisions, 20 cases per risk level
- Architecture: **Hybrid Cascaded** (Layer 1 Fast-Path <1ms rule engine + Layer 2 LRU Decision Cache + Layer 3 th1200 Neural model with CPU INT8 quantization)

### Locked Test Results

| Metric | laya-th1200 (Latest) | laya-th960 (Baseline) |
| --- | ---: | ---: |
| Overall accuracy | **86.75%** | 88.00% |
| Action choice accuracy | **88.00%** | 89.00% |
| Noul accuracy (`needs_review` / `prohibited`) | **93.50%** | 92.50% |
| Level 4 risk recall (Catastrophic/Destructive) | **100.00%** | 90.00% |
| Score within-one accuracy | **95.00%** | 96.00% |

#### Why th1200?
Earlier checkpoints (`th960`) were trained on short 1-line requests (mean ~65 chars). Real coding agent contexts sent to `jev_task_start` are 120–460 characters long and include meta-phrases like *"ผู้ใช้สั่งให้ทำแล้ว"* or *"ไม่มีการ push"*. This previously caused false-positive `prohibited` spikes on routine tasks. `th1200` incorporates 120 agent-task context cases in `th_17_agent_task_context_train.csv`, resolving false alarms while achieving **flawless 100% Level 4 risk recall**.

The previous experiment was `laya-th960`:
- Model: [JonusNattapong/jev-my-bro-th960](https://huggingface.co/JonusNattapong/jev-my-bro-th960)
- Dataset: [JonusNattapong/jev-my-bro-dataset-th960](https://huggingface.co/datasets/JonusNattapong/jev-my-bro-dataset-th960)
- Model Card: [`docs/model-cards/jev-my-bro-th960.md`](docs/model-cards/jev-my-bro-th960.md)
- Benchmark visualization: [`docs/assets/laya-th960-benchmark.svg`](docs/assets/laya-th960-benchmark.svg)

## Use the published model

The model repository is private unless it is made public by its owner. In
Colab, authenticate without placing a token in a notebook cell:

```python
%cd /content/jev-my-bro
!pip install -q -r requirements.txt

from huggingface_hub import notebook_login
notebook_login()
```

Load the model and make a Thai decision:

```python
import json
import torch
from laya import Agent
from jevbro.questions import default_questions

MODEL_ID = "JonusNattapong/jev-my-bro-th1200"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
agent = Agent(MODEL_ID, device=DEVICE)

def decide(request: str, domain: str = "general") -> dict:
    result = agent.predict(
        {"request": request, "domain": domain},
        default_questions("th"),
    )
    answers = result["answers"]
    return {
        "request": request,
        "action": answers["action"]["choice"],
        "needs_review": answers["needs_review"]["noul"],
        "prohibited": answers["prohibited"]["noul"],
        "risk_score": answers["risk"]["score"],
        "action_probabilities": answers["action"]["probabilities"],
        "risk_probabilities": answers["risk"]["probabilities"],
    }

print(json.dumps(
    decide("เปลี่ยนค่า firewall ของ production ให้เปิดจากอินเทอร์เน็ต", "network"),
    ensure_ascii=False,
    indent=2,
))
```

For a checkpoint already downloaded in Colab, use:

```python
agent = Agent("artifacts/laya-th1200", device="cuda")
```

Always put the policy gate outside the model:

```python
result = decide("ลบข้อมูลใน production ที่ไม่ใช้แล้ว", "database")

if result["prohibited"] >= 0.7 or result["action"] == "reject":
    decision = "reject"
elif (
    result["needs_review"] >= 0.5
    or result["action"] == "ask_user"
    or result["prohibited"] >= 0.5
):
    decision = "human_review"
else:
    decision = "execute"
```

The prohibited signal has priority over the raw action only when it is high
enough (`>= 0.7`) to indicate a likely prohibition. A medium score from
`0.5` to below `0.7` is treated as uncertainty and sent to human review,
which prevents ordinary but underspecified work from being rejected solely
because the model is uncertain.

## Run the local HTTP & MCP service

Install the project and start the native Laya/PyTorch service:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps

# Start HTTP and MCP server (with INT8 quantization on CPU and auto-loaded rules.yaml)
.\.venv\Scripts\jev.exe serve `
  --model JonusNattapong/jev-my-bro-th1200 `
  --host 127.0.0.1 `
  --port 8787 `
  --quantize
```

Check health (includes cache statistics and rule hit counts):

```powershell
curl http://127.0.0.1:8787/health
```

Ask for a typed governance decision:

```powershell
curl -X POST http://127.0.0.1:8787/v1/decide `
  -H "content-type: application/json" `
  -d '{"context":"Deploy the payment service to production","language":"en"}'
```

The service also exposes `/v1/predict`, `/v1/jev-my-bro/decide`,
`/v1/jev-my-bro/predict`, and `/v1/systemone`. See
[`docs/MCP_INTEGRATION.md`](docs/MCP_INTEGRATION.md) for connecting Codex,
Claude Code, or OpenCode through MCP.

## Reproduce the Thai 1,200 training run

The checked-in source rows are manually authored. The builder creates the
JSONL splits and checks unique IDs, unique requests, action-label consistency,
and balanced holdouts.

```bash
python scripts/build_th_curated_1200.py
python scripts/validate_dataset.py --root data/th_curated_1200
```

The split layout is:

| Split | Cases | Decisions | Purpose |
| --- | ---: | ---: | --- |
| Train | 1,200 | 4,800 | Fit model weights (including 120 agent task contexts) |
| Validation | 100 | 400 | Select checkpoint |
| Calibration | 100 | 400 | Fit temperature and score decoder |
| Test | 100 | 400 | One locked final evaluation |

On a Colab T4:

```bash
!python -m jevbro.train --config configs/colab-th1200.yaml

!python -m jevbro.calibrate \
  --model artifacts/laya-th1200 \
  --data data/th_curated_1200/calibration.jsonl \
  --report artifacts/laya-th1200/calibration-report.json

!python -m jevbro.evaluate \
  --model artifacts/laya-th1200 \
  --data data/th_curated_1200/test.jsonl \
  --device cuda \
  --lock-decoder \
  --report artifacts/laya-th1200/test-report.json
```

`calibrate` refuses a path named `test.jsonl`. The `--lock-decoder` flag
requires calibration provenance in `rl_agent_config.json`, including the
calibration dataset hash, before evaluating test.

## Hybrid Cascaded Architecture

Jev combines instant deterministic safety with neural semantic generalization across 3 tiers:

```text
Incoming Operational Request / Tool Call
                   |
                   v
  +---------------------------------+
  | Layer 1: Fast-Path Rule Engine  | < 1 ms  (Deterministic rules.yaml / hard blocks)
  +---------------------------------+
         | Match?
        / \
      YES  NO
      /     \
  [Return]   v
  +---------------------------------+
  | Layer 2: LRU Decision Cache     | ~15 ms  (1024-entry normalized LRU hit)
  +---------------------------------+
         | Cache Hit?
        / \
      YES  NO
      /     \
  [Return]   v
  +---------------------------------+
  | Layer 3: Neural Model Inference | ~200-300 ms (CPU Dynamic INT8 Quantization)
  |  - Base: convaiinnovations/laya |
  |  - Heads: choice / noul / score |
  +---------------------------------+
                   |
                   v
         Calibrated Outer Policy Gate
          (prohibited >= 0.5 -> reject,
           needs_review >= 0.5 -> ask_user,
           else -> execute)
```

## Automated Claude Code Governance Hook

Jev can intercept tool execution proactively before shell commands or file writes run:
- Script: [`hooks/claude_pre_tool_use.py`](hooks/claude_pre_tool_use.py)
- Configuration: `.claude/settings.json`
- Protocol: returns JSON with `permissionDecision` (`allow`, `ask`, `deny`)
- Full documentation: [`docs/CLAUDE_HOOK_SETUP.md`](docs/CLAUDE_HOOK_SETUP.md)

## Run the public JevBench tasks

This repository includes a runner that reuses JevBench's own public task
loader, serial runner, and scoring implementation. It loads the local Jev
checkpoint through `laya.Agent`, so a Colab GPU can be used without changing
the benchmark protocol.

Clone JevBench beside this repository, then run its published tasks:

```bash
git clone --depth 1 https://github.com/fstandhartinger/jevbench.git /content/jevbench
python -m pip install -e /content/jevbench
python scripts/run_jevbench_public.py \
  --jevbench-dir /content/jevbench \
  --model artifacts/laya-th1200 \
  --device cuda
```

The run writes raw responses and the public summary under
`private/jevbench-public/`; that directory is ignored by Git. The local run
covers only the published task files. The official JevBench board also uses
held-out tasks that are not distributed publicly, so this output must not be
called an official full-score submission. Keep its dataset hash, model
revision, device, and summary when reporting the result.

## Repository layout

```text
jevbro/                         core runtime, rules engine, caching, MCP server
hooks/claude_pre_tool_use.py    Claude Code PreToolUse governance hook
configs/colab-th1200.yaml       reproducible T4 training config for th1200
data/th_curated_1200/           Thai 1,200-case JSONL splits and README
rules.example.yaml              template for custom fast-path rules
scripts/build_th_curated_1200.py dataset builder and integrity checks
scripts/analyze_rule_hit_rate.py rule engine telemetry analyzer
scripts/probe_real_requests.py  qualitative real-request probe
scripts/run_jevbench_public.py  JevBench public-task runner
docs/model-cards/jev-my-bro-th1200.md  model card for th1200
tests/                          unit, hook, cache, data, and workflow tests
```

## Verification

```bash
python scripts/validate_dataset.py --root data/th_curated_1200
python -m compileall -q jevbro hooks scripts
pytest -q
```

The latest repository verification passed with **89 tests (100% pass rate)**. GPU training and
the final laya-th1200 metrics were run in Google Colab; they are not reproduced
by the local test suite.

## Limitations and safety

- The Thai 1,200-case set is manually curated development data, not a complete
  production authorization policy.
- The test report is a benchmark on this held-out set, not proof of safety on
  unseen organizations or domains.
- Risk 1 and risk 3 remain the weakest per-level recall slices.
- Keep an independent policy layer, explicit approvals, human escalation,
  audit logs, and outcome feedback around the model.
- Never put Hugging Face tokens, private keys, or production credentials in
  this repository or a notebook.

## License

Project code and project-authored data are licensed under the Apache License
2.0; see [`LICENSE`](LICENSE). Laya is also Apache-2.0. Review
[`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md) before redistributing datasets or
derived artifacts.
