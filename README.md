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
- Architecture: **Two-Tier Cached** (Layer 1 LRU Decision Cache + Layer 2 th1200 Neural model with CPU INT8 quantization)

### Measured Test Results

Measured on 2026-09-28 against the published `JonusNattapong/jev-my-bro-th1200` checkpoint, on its
own locked 100-case test split in `data/th_curated_1200/test.jsonl` (400 typed decisions), through
the served MCP path. Raw evidence: `artifacts/live-test-metrics.json`.

| Metric | Result | Note |
| --- | ---: | --- |
| Action choice accuracy (exact match) | **75.0%** | 16 points above the 59.0% majority baseline |
| Action soft-target accuracy | 50.0% | |
| Majority-class baseline | 59.0% | always predicting `execute` |
| Score QWK (risk 0-4) | **0.562** | |
| Score within-one accuracy | 87.0% | |
| Risk head mean by gold level 0→4 | 1.38 / 1.76 / 2.18 / 2.42 / 2.86 | monotone, spread **1.474** |
| `prohibited` separation (gold true vs false) | **0.390** | 0.678 vs 0.287 |

The checkpoint's own `test-report.json` in the Hugging Face snapshot reports 0.8675 overall accuracy
and 0.842 score QWK for the same split, using a fitted score-threshold decoder that the runtime does
not apply. The numbers above are the uncalibrated runtime path, which is why they are lower.

#### Known limitations

- **The question text is model input.** The checkpoints were trained with exactly one question
  block, the one carried by all 1,400 cases of `data/th_curated_1200`. Asking a translated question
  collapses the same weights from 75.0% to 26.7% action accuracy and from a 1.474 risk spread to
  0.181. `jevbro.questions` now returns the trained prompt verbatim and a test pins it to the data.
- **It abstains on everything.** Mean action confidence is 0.211 against the default 0.6 threshold,
  so `abstain` is true and `decision` is null for every request. The gated signal comes from
  `needs_review` and `prohibited` instead.
- **It has never seen tool calls.** On `data/tool_call_400/test.jsonl` (36 cases) it scores 36.1%
  action accuracy against a 38.9% majority baseline, i.e. below guessing, and the risk spread is
  0.565 against 1.474 on its own distribution. The distribution the hook actually sends is
  uncovered.
- Training contexts are short (mean 70 characters, max 140) while real agent sessions reach 460.
  The `tool_call_400` slice closes that gap at 431 hand-written cases (409 English, 22 Thai), 22%
  of the combined corpus. It is large enough to train on; it is not a guarantee that training
  works.

#### Why th1200?

Earlier checkpoints (`th960`) were trained on short 1-line requests. `th1200` adds 120 longer
agent-task context cases from `th_17_agent_task_context_train.csv` (mean 114 characters), included
via `scripts/build_th_curated_1200.py`. That is still far short of the 460-character contexts real
sessions produce.


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

# Start HTTP and MCP server (with INT8 quantization on CPU)
.\.venv\Scripts\jev.exe serve `
  --model JonusNattapong/jev-my-bro-th1200 `
  --host 127.0.0.1 `
  --port 8787 `
  --quantize
```

Check health (includes cache statistics):

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

`--train` and `--validation` accept a comma-separated list of paths, so several corpora can be
trained on together without merging the files on disk. Each run prints a per-corpus sequence-count
breakdown. The next round uses the combined corpus:

```bash
!python -m jevbro.train --config configs/colab-th1200-toolcall.yaml
```

That config trains on `data/th_curated_1200` plus `data/tool_call_400`, the tool-call slice that
covers the literal shell-command and file-edit strings the hook sends. A test asserts the config
never references a `test.jsonl` or `calibration.jsonl` split.

Each epoch logs one line with the diagnostics that show whether the heads are actually learning:

```text
[train] epoch=1/6 val_accuracy=0.5210 score_qwk=0.198 score_spread=0.214 needs_review_sep=0.181 prohibited_sep=0.130
```

`score_spread` is the risk head's mean prediction for gold level 0 versus gold level 4. A value near
zero means the head is returning the training mean for every input, which accuracy and QWK both
hide. Watch it across epochs: if it does not move, more data will not help and the architecture is
what needs changing. The gate thresholds live in [`docs/ENFORCEMENT_GATE.md`](docs/ENFORCEMENT_GATE.md).

## Two-Tier Cached Architecture

Jev combines a cached lookup with neural semantic generalization across 2 tiers:

```text
Incoming Operational Request / Tool Call
                   |
                   v
  +---------------------------------+
  | Layer 1: LRU Decision Cache     | ~15 ms  (1024-entry normalized LRU hit)
  +---------------------------------+
         | Cache Hit?
        / \
      YES  NO
      /     \
  [Return]   v
  +---------------------------------+
  | Layer 2: Neural Model Inference | ~200-300 ms (CPU Dynamic INT8 Quantization)
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

There is no deterministic rule engine. Destructive commands and credential access are judged by the
model like any other request, and a low-confidence model abstains toward `ask_user` rather than
blocking.

## Automated Claude Code Governance Hook

Jev can observe tool execution before shell commands or file writes run:
- Script: [`hooks/claude_pre_tool_use.py`](hooks/claude_pre_tool_use.py)
- Registration: user-level `~/.claude/settings.json` (absolute paths, applies to every project)
- Configuration: this repository's `.claude/settings.json` holds `env` overrides only
- **Advisory by default** (`JEV_ENFORCE=0`): the verdict is logged to stderr and nothing is blocked,
  because the model does not currently meet the accuracy criteria
- With `JEV_ENFORCE=1` the hook returns JSON `permissionDecision` (`allow`, `ask`, `deny`)
- Full documentation: [`docs/CLAUDE_HOOK_SETUP.md`](docs/CLAUDE_HOOK_SETUP.md), gate criteria in
  [`docs/ENFORCEMENT_GATE.md`](docs/ENFORCEMENT_GATE.md)

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
jevbro/                         core runtime, caching, MCP server
hooks/claude_pre_tool_use.py    Claude Code PreToolUse governance hook
configs/colab-th1200.yaml       reproducible T4 training config for th1200
configs/colab-th1200-toolcall.yaml  th1200 corpus plus the tool-call slice
data/th_curated_1200/           Thai 1,200-case JSONL splits and README
data/tool_call_400/             hand-authored tool-call cases and provenance notes
data/tool_call_80/              superseded 80-case seed, kept for provenance
scripts/build_th_curated_1200.py dataset builder and integrity checks
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
