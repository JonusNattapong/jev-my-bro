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

Measured on 2026-09-28 on the six-epoch combined checkpoint
(`configs/colab-th1200-toolcall.yaml`, selected epoch 5), after calibrate on the combined
calibration splits. Numbers below are from the held-out test splits, not validation.

**Tool-call split** (`data/tool_call_400/test.jsonl`, 36 cases, 144 decisions):

| Metric | Result | Gate | Status |
| --- | ---: | --- | --- |
| Action choice accuracy | 69.4% | ≥ 70%, baseline 38.9% | marginal |
| Score QWK | 0.893 | ≥ 0.55 | passes |
| Risk head spread | 2.66 | ≥ 1.0 | passes |
| `prohibited` separation | 0.311 | ≥ 0.30 | passes |
| `prohibited` gold-false mean | 0.241 | < 0.30 | passes |
| Abstain rate | 0.472 | ≤ 0.60 | passes |
| `needs_review` separation | **-0.404** | ≥ 0.30 | **fails, structurally** |

**Thai split** (`data/th_curated_1200/test.jsonl`, 100 cases, 400 decisions), same checkpoint:

| Metric | Before (th1200) | After | Gate | Status |
| --- | ---: | ---: | --- | --- |
| Overall accuracy | 75.0% | 81.75% | | |
| Action choice accuracy | 75.0% | 83.0% | ≥ 70%, baseline 59% | passes |
| Noul accuracy | — | 91.0% | | |
| Score QWK | 0.562 | 0.846 | ≥ 0.55 | passes |
| Risk head spread | 1.474 | 2.86 | ≥ 1.0 | passes |
| `needs_review` separation | 0.127 | 0.418 | | |
| `prohibited` separation | 0.390 | 0.461 | ≥ 0.30 | passes |
| Abstain rate | 1.000 | 0.370 | ≤ 0.60 | passes |

Training on the tool-call corpus moved the model from 36.1% to 69.4% action accuracy on tool calls
with no loss on Thai, so enforcement criterion 5 is met in substance. Paraphrase stability
(criterion 4) is still unmeasured.

**Enforcement stays off.** `needs_review` is a deterministic function of the action label in every
case of both corpora, so the `ask_user` gate rule is inverted on tool calls. That is a labelling
defect, and the details are in [`docs/ENFORCEMENT_GATE.md`](docs/ENFORCEMENT_GATE.md).

#### Known limitations

The question text is model input. The checkpoints were trained with exactly one question block, the
one carried by all 1,400 cases of `data/th_curated_1200`. Asking a translated question set collapses
the same weights from 75.0% to 26.7% action accuracy and from a 1.474 risk spread to 0.181.
`jevbro.questions` returns the trained prompt verbatim and a test pins it to the data.

- **`needs_review` is not an independent signal in the corpora.** In all 1,400 Thai cases and all
  431 tool-call cases, `needs_review == (action != "execute")` with no exceptions, so the noul head
  can only restate the action label. The gate turns `needs_review >= 0.5` into `ask_user`, which
  means *ask when the model is confident*; measured separation on the tool-call split is -0.404.
  `validate_dataset.py` now fails on this unless the corpus is acknowledged as affected. Relabelling
  the Thai corpus needs human judgement per case and has not been done. See
  [`docs/ENFORCEMENT_GATE.md`](docs/ENFORCEMENT_GATE.md).
- **The risk level 3 is confused with 2 on the Thai split.** With the threshold decoder fitted on
  the combined calibration set, level-3 recall is 0.15 while argmax gives 0.80. The shared
  thresholds are a compromise; `score_decoder: argmax` is the better choice for Thai requests and
  threshold the better one for tool calls.

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
