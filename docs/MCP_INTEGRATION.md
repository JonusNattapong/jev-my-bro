# MCP integration

`jev-my-bro` runs one shared MCP decision/task service for Codex, Claude Code,
and OpenCode. Streamable HTTP is recommended so the Laya checkpoint is loaded
once.

## Install and serve

```powershell
cd D:\Projects\Github\jev-my-bro
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps

.\.venv\Scripts\jev.exe serve --model JonusNattapong/jev-my-bro-th1200 --host 127.0.0.1 --port 8787 --quantize
```

`--model` accepts a Hugging Face model ID or a local checkpoint directory. The
recommended model is `JonusNattapong/jev-my-bro-th1200`; it is downloaded to the
Hugging Face cache on first start. It was trained on 1,200 curated Thai governance
examples with 86.75% accuracy. Agents should send decision `context` in Thai
with `language="th"` for optimal semantic precision. Pass `--quantize` on CPU
for dynamic INT8 acceleration.

The older local checkpoint `.\artifacts\laya-model` also works and contains
`model.safetensors`, `rl_agent_config.json`, `encoder\`, and `tokenizer\`.
Directories such as `artifacts\v41` contain evaluation reports only and are not
valid values for `--model`.

MCP endpoint:

```text
http://127.0.0.1:8787/mcp
```

## Primary tools

- `jev_task_start`: create a tracked task and initial Jev decision.
- `jev_task_complete`: finish successfully with verification evidence.
- `jev_task_fail`: record a blocked/failed task with failure evidence.
- `jev_task_status`: inspect one task and its decisions.
- `jev_task_list`: list/filter tasks; returns `{count, items}`.
- `jev_decide`: attach additional advisory decisions to a task.
- `jev_feedback_stats`: aggregate real-use metrics.
- `jev_predict`: low-level typed `choice` / `noul` / `score` inference.
- `jev_model_info`: model/task-lifecycle capabilities.
- `jev_health`: readiness and feedback-store status.

Compatibility tools remain: `jev_feedback_start`, `jev_feedback_complete`,
`jev_feedback_get`, and `jev_record_outcome`.

Jev is advisory. It does not replace repository policy, approvals, tests, or
the coding agent's own verification.

## Connect Codex

```powershell
codex mcp add jev --url http://127.0.0.1:8787/mcp
codex mcp list
```

Project config:

```toml
[mcp_servers.jev]
url = "http://127.0.0.1:8787/mcp"
```

## Connect Claude Code

```powershell
claude mcp add --transport http jev http://127.0.0.1:8787/mcp
claude mcp list
```

Inside Claude Code, `/mcp` shows the connection.

## Connect OpenCode

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "servers": {
      "jev": {
        "type": "remote",
        "url": "http://127.0.0.1:8787/mcp",
        "oauth": false,
        "protocol": "auto"
      }
    }
  }
}
```

## Agent contract

For every non-trivial coding task:

1. Call `jev_task_start` with the correct `source_agent`.
2. Preserve `task_id`.
3. Do repository work and verification with normal coding-agent tools.
4. Attach extra bounded decisions with `jev_decide(task_id=...)` when useful.
5. Before the final response, call `jev_task_complete` with actual evidence.
6. If blocked/failed, call `jev_task_fail` instead.
7. Never report `tests_passed=true` unless relevant tests actually ran and passed.

See [`FEEDBACK_LOOP.md`](FEEDBACK_LOOP.md) for CLI/evaluation/export details.

## Optimization and Rules Configuration

Jev MCP server features a 3-layer hybrid architecture:
- **Layer 1 Fast-Path Rules**: Deterministic regex matching (<1ms) for instant allows or catastrophic blocks.
  Pass `--rules-config path/to/rules.yaml` or place a `rules.yaml` in the repo root / `.jev/rules.yaml`. See `rules.example.yaml`.
- **Layer 2 LRU Decision Cache**: In-memory LRU cache (1024 entries) yielding ~15ms responses on repeated tool calls. Cache metrics are exposed via `jev_health`.
- **Layer 3 Neural Model**: `th1200` with optional `--quantize` on CPU (~200–300ms).

## Automated Claude Code Hook Integration

For zero-overhead proactive protection in Claude Code, configure the PreToolUse hook to automatically inspect every tool execution before it runs:
See [`CLAUDE_HOOK_SETUP.md`](CLAUDE_HOOK_SETUP.md).

## STDIO alternative

```powershell
.\.venv\Scripts\jev.exe serve --model JonusNattapong/jev-my-bro-th1200 --transport stdio --quantize
```

Do not run three separate stdio instances for three coding agents on a
memory-constrained workstation; each process may load its own checkpoint. Streamable HTTP on `http://127.0.0.1:8787/mcp` is strongly recommended.
