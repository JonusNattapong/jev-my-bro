# Claude Code guidance

Follow the repository-wide rules in `AGENTS.md`.

Jev is available through the project MCP server named `jev`. Use
`jev_decide` when an operational action has multiple plausible governance
outcomes such as execute now, ask for explicit approval, or reject. Treat Jev as
advisory only. If it returns `abstain: true`, rely on repository policy,
stronger evidence, or human review instead of the raw model decision.

For each non-trivial coding task, call `jev_task_start` with
`source_agent="claude_code"` before implementation and keep the returned
`task_id`. Pass that id to any additional `jev_decide` calls.

Read `gated_decision`, not only `decision`. `decision` comes from the action
head alone; `gated_decision` also applies the `prohibited` and `needs_review`
signals (`>= 0.5`) and is computed even when Jev abstains. If it is `reject` or
`ask_user`, get explicit user approval before any outward-facing or
hard-to-reverse action; an explicit user instruction for that action counts. Implement and
run tests with normal Claude Code tools. Before giving the final task response,
call `jev_task_complete` with the final choice and actual verification evidence.
If the task is blocked or fails, call `jev_task_fail` instead. Use null for
`tests_passed` when tests were not run.

Do not use Jev as a code generator or as a replacement for tests, verification,
or approval boundaries.

## Running the Jev server

The `jev` MCP server (`http://127.0.0.1:8787/mcp`) must be running before the
session starts. If `jev` fails to connect, start it and reconnect with `/mcp`:

```powershell
.\.venv\Scripts\jev.exe serve --model JonusNattapong/jev-my-bro-th1200 --host 127.0.0.1 --port 8787 --quantize
```

## Language and Governance Engine

The active model (`jev-my-bro-th1200`) was trained on 1,200 curated Thai governance
cases with 86.75% accuracy and 100% Level-4 risk recall.
Write the `context` for `jev_task_start` and `jev_decide` in Thai and pass
`language="th"` for optimal semantic decisions.

The server operates a 3-layer hybrid cascaded architecture:
- **Layer 1 (Fast-Path Rules)**: Instant (<1ms) deterministic blocks for destructive acts (`rm -rf`) and allows for pure read-only inspections (`git status`). Custom rules can be placed in `rules.yaml` (see `rules.example.yaml`).
- **Layer 2 (LRU Cache)**: 1024-slot in-memory cache returning ~15ms decisions on repeated commands without touching the neural model.
- **Layer 3 (Neural Model)**: Semantic evaluation via `th1200` with INT8 dynamic quantization on CPU (~200–300ms).

## Automated PreToolUse Hook Integration

This repository includes `.claude/settings.json` configured with `hooks/claude_pre_tool_use.py`.
Every tool execution (`Bash`, `Write`, `Edit`, `MultiEdit`, `NotebookEdit`) is proactively evaluated:
- **Safe commands**: allowed automatically without latency overhead.
- **Destructive operations**: rejected immediately (`permissionDecision: deny`).
- **High-risk modifications**: prompt user for confirmation (`permissionDecision: ask`).

See [`docs/CLAUDE_HOOK_SETUP.md`](docs/CLAUDE_HOOK_SETUP.md) for full configuration and environment options (`JEV_TIMEOUT`, `JEV_FAIL_MODE`).
