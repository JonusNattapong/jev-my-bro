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

The active model (`jev-my-bro-th1200`) was trained on 1,200 curated Thai governance cases.
Measured on its own locked 100-case test split it scores **49.0%** action accuracy against a **59.0%**
majority baseline, with score QWK 0.205 and a risk head that outputs ~2.4 for every input. It is
advisory only: do not treat its verdict as authorization. See
[`docs/ENFORCEMENT_GATE.md`](docs/ENFORCEMENT_GATE.md).
Write the `context` for `jev_task_start` and `jev_decide` in Thai and pass
`language="th"` for optimal semantic decisions; never pin `language` in automated callers, because a
mismatched pin silently changes the score.

The server operates a 2-layer cached architecture:
- **Layer 1 (LRU Cache)**: 1024-slot in-memory cache returning ~15ms decisions on repeated commands without touching the neural model.
- **Layer 2 (Neural Model)**: Semantic evaluation via `th1200` with INT8 dynamic quantization on CPU (~200–300ms).

There is no deterministic rule engine. The model is advisory and abstains on most inputs, so low-confidence decisions surface as `ask_user` rather than an automatic block.

## Automated PreToolUse Hook Integration

`hooks/claude_pre_tool_use.py` is registered as a user-level `PreToolUse` hook in `~/.claude/settings.json` with absolute paths, so it observes every project; this repository's `.claude/settings.json` only sets `env` overrides (`JEV_ENFORCE=0`, `JEV_FAIL_MODE=ask`). The hook **does not gate on the model**; it logs the verdict and lets Claude Code apply its own rules. It does deny a small hardcoded set of catastrophic operations (recursive system deletes, credential reads, secret exfiltration, destructive SQL, force-push to main, audit bypass) regardless of `JEV_ENFORCE`. See [`docs/ENFORCEMENT_GATE.md`](docs/ENFORCEMENT_GATE.md).
Every tool execution (`Bash`, `Write`, `Edit`, `MultiEdit`, `NotebookEdit`) is recorded:
- **Model verdicts are advisory.** The model reaches 75.0% action accuracy on its own test split
  but abstains on 100% of requests and has no discrimination on tool-call inputs, so its verdict is
  logged, not enforced.
- The hard safety floor is enforced: catastrophic operations are refused before the model is
  consulted.

See [`docs/CLAUDE_HOOK_SETUP.md`](docs/CLAUDE_HOOK_SETUP.md) for full configuration and environment options (`JEV_TIMEOUT`, `JEV_ENFORCE`, `JEV_FAIL_MODE`).
