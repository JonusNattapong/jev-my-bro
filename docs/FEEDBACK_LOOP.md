# Feedback and evaluation loop

Jev tracks one task lifecycle per non-trivial coding task. The coding agent still
does the real repository work; Jev records the initial model decision, later
bounded decisions, verification evidence, and final task result.

## Primary lifecycle

```text
Claude Code / Codex / OpenCode
  -> jev_task_start
  -> task_id + initial decision
  -> agent inspects / edits / builds / tests with normal tools
  -> jev_decide(task_id=...) zero or more times
  -> jev_task_complete OR jev_task_fail
  -> SQLite feedback store
  -> stats / eval / export
  -> reviewed v0.3 evidence
```

Default database:

```text
artifacts/feedback/jev_feedback.sqlite3
```

Runtime feedback stays under `artifacts/` and is ignored by Git.

## Start

```json
{
  "context": "Fix the replication regression",
  "source_agent": "claude_code",
  "repo": "warz",
  "agent_model": "optional model name",
  "abstain_threshold": 0.6
}
```

The result contains a durable `task_id` and the initial Jev decision:

```json
{
  "task_id": "jevtask-...",
  "status": "running",
  "decision_id": "jev-...",
  "decision": "ask_user",
  "raw_decision": "ask_user",
  "confidence": 0.81,
  "abstain": false
}
```

## Additional decisions

Attach later decisions to the same task:

```json
{
  "task_id": "jevtask-...",
  "context": "Should this bounded patch execute now?"
}
```

Task-level outcome and intermediate decisions remain separate. A passing task
does not automatically label every intermediate Jev decision as correct.

## Complete

After successful implementation and verification:

```json
{
  "task_id": "jevtask-...",
  "agent_choice": "minimal_patch",
  "tests_passed": true,
  "test_command": "pytest -q",
  "test_exit_code": 0,
  "test_summary": "all passed",
  "implementation_summary": "Applied and verified the bounded fix.",
  "changed_files": ["jevbro/core.py"],
  "git_commit": "optional",
  "duration_ms": 1200
}
```

`tests_passed=true` means relevant tests actually ran and passed. Use `false`
for an observed failure and null when tests were not run or are unknown.

## Fail

If the coding task cannot be completed:

```json
{
  "task_id": "jevtask-...",
  "failure_reason": "required SDK unavailable",
  "tests_passed": false,
  "test_command": "build.cmd",
  "test_exit_code": 1
}
```

A repeated complete/fail call is retry-safe for the same terminal state. A task
that already completed cannot later be failed, and a failed task cannot later
be completed.

## MCP task tools

- `jev_task_start`
- `jev_task_complete`
- `jev_task_fail`
- `jev_task_status`
- `jev_task_list`
- `jev_decide`
- `jev_feedback_stats`

Older `jev_feedback_start`, `jev_feedback_complete`, `jev_feedback_get`, and
`jev_record_outcome` remain available for compatibility.

## CLI

Install the editable console entry point once:

```powershell
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

Use the installed Windows console entry point directly:

```powershell
.\.venv\Scripts\jev.exe health
.\.venv\Scripts\jev.exe task start "Fix regression" --agent codex --repo jev-my-bro
.\.venv\Scripts\jev.exe task status jevtask-...
.\.venv\Scripts\jev.exe task complete jevtask-... --choice minimal_patch --tests-passed --test-command "pytest -q" --test-exit-code 0
.\.venv\Scripts\jev.exe task fail jevtask-... --reason "dependency unavailable"
.\.venv\Scripts\jev.exe task list --status completed
.\.venv\Scripts\jev.exe stats
.\.venv\Scripts\jev.exe eval
.\.venv\Scripts\jev.exe export
.\.venv\Scripts\jev.exe harvest
```

The legacy wrappers live under `scripts\` rather than the repository root.
New documentation uses `.\.venv\Scripts\jev.exe` as the canonical path.

Task commands call the running MCP service, so they reuse the single loaded Laya
checkpoint instead of loading the model for every command. `eval`, `export`, and `harvest`
operate directly on the local SQLite feedback store.

## Evaluation, export, and active learning harvest

```powershell
# Evaluate model calibration against task outcomes
.\.venv\Scripts\jev.exe eval

# Export session logs for external audits
.\.venv\Scripts\jev.exe export

# Mine edge cases and high-uncertainty decisions for dataset expansion (e.g. th1400)
.\.venv\Scripts\jev.exe harvest

# Analyze Fast-Path rule hit rates from real-world telemetry
python scripts/analyze_rule_hit_rate.py
```

Defaults:

```text
artifacts/feedback/evaluation.json
artifacts/feedback/v0.3-feedback.jsonl
artifacts/feedback/active_learning_report.json
data/active_learning/candidates_th1400.csv
```

The default export includes terminal tasks (completed and failed) and excludes
running tasks. Use `.\.venv\Scripts\jev.exe export --include-running` when an audit needs them.

The export is evidence, not gold training data. Do not append it directly to
`data/hf_expanded/`. Review, deduplicate, remove sensitive/proprietary context,
assign provenance, and build new independent train/validation/calibration/test
splits before v0.3 training.
