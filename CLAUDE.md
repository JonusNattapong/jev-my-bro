# Claude Code guidance

Follow the repository-wide rules in `AGENTS.md`.

Jev is available through the project MCP server named `jev`. Use
`jev_decide` when an operational action has multiple plausible governance
outcomes such as execute now, ask for explicit approval, or reject. Treat Jev as
advisory only. If it returns `abstain: true`, rely on repository policy,
stronger evidence, or human review instead of the raw model decision.

For each non-trivial coding task, call `jev_task_start` with
`source_agent="claude_code"` before implementation and keep the returned
`task_id`. Pass that id to any additional `jev_decide` calls. Implement and
run tests with normal Claude Code tools. Before giving the final task response,
call `jev_task_complete` with the final choice and actual verification evidence.
If the task is blocked or fails, call `jev_task_fail` instead. Use null for
`tests_passed` when tests were not run.

Do not use Jev as a code generator or as a replacement for tests, verification,
or approval boundaries.
