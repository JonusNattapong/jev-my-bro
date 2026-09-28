# Claude Code PreToolUse Hook Setup for Jev

Integrate Jev Governance directly into Claude Code so **every single Bash and file modification command is verified automatically**, eliminating reliance on agent memory.

---

## ⚡ How it works

When Claude Code is about to run a tool (`Bash`, `Write`, `Edit`, `MultiEdit`, `NotebookEdit`), Claude Code executes the `PreToolUse` hook first:
1. **Low-risk command** (`git status`, reading docs) ➔ Jev returns `allow` ➔ Tool executes immediately.
2. **Ambiguous or High-Risk action** (refactoring auth, migrations) ➔ Jev returns `ask` ➔ Claude Code **pauses and asks the human** for explicit confirmation.
3. **Prohibited action** (`rm -rf /`, exfiltrating `.env`, `DROP DATABASE`) ➔ Jev returns `deny` ➔ Claude Code blocks execution completely.

Jev is advisory for model verdicts: it does not execute commands, and a low-confidence model abstains
toward `ask` rather than blocking. **The hook does not gate on the model by default.**
`JEV_ENFORCE=0` records the verdict to stderr and lets Claude Code apply its own permission rules;
set `JEV_ENFORCE=1` only after the measured criteria in [`ENFORCEMENT_GATE.md`](ENFORCEMENT_GATE.md)
are met.

### Hard safety floor

Independent of `JEV_ENFORCE` and of whether the server is reachable, the hook denies a small set of
catastrophic operations before the model is consulted. It is deny-only, hardcoded, and deliberately
minimal:

- recursive force delete of a system or home path (`rm -rf /`, `rm -rf ~`)
- disk format or raw device write (`mkfs`, `dd of=/dev/…`, `shutil.rmtree(…)`)
- recursive delete driven by `find / … -delete`
- reading credential material (`cat .env`, `cat ~/.ssh/id_rsa`, `printenv …SECRET…`)
- sending credential material off-host (`curl … -d @.env`, piping a secret to a URL)
- destructive database statements (`DROP DATABASE/SCHEMA/TABLE`, `TRUNCATE`)
- irreversible git destruction (force-push to `main`/`master`, `git clean -fdx`,
  `git reset --hard origin/main`)
- audit bypass (`git commit --no-verify`, disabling branch protection or required checks)
- writing to credential files or trust boundaries (`.env`, private keys, `.git/hooks/`,
  `.github/workflows/`)

The command patterns apply only to `Bash`-style tools, so documentation that quotes a dangerous
command is not blocked. `.env.example` and similar templates stay writable. There is no allow side:
this list never permits anything, it only refuses. Extending it beyond catastrophic operations
re-creates the rule engine that was removed, so keep it short and explicit.

The hook returns decisions via **JSON `permissionDecision`** field, which Claude Code natively understands.

---

## 🚀 Configuration

### Hook Registration

The hook is registered once at user level in `~/.claude/settings.json`, with absolute paths so it
runs in every project:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash|Write|Edit|MultiEdit|NotebookEdit",
        "hooks": [
          {
            "type": "command",
            "command": "<repo>/.venv/Scripts/python.exe <repo>/hooks/claude_pre_tool_use.py"
          }
        ]
      }
    ]
  }
}
```

Do **not** register the hook again in a project `.claude/settings.json`: the tool call would be
evaluated twice and a relative `python hooks/...` command may resolve to an interpreter without
the project dependencies.

### Project-level Settings

This repository's `.claude/settings.json` only sets `env` overrides:

- **`JEV_TIMEOUT`**: Seconds to wait for Jev server (default: 8.0)
- **`JEV_ENFORCE`**: `0` (advisory, default — never blocks) or `1` (return `permissionDecision`)
- **`JEV_FAIL_MODE`**: `open` (allow if offline), `ask` (pause for human), or `deny` (block if offline)
- **`JEV_MCP_URL`**: URL of Jev MCP server (default: http://127.0.0.1:8787/mcp)
- **`JEV_RETRY_TIMEOUT`**: Shorter timeout for the single retry after a timeout (default: 3.0)

### Context sent to Jev

For `Write`/`Edit`/`MultiEdit` the hook sends the file path, a file classification
(`documentation`, `configuration`, `test`, `source code`, `unknown`), the payload size, and a
bounded snippet (max 600 chars) of the text being written. The snippet is what makes the request
judgeable — a contentless `Edit file <path> (replacing ~0.0KB)` string carries no intent and gates
to `reject` on most inputs.

The hook does **not** send a `language` argument; the server detects it from the context. Pinning
it to `th` scored an English documentation edit at `prohibited=0.65` (reject) versus `0.18`
(`ask_user`) on the identical context, so a wrong pin silently denies every English tool call.

### Tool Matching

The hook matcher covers: `Bash|Write|Edit|MultiEdit|NotebookEdit`

---

## 🛠️ Testing

Test the hook manually:

```powershell
# Safe command
python hooks/claude_pre_tool_use.py "git status"

# Destructive command
python hooks/claude_pre_tool_use.py "rm -rf /"

# Via JSON payload
Write-Output '{"tool_name":"Bash","tool_input":{"command":"git diff"}}' | python hooks/claude_pre_tool_use.py
```

---

## 📊 Requirements

Before using the hook:
1. **Jev server running**: `jev serve --model JonusNattapong/jev-my-bro-th1200 --host 127.0.0.1 --port 8787`
2. **Virtual environment activated**: `.\.venv\Scripts\Activate.ps1`
3. **Hook script in place**: `/hooks/claude_pre_tool_use.py`

---

## 🔍 Troubleshooting

- **Hook not triggering**: Verify `~/.claude/settings.json` has the hook registered with absolute
  paths and the correct matcher tool names
- **Every file edit denied**: Jev is gating the context, not the path. Inspect the
  `permissionDecisionReason` — the `context:` tail shows exactly what Jev saw. An empty or
  path-only context means the snippet is missing; an explicit `writing: ...` snippet means Jev
  judged the content itself
- **Jev server offline**: No rule engine remains behind the hook, so the decision is
  `JEV_FAIL_MODE` alone (`ask` in this repository). Entries are appended to `JEV_FALLBACK_LOG`
- **Timeouts**: Increase `JEV_TIMEOUT` or set env var `JEV_TIMEOUT=10.0`
