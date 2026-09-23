# Claude Code PreToolUse Hook Setup for Jev

Integrate Jev Governance directly into Claude Code so **every single Bash and file modification command is verified automatically**, eliminating reliance on agent memory.

---

## ⚡ How it works

When Claude Code is about to run a tool (`Bash`, `Write`, `Edit`, `MultiEdit`, `NotebookEdit`), Claude Code executes the `PreToolUse` hook first:
1. **Safe command** (`git status`, reading docs) ➔ Jev returns `allow` ➔ Tool executes immediately (< 1ms with cache/rules).
2. **Ambiguous or High-Risk action** (refactoring auth, migrations) ➔ Jev returns `ask` ➔ Claude Code **pauses and asks the human** for explicit confirmation.
3. **Destructive command** (`rm -rf /`, exfiltrating `.env`, `DROP DATABASE`) ➔ Jev returns `deny` ➔ Claude Code blocks execution completely.

The hook returns decisions via **JSON `permissionDecision`** field, which Claude Code natively understands.

---

## 🚀 Configuration

### Project-level Settings

The repository includes `.claude/settings.json` with the hook pre-configured.

Customize these environment variables:
- **`JEV_TIMEOUT`**: Seconds to wait for Jev server (default: 5.0)
- **`JEV_FAIL_MODE`**: `open` (allow if offline) or `deny` (ask if offline)
- **`JEV_MCP_URL`**: URL of Jev MCP server (default: http://127.0.0.1:8787/mcp)

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

- **Hook not triggering**: Verify `.claude/settings.json` has correct tool names
- **Jev server offline**: Falls back to local rules (Layer 1)
- **Timeouts**: Increase `JEV_TIMEOUT` or set env var `JEV_TIMEOUT=10.0`
