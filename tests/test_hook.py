"""Tests for Claude Code PreToolUse Hook."""

import json
import subprocess
import sys
from pathlib import Path


def run_hook(tool_name: str, tool_input: dict) -> dict:
    """Run hook via subprocess and parse JSON output."""
    payload = json.dumps({"tool_name": tool_name, "tool_input": tool_input})
    result = subprocess.run(
        [sys.executable, "hooks/claude_pre_tool_use.py"],
        input=payload,
        text=True,
        capture_output=True,
        cwd=Path(__file__).parent.parent,
    )
    if result.stdout:
        return json.loads(result.stdout)
    return {}


def test_hook_safe_git_command() -> None:
    """Test that git status is allowed."""
    output = run_hook("Bash", {"command": "git status"})
    
    assert "hookSpecificOutput" in output
    hook_out = output["hookSpecificOutput"]
    # Either allow or ask is fine - depends on server
    assert hook_out["permissionDecision"] in ("allow", "ask")


def test_hook_destructive_database_command() -> None:
    """Test that DROP DATABASE is blocked."""
    output = run_hook("Bash", {"command": "DROP TABLE users"})
    
    assert "hookSpecificOutput" in output
    hook_out = output["hookSpecificOutput"]
    # Should block destructive database operations
    assert hook_out["permissionDecision"] in ("deny", "ask")


def test_hook_write_file() -> None:
    """Test that Write tool sends path info in decision reason."""
    output = run_hook("Write", {
        "file_path": "myconfig.py",
        "content": "x" * 50000
    })
    
    assert "hookSpecificOutput" in output
    hook_out = output["hookSpecificOutput"]
    # Reason should mention file, not contain full content
    reason = hook_out["permissionDecisionReason"]
    assert "myconfig.py" in reason or "Write" in reason


def test_hook_fallback_on_bad_payload() -> None:
    """Test that hook gracefully handles invalid JSON."""
    result = subprocess.run(
        [sys.executable, "hooks/claude_pre_tool_use.py"],
        input="definitely not json at all",
        text=True,
        capture_output=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0
