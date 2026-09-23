#!/usr/bin/env python3
"""Claude Code PreToolUse Hook for Jev Governance.

Intercepts Claude Code tool calls (Bash, Write, Edit, MultiEdit, NotebookEdit),
evaluates safety via Jev (Layer 1 Fast-Path + Layer 2 Model), and enforces governance:
- execute    -> allow immediately
- ask_user   -> JSON permissionDecision "ask" (Claude Code pauses for user)
- reject     -> JSON permissionDecision "deny" (Claude Code blocks execution)

Returns permissionDecision via stdout JSON for Claude Code.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

JEV_MCP_URL = os.environ.get("JEV_MCP_URL", "http://127.0.0.1:8787/mcp")
JEV_TIMEOUT = float(os.environ.get("JEV_TIMEOUT", "5.0"))
JEV_FAIL_MODE = os.environ.get("JEV_FAIL_MODE", "open")  # "open" or "deny"


def extract_context(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Extract semantic context string from tool arguments.

    For Write/Edit tools, only send path + size, not file contents.
    """
    normalized_tool = (tool_name or "").lower()

    if normalized_tool in ("bash", "run_command", "shell_command", "terminal"):
        return str(
            tool_input.get("command") or tool_input.get("CommandLine") or tool_input.get("cmd") or ""
        ).strip()

    if normalized_tool in ("write", "writefile", "write_to_file", "create_file"):
        path = tool_input.get("file_path") or tool_input.get("path") or "unknown"
        # Estimate size if content is present
        content = tool_input.get("content") or ""
        size_kb = len(str(content)) / 1024.0
        return f"Write file {path} (~{size_kb:.1f}KB)"

    if normalized_tool in ("edit", "editfile", "replace_file_content", "edit_file", "multiedit"):
        path = tool_input.get("file_path") or tool_input.get("path") or "unknown"
        # For edits, extract just the intent, not the actual content
        old_str = tool_input.get("old_string") or ""
        size_kb = len(str(old_str)) / 1024.0
        return f"Edit file {path} (replacing ~{size_kb:.1f}KB)"

    if normalized_tool in ("notebookcell", "run_notebook_cell"):
        code = str(tool_input.get("code") or tool_input.get("cell_source") or "").strip()
        return f"Run notebook cell: {code[:100]}"

    # Generic fallback
    if "command" in tool_input:
        return str(tool_input["command"]).strip()

    return f"Execute {tool_name}"


def query_jev_remote(context: str) -> dict[str, Any] | None:
    """Call Jev MCP server via HTTP JSON-RPC with configurable timeout."""
    payload = {
        "jsonrpc": "2.0",
        "id": "hook-query-1",
        "method": "tools/call",
        "params": {
            "name": "jev_decide",
            "arguments": {
                "context": context,
                "source_agent": "claude-code",
                "language": "th",
            },
        },
    }
    req = urllib.request.Request(
        JEV_MCP_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=JEV_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "result" in data and "content" in data["result"]:
                for item in data["result"]["content"]:
                    if item.get("type") == "text":
                        return json.loads(item["text"])
            return data.get("result")
    except Exception:
        return None


def query_jev_local_fallback(context: str) -> dict[str, Any]:
    """Fallback to deterministic local rules if Jev server is offline or slow."""
    try:
        from jevbro.rules import evaluate_rules

        match = evaluate_rules(context)
        if match is not None:
            return {
                "engine": "jev-rules-fastpath-local",
                "gated_decision": match.action,
                "risk": match.risk,
                "rule_matched": match.rule_id,
                "rule_reason": match.reason,
            }
    except Exception:
        pass

    # Default based on fail mode (open or deny)
    # Fail safely: deny/ask if server offline + rule miss
    default_decision = "execute" if JEV_FAIL_MODE == "open" else ("reject" if JEV_FAIL_MODE == "deny" else "ask_user")
    return {
        "engine": "fallback-offline",
        "gated_decision": default_decision,
        "risk": 0.0,
    }


def evaluate_action(context: str) -> dict[str, Any]:
    """Evaluate context through Jev server, falling back to local rules."""
    decision = query_jev_remote(context)
    if decision is None:
        decision = query_jev_local_fallback(context)
    return decision


def encode_permission_decision(
    decision: str, context: str, risk: float, reason: str, engine: str
) -> str:
    """Encode the decision as JSON for Claude Code's permissionDecision field."""
    decision_map = {
        "execute": "allow",
        "ask_user": "ask",
        "reject": "deny",
    }
    permission = decision_map.get(decision, "allow")

    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": permission,
            "permissionDecisionReason": f"[{engine}] Risk={risk:.1f}/4 | {reason}" if reason else f"[{engine}] {permission}",
            "context": context[:150],
        }
    }
    return json.dumps(output, ensure_ascii=False)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    # Read tool execution payload from stdin or command line arguments
    raw_input = ""
    tool_name = "Bash"
    tool_input: dict[str, Any] = {}

    if len(sys.argv) > 1:
        raw_input = " ".join(sys.argv[1:])
        tool_input = {"command": raw_input}
    else:
        try:
            raw_input = sys.stdin.read().strip()
            idx = raw_input.find("{")
            if idx != -1:
                payload = json.loads(raw_input[idx:])
                tool_name = (
                    payload.get("tool_name")
                    or payload.get("tool")
                    or payload.get("name")
                    or "Bash"
                )
                tool_input = (
                    payload.get("tool_input")
                    or payload.get("parameters")
                    or payload.get("args")
                    or payload.get("input")
                    or {}
                )
            else:
                sys.exit(0)
        except Exception:
            sys.exit(0)

    context = extract_context(tool_name, tool_input)
    if not context:
        sys.exit(0)

    decision = evaluate_action(context)
    gated = decision.get("gated_decision", "execute")
    risk = decision.get("risk", 0.0)
    engine = decision.get("engine", "jev")
    reason = decision.get("rule_reason", "")

    # Output permission decision as JSON for Claude Code
    print(encode_permission_decision(gated, context, risk, reason, engine))
    sys.exit(0)


if __name__ == "__main__":
    main()

