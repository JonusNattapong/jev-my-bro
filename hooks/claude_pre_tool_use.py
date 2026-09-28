#!/usr/bin/env python3
"""Claude Code PreToolUse Hook for Jev Governance.

Intercepts Claude Code tool calls (Bash, Write, Edit, MultiEdit, NotebookEdit) and asks Jev for a
governance verdict:
- execute    -> allow immediately
- ask_user   -> JSON permissionDecision "ask" (Claude Code pauses for user)
- reject     -> JSON permissionDecision "deny" (Claude Code blocks execution)

Enforcement is off by default (`JEV_ENFORCE=0`): the hook records the verdict to stderr and lets
Claude Code apply its own permission rules. Set `JEV_ENFORCE=1` only after the measured criteria in
`docs/ENFORCEMENT_GATE.md` are met — the model is not currently accurate enough to gate tool calls.
"""

from __future__ import annotations

import datetime
import json
import os
import socket
import sys
import urllib.error
import urllib.request
from typing import Any

JEV_MCP_URL = os.environ.get("JEV_MCP_URL", "http://127.0.0.1:8787/mcp")
JEV_TIMEOUT = float(os.environ.get("JEV_TIMEOUT", "8.0"))
JEV_RETRY_TIMEOUT = float(os.environ.get("JEV_RETRY_TIMEOUT", "3.0"))
JEV_FAIL_MODE = os.environ.get("JEV_FAIL_MODE", "open")  # "open", "deny", or "ask"
JEV_ENFORCE = os.environ.get("JEV_ENFORCE", "0") not in ("0", "false", "False", "")
JEV_FALLBACK_LOG = os.environ.get("JEV_FALLBACK_LOG", "artifacts/feedback/hook_fallback.log")


MAX_SNIPPET_CHARS = 600

_DOC_SUFFIXES = (".md", ".mdx", ".rst", ".txt", ".adoc")
_CONFIG_SUFFIXES = (".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf")
_CONFIG_NAMES = (".env", ".claude", ".github", "settings.json", "Dockerfile", "Makefile")


def _collapse(text: Any, limit: int = MAX_SNIPPET_CHARS) -> str:
    """Flatten text to a single bounded line so the model sees intent, not a wall of source."""
    flat = " ".join(str(text or "").split())
    if len(flat) > limit:
        return flat[:limit] + " ..."
    return flat


def _file_kind(path: Any) -> str:
    """Classify the target file so governance can weigh intent (docs vs config vs code)."""
    normalized = str(path or "").replace("\\", "/").lower()
    name = normalized.rsplit("/", 1)[-1]
    if name.endswith(_DOC_SUFFIXES) or "/docs/" in normalized or "/doc/" in normalized:
        return "documentation"
    if name.endswith(_CONFIG_SUFFIXES) or any(token in normalized for token in _CONFIG_NAMES):
        return "configuration"
    if name.startswith("test_") or "_test." in name or "/tests/" in normalized:
        return "test"
    if name.endswith((".py", ".ts", ".tsx", ".js", ".go", ".rs", ".java", ".sh", ".ps1")):
        return "source code"
    return "unknown"


def _proposed_text(tool_input: dict[str, Any]) -> str:
    """Collect what the edit would actually introduce, across Write/Edit/MultiEdit shapes."""
    edits = tool_input.get("edits")
    if isinstance(edits, list) and edits:
        parts = [
            str(item.get("new_string") or item.get("new_str") or item.get("content") or "")
            for item in edits
            if isinstance(item, dict)
        ]
        return _collapse(" | ".join(part for part in parts if part))
    return _collapse(
        tool_input.get("new_string")
        or tool_input.get("new_str")
        or tool_input.get("content")
        or ""
    )


def extract_context(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Extract semantic context string from tool arguments.

    For Write/Edit tools, send path, file kind, size, and a bounded snippet of the proposed
    text. The snippet is what makes the request judgeable: a bare "Edit file X (replacing
    ~0.0KB)" carries no intent and gates to reject on most inputs.
    """
    normalized_tool = (tool_name or "").lower()

    if normalized_tool in ("bash", "run_command", "shell_command", "terminal"):
        return str(
            tool_input.get("command") or tool_input.get("CommandLine") or tool_input.get("cmd") or ""
        ).strip()

    if normalized_tool in ("write", "writefile", "write_to_file", "create_file"):
        path = tool_input.get("file_path") or tool_input.get("path") or "unknown"
        content = str(tool_input.get("content") or "")
        size_kb = len(content) / 1024.0
        proposed = _proposed_text(tool_input)
        suffix = f" writing: {proposed}" if proposed else ""
        return f"Write {_file_kind(path)} file {path} (~{size_kb:.1f}KB){suffix}"

    if normalized_tool in ("edit", "editfile", "replace_file_content", "edit_file", "multiedit"):
        path = tool_input.get("file_path") or tool_input.get("path") or "unknown"
        old_str = str(tool_input.get("old_string") or "")
        size_kb = len(old_str) / 1024.0
        proposed = _proposed_text(tool_input)
        suffix = f" writing: {proposed}" if proposed else ""
        return f"Edit {_file_kind(path)} file {path} (replacing ~{size_kb:.1f}KB){suffix}"

    if normalized_tool in ("notebookcell", "run_notebook_cell"):
        code = str(tool_input.get("code") or tool_input.get("cell_source") or "").strip()
        return f"Run notebook cell: {code[:100]}"

    # Generic fallback
    if "command" in tool_input:
        return str(tool_input["command"]).strip()

    return f"Execute {tool_name}"


def _write_fallback_log(context: str, error_kind: str) -> None:
    """Append one JSON line recording why the hook fell back to the fail mode.

    Logging failures are swallowed on purpose: the audit trail must never be
    able to crash or block a governance decision.
    """
    if not JEV_FALLBACK_LOG:
        return
    try:
        entry = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "error_kind": error_kind,
            "fail_mode": JEV_FAIL_MODE,
            "context": context[:200],
        }
        log_path = JEV_FALLBACK_LOG
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def query_jev_remote(context: str, timeout: float) -> tuple[dict[str, Any] | None, str]:
    """    Call Jev MCP server via HTTP JSON-RPC.

    `language` is intentionally not sent: the server auto-detects it from the context. Pinning
    it to "th" scored an English edit at prohibited=0.65 (reject) versus 0.18 (ask_user) on the
    same context, so a wrong pin silently denies every English tool call.

    Returns (decision, error_kind). error_kind is "" on success, otherwise
    one of "timeout", "connection_error", or "unknown_error" so the caller
    can decide whether a retry is worthwhile and so failures are logged with
    an accurate cause instead of a blanket except.
    """
    payload = {
        "jsonrpc": "2.0",
        "id": "hook-query-1",
        "method": "tools/call",
        "params": {
            "name": "jev_decide",
            "arguments": {
                "context": context,
                "source_agent": "claude-code",
            },
        },
    }
    req = urllib.request.Request(
        JEV_MCP_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "result" in data and "content" in data["result"]:
                for item in data["result"]["content"]:
                    if item.get("type") == "text":
                        return json.loads(item["text"]), ""
            return data.get("result"), ""
    except socket.timeout:
        return None, "timeout"
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, socket.timeout):
            return None, "timeout"
        return None, "connection_error"
    except Exception:
        return None, "unknown_error"


def query_jev_local_fallback(context: str) -> dict[str, Any]:
    """Decision when the Jev server is unreachable: apply the configured fail mode only.

    There is no deterministic rule engine behind this fallback anymore, so an offline server
    means the hook can no longer distinguish safe from destructive calls.
    """
    default_decision = "execute" if JEV_FAIL_MODE == "open" else ("reject" if JEV_FAIL_MODE == "deny" else "ask_user")
    return {
        "engine": "fallback-offline",
        "gated_decision": default_decision,
        "risk": 0.0,
    }


def evaluate_action(context: str) -> dict[str, Any]:
    """Evaluate context through Jev server, with one bounded retry on timeout,
    falling back to the configured fail mode only after both attempts fail.
    """
    decision, error_kind = query_jev_remote(context, JEV_TIMEOUT)

    if decision is None and error_kind == "timeout":
        # A single retry with a shorter budget: recovers from a transient
        # queue spike (common under concurrent agent sessions) without
        # doubling the worst-case wait on a genuinely offline server.
        decision, error_kind = query_jev_remote(context, JEV_RETRY_TIMEOUT)

    if decision is None:
        _write_fallback_log(context, error_kind or "unknown_error")
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

    # Always surface the evaluated context so the reason is actionable, not just a verdict.
    if reason:
        reason_str = f"[{engine}] Risk={risk:.1f}/4 | {reason} | context: {context[:160]}"
    else:
        reason_str = f"[{engine}] {permission} | context: {context[:160]}"

    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": permission,
            "permissionDecisionReason": reason_str,
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
    reason = ""

    # Advisory mode (default): record the decision but never gate the tool call. The model
    # currently scores below the majority baseline on its own test split and its risk head is
    # flat, so enforcing its verdict would block work without protecting anything. Re-enable
    # with JEV_ENFORCE=1 only after the criteria in docs/ENFORCEMENT_GATE.md are met.
    if not JEV_ENFORCE:
        print(f"[jev] advisory | {gated} | risk={risk:.2f} | {context[:120]}", file=sys.stderr)
        sys.exit(0)

    # Output permission decision as JSON for Claude Code
    print(encode_permission_decision(gated, context, risk, reason, engine))
    sys.exit(0)


if __name__ == "__main__":
    main()

