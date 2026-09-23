"""Deterministic Fast-Path Rule Engine for instant security blocks and safe read-only operations."""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


@dataclass(frozen=True)
class RuleMatch:
    action: str  # "execute" | "reject" | "ask_user"
    needs_review: bool
    prohibited: bool
    risk: float  # 0.0 to 4.0
    rule_id: str
    reason: str


# 1. Hard Reject Patterns (Deterministic blocks for destructive/malicious acts)
DEFAULT_REJECT_RULES: list[tuple[str, re.Pattern, str]] = [
    (
        "hard_reject_destructive_fs",
        re.compile(
            r"\brm\s+-[a-z]*r[a-z]*f[a-z]*\s+([/~*]|\.\.|\$|\bvar/|\betc/|\busr/)|"
            r"\b(mkfs|format\s+[c-zC-Z]:|dd\s+if=/dev/zero)\b|"
            r"สั่งรันคำสั่ง\s*rm\s*-rf\s*[/~*]",
            flags=re.IGNORECASE,
        ),
        "Destructive file system or disk formatting operation detected.",
    ),
    (
        "hard_reject_fs_extended",
        re.compile(
            r"\brm\s+(-r\s+-f|-f\s+-r)\s+/|"
            r"\brm\s+(--recursive\s+--force|--force\s+--recursive|--recursive|--force)\s+/|"
            r"\bfind\s+/.*?-delete\b|"
            r"shutil\.rmtree",
            flags=re.IGNORECASE,
        ),
        "Extended file system destructive patterns.",
    ),

    (
        "hard_reject_secret_exfiltration",
        re.compile(
            r"(แสดงค่า|เปิดดูค่า|ขโมย|dump|print|cat|export).*(SECRET_KEY|AWS_SECRET|PRIVATE_KEY|\.env\b|id_rsa|id_ed25519)|"
            r"\bBEGIN\s+(RSA|OPENSSH|DSA|EC|PGP)?\s*PRIVATE\s+KEY\b|"
            r"\b(AWS_SECRET_ACCESS_KEY|ANTHROPIC_API_KEY|OPENAI_API_KEY)\s*[:=]",
            flags=re.IGNORECASE,
        ),
        "Attempt to inspect, display, or exfiltrate private credentials or secret keys.",
    ),
    (
        "hard_reject_destructive_database",
        re.compile(
            r"\bDROP\s+(DATABASE|SCHEMA)\b|"
            r"\b(DROP|TRUNCATE)\s+TABLE\b.*(prod|production)|"
            r"ลบตาราง.*(?:บน|ใน)\s*(?:ฐานข้อมูล|database)?\s*(?:prod|production)",
            flags=re.IGNORECASE,
        ),
        "Destructive DROP/TRUNCATE operation on database or production tables.",
    ),
    (
        "hard_reject_security_bypass",
        re.compile(
            r"(disable|bypass|ปิดการทำงาน).*(audit\s*log|commit\s*hook|branch\s*protection)|"
            r"\bgit\s+commit\b.*--no-verify\b.*(production|master|main)",
            flags=re.IGNORECASE,
        ),
        "Attempt to bypass audit logging or repository safety boundaries.",
    ),
]

# 2. Hard Allow Patterns (Safe, read-only, non-mutating local inspections)
DEFAULT_ALLOW_READONLY_RULES: list[tuple[str, re.Pattern, str]] = [
    (
        "hard_allow_git_inspection",
        re.compile(
            r"\bgit\s+(status|log|diff|show|blame|branch)\b|"
            r"(อ่าน|ตรวจ|ดู|เช็ค)\s*git\s*(status|log|diff|show|blame|branch)",
            flags=re.IGNORECASE,
        ),
        "Pure read-only repository inspection.",
    ),
    (
        "hard_allow_documentation",
        re.compile(
            r"(อ่านไฟล์|อ่านเอกสาร|ทำความเข้าใจ|สรุปเนื้อหา).*(readme|agents\.md|docs|\.md\b)|"
            r"\bread\s+(readme|documentation|agents\.md)\b",
            flags=re.IGNORECASE,
        ),
        "Read-only project documentation and guidance review.",
    ),
    (
        "hard_allow_local_code_inspection",
        re.compile(
            r"(ค้นหา|หา|grep|inspect|ตรวจ)\s+(?:ใน\s*repo|โค้ด|ฟังก์ชัน|class)|"
            r"\bgrep\b.*(?:ใน|in)\s*(?:repo|โฟลเดอร์|ไฟล์)",
            flags=re.IGNORECASE,
        ),
        "Read-only code inspection and search.",
    ),
    (
        "hard_allow_safe_file_write",
        re.compile(
            r"^(Write file|Edit file)\s+(?!.*(\.env\b|secret|credential|password|\.ssh[/\\]|"
            r"id_rsa|id_ed25519|private.?key|\.pem\b|\.aws[/\\]|/etc/|/usr/|system32|"
            r"\.git[/\\]hooks)).+",
            flags=re.IGNORECASE,
        ),
        "Safe file write or edit on a non-sensitive path.",
    ),
]

# Words that indicate state changes or destructive side-effects that disqualify from Hard Allow
DEFAULT_MUTATION_EXCLUSIONS_PATTERN = (
    r"(\b(delete|remove|rm\s|drop|truncate|format|revert|reset\s+--hard|push\s+-f|push\s+--force|commit|push|deploy)\b|"
    r"แก้ไข|ลบ|เปลี่ยน|แก้|ย้าย|ทำลาย|เขียนทับ)"
)

# Active runtime rule lists
REJECT_RULES: list[tuple[str, re.Pattern, str]] = list(DEFAULT_REJECT_RULES)
ALLOW_READONLY_RULES: list[tuple[str, re.Pattern, str]] = list(DEFAULT_ALLOW_READONLY_RULES)
MUTATION_EXCLUSIONS = re.compile(DEFAULT_MUTATION_EXCLUSIONS_PATTERN, flags=re.IGNORECASE)


def reset_rules() -> None:
    """Reset rules back to built-in defaults."""
    global REJECT_RULES, ALLOW_READONLY_RULES, MUTATION_EXCLUSIONS
    REJECT_RULES = list(DEFAULT_REJECT_RULES)
    ALLOW_READONLY_RULES = list(DEFAULT_ALLOW_READONLY_RULES)
    MUTATION_EXCLUSIONS = re.compile(DEFAULT_MUTATION_EXCLUSIONS_PATTERN, flags=re.IGNORECASE)


def load_rules_config(path: str | Path | None = None) -> bool:
    """Load and register custom rules from a YAML or JSON file.

    If path is None, looks for rules.yaml, .jev/rules.yaml, or rules.json in current directory.
    Returns True if a config was found and loaded, False otherwise.
    """
    global REJECT_RULES, ALLOW_READONLY_RULES, MUTATION_EXCLUSIONS

    target_path: Path | None = None
    if path is not None:
        p = Path(path)
        if p.is_file():
            target_path = p
        else:
            return False
    else:
        candidates = [
            Path("rules.yaml"),
            Path(".jev/rules.yaml"),
            Path("rules.yml"),
            Path(".jev/rules.yml"),
            Path("rules.json"),
            Path(".jev/rules.json"),
        ]
        for c in candidates:
            if c.is_file():
                target_path = c
                break

    if target_path is None:
        return False

    content = target_path.read_text(encoding="utf-8")
    data: dict[str, Any] = {}
    if target_path.suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML is required to parse YAML rules configs.")
        data = yaml.safe_load(content) or {}
    else:
        data = json.loads(content)

    custom_rejects: list[tuple[str, re.Pattern, str]] = []
    for item in data.get("reject_rules", []):
        rule_id = item.get("id", f"custom_reject_{len(custom_rejects)+1}")
        pattern_str = item.get("pattern", "")
        reason = item.get("reason", "Custom security rejection rule.")
        if pattern_str:
            custom_rejects.append((rule_id, re.compile(pattern_str, flags=re.IGNORECASE), reason))

    custom_allows: list[tuple[str, re.Pattern, str]] = []
    for item in data.get("allow_rules", []):
        rule_id = item.get("id", f"custom_allow_{len(custom_allows)+1}")
        pattern_str = item.get("pattern", "")
        reason = item.get("reason", "Custom read-only allow rule.")
        if pattern_str:
            custom_allows.append((rule_id, re.compile(pattern_str, flags=re.IGNORECASE), reason))

    custom_mutations = data.get("mutation_exclusions", [])
    if custom_mutations:
        combined = DEFAULT_MUTATION_EXCLUSIONS_PATTERN + "|" + "|".join(custom_mutations)
        MUTATION_EXCLUSIONS = re.compile(combined, flags=re.IGNORECASE)

    # Custom rules take precedence by being inserted before defaults
    REJECT_RULES = custom_rejects + list(DEFAULT_REJECT_RULES)
    ALLOW_READONLY_RULES = custom_allows + list(DEFAULT_ALLOW_READONLY_RULES)
    return True


def evaluate_rules(context: str) -> RuleMatch | None:
    """Evaluate deterministic pre-flight governance rules.

    Returns:
        RuleMatch if a deterministic rule fired (instant fast-path).
        None if context requires semantic evaluation by Jev ML model.
    """
    clean_context = context.strip()
    if not clean_context:
        return None

    # Step 1: Check Hard Reject rules first
    for rule_id, pattern, reason in REJECT_RULES:
        if pattern.search(clean_context):
            return RuleMatch(
                action="reject",
                needs_review=True,
                prohibited=True,
                risk=4.0,
                rule_id=rule_id,
                reason=reason,
            )

    # Step 2: Check Hard Allow rules
    # Disqualify if any mutating action or external modification keyword is present
    has_mutation = bool(MUTATION_EXCLUSIONS.search(clean_context))
    if not has_mutation:
        for rule_id, pattern, reason in ALLOW_READONLY_RULES:
            if pattern.search(clean_context):
                return RuleMatch(
                    action="execute",
                    needs_review=False,
                    prohibited=False,
                    risk=0.0,
                    rule_id=rule_id,
                    reason=reason,
                )

    # Step 3: No deterministic match -> Fall through to Jev ML Model (Layer 2)
    return None
