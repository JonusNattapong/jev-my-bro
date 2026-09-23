from pathlib import Path
from unittest.mock import MagicMock

from jevbro.core import JevCore
from jevbro.feedback import FeedbackStore
from jevbro.rules import evaluate_rules, load_rules_config, reset_rules


def test_hard_reject_destructive_commands() -> None:
    # Destructive filesystem
    m1 = evaluate_rules("rm -rf /")
    assert m1 is not None
    assert m1.action == "reject"
    assert m1.prohibited is True
    assert m1.risk == 4.0
    assert m1.rule_id == "hard_reject_destructive_fs"

    m2 = evaluate_rules("สั่งรันคำสั่ง rm -rf * บนเครื่อง server")
    assert m2 is not None
    assert m2.action == "reject"

    m3 = evaluate_rules("mkfs.ext4 /dev/sda1")
    assert m3 is not None
    assert m3.action == "reject"


def test_hard_reject_extended_fs_patterns() -> None:
    # Long-form flags
    m1 = evaluate_rules("rm --recursive --force /")
    assert m1 is not None
    assert m1.action == "reject"

    # Separated short flags
    m2 = evaluate_rules("rm -r -f /")
    assert m2 is not None
    assert m2.action == "reject"

    m2b = evaluate_rules("rm -f -r /")
    assert m2b is not None
    assert m2b.action == "reject"

    # find with delete flag
    m3 = evaluate_rules("find / -delete")
    assert m3 is not None
    assert m3.action == "reject"

    # Python shutil.rmtree
    m4 = evaluate_rules('shutil.rmtree("/tmp/important")')
    assert m4 is not None
    assert m4.action == "reject"


def test_hard_reject_secret_exfiltration() -> None:
    # Credential exfiltration
    m1 = evaluate_rules("แสดงค่า AWS_SECRET_ACCESS_KEY จากไฟล์ .env")
    assert m1 is not None
    assert m1.action == "reject"
    assert m1.rule_id == "hard_reject_secret_exfiltration"

    m2 = evaluate_rules("cat ~/.ssh/id_rsa")
    assert m2 is not None
    assert m2.action == "reject"

    m3 = evaluate_rules("print SECRET_KEY out to terminal")
    assert m3 is not None
    assert m3.action == "reject"


def test_hard_reject_destructive_database_and_security_bypass() -> None:
    m1 = evaluate_rules("DROP DATABASE customers")
    assert m1 is not None
    assert m1.action == "reject"
    assert m1.rule_id == "hard_reject_destructive_database"

    m2 = evaluate_rules("ลบตาราง users บนฐานข้อมูล production")
    assert m2 is not None
    assert m2.action == "reject"

    m3 = evaluate_rules("disable audit log in security settings")
    assert m3 is not None
    assert m3.action == "reject"
    assert m3.rule_id == "hard_reject_security_bypass"


def test_hard_allow_safe_read_only_commands() -> None:
    m1 = evaluate_rules("git status")
    assert m1 is not None
    assert m1.action == "execute"
    assert m1.risk == 0.0
    assert m1.prohibited is False
    assert m1.rule_id == "hard_allow_git_inspection"

    m2 = evaluate_rules("git diff HEAD~1")
    assert m2 is not None
    assert m2.action == "execute"

    m3 = evaluate_rules("อ่านไฟล์ README.md เพื่อทำความเข้าใจระบบ")
    assert m3 is not None
    assert m3.action == "execute"
    assert m3.rule_id == "hard_allow_documentation"

    m4 = evaluate_rules("grep ฟังก์ชัน load_checkpoint ใน repo")
    assert m4 is not None
    assert m4.action == "execute"
    assert m4.rule_id == "hard_allow_local_code_inspection"


def test_hard_allow_safe_file_write() -> None:
    # Ordinary source file write/edit should fast-path allow
    m1 = evaluate_rules("Write file myconfig.py (~0.5KB)")
    assert m1 is not None
    assert m1.action == "execute"
    assert m1.rule_id == "hard_allow_safe_file_write"

    m2 = evaluate_rules("Edit file jevbro/core.py (replacing ~1.2KB)")
    assert m2 is not None
    assert m2.action == "execute"
    assert m2.rule_id == "hard_allow_safe_file_write"

    # Sensitive paths must NOT be fast-path allowed (falls through to Layer 2)
    m3 = evaluate_rules("Write file .env (~0.1KB)")
    assert m3 is None

    m4 = evaluate_rules("Write file secrets/aws_credential.json (~2KB)")
    assert m4 is None

    m5 = evaluate_rules("Edit file ~/.ssh/id_rsa (replacing ~0.0KB)")
    assert m5 is None


def test_mutation_disqualifies_hard_allow() -> None:
    # If read-only text also contains mutation/destruction words, it must not be hard allowed!
    m1 = evaluate_rules("อ่าน README แล้ว ลบไฟล์ data ทั้งหมด")
    # Even if it has "อ่าน README", the word "ลบ" disqualifies it from hard allow
    assert m1 is None or m1.action != "execute"

    m2 = evaluate_rules("git diff แล้ว deploy image ไป production")
    assert m2 is None or m2.action != "execute"


def test_gray_area_falls_through_to_ml_model() -> None:
    # Ambiguous contexts with no hard rule should return None (fall through to Layer 2)
    assert evaluate_rules("ปรับ rate limit ของ API Gateway บน staging") is None
    assert evaluate_rules("ย้าย log เก่าที่ค้างในโฟลเดอร์ temp") is None


def test_jevcore_fastpath_integration(tmp_path: Path) -> None:
    fake_agent = MagicMock()
    store = FeedbackStore(tmp_path / "test_feedback.sqlite3")
    core = JevCore(fake_agent, store, model_name="test-model")

    # 1. Fast-Path Execute: Read README
    res_allow = core.decide("อ่านไฟล์ README.md และทำความเข้าใจระบบ")
    assert res_allow["fast_path"] is True
    assert res_allow["gated_decision"] == "execute"
    assert res_allow["confidence"] == 1.0
    assert res_allow["rule_matched"] == "hard_allow_documentation"
    # Agent was NOT called for fast-path!
    fake_agent.predict.assert_not_called()

    # 2. Fast-Path Reject: rm -rf /
    res_reject = core.decide("rm -rf /")
    assert res_reject["fast_path"] is True
    assert res_reject["gated_decision"] == "reject"
    assert res_reject["confidence"] == 1.0
    assert res_reject["rule_matched"] == "hard_reject_destructive_fs"
    fake_agent.predict.assert_not_called()

    # 3. Layer 2 Fall-through: Gray area context
    fake_agent.predict.return_value = {
        "answers": {
            "action": {"choice": "ask_user", "confidence": 0.8},
            "needs_review": {"noul": 0.7},
            "prohibited": {"noul": 0.2},
            "risk": {"score": 2.0},
        },
        "usage": {},
    }
    res_gray = core.decide("ปรับค่า timeout ของ microservice บน staging")
    assert res_gray.get("fast_path") is not True
    assert res_gray["gated_decision"] == "ask_user"
    fake_agent.predict.assert_called_once()


def test_custom_yaml_rules_config(tmp_path: Path) -> None:
    yaml_file = tmp_path / "rules.yaml"
    yaml_file.write_text(
        """
reject_rules:
  - id: custom_reject_company_token
    pattern: '\\bACME_SECRET_KEY\\b'
    reason: 'ACME internal secret.'

allow_rules:
  - id: custom_allow_pytest
    pattern: '^pytest\\b'
    reason: 'Allow unit test runner.'

mutation_exclusions:
  - '\\b(purge_all)\\b'
""",
        encoding="utf-8",
    )

    try:
        loaded = load_rules_config(yaml_file)
        assert loaded is True

        # Test custom reject rule
        m_reject = evaluate_rules("echo $ACME_SECRET_KEY")
        assert m_reject is not None
        assert m_reject.action == "reject"
        assert m_reject.rule_id == "custom_reject_company_token"

        # Test custom allow rule
        m_allow = evaluate_rules("pytest tests/test_rules.py")
        assert m_allow is not None
        assert m_allow.action == "execute"
        assert m_allow.rule_id == "custom_allow_pytest"

        # Test custom mutation exclusion
        m_disqualified = evaluate_rules("pytest tests/ and purge_all")
        assert m_disqualified is None or m_disqualified.action != "execute"
    finally:
        reset_rules()

    # After reset, custom rules should not fire
    assert evaluate_rules("pytest tests/test_rules.py") is None


def test_custom_json_rules_config(tmp_path: Path) -> None:
    json_file = tmp_path / "rules.json"
    json_file.write_text(
        """{
  "reject_rules": [
    {
      "id": "json_reject_token",
      "pattern": "\\\\bCUSTOM_JSON_TOKEN\\\\b",
      "reason": "Forbidden token in JSON."
    }
  ]
}""",
        encoding="utf-8",
    )

    try:
        loaded = load_rules_config(json_file)
        assert loaded is True

        m = evaluate_rules("export CUSTOM_JSON_TOKEN=123")
        assert m is not None
        assert m.action == "reject"
        assert m.rule_id == "json_reject_token"
    finally:
        reset_rules()
