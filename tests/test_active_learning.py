import csv
from pathlib import Path

from jevbro.active_learning import ActiveLearningHarvester, infer_domain
from jevbro.feedback import FeedbackStore


def test_infer_domain() -> None:
    assert infer_domain("อ่านไฟล์ README.md ของโปรเจกต์") == "docs"
    assert infer_domain("รัน pytest -q ใน virtualenv") == "testing"
    assert infer_domain("สร้าง branch feat/active-learning และ commit") == "git"
    assert infer_domain("รัน migration ตาราง users บน database") == "database"
    assert infer_domain("deploy image container ไปที่ k8s cluster") == "deployment"
    assert infer_domain("rm -rf /var/log บน server linux") == "os"
    assert infer_domain("ซ่อน secret key และ token") == "security"


def test_harvest_detects_false_positive_and_false_negative(tmp_path: Path) -> None:
    db_file = tmp_path / "test_feedback.sqlite3"
    store = FeedbackStore(db_file)

    # 1. False Positive: Model rejected routine work, but agent completed it with passing tests
    fp_session = store.start_session(
        source_agent="antigravity",
        task="อ่านไฟล์ README.md เพื่อตรวจสอบวิธีใช้งานโปรเจกต์",
        task_id="task-fp-1",
        feedback_id="task-fp-1",
    )
    store.record_decision(
        decision_id="dec-fp-1",
        feedback_id="task-fp-1",
        task_id="task-fp-1",
        source_agent="antigravity",
        context="อ่านไฟล์ README.md เพื่อตรวจสอบวิธีใช้งานโปรเจกต์",
        language="th",
        model_name="test-model",
        abstain_threshold=0.6,
        result={
            "raw_decision": "ask_user",
            "gated_decision": "reject",
            "prohibited": 0.72,
            "needs_review": 0.65,
            "risk": 1.2,
            "abstain": True,
            "confidence": 0.45,
        },
    )
    store.complete_session(
        "task-fp-1",
        final_choice="execute",
        tests_passed=True,
        task_success=True,
    )

    # 2. False Negative: Model said execute, but task failed
    fn_session = store.start_session(
        source_agent="antigravity",
        task="แก้ไขการตั้งค่า firewall และ network port ของ production",
        task_id="task-fn-1",
        feedback_id="task-fn-1",
    )
    store.record_decision(
        decision_id="dec-fn-1",
        feedback_id="task-fn-1",
        task_id="task-fn-1",
        source_agent="antigravity",
        context="แก้ไขการตั้งค่า firewall และ network port ของ production",
        language="th",
        model_name="test-model",
        abstain_threshold=0.6,
        result={
            "raw_decision": "execute",
            "gated_decision": "execute",
            "prohibited": 0.15,
            "needs_review": 0.20,
            "risk": 1.0,
            "abstain": False,
            "confidence": 0.85,
        },
    )
    store.fail_session(
        "task-fn-1",
        failure_reason="Network disrupted, rollback required",
        final_choice="execute",
        tests_passed=False,
    )

    harvester = ActiveLearningHarvester(store)
    candidates = harvester.harvest(min_context_len=10)

    assert len(candidates) == 2

    # Check FP
    fp = next(c for c in candidates if c.task_id == "task-fp-1")
    assert fp.discrepancy_type == "false_positive"
    assert fp.suggested_action == "execute"
    assert fp.suggested_prohibited == "false"
    assert fp.suggested_needs_review == "false"

    # Check FN
    fn = next(c for c in candidates if c.task_id == "task-fn-1")
    assert fn.discrepancy_type == "false_negative"
    assert fn.suggested_action == "ask_user"
    assert fn.suggested_needs_review == "true"


def test_harvest_deduplication_and_export(tmp_path: Path) -> None:
    db_file = tmp_path / "test_feedback.sqlite3"
    store = FeedbackStore(db_file)

    task_text = "รัน unit test สำหรับโมดูล auth ใน dev environment"
    store.start_session(
        source_agent="antigravity",
        task=task_text,
        task_id="task-dedupe-1",
        feedback_id="task-dedupe-1",
    )
    store.record_decision(
        decision_id="dec-dedupe-1",
        feedback_id="task-dedupe-1",
        task_id="task-dedupe-1",
        source_agent="antigravity",
        context=task_text,
        language="th",
        model_name="test-model",
        abstain_threshold=0.6,
        result={
            "raw_decision": "ask_user",
            "gated_decision": "reject",
            "prohibited": 0.8,
            "needs_review": 0.5,
            "risk": 1.0,
            "abstain": False,
            "confidence": 0.75,
        },
    )
    store.complete_session(
        "task-dedupe-1",
        final_choice="execute",
        tests_passed=True,
    )

    harvester = ActiveLearningHarvester(store)

    # When deduplication excludes this task
    filtered = harvester.harvest(existing_requests={task_text})
    assert len(filtered) == 0

    # When deduplication does not exclude this task
    candidates = harvester.harvest(existing_requests=set())
    assert len(candidates) == 1

    # Export to CSV
    csv_file = tmp_path / "output.csv"
    count = harvester.export_csv(candidates, csv_file, prefix="th-act-")
    assert count == 1
    assert csv_file.exists()

    with csv_file.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert reader[0]["id"] == "th-act-001"
        assert reader[0]["request"] == task_text
        assert reader[0]["action"] == "execute"
        assert reader[0]["discrepancy_type"] == "false_positive"
