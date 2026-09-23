"""Active learning harvester for identifying hard negatives and edge cases from feedback."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jevbro.feedback import FeedbackStore


@dataclass
class HarvestCandidate:
    task_id: str
    request: str
    domain: str
    discrepancy_type: str  # "false_positive", "false_negative", "disagreement", "high_uncertainty"
    model_raw_decision: str | None
    model_gated_decision: str | None
    model_prohibited: float | None
    model_needs_review: float | None
    model_risk: float | None
    model_abstain: bool
    agent_choice: str | None
    status: str
    tests_passed: bool | None
    task_success: bool | None
    suggested_action: str
    suggested_needs_review: str  # "true" | "false"
    suggested_prohibited: str  # "true" | "false"
    suggested_risk: int  # 0 to 4
    reason: str


def infer_domain(text: str) -> str:
    """Heuristic domain classifier from text keywords."""
    lowered = text.lower()
    if any(k in lowered for k in ("git", "commit", "branch", "diff", "merge", "revert")):
        return "git"
    if any(k in lowered for k in ("test", "pytest", "unit", "spec", "flaky")):
        return "testing"
    if any(k in lowered for k in ("sql", "database", "migration", "table", "schema", "drop")):
        return "database"
    if any(k in lowered for k in ("deploy", "cluster", "k8s", "docker", "container", "pipeline")):
        return "deployment"
    if any(k in lowered for k in ("api", "gateway", "endpoint", "route", "http", "curl")):
        return "api"
    if any(k in lowered for k in ("rm -rf", "delete", "format", "server", "linux", "os", "disk")):
        return "os"
    if any(k in lowered for k in ("npm", "pip", "package", "dependency", "gem", "yarn")):
        return "dependency"
    if any(k in lowered for k in ("readme", "doc", "document", "markdown", "guide")):
        return "docs"
    if any(k in lowered for k in ("secret", "token", "key", "password", "auth", "credential")):
        return "security"
    return "general"


def find_existing_requests(data_roots: list[str | Path]) -> set[str]:
    """Scan dataset directories for existing request strings to prevent duplicates."""
    seen: set[str] = set()
    for root in data_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        # Read JSONL files
        for jsonl_file in root_path.glob("**/*.jsonl"):
            try:
                with jsonl_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        req = data.get("state", {}).get("request") or data.get("request")
                        if req:
                            seen.add(req.strip())
            except Exception:
                pass
        # Read CSV files
        for csv_file in root_path.glob("**/*.csv"):
            try:
                with csv_file.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        req = row.get("request")
                        if req:
                            seen.add(req.strip())
            except Exception:
                pass
    return seen


class ActiveLearningHarvester:
    """Extracts hard negatives and edge cases from Jev task feedback."""

    def __init__(self, store: FeedbackStore) -> None:
        self.store = store

    def harvest(
        self,
        *,
        existing_requests: set[str] | None = None,
        min_context_len: int = 15,
        include_uncertain: bool = True,
    ) -> list[HarvestCandidate]:
        sessions = self.store.sessions(status=None, include_decisions=True)
        candidates: list[HarvestCandidate] = []
        known_requests = existing_requests or set()
        seen_in_batch: set[str] = set()

        for session in sessions:
            status = session.get("status")
            if status == "active":
                continue

            request_text = (session.get("task") or "").strip()
            if len(request_text) < min_context_len:
                continue
            if request_text in known_requests or request_text in seen_in_batch:
                continue

            decisions = session.get("decisions", [])
            if not decisions:
                continue

            # Pick the primary decision (last or initial)
            primary_dec = decisions[-1]
            tests_passed = session.get("tests_passed")
            task_success = session.get("task_success")
            final_choice = session.get("final_choice")

            raw_decision = primary_dec.get("raw_decision")
            response = primary_dec.get("response", {})
            gated_decision = response.get("gated_decision") or primary_dec.get("promoted_decision")
            prohibited = primary_dec.get("prohibited")
            needs_review = primary_dec.get("needs_review")
            risk = primary_dec.get("risk")
            abstain = bool(primary_dec.get("abstain"))

            p_val = float(prohibited) if prohibited is not None else 0.0
            r_val = float(needs_review) if needs_review is not None else 0.0
            risk_val = float(risk) if risk is not None else 1.0

            domain = infer_domain(request_text)

            candidate: HarvestCandidate | None = None

            # Pattern 1: False Positive (Over-conservative)
            # Model gated to reject or high prohibited, but task completed successfully with tests passing!
            if (
                status == "completed"
                and tests_passed is True
                and (gated_decision == "reject" or p_val >= 0.50)
            ):
                # Safe operational work that the model overly blocked
                suggested_risk = min(int(round(risk_val)), 2)
                candidate = HarvestCandidate(
                    task_id=session["task_id"],
                    request=request_text,
                    domain=domain,
                    discrepancy_type="false_positive",
                    model_raw_decision=raw_decision,
                    model_gated_decision=gated_decision,
                    model_prohibited=p_val,
                    model_needs_review=r_val,
                    model_risk=risk_val,
                    model_abstain=abstain,
                    agent_choice=final_choice,
                    status=status,
                    tests_passed=tests_passed,
                    task_success=task_success,
                    suggested_action="execute",
                    suggested_needs_review="false",
                    suggested_prohibited="false",
                    suggested_risk=suggested_risk,
                    reason=f"Model gated to '{gated_decision}' (prohibited: {p_val:.2f}) but task succeeded with passing tests.",
                )

            # Pattern 2: False Negative (Under-conservative)
            # Model advised execute / low risk, but task failed or broke tests!
            elif (
                (status == "failed" or tests_passed is False or task_success is False)
                and (raw_decision == "execute" or gated_decision == "execute")
                and p_val < 0.50
            ):
                suggested_risk = max(int(round(risk_val)), 2)
                candidate = HarvestCandidate(
                    task_id=session["task_id"],
                    request=request_text,
                    domain=domain,
                    discrepancy_type="false_negative",
                    model_raw_decision=raw_decision,
                    model_gated_decision=gated_decision,
                    model_prohibited=p_val,
                    model_needs_review=r_val,
                    model_risk=risk_val,
                    model_abstain=abstain,
                    agent_choice=final_choice,
                    status=status,
                    tests_passed=tests_passed,
                    task_success=task_success,
                    suggested_action="ask_user",
                    suggested_needs_review="true",
                    suggested_prohibited="false",
                    suggested_risk=suggested_risk,
                    reason=f"Model suggested '{gated_decision}' but task failed or tests did not pass.",
                )

            # Pattern 3: Disagreement
            # Agent executed directly while model asked for user review, and task succeeded cleanly
            elif (
                status == "completed"
                and tests_passed is True
                and gated_decision == "ask_user"
                and final_choice in ("execute", "minimal_patch", "routine")
                and p_val < 0.40
            ):
                candidate = HarvestCandidate(
                    task_id=session["task_id"],
                    request=request_text,
                    domain=domain,
                    discrepancy_type="disagreement",
                    model_raw_decision=raw_decision,
                    model_gated_decision=gated_decision,
                    model_prohibited=p_val,
                    model_needs_review=r_val,
                    model_risk=risk_val,
                    model_abstain=abstain,
                    agent_choice=final_choice,
                    status=status,
                    tests_passed=tests_passed,
                    task_success=task_success,
                    suggested_action="execute",
                    suggested_needs_review="false",
                    suggested_prohibited="false",
                    suggested_risk=min(int(round(risk_val)), 1),
                    reason="Model requested review for routine work that agent executed cleanly with tests passing.",
                )

            # Pattern 4: High Uncertainty (Abstain)
            elif include_uncertain and abstain and (status in ("completed", "failed")):
                is_safe = (status == "completed" and tests_passed is True)
                candidate = HarvestCandidate(
                    task_id=session["task_id"],
                    request=request_text,
                    domain=domain,
                    discrepancy_type="high_uncertainty",
                    model_raw_decision=raw_decision,
                    model_gated_decision=gated_decision,
                    model_prohibited=p_val,
                    model_needs_review=r_val,
                    model_risk=risk_val,
                    model_abstain=abstain,
                    agent_choice=final_choice,
                    status=status,
                    tests_passed=tests_passed,
                    task_success=task_success,
                    suggested_action="execute" if is_safe else "ask_user",
                    suggested_needs_review="false" if is_safe else "true",
                    suggested_prohibited="false",
                    suggested_risk=min(int(round(risk_val)), 2) if is_safe else max(int(round(risk_val)), 2),
                    reason=f"Model abstained (confidence below threshold); real outcome was {'success' if is_safe else 'failure'}.",
                )

            if candidate is not None:
                candidates.append(candidate)
                seen_in_batch.add(request_text)

        return candidates

    @staticmethod
    def to_dataset_rows(
        candidates: list[HarvestCandidate],
        *,
        start_id: int = 1,
        prefix: str = "th-act-",
        split: str = "train",
        writer_id: str = "active_learning_harvester",
    ) -> list[dict[str, Any]]:
        """Format harvest candidates into the standard curated dataset CSV rows."""
        rows: list[dict[str, Any]] = []
        for i, cand in enumerate(candidates, start=start_id):
            case_id = f"{prefix}{i:03d}"
            scenario_family = f"active_{cand.discrepancy_type}_{cand.domain}_{i:03d}"
            rows.append({
                "id": case_id,
                "split": split,
                "domain": cand.domain,
                "request": cand.request,
                "action": cand.suggested_action,
                "needs_review": cand.suggested_needs_review,
                "prohibited": cand.suggested_prohibited,
                "risk": str(cand.suggested_risk),
                "scenario_family": scenario_family,
                "writer_id": writer_id,
                "discrepancy_type": cand.discrepancy_type,
                "reason": cand.reason,
            })
        return rows

    @classmethod
    def export_csv(
        cls,
        candidates: list[HarvestCandidate],
        output_file: str | Path,
        *,
        start_id: int = 1,
        prefix: str = "th-act-",
        split: str = "train",
        writer_id: str = "active_learning_harvester",
    ) -> int:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = cls.to_dataset_rows(
            candidates,
            start_id=start_id,
            prefix=prefix,
            split=split,
            writer_id=writer_id,
        )
        if not rows:
            return 0

        fieldnames = [
            "id",
            "split",
            "domain",
            "request",
            "action",
            "needs_review",
            "prohibited",
            "risk",
            "scenario_family",
            "writer_id",
            "discrepancy_type",
            "reason",
        ]
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        return len(rows)
