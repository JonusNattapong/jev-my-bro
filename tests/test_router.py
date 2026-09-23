from fastapi.testclient import TestClient

from jevbro.router import gate_decision
from jevbro.serve import create_app


class FakeAgent:
    def predict(self, state, questions):
        return {
            "answers": {
                "action": {"choice": "ask_user", "confidence": 0.8},
                "needs_review": {"noul": 0.9},
                "prohibited": {"noul": 0.1},
                "risk": {"score": 3.0},
            },
            "usage": {"input_tokens": 10},
        }


def test_jev_my_bro_router_decide_and_health() -> None:
    client = TestClient(create_app(FakeAgent()))

    health = client.get("/v1/jev-my-bro/health")
    assert health.status_code == 200
    assert health.json() == {"ok": True, "engine": "jev-my-bro", "runtime": "native"}

    response = client.post(
        "/v1/jev-my-bro/decide",
        json={"context": "Deploy the service to production"},
    )
    assert response.status_code == 200
    assert response.json()["engine"] == "jev-my-bro"
    assert response.json()["decision"] == "ask_user"


def test_legacy_decide_route_remains_compatible() -> None:
    client = TestClient(create_app(FakeAgent()))
    response = client.post("/v1/decide", json={"context": "Review this change"})
    assert response.status_code == 200
    assert response.json()["decision"] == "ask_user"


def test_typesafe_systemone_wire_format() -> None:
    client = TestClient(create_app(FakeAgent()))
    response = client.post(
        "/v1/systemone",
        json={
            "model": "jev-my-bro",
            "state": {"request": "Deploy the service to production"},
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": "What should the agent do?",
                    "criteria": {
                        "execute": "Proceed now",
                        "ask_user": "Request approval",
                        "reject": "Do not proceed",
                    },
                }
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "model": "jev-my-bro",
        "answers": FakeAgent().predict(None, None)["answers"],
        "usage": {"input_tokens": 10},
    }


def test_typesafe_systemone_validates_score_levels_and_model() -> None:
    client = TestClient(create_app(FakeAgent()))
    response = client.post(
        "/v1/systemone",
        json={
            "model": "unknown-model",
            "state": "state",
            "questions": {
                "risk": {
                    "type": "score",
                    "instructions": "How risky?",
                    "criteria": ["only one level"],
                }
            },
        },
    )

    assert response.status_code == 422


def test_typesafe_systemone_maps_laya_rubric_budget_error_to_422() -> None:
    class RejectingAgent:
        def predict(self, state, questions):
            raise ValueError("question 'action' options exceed head_max_len=256")

    client = TestClient(create_app(RejectingAgent()))
    response = client.post(
        "/v1/systemone",
        json={
            "model": "jev-latest",
            "state": "state",
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": "Choose",
                    "criteria": {"a": "A", "b": "B"},
                }
            },
        },
    )

    assert response.status_code == 422
    assert "head_max_len" in response.json()["detail"]


def test_gate_decision_prioritizes_prohibited_over_action() -> None:
    assert gate_decision("execute", needs_review=0.1, prohibited=0.9) == "reject"
    assert gate_decision("ask_user", needs_review=0.9, prohibited=0.5) == "reject"
    assert gate_decision("reject", needs_review=0.0, prohibited=0.0) == "reject"


def test_gate_decision_escalates_review_and_passes_clean_execute() -> None:
    assert gate_decision("execute", needs_review=0.7, prohibited=0.2) == "ask_user"
    assert gate_decision("ask_user", needs_review=0.1, prohibited=0.1) == "ask_user"
    assert gate_decision("execute", needs_review=0.49, prohibited=0.49) == "execute"


def test_decide_reports_gated_decision_without_changing_raw_decision() -> None:
    class ConflictingAgent:
        def predict(self, state, questions):
            return {
                "answers": {
                    "action": {"choice": "execute", "confidence": 0.95},
                    "needs_review": {"noul": 0.2},
                    "prohibited": {"noul": 0.9},
                    "risk": {"score": 4.0},
                },
                "usage": {},
            }

    client = TestClient(create_app(ConflictingAgent()))
    body = client.post("/v1/jev-my-bro/decide", json={"context": "Disable audit logging"}).json()
    assert body["decision"] == "execute"
    assert body["gated_decision"] == "reject"
