from fastapi.testclient import TestClient

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
