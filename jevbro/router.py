"""HTTP router for integrating jev-my-bro with other systems."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from jevbro.questions import default_questions, detect_question_language


class PredictRequest(BaseModel):
    state: str | dict | list
    questions: dict[str, dict]


class DecideRequest(BaseModel):
    context: str = Field(min_length=1, max_length=20000)
    language: str | None = None


def decision_response(agent, request: DecideRequest) -> dict:
    language = request.language or detect_question_language(request.context)
    questions = default_questions(language)
    result = agent.predict({"request": request.context}, questions)
    answers = result["answers"]
    action = answers["action"]
    return {
        "engine": "jev-my-bro",
        "decision": action["choice"],
        "confidence": action["confidence"],
        "needs_review": answers["needs_review"]["noul"],
        "prohibited": answers["prohibited"]["noul"],
        "risk": answers["risk"]["score"],
        "answers": answers,
        "usage": result.get("usage", {}),
    }


def create_router(agent, prefix: str = "/v1/jev-my-bro") -> APIRouter:
    """Create the integration router around an already-loaded model agent."""

    router = APIRouter(prefix=prefix, tags=["jev-my-bro"])

    @router.get("/health")
    def health() -> dict:
        return {"ok": True, "engine": "jev-my-bro", "runtime": "native"}

    @router.post("/predict")
    def predict(request: PredictRequest) -> dict:
        return agent.predict(request.state, request.questions)

    @router.post("/decide")
    def decide(request: DecideRequest) -> dict:
        return decision_response(agent, request)

    return router
