"""HTTP router for integrating jev-my-bro with other systems."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from jevbro.questions import default_questions, detect_question_language


class PredictRequest(BaseModel):
    state: str | dict | list
    questions: dict[str, dict]


class SystemOneRequest(BaseModel):
    """TypeSafe-compatible request envelope for Jev-class clients."""

    state: str | dict | list
    model: str = Field(min_length=1)
    questions: dict[str, dict[str, Any]]

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        supported = {"jev-my-bro", "jev-my-bro-latest", "jev-latest"}
        if value not in supported:
            raise ValueError(f"unsupported model {value!r}; use one of {sorted(supported)}")
        return value

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        if not questions:
            raise ValueError("questions must contain at least one question")

        for question_id, question in questions.items():
            if not question_id:
                raise ValueError("question ids must not be empty")
            qtype = question.get("type")
            if qtype not in {"choice", "score", "noul"}:
                raise ValueError(f"question {question_id!r} has unsupported type {qtype!r}")
            if "instructions" not in question or not isinstance(question["instructions"], (str, dict, list)):
                raise ValueError(
                    f"question {question_id!r} instructions must be a string, object, or array"
                )

            criteria = question.get("criteria")
            if qtype == "choice":
                if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                    raise ValueError(f"choice question {question_id!r} requires 2 to 255 criteria")
            elif qtype == "score":
                if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                    raise ValueError(f"score question {question_id!r} requires 2 to 10 ordered levels")
            elif criteria is not None:
                if not isinstance(criteria, dict) or not set(criteria).issubset({"true", "false"}):
                    raise ValueError(
                        f"noul question {question_id!r} criteria may only define true and false"
                    )

        return questions


class DecideRequest(BaseModel):
    context: str = Field(min_length=1, max_length=20000)
    language: str | None = None
    prohibited_threshold: float | None = None
    review_threshold: float | None = None


REVIEW_THRESHOLD = 0.5
PROHIBITED_REJECT_THRESHOLD = 0.7
HIGH_RISK_THRESHOLD = 3.0


def gate_decision(
    action: str,
    needs_review: float,
    prohibited: float,
    *,
    risk: float | None = None,
    review_threshold: float = REVIEW_THRESHOLD,
    prohibited_threshold: float = PROHIBITED_REJECT_THRESHOLD,
    high_risk_threshold: float = HIGH_RISK_THRESHOLD,
) -> str:
    """Combine raw signals while separating review uncertainty from prohibition.

    The raw action head is trained to agree with the noul heads, but nothing enforces that
    at inference time. A prohibited score from review_threshold to prohibited_threshold is
    treated as uncertainty and escalated for review; only a calibrated high prohibited score
    (prohibited >= prohibited_threshold) or an explicit reject action is an automatic reject.
    High operational risk (risk >= high_risk_threshold) is also escalated to ask_user.
    """
    if prohibited >= prohibited_threshold or action == "reject":
        return "reject"
    if (
        needs_review >= review_threshold
        or action == "ask_user"
        or prohibited >= review_threshold
        or (risk is not None and risk >= high_risk_threshold)
    ):
        return "ask_user"
    return "execute"


def decision_response(
    agent,
    request: DecideRequest,
    *,
    review_threshold: float | None = None,
    prohibited_threshold: float | None = None,
    high_risk_threshold: float = HIGH_RISK_THRESHOLD,
) -> dict:
    language = request.language or detect_question_language(request.context)
    questions = default_questions(language)
    result = agent.predict({"request": request.context}, questions)
    answers = result["answers"]
    action = answers["action"]
    risk_score = float(answers["risk"]["score"])
    r_thresh = review_threshold if review_threshold is not None else (request.review_threshold or REVIEW_THRESHOLD)
    p_thresh = prohibited_threshold if prohibited_threshold is not None else (request.prohibited_threshold or PROHIBITED_REJECT_THRESHOLD)
    return {
        "engine": "jev-my-bro",
        "decision": action["choice"],
        "confidence": action["confidence"],
        "needs_review": answers["needs_review"]["noul"],
        "prohibited": answers["prohibited"]["noul"],
        "risk": answers["risk"]["score"],
        "gated_decision": gate_decision(
            action["choice"],
            float(answers["needs_review"]["noul"]),
            float(answers["prohibited"]["noul"]),
            risk=risk_score,
            review_threshold=r_thresh,
            prohibited_threshold=p_thresh,
            high_risk_threshold=high_risk_threshold,
        ),
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


def create_systemone_router(agent) -> APIRouter:
    """Expose the published TypeSafe wire format for compatible clients and eval harnesses."""

    router = APIRouter(tags=["system-one"])

    @router.post("/v1/systemone")
    def system_one(request: SystemOneRequest) -> dict:
        try:
            result = agent.predict(request.state, request.questions)
        except ValueError as exc:
            # Laya can reject a rubric whose rendered options exceed the model's
            # configured question-head token budget. Surface this as bad input.
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
            raise HTTPException(status_code=502, detail="model returned an invalid answer envelope")
        return {
            "model": "jev-my-bro",
            "answers": result["answers"],
            "usage": result.get("usage", {"input_tokens": 0, "output_tokens": 0}),
        }

    return router
