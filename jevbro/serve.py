from __future__ import annotations

import argparse

import laya
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from jevbro.questions import default_questions, detect_question_language


class PredictRequest(BaseModel):
    state: str | dict | list
    questions: dict[str, dict]


class DecideRequest(BaseModel):
    context: str = Field(min_length=1, max_length=20000)
    language: str | None = None


def create_app(agent) -> FastAPI:
    app = FastAPI(title="jev-my-bro", version="0.2.0")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "engine": "laya"}

    @app.post("/v1/predict")
    def predict(request: PredictRequest) -> dict:
        return agent.predict(request.state, request.questions)

    @app.post("/v1/decide")
    def decide(request: DecideRequest) -> dict:
        language = request.language or detect_question_language(request.context)
        questions = default_questions(language)
        result = agent.predict({"request": request.context}, questions)
        answers = result["answers"]
        action = answers["action"]
        return {
            "decision": action["choice"],
            "confidence": action["confidence"],
            "needs_review": answers["needs_review"]["noul"],
            "prohibited": answers["prohibited"]["noul"],
            "risk": answers["risk"]["score"],
            "answers": answers,
            "usage": result.get("usage", {}),
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/laya-model")
    parser.add_argument("--device")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = laya.Agent(args.model, device=args.device)
    uvicorn.run(create_app(agent), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
