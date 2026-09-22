from __future__ import annotations

import argparse
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import laya
import uvicorn
from fastapi import FastAPI

from jevbro.router import (
    DecideRequest,
    PredictRequest,
    create_router,
    create_systemone_router,
    decision_response,
)


def create_app(agent) -> FastAPI:
    app = FastAPI(title="jev-my-bro", version="0.2.0")
    app.include_router(create_router(agent))
    app.include_router(create_systemone_router(agent))

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "engine": "jev-my-bro", "runtime": "native"}

    @app.post("/v1/predict")
    def predict(request: PredictRequest) -> dict:
        return agent.predict(request.state, request.questions)

    @app.post("/v1/decide")
    def decide(request: DecideRequest) -> dict:
        return decision_response(agent, request)

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
