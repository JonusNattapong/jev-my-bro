from __future__ import annotations

import argparse

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from inference.predictor import DecisionPredictor


class DecideRequest(BaseModel):
    context: str = Field(min_length=1, max_length=12000)


class DecideResponse(BaseModel):
    decision: str
    confidence: float
    probabilities: dict[str, float]


def create_app(predictor: DecisionPredictor) -> FastAPI:
    app = FastAPI(title="jev-my-bro", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.post("/v1/decide", response_model=DecideResponse)
    def decide(request: DecideRequest) -> dict:
        return predictor.decide(request.context)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = DecisionPredictor(
        model_dir=args.model,
        onnx_path=args.onnx,
        calibration_path=args.calibration,
    )
    uvicorn.run(create_app(predictor), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
