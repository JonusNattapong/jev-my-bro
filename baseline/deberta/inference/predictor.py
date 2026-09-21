from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from training import ID2LABEL
from training.common import build_text, softmax_np


class DecisionPredictor:
    def __init__(
        self,
        *,
        model_dir: str,
        onnx_path: str,
        calibration_path: str | None = None,
        max_length: int = 256,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.max_length = max_length
        self.temperature = 1.0
        if calibration_path:
            payload = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
            self.temperature = float(payload["temperature"])

    def decide(self, context: str) -> dict:
        row = {
            "context": context,
            "question": "What should the agent do?",
            "options": ["execute", "ask_user", "reject"],
        }
        encoded = self.tokenizer(
            build_text(row),
            return_tensors="np",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        logits = self.session.run(["logits"], inputs)[0]
        probabilities = softmax_np(logits / self.temperature)[0]
        values = {ID2LABEL[idx]: float(probabilities[idx]) for idx in range(len(probabilities))}
        best = int(np.argmax(probabilities))
        return {
            "decision": ID2LABEL[best],
            "confidence": float(probabilities[best]),
            "probabilities": values,
        }
