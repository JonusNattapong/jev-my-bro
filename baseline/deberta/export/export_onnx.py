from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class LogitsWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()
    wrapper = LogitsWrapper(model)
    sample = tokenizer(
        "Context:\nAgent wants to inspect git status\n\nQuestion:\nWhat should the agent do?\n\nOptions:\nexecute | ask_user | reject",
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    program = torch.onnx.export(
        wrapper,
        (sample["input_ids"], sample["attention_mask"]),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamo=True,
    )
    program.save(str(output))
    print(f"exported {output}")


if __name__ == "__main__":
    main()
