from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from training.common import read_jsonl, to_hf_dataset


def collect_logits(model_path: str, data_path: str, *, max_length: int = 256, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    dataset = to_hf_dataset(read_jsonl(data_path))

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format(type="torch")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer))

    logits_parts = []
    label_parts = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").cpu().numpy()
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits.detach().cpu().numpy()
            logits_parts.append(logits)
            label_parts.append(labels)

    return np.concatenate(logits_parts), np.concatenate(label_parts)
