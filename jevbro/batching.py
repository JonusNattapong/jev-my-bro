from __future__ import annotations

import torch


def collate_items(items: list[dict], pad_id: int) -> dict:
    count = len(items)
    max_length = max(len(item["ids"]) for item in items)
    max_options = max(len(item["markers"]) for item in items)
    input_ids = torch.full((count, max_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((count, max_length), dtype=torch.long)
    marker_pos = torch.zeros((count, max_options), dtype=torch.long)
    marker_mask = torch.zeros((count, max_options), dtype=torch.bool)
    target = torch.zeros((count, max_options), dtype=torch.float32)

    for index, item in enumerate(items):
        ids = torch.tensor(item["ids"], dtype=torch.long)
        input_ids[index, : len(ids)] = ids
        attention_mask[index, : len(ids)] = 1
        option_count = len(item["markers"])
        marker_pos[index, :option_count] = torch.tensor(item["markers"], dtype=torch.long)
        marker_mask[index, :option_count] = True
        target[index, : len(item["target"])] = torch.tensor(item["target"], dtype=torch.float32)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "target": target,
        "qtype": torch.tensor([item["qtype"] for item in items], dtype=torch.long),
        "label": torch.tensor([item["label"] for item in items], dtype=torch.long),
    }
