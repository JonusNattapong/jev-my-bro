from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config
from laya.common import build_model
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer


def resolve_model(model_id_or_path: str) -> str:
    if os.path.isdir(model_id_or_path):
        model_dir = model_id_or_path
    else:
        model_dir = snapshot_download(model_id_or_path)
    _fix_tokenizer_config(model_dir)
    return model_dir


def load_trainable(model_id_or_path: str, device: torch.device):
    model_dir = resolve_model(model_id_or_path)
    cfg_path = os.path.join(model_dir, "rl_agent_config.json")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    encoder_dir = os.path.join(model_dir, "encoder")
    model = build_model(cfg, encoder_dir=encoder_dir if os.path.isdir(encoder_dir) else None)
    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    model.to(device)
    return model, tokenizer, cfg, model_dir


def save_checkpoint(model, tokenizer, cfg: dict, output_dir: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().half().contiguous().cpu() for key, value in model.state_dict().items()}
    save_file(state, str(output / "model.safetensors"))
    model.encoder.config.save_pretrained(str(output / "encoder"))
    tokenizer.save_pretrained(str(output / "tokenizer"))
    (output / "rl_agent_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
