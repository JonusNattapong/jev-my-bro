from __future__ import annotations

import argparse
import json
import math
import random
import time

import numpy as np
import torch
from laya.common import proper_reward

from jevbro.batching import collate_items
from jevbro.checkpoint import load_trainable, save_checkpoint
from jevbro.data import load_items

DEFAULT_BASE = "convaiinnovations/laya-multilingual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Laya with RLCD on jev-my-bro data")
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--validation", default="data/validation.jsonl")
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--output", default="artifacts/laya-model")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=2.5e-5)
    parser.add_argument("--head-lr", type=float, default=1.0e-4)
    parser.add_argument("--sigma-start", type=float, default=0.4)
    parser.add_argument("--sigma-end", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validation_metrics(model, items: list[dict], tokenizer, device: torch.device, batch_size: int = 16) -> dict:
    model.eval()
    correct = 0
    total = 0
    nll_sum = 0.0
    for start in range(0, len(items), batch_size):
        batch = collate_items(items[start : start + batch_size], tokenizer.pad_token_id)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        marker_pos = batch["marker_pos"].to(device)
        marker_mask = batch["marker_mask"].to(device)
        qtype = batch["qtype"].to(device)
        target = batch["target"].to(device)
        labels = batch["label"].to(device)
        logits, _ = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        logits = logits.float().masked_fill(~marker_mask, -1e4)
        log_probs = torch.log_softmax(logits, -1)
        nll_sum += float((-(target * log_probs).sum(-1)).sum().item())
        correct += int((logits.argmax(-1) == labels).sum().item())
        total += len(labels)
    model.train()
    return {"accuracy": correct / max(1, total), "soft_nll": nll_sum / max(1, total)}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for RLCD fine-tuning. Use the Colab notebook with GPU enabled.")

    seed_everything(args.seed)
    device = torch.device("cuda")
    model, tokenizer, cfg, resolved_base = load_trainable(args.base_model, device)

    cfg["max_len"] = int(cfg.get("max_len", 1024))
    cfg["head_max_len"] = int(cfg.get("head_max_len", 256))
    cfg["gradient_checkpointing"] = True

    try:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except (AttributeError, TypeError):
        try:
            model.encoder.gradient_checkpointing_enable()
        except AttributeError:
            pass
    model.head_checkpointing = True
    model.train()

    train_items = load_items(tokenizer, cfg, args.train)
    validation_items = load_items(tokenizer, cfg, args.validation)
    print(f"[train] base={resolved_base}")
    print(f"[train] sequences={len(train_items)} validation={len(validation_items)}")

    encoder_params = [param for name, param in model.named_parameters() if name.startswith("encoder.")]
    head_params = [param for name, param in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.encoder_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=0.01,
    )

    updates_per_epoch = max(1, math.ceil(len(train_items) / (args.micro_batch * args.grad_accum)))
    total_updates = max(1, updates_per_epoch * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_updates,
        eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    started = time.time()

    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(train_items)
        progress = epoch / max(1, args.epochs - 1)
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * progress
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_reward = 0.0
        batches = 0

        for start in range(0, len(train_items), args.micro_batch):
            chunk = train_items[start : start + args.micro_batch]
            batch = collate_items(chunk, tokenizer.pad_token_id)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            marker_pos = batch["marker_pos"].to(device)
            marker_mask = batch["marker_mask"].to(device)
            qtype = batch["qtype"].to(device)
            target = batch["target"].to(device)

            with torch.autocast("cuda", dtype=torch.float16):
                logits, action_logits = model(
                    input_ids,
                    attention_mask,
                    marker_pos,
                    marker_mask,
                    qtype,
                )

            logits = logits.float()
            option_count = marker_mask.sum(-1, keepdim=True).float()
            noise = torch.randn((args.group_size,) + logits.shape, device=device)
            noise = noise * sigma * marker_mask
            noise = (noise - noise.sum(-1, keepdim=True) / option_count) * marker_mask
            sampled_logits = logits.detach().unsqueeze(0) + noise
            sampled_probs = torch.softmax(
                sampled_logits.masked_fill(~marker_mask, -1e4),
                -1,
            )

            with torch.no_grad():
                reward = proper_reward(
                    sampled_probs,
                    target.unsqueeze(0),
                    qtype,
                    marker_mask,
                    w_sph=0.75,
                    w_rps=1.0,
                )
                advantage = reward - reward.mean(0, keepdim=True)
                advantage = advantage / (advantage.std() + 1e-6)

            log_policy = -(
                ((sampled_logits - logits.unsqueeze(0)) ** 2) * marker_mask
            ).sum(-1) / (2 * sigma**2)
            loss_rl = -(advantage * log_policy).mean()
            loss_ce = -(
                target
                * torch.log_softmax(logits.masked_fill(~marker_mask, -1e4), -1)
            ).sum(-1).mean()
            loss = (loss_rl + loss_ce) / args.grad_accum + 0.0 * action_logits.sum()

            scaler.scale(loss).backward()
            batches += 1
            should_step = batches % args.grad_accum == 0 or start + args.micro_batch >= len(train_items)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item() * args.grad_accum)
            running_reward += float(reward.mean().item())

        metrics = validation_metrics(model, validation_items, tokenizer, device)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "epochs": args.epochs,
                    "loss": running_loss / max(1, batches),
                    "reward": running_reward / max(1, batches),
                    "sigma": sigma,
                    "validation": metrics,
                    "elapsed_seconds": round(time.time() - started, 1),
                },
                indent=2,
            )
        )

    cfg["fine_tuned"] = True
    cfg["model_name"] = "jev-my-bro"
    cfg["base_model"] = args.base_model
    cfg["temperature"] = [1.0, 1.0, 1.0]
    cfg["training"] = {
        "method": "rlcd_plus_soft_cross_entropy",
        "epochs": args.epochs,
        "seed": args.seed,
        "train_sequences": len(train_items),
        "validation_sequences": len(validation_items),
    }
    model.eval()
    save_checkpoint(model, tokenizer, cfg, args.output)
    print(f"[train] saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
