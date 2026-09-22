from __future__ import annotations

import argparse
import json
import math
import random
import time

import numpy as np
import torch
from laya.common import QTYPES, proper_reward

from jevbro.batching import collate_items
from jevbro.checkpoint import load_trainable, save_checkpoint
from jevbro.config import apply_config_defaults
from jevbro.data import load_items
from jevbro.ordinal import cumulative_ordinal_loss, effective_number_weights, ranked_probability_loss

DEFAULT_BASE = "convaiinnovations/laya-multilingual"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Laya with RLCD on jev-my-bro data")
    parser.add_argument("--config", default=None, help="YAML config with the same keys as CLI options")
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--validation", default="data/validation.jsonl")
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--output", default="artifacts/laya-model")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--checkpoint-each-epoch", action="store_true")
    parser.add_argument("--english-weight", type=float, default=1.5)
    parser.add_argument("--choice-weight", type=float, default=1.5)
    parser.add_argument("--score-weight", type=float, default=2.0)
    parser.add_argument("--score-ce-weight", type=float, default=0.5)
    parser.add_argument("--score-rps-weight", type=float, default=1.0)
    parser.add_argument("--score-class-balance-beta", type=float, default=0.0)
    parser.add_argument("--score-level-weights", default="")
    parser.add_argument("--score-cumulative-weight", type=float, default=0.0)
    parser.add_argument("--encoder-lr", type=float, default=2.5e-5)
    parser.add_argument("--head-lr", type=float, default=1.0e-4)
    parser.add_argument("--sigma-start", type=float, default=0.4)
    parser.add_argument("--sigma-end", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    apply_config_defaults(parser, argv)
    return parser.parse_args(argv)


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
    score_n = 0
    score_correct = 0
    score_hard_error = 0.0
    score_expected_error = 0.0
    score_rps_sum = 0.0
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
        probs = torch.softmax(logits, -1)
        nll_sum += float((-(target * log_probs).sum(-1)).sum().item())
        correct += int((logits.argmax(-1) == labels).sum().item())
        is_score = qtype == QTYPES["score"]
        if is_score.any():
            selected_probs = probs[is_score]
            selected_target = target[is_score]
            selected_mask = marker_mask[is_score]
            selected_labels = labels[is_score]
            pred_levels = selected_probs.argmax(-1)
            level_axis = torch.arange(selected_probs.shape[-1], device=device, dtype=selected_probs.dtype)
            pred_expected = (selected_probs * level_axis).sum(-1)
            target_expected = (selected_target * level_axis).sum(-1)
            count = int(is_score.sum().item())
            score_n += count
            score_correct += int((pred_levels == selected_labels).sum().item())
            score_hard_error += float((pred_levels - selected_labels).abs().float().sum().item())
            score_expected_error += float((pred_expected - target_expected).abs().sum().item())
            score_rps_sum += float(
                ranked_probability_loss(selected_probs, selected_target, selected_mask).sum().item()
            )
        total += len(labels)
    model.train()
    return {
        "accuracy": correct / max(1, total),
        "soft_nll": nll_sum / max(1, total),
        "score": {
            "n": score_n,
            "accuracy": score_correct / max(1, score_n),
            "hard_mae": score_hard_error / max(1, score_n),
            "expected_mae": score_expected_error / max(1, score_n),
            "rps": score_rps_sum / max(1, score_n),
        },
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    seed_everything(args.seed)
    print(f"[train] device={device}", flush=True)
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
    print(f"[train] base={resolved_base}", flush=True)
    print(f"[train] sequences={len(train_items)} validation={len(validation_items)}", flush=True)
    score_labels = [item["label"] for item in train_items if item["qtype"] == QTYPES["score"]]
    score_level_weights = torch.tensor(
        effective_number_weights(score_labels, args.score_class_balance_beta, levels=5),
        device=device,
        dtype=torch.float32,
    )
    if args.score_level_weights:
        manual_weights = [float(value.strip()) for value in args.score_level_weights.split(",")]
        if len(manual_weights) != 5 or any(value <= 0 for value in manual_weights):
            raise SystemExit("--score-level-weights requires five positive comma-separated values")
        score_level_weights = score_level_weights * torch.tensor(
            manual_weights,
            device=device,
            dtype=torch.float32,
        )
    print(f"[train] score_level_weights={score_level_weights.tolist()}", flush=True)

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
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
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
            labels = batch["label"].to(device)
            language = batch["language"].to(device)

            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
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

            item_weight = torch.ones_like(qtype, dtype=torch.float32)
            item_weight = item_weight * torch.where(
                language == 0,
                torch.tensor(args.english_weight, device=device),
                torch.tensor(1.0, device=device),
            )
            item_weight = item_weight * torch.where(
                qtype == QTYPES["choice"],
                torch.tensor(args.choice_weight, device=device),
                torch.tensor(1.0, device=device),
            )
            item_weight = item_weight * torch.where(
                qtype == QTYPES["score"],
                torch.tensor(args.score_weight, device=device),
                torch.tensor(1.0, device=device),
            )
            score_balance = score_level_weights[labels.clamp(min=0, max=4)]
            item_weight = item_weight * torch.where(
                qtype == QTYPES["score"],
                score_balance,
                torch.tensor(1.0, device=device),
            )
            item_weight = item_weight / item_weight.mean().clamp_min(1e-6)

            log_policy = -(
                ((sampled_logits - logits.unsqueeze(0)) ** 2) * marker_mask
            ).sum(-1) / (2 * sigma**2)
            loss_rl = -(advantage * log_policy).mean(0)
            loss_ce = -(
                target
                * torch.log_softmax(logits.masked_fill(~marker_mask, -1e4), -1)
            ).sum(-1)
            probs = torch.softmax(logits.masked_fill(~marker_mask, -1e4), -1)
            loss_rps = ranked_probability_loss(probs, target, marker_mask)
            loss_cumulative = cumulative_ordinal_loss(probs, target, marker_mask)
            is_score = (qtype == QTYPES["score"]).float()
            supervised = loss_ce * (1.0 - is_score + is_score * args.score_ce_weight)
            supervised = supervised + is_score * args.score_rps_weight * loss_rps
            supervised = supervised + is_score * args.score_cumulative_weight * loss_cumulative
            loss = ((loss_rl + supervised) * item_weight).mean() / args.grad_accum + 0.0 * action_logits.sum()

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

            if batches == 1 or batches % 25 == 0 or start + args.micro_batch >= len(train_items):
                print(
                    f"[train] epoch={epoch + 1}/{args.epochs} "
                    f"batch={batches}/{math.ceil(len(train_items) / args.micro_batch)} "
                    f"loss={running_loss / batches:.4f} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

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
            ),
            flush=True,
        )

        if args.checkpoint_each_epoch:
            epoch_cfg = dict(cfg)
            epoch_cfg["fine_tuned"] = True
            epoch_cfg["model_name"] = "jev-my-bro"
            epoch_cfg["base_model"] = args.base_model
            epoch_cfg["temperature"] = [1.0, 1.0, 1.0]
            epoch_cfg["training"] = {
                "method": "rlcd_plus_ordinal_soft_ce_rps",
                "epochs": epoch + 1,
                "seed": args.seed,
                "train_sequences": len(train_items),
                "validation_sequences": len(validation_items),
                "score_ce_weight": args.score_ce_weight,
                "score_rps_weight": args.score_rps_weight,
                "score_class_balance_beta": args.score_class_balance_beta,
                "score_level_weights": args.score_level_weights,
                "score_cumulative_weight": args.score_cumulative_weight,
            }
            save_checkpoint(model, tokenizer, epoch_cfg, f"{args.output}/epoch-{epoch + 1}")
            print(f"[train] saved epoch checkpoint to {args.output}/epoch-{epoch + 1}", flush=True)

    cfg["fine_tuned"] = True
    cfg["model_name"] = "jev-my-bro"
    cfg["base_model"] = args.base_model
    cfg["temperature"] = [1.0, 1.0, 1.0]
    cfg["training"] = {
        "method": "rlcd_plus_ordinal_soft_ce_rps",
        "epochs": args.epochs,
        "seed": args.seed,
        "train_sequences": len(train_items),
        "validation_sequences": len(validation_items),
        "score_ce_weight": args.score_ce_weight,
        "score_rps_weight": args.score_rps_weight,
        "score_class_balance_beta": args.score_class_balance_beta,
        "score_level_weights": args.score_level_weights,
        "score_cumulative_weight": args.score_cumulative_weight,
    }
    model.eval()
    save_checkpoint(model, tokenizer, cfg, args.output)
    print(f"[train] saved checkpoint to {args.output}", flush=True)


if __name__ == "__main__":
    main()
