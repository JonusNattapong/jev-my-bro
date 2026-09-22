"""Publish a completed checkpoint to Hugging Face without storing credentials."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repository")
    parser.add_argument("--commit-message", default="Publish Jev checkpoint")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    checkpoint = Path(args.checkpoint)
    required = ("model.safetensors", "rl_agent_config.json", "tokenizer")
    missing = [name for name in required if not (checkpoint / name).exists()]
    if missing:
        raise SystemExit(f"checkpoint is incomplete: missing {', '.join(missing)}")
    if args.dry_run:
        print(f"[publish] validated checkpoint={checkpoint} repo={args.repo_id}")
        return

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(checkpoint),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )
    print(f"[publish] uploaded {checkpoint} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
