"""Train, calibrate, and evaluate the provenance-aware HF dataset on Colab."""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


ROOT = Path("/content/jev-my-bro")
ARTIFACTS = Path("/content/jev-artifacts")
DATA = ROOT / "data" / "hf_expanded"


def maybe_limit_split(path: Path, limit: int | None) -> Path:
    """Create a deterministic small split for Colab smoke tests."""
    if limit is None:
        return path
    smoke_dir = ARTIFACTS / "smoke-data"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    output = smoke_dir / path.name
    lines = path.read_text(encoding="utf-8").splitlines()
    output.write_text("\n".join(lines[:limit]) + "\n", encoding="utf-8")
    return output

if not ROOT.exists():
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/JonusNattapong/jev-my-bro.git", str(ROOT)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "laya==0.3.4", "datasets>=3.0"], check=True)
ARTIFACTS.mkdir(parents=True, exist_ok=True)
smoke_cases = int(os.environ["SMOKE_CASES"]) if os.environ.get("SMOKE_CASES") else None
epochs = int(os.environ.get("TRAIN_EPOCHS", "4"))
train_data = maybe_limit_split(DATA / "train.jsonl", smoke_cases)
validation_data = maybe_limit_split(DATA / "validation.jsonl", smoke_cases)
calibration_data = maybe_limit_split(DATA / "calibration.jsonl", smoke_cases)
test_data = maybe_limit_split(DATA / "test.jsonl", smoke_cases)

subprocess.run(
    [
        sys.executable,
        "-m",
        "jevbro.train",
        "--train",
        str(train_data),
        "--validation",
        str(validation_data),
        "--base-model",
        "convaiinnovations/laya-multilingual",
        "--output",
        str(ARTIFACTS / "jev-my-bro-model"),
        "--epochs",
        str(epochs),
        "--micro-batch",
        "4",
        "--grad-accum",
        "8",
        "--group-size",
        "4",
        "--seed",
        "42",
    ],
    cwd=ROOT,
    check=True,
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "jevbro.calibrate",
        "--model",
        str(ARTIFACTS / "jev-my-bro-model"),
        "--data",
        str(calibration_data),
        "--report",
        str(ARTIFACTS / "calibration-report.json"),
    ],
    cwd=ROOT,
    check=True,
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "jevbro.evaluate",
        "--model",
        str(ARTIFACTS / "jev-my-bro-model"),
        "--data",
        str(test_data),
        "--device",
        "cuda",
        "--report",
        str(ARTIFACTS / "test-report.json"),
    ],
    cwd=ROOT,
    check=True,
)
print(f"Artifacts written to {ARTIFACTS}")
