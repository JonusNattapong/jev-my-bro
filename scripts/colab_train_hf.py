"""Train, calibrate, and evaluate the provenance-aware HF dataset on Colab."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path("/content/jev-my-bro")
ARTIFACTS = Path("/content/jev-artifacts")
DATA = ROOT / "data" / "hf_expanded"

subprocess.run(["git", "clone", "--depth", "1", "https://github.com/JonusNattapong/jev-my-bro.git", str(ROOT)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "laya==0.3.4", "datasets>=3.0"], check=True)
ARTIFACTS.mkdir(parents=True, exist_ok=True)

subprocess.run(
    [
        sys.executable,
        "-m",
        "jevbro.train",
        "--train",
        str(DATA / "train.jsonl"),
        "--validation",
        str(DATA / "validation.jsonl"),
        "--base-model",
        "convaiinnovations/laya-multilingual",
        "--output",
        str(ARTIFACTS / "jev-my-bro-model"),
        "--epochs",
        "4",
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
        str(DATA / "calibration.jsonl"),
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
        str(DATA / "test.jsonl"),
        "--device",
        "cuda",
        "--report",
        str(ARTIFACTS / "test-report.json"),
    ],
    cwd=ROOT,
    check=True,
)
print(f"Artifacts written to {ARTIFACTS}")
