"""Recalibrate and re-evaluate an existing Score v4.1 Colab checkpoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path("/content/jev-my-bro")
ARTIFACTS = Path("/content/jev-artifacts")
MODEL = ARTIFACTS / "jev-my-bro-v4"

subprocess.run(
    [
        sys.executable,
        "-m",
        "jevbro.calibrate",
        "--model",
        str(MODEL),
        "--data",
        str(ROOT / "data/calibration.jsonl"),
        "--report",
        str(ARTIFACTS / "calibration-threshold-report.json"),
    ],
    cwd=ROOT,
    check=True,
)
for data_name, report_name in (
    ("test.jsonl", "test-threshold-report.json"),
    ("score_v41_challenge.jsonl", "challenge-threshold-report.json"),
):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jevbro.evaluate",
            "--model",
            str(MODEL),
            "--data",
            str(ROOT / "data" / data_name),
            "--device",
            "cuda",
            "--report",
            str(ARTIFACTS / report_name),
        ],
        cwd=ROOT,
        check=True,
    )
