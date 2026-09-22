"""Bootstrap a Colab checkout and run the shared train/calibrate/evaluate flow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("[colab]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/colab-t4.yaml")
    parser.add_argument("--data-root", default="data/hf_expanded")
    parser.add_argument("--output", default="artifacts/laya-colab-t4")
    args = parser.parse_args()
    run([sys.executable, "scripts/validate_dataset.py", "--root", args.data_root])
    run([sys.executable, "-m", "jevbro.train", "--config", args.config])
    run([sys.executable, "-m", "jevbro.calibrate", "--model", args.output,
         "--data", f"{args.data_root}/calibration.jsonl",
         "--report", f"{args.output}/calibration-report.json"])
    run([sys.executable, "-m", "jevbro.evaluate", "--model", args.output,
         "--data", f"{args.data_root}/test.jsonl", "--device", "cuda",
         "--report", f"{args.output}/test-report.json"])


if __name__ == "__main__":
    main()
