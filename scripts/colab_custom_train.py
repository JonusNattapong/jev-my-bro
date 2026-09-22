"""Train on the hand-written Thai dataset (data/custom_v1) in Colab.

colab_train_hf.py clones the repository from GitHub, which does not contain
data/custom_v1 until it is pushed. If the dataset is missing after the clone,
upload data/custom_v1/*.jsonl into /content/jev-my-bro/data/custom_v1 first.
"""

import os
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path("/content/jev-my-bro")
if not ROOT.exists():
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/JonusNattapong/jev-my-bro.git", str(ROOT)], check=True)

data_root = ROOT / "data" / "custom_v1"
missing = [name for name in ("train", "validation", "calibration", "test") if not (data_root / f"{name}.jsonl").exists()]
if missing:
    sys.exit(f"{data_root} is missing {missing}; push data/custom_v1 or upload the JSONL files first")

os.environ.update(
    {
        "TRAIN_DATA_ROOT": "data/custom_v1",
        "TRAIN_OUTPUT_NAME": "jev-my-bro-custom-v1",
        "TRAIN_EPOCHS": "4",
        "TRAIN_MICRO_BATCH": "8",
        "TRAIN_GRAD_ACCUM": "4",
        "CHECKPOINT_EACH_EPOCH": "1",
        "SCORE_CE_WEIGHT": "1.0",
        "SCORE_RPS_WEIGHT": "0.5",
        "SCORE_CUMULATIVE_WEIGHT": "1.0",
        "SCORE_CLASS_BALANCE_BETA": "0.0",
        "SCORE_LEVEL_WEIGHTS": "1.0,1.0,1.0,1.0,1.0",
    }
)
runpy.run_path(str(ROOT / "scripts" / "colab_train_hf.py"), run_name="__main__")
