"""Run the Score v4 effective-number class-balance ablation."""

from __future__ import annotations

import os
import subprocess


ROOT = "/content/jev-my-bro"
if not os.path.exists(ROOT):
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/JonusNattapong/jev-my-bro.git", ROOT],
        check=True,
    )

os.environ.update({
    "TRAIN_DATA_ROOT": "data",
    "TRAIN_OUTPUT_NAME": "jev-my-bro-v4-cb",
    "TRAIN_EPOCHS": "4",
    "TRAIN_MICRO_BATCH": "8",
    "TRAIN_GRAD_ACCUM": "4",
    "CHECKPOINT_EACH_EPOCH": "1",
    "SCORE_CE_WEIGHT": "0.5",
    "SCORE_RPS_WEIGHT": "1.0",
    "SCORE_CLASS_BALANCE_BETA": "0.99",
})
runpy = os.path.join(ROOT, "scripts", "colab_train_hf.py")
with open(runpy, encoding="utf-8") as handle:
    exec(compile(handle.read(), runpy, "exec"), {"__name__": "__main__"})
