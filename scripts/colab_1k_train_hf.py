"""Run a four-epoch 1,000-case Colab training trial."""

from __future__ import annotations

import os
import subprocess


ROOT = "/content/jev-my-bro"
if not os.path.exists(ROOT):
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/JonusNattapong/jev-my-bro.git", ROOT],
        check=True,
    )

os.environ["SMOKE_CASES"] = "1000"
os.environ["TRAIN_EPOCHS"] = "1"
os.environ["TRAIN_MICRO_BATCH"] = "8"
os.environ["TRAIN_GRAD_ACCUM"] = "4"
os.environ["CHECKPOINT_EACH_EPOCH"] = "1"
runpy = os.path.join(ROOT, "scripts", "colab_train_hf.py")
with open(runpy, encoding="utf-8") as handle:
    exec(compile(handle.read(), runpy, "exec"), {"__name__": "__main__"})
