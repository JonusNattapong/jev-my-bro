"""Train the evidence-first v5 dataset for one Colab smoke-test epoch."""

import os
import runpy


os.environ.update(
    {
        "TRAIN_DATA_ROOT": "data/reviewed_v5",
        "TRAIN_OUTPUT_NAME": "jev-my-bro-v5-1e",
        "TRAIN_EPOCHS": "1",
        "SMOKE_CASES": "500",
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
runpy.run_path("/content/jev-my-bro/scripts/colab_train_hf.py", run_name="__main__")
