# Thai curated 700-case score-disambiguation training set

This set preserves the 40-case validation, calibration, and test splits from
the 620-case baseline and adds 80 manually authored training cases. The new
cases focus on separating risk levels 1/2, 2/3, and 3/4 while covering both
bounded authorized operations and prohibited critical operations.

Build and validate:

    python scripts/build_th_curated_700.py
    python scripts/validate_dataset.py --root data/th_curated_700

Train on Colab:

    python -m jevbro.train --config configs/colab-th700.yaml
    python -m jevbro.calibrate --model artifacts/laya-th700 --data data/th_curated_700/calibration.jsonl
    python -m jevbro.evaluate --model artifacts/laya-th700 --data data/th_curated_700/test.jsonl --device cuda
