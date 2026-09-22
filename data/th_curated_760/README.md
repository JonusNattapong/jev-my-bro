# Thai curated 760-case risk 1/3 contrast set

This set keeps the original 700-case training data, adds 60 manually authored
training cases focused on separating risk 1 from risk 3, and expands each
holdout split to 100 cases. Validation, calibration, and test each contain
exactly 20 cases at every risk level.

Build and validate:

    python scripts/build_th_curated_760.py
    python scripts/validate_dataset.py --root data/th_curated_760

Train on Colab:

    python -m jevbro.train --config configs/colab-th760.yaml
    python -m jevbro.calibrate --model artifacts/laya-th760 --data data/th_curated_760/calibration.jsonl
    python -m jevbro.evaluate --model artifacts/laya-th760 --data data/th_curated_760/test.jsonl --device cuda
