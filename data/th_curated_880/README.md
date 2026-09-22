# Thai curated 880-case paired ordinal-boundary set

This set keeps the 760-case baseline and adds 120 manually authored training
cases as 60 paired contrasts. The pairs target risk boundaries 1/2, 2/3, and
3/4 by changing scope, blast radius, reversibility, or production impact while
keeping the operation family similar. Validation, calibration, and test remain
independent 100-case splits with exactly 20 examples at every risk level.

Build and validate:

    python scripts/build_th_curated_880.py
    python scripts/validate_dataset.py --root data/th_curated_880

Train on Colab:

    python -m jevbro.train --config configs/colab-th880.yaml
    python -m jevbro.calibrate --model artifacts/laya-th880 --data data/th_curated_880/calibration.jsonl
    python -m jevbro.evaluate --model artifacts/laya-th880 --data data/th_curated_880/test.jsonl --device cuda
