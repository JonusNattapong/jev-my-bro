# Thai curated 620-case boundary-contrast training set

This set preserves the 40-case validation, calibration, and test splits from
the 560-case baseline and adds 60 manually authored training cases. The new
cases target risk boundaries 1/2 and 3/4 and contrast authorized versus
unauthorized versions of similar operations.

Build and validate:

    python scripts/build_th_curated_620.py
    python scripts/validate_dataset.py --root data/th_curated_620

Train on Colab:

    python -m jevbro.train --config configs/colab-th620.yaml
    python -m jevbro.calibrate --model artifacts/laya-th620 --data data/th_curated_620/calibration.jsonl
    python -m jevbro.evaluate --model artifacts/laya-th620 --data data/th_curated_620/test.jsonl --device cuda
