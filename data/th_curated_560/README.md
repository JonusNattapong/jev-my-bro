# Thai curated 560-case ordinal-balanced training set

This set keeps the original independent 40-case validation, calibration, and
test splits, and adds 60 authored training cases focused on risk levels 1, 2,
and 4. The new training rows are marked with
`jev-my-bro-handwritten-th-ordinal-balance` provenance.

Build and validate:

```bash
python scripts/build_th_curated_560.py
python scripts/validate_dataset.py --root data/th_curated_560
```

Train on Colab:

```bash
python -m jevbro.train --config configs/colab-th560.yaml
python -m jevbro.calibrate --model artifacts/laya-th560 --data data/th_curated_560/calibration.jsonl
python -m jevbro.evaluate --model artifacts/laya-th560 --data data/th_curated_560/test.jsonl --device cuda
```

The selected root checkpoint is the epoch with the best validation score QWK,
breaking ties with lower RPS. The last epoch is retained under `last/`.
