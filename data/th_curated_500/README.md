# Thai curated 500 training set

This is a Thai-only, hand-authored training set with **500 cases / 2,000 typed decisions**.

- 380 cases are the previously hand-authored `custom_v1` training rows.
- 120 cases are newly authored in `th_06_new_handwritten.csv`.
- `validation`, `calibration`, and `test` each contain 40 separate hand-authored cases.
- No imported English or upstream rows are used in this dataset.
- Calibration is used only for temperature fitting; test remains untouched until final evaluation.

Build it with:

```bash
python scripts/build_th_curated_500.py
python scripts/validate_dataset.py --root data/th_curated_500
```

Train on Colab with the dedicated configuration:

```bash
python -m jevbro.train --config configs/colab-th500.yaml
python -m jevbro.calibrate --model artifacts/laya-th500 --data data/th_curated_500/calibration.jsonl
python -m jevbro.evaluate --model artifacts/laya-th500 --data data/th_curated_500/test.jsonl --device cuda
```

The configuration enables the cumulative ordinal loss and effective-number
score balancing. These are required for this handwritten split because its
training labels are not uniformly distributed across the five risk levels.
Calibration must still run only on `calibration.jsonl`; do not tune the model
or thresholds against `test.jsonl`.
