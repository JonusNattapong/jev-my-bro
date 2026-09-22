# Thai curated 960-case paired-boundary set

This set keeps the 880-case baseline and adds 80 manually authored training
cases as 40 paired contrasts. The new pairs focus on risk boundaries 1/2 and
2/3 across varied operation domains. Validation, calibration, and test remain
independent 100-case splits with exactly 20 examples at every risk level.

Build and validate:

    python scripts/build_th_curated_960.py
    python scripts/validate_dataset.py --root data/th_curated_960

Train, calibrate, then lock the final test evaluation:

    python -m jevbro.train --config configs/colab-th960.yaml
    python -m jevbro.calibrate --model artifacts/laya-th960 --data data/th_curated_960/calibration.jsonl --report artifacts/laya-th960/calibration-report.json
    python -m jevbro.evaluate --model artifacts/laya-th960 --data data/th_curated_960/test.jsonl --device cuda --lock-decoder --report artifacts/laya-th960/test-report.json

The final command only consumes the decoder and temperatures written by the
calibration command. Calibration refuses a path named `test.jsonl`.
