# Reproducible Google Colab training

This workflow uses the repository's shared RLCD + ordinal soft-target cross-entropy trainer. The notebook is optional; the commands below are the source of truth.

## 1. Start a T4 runtime

In Colab, choose **Runtime -> Change runtime type -> T4 GPU**, then run:

```bash
!nvidia-smi
!git clone https://github.com/JonusNattapong/jev-my-bro.git /content/jev-my-bro
%cd /content/jev-my-bro
!pip install -r requirements.txt
```

The base model is downloaded by `huggingface_hub` at runtime. If the model is gated or private, authenticate with Colab's prompt or an environment-managed token; never put a token in this repository or notebook.

## 2. Validate all independent splits

```bash
!python scripts/validate_dataset.py --root data/hf_expanded
```

The active dataset must contain `train.jsonl`, `validation.jsonl`, `calibration.jsonl`, and `test.jsonl`. Calibration is only used after training; test is only used for the final report.

## 3. Train from the checked-in T4 config

```bash
!python -m jevbro.train --config configs/colab-t4.yaml
```

This writes epoch checkpoints and the final checkpoint under `artifacts/laya-colab-t4`. The seed, split paths, batch sizes, learning rates, and RLCD/ordinal weights are all in the config. Explicit CLI flags override config values.

## 4. Fit calibration and lock the final test evaluation

```bash
!python -m jevbro.calibrate \
  --model artifacts/laya-colab-t4 \
  --data data/hf_expanded/calibration.jsonl \
  --report artifacts/laya-colab-t4/calibration-report.json

!python -m jevbro.evaluate \
  --model artifacts/laya-colab-t4 \
  --data data/hf_expanded/test.jsonl \
  --device cuda \
  --lock-decoder \
  --report artifacts/laya-colab-t4/test-report.json
```

The calibration command writes temperature, score thresholds, decoder choice, and
the calibration-file hash into `rl_agent_config.json`. `--lock-decoder` refuses
to evaluate unless that provenance exists, so the test split cannot select or
fit the decoder. The final report contains overall accuracy, English/Thai
accuracy, per-primitive accuracy and Brier score, ECE, NLL, and score-specific
ordinal metrics. Do not tune against `test.jsonl`.

For the current Thai 960-case experiment, build and validate the checked-in
splits first, then use the matching T4 config:

```bash
!python scripts/build_th_curated_960.py
!python scripts/validate_dataset.py --root data/th_curated_960
!python -m jevbro.train --config configs/colab-th960.yaml
!python -m jevbro.calibrate \
  --model artifacts/laya-th960 \
  --data data/th_curated_960/calibration.jsonl \
  --report artifacts/laya-th960/calibration-report.json
!python -m jevbro.evaluate \
  --model artifacts/laya-th960 \
  --data data/th_curated_960/test.jsonl \
  --device cuda \
  --lock-decoder \
  --report artifacts/laya-th960/test-report.json
```

## 5. Publish without hardcoded credentials

Set `HF_TOKEN` in the Colab runtime secret/environment, then run:

```bash
!python -m jevbro.publish \
  --checkpoint artifacts/laya-colab-t4 \
  --repo-id YOUR_ACCOUNT/jev-my-bro
```

Use `--dry-run` first to validate that the checkpoint has weights, config, and tokenizer files. Publishing is separate from training and evaluation.

For a single non-notebook run, use `python scripts/colab_runner.py`; it performs dataset validation, training, calibration, and final evaluation through the same module entry points.
