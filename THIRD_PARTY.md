# Third-party foundations

## Laya

jev-my-bro v0.2 uses the open-source Laya decision-model runtime and model family.

- Project: https://github.com/NandhaKishorM/laya
- Package: https://pypi.org/project/laya/
- License: Apache License 2.0
- Upstream author: Convai Innovations / NandhaKishorM

The training approach in this repository follows Laya's published RLCD design: noisy policy samples are scored with strictly proper scoring rules and trained together with soft-target cross entropy. jev-my-bro keeps its own domain dataset, training entrypoints, calibration split, evaluation, and serving contract.

The original DeBERTa experiment remains under baseline/deberta for comparison.
