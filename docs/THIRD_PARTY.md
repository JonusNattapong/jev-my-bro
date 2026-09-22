# Third-party foundations

## Laya

jev-my-bro v0.2 uses the open-source Laya decision-model runtime and model family.

- Project: https://github.com/NandhaKishorM/laya
- Package: https://pypi.org/project/laya/
- License: Apache License 2.0
- Upstream author: Convai Innovations / NandhaKishorM

The training approach in this repository follows Laya's published RLCD design:
noisy policy samples are scored with strictly proper scoring rules. The current
Score v4 development line adds distance-aware ordinal targets and a direct RPS
loss for the score primitive. jev-my-bro keeps its own domain dataset, training
entrypoints, calibration split, evaluation, and serving contract.

The original DeBERTa experiment remains under baseline/deberta for comparison.

## Training-data sources

The active `data/hf_expanded/` dataset is mixed-source. Imported rows retain
source-specific provenance, license, and attribution metadata.

### MASSIVE

- Dataset: `AmazonScience/massive`
- Configuration: `th-TH`
- License recorded in `SOURCE_MANIFEST.json`: CC BY 4.0
- Attribution: FitzGerald et al., MASSIVE (2022); Amazon Science
- Source: https://huggingface.co/datasets/AmazonScience/massive

### BANKING77

- Dataset: `mteb/banking77`
- License recorded in `SOURCE_MANIFEST.json`: MIT
- Attribution: Casanueva et al., BANKING77 (2020); MTEB mirror
- Source: https://huggingface.co/datasets/mteb/banking77

### Hermes function calling

- Dataset: `NousResearch/hermes-function-calling-v1`
- Configuration: `func_calling_singleturn`
- License recorded in `SOURCE_MANIFEST.json`: Apache-2.0
- Attribution: Nous Research Hermes function-calling dataset
- Source: https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1

These upstream licenses apply to their corresponding source-derived rows.
The 1,008 project-authored bootstrap rows are licensed under Apache-2.0; see
[`../LICENSE`](../LICENSE). The combined dataset remains mixed-license; the
repository Apache-2.0 license does not replace upstream terms. See
[`../data/hf_expanded/README.md`](../data/hf_expanded/README.md) and
[`../data/hf_expanded/SOURCE_MANIFEST.json`](../data/hf_expanded/SOURCE_MANIFEST.json)
for the published dataset documentation.
