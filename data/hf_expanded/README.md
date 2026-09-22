---
pretty_name: jev-my-bro Governance Dataset
language:
  - en
  - th
license: other
tags:
  - agent-governance
  - tool-governance
  - decision-model
  - thai
  - english
  - jsonl
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.jsonl
      - split: validation
        path: validation.jsonl
      - split: calibration
        path: calibration.jsonl
      - split: test
        path: test.jsonl
---

# jev-my-bro Governance Dataset

Provenance-aware English/Thai dataset for training and evaluating the typed
`jev-my-bro` decision model. Each case asks four structured governance
questions about an operation:

- `action`: `execute`, `ask_user`, or `reject`
- `needs_review`: whether explicit human review/approval is required
- `prohibited`: whether the operation should be prohibited
- `risk`: five-level operational risk

This snapshot contains **8,508 cases / 34,032 typed decisions**.

> This is research/development data, not a production authorization policy.
> Imported labels marked `rule_reviewed` are rule-derived and consistency
> reviewed; they are **not human annotations**.

## Files and splits

| Split | Cases | Decisions | English | Thai |
| --- | ---: | ---: | ---: | ---: |
| `train` | 5,731 | 22,924 | 3,829 | 1,902 |
| `validation` | 879 | 3,516 | 586 | 293 |
| `calibration` | 944 | 3,776 | 641 | 303 |
| `test` | 954 | 3,816 | 616 | 338 |
| **Total** | **8,508** | **34,032** | **5,672** | **2,836** |

The four splits are kept independent. Cases derived from the same
`scenario_family` stay in one split to prevent family/paraphrase leakage.

## Source composition

| Source | Cases | Language | Upstream license |
| --- | ---: | --- | --- |
| jev-my-bro static bootstrap | 1,008 | EN + TH | Apache-2.0 |
| `AmazonScience/massive` (`th-TH`) | 2,500 | TH | CC BY 4.0 |
| `mteb/banking77` | 2,500 | EN | MIT |
| `NousResearch/hermes-function-calling-v1` (`func_calling_singleturn`) | 2,500 | EN | Apache-2.0 |

Every imported row records provenance fields such as `source_dataset`,
`source_record`, `source_license`, and `attribution`. See
`SOURCE_MANIFEST.json` for the machine-readable source manifest.

## Transformations

The imported upstream text is selected and transformed into the jev-my-bro
governance contract. Source labels are not presented as original governance
annotations.

The provenance-aware imported families use these variants:

| Variant | Purpose |
| --- | --- |
| `base` | Source-derived base request |
| `approval_context` | Verified explicit authorization context |
| `revoked_approval` | Previously granted authorization was revoked |
| `conflict_context` | Conflicting authorization/scope instructions |
| `ood_mixed_indirect` | Indirect/OOD framing with Thai-English mixed wording |

Current variant counts are:

| Variant | Cases |
| --- | ---: |
| `base` | 2,508 |
| `approval_context` | 1,500 |
| `revoked_approval` | 1,500 |
| `conflict_context` | 1,500 |
| `ood_mixed_indirect` | 1,500 |

## Schema

Each JSONL row has this high-level shape:

```json
{
  "id": "stable-case-id",
  "workflow": "agent_operation_governance",
  "language": "en",
  "state": {
    "request": "operation or request text",
    "domain": "domain-name",
    "difficulty": "easy|medium|hard",
    "source_dataset": "upstream-or-project-source",
    "source_record": "upstream-record-id",
    "source_license": "license-id",
    "attribution": "source attribution",
    "scenario_family": "family-id",
    "variant_type": "base|approval_context|revoked_approval|conflict_context|ood_mixed_indirect",
    "label_status": "existing-static|rule_reviewed"
  },
  "questions": {
    "action": {"type": "choice"},
    "needs_review": {"type": "noul"},
    "prohibited": {"type": "noul"},
    "risk": {"type": "score"}
  },
  "gold": {
    "action": {"label": "execute", "probabilities": {}},
    "needs_review": {"label": "false", "noul": 0.08, "probabilities": {}},
    "prohibited": {"label": "false", "noul": 0.08, "probabilities": {}},
    "risk": {"label": "0", "score": 0.2, "probabilities": {}}
  }
}
```

All training targets are probability distributions. For `risk`, `score` is
kept consistent with the probability-weighted expected level. Score targets use
a distance-aware Gaussian ordinal distribution (`sigma=0.75`), so neighboring
risk levels receive more probability mass than distant levels. Risk measures
operational impact, blast radius, and reversibility if executed; authorization
and prohibition are handled by their dedicated primitives.

## Action distribution

Across all splits:

| Action | Cases |
| --- | ---: |
| `ask_user` | 4,766 |
| `execute` | 3,406 |
| `reject` | 336 |

This distribution is intentionally reported because the dataset is not
class-balanced. Consumers should evaluate per-class behavior instead of relying
only on aggregate accuracy.

## Risk distribution

| Risk level | Cases |
| ---: | ---: |
| 0 | 2,623 |
| 1 | 68 |
| 2 | 3,168 |
| 3 | 2,313 |
| 4 | 336 |

The project-authored 1,008-case bootstrap subset was relabeled for the v4
ordinal-risk rubric so all five levels are represented. Imported `rule_reviewed`
rows retain their semantic labels but use the same ordinal target distribution.
Risk level 1 remains sparse in the expanded corpus and should be treated as a
known coverage limitation until further human-reviewed curation is added.

## Load from Hugging Face

```python
from datasets import load_dataset

dataset = load_dataset("JonusNattapong/jev-my-bro-dataset")

print(dataset["train"][0])
```

Repository training uses the checked-out JSONL files directly:

```bash
python -m jevbro.train \
  --train data/hf_expanded/train.jsonl \
  --validation data/hf_expanded/validation.jsonl
```

Calibration and final evaluation use their dedicated held-out splits:

```bash
python -m jevbro.calibrate \
  --model artifacts/laya-model \
  --data data/hf_expanded/calibration.jsonl

python -m jevbro.evaluate \
  --model artifacts/laya-model \
  --data data/hf_expanded/test.jsonl
```

Do not fit calibration or tune hyperparameters on the test split.

## Validation and quality checks

The source repository provides two checks:

```bash
python scripts/validate_dataset.py --root data/hf_expanded
python scripts/audit_hf_dataset.py --root data/hf_expanded
```

For the published snapshot on **2026-09-21**:

- cases: **8,508**
- decisions: **34,032**
- scenario families: **2,508**
- audit flags: **0**
- normalized duplicate request leakage across splits: **0**
- scenario-family leakage across splits: **0**

The checks are structural and rubric-consistency checks; they do not substitute
for independent human review.

## Evaluation status

Previously published jev-my-bro v0.2 metrics were measured on the older
1,008-case bootstrap dataset. Those scores should **not** be treated as metrics
for this updated 8,508-case snapshot.

A new training, calibration, and test run is required before publishing model
quality claims for this dataset revision.

## Licenses and attribution

This is a **mixed-source dataset**. There is no claim that all rows can be
relicensed under one upstream license. Consumers and redistributors should
respect the license and attribution metadata associated with each source.

### MASSIVE

- Dataset: `AmazonScience/massive`
- Configuration: `th-TH`
- License recorded by this dataset: **CC BY 4.0**
- Attribution: **FitzGerald et al., MASSIVE (2022); Amazon Science**
- Source: https://huggingface.co/datasets/AmazonScience/massive

### BANKING77

- Dataset: `mteb/banking77`
- License recorded by this dataset: **MIT**
- Attribution: **Casanueva et al., BANKING77 (2020); MTEB mirror**
- Source: https://huggingface.co/datasets/mteb/banking77

### Hermes Function Calling

- Dataset: `NousResearch/hermes-function-calling-v1`
- Configuration: `func_calling_singleturn`
- License recorded by this dataset: **Apache-2.0**
- Attribution: **Nous Research Hermes function-calling dataset**
- Source: https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1

### jev-my-bro bootstrap data

- Dataset: `jev-my-bro-static-v0.1`
- Cases: **1,008**
- License: **Apache-2.0**
- Project license text: https://github.com/JonusNattapong/jev-my-bro/blob/main/LICENSE

The combined dataset remains mixed-license. The project Apache-2.0 license does
not replace or override MASSIVE CC BY 4.0, BANKING77 MIT, or other applicable
upstream terms.

## Known limitations

- `rule_reviewed` cases are not independently human-labeled.
- The `reject` action is substantially less frequent than `ask_user` and
  `execute`.
- The dataset is specialized for agent/tool governance and should not be
  interpreted as a general safety or authorization benchmark.
- Source-domain language and transformation templates can introduce shortcuts;
  per-source, per-language, per-variant, and per-class evaluation is recommended.
- The model consuming this data should remain advisory unless paired with
  explicit policy enforcement, human escalation, and audit logging.

## Related project

- Source code: https://github.com/JonusNattapong/jev-my-bro
- Model: https://huggingface.co/JonusNattapong/jev-my-bro
- Dataset: https://huggingface.co/datasets/JonusNattapong/jev-my-bro-dataset
