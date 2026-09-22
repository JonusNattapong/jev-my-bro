# Dataset expansion plan

This document started as the acceptance plan for expanding the original 1,008-case
bootstrap corpus. The target has been reached: the active provenance-aware
snapshot is `data/hf_expanded/` with **8,508 cases** (5,731 train, 879 validation,
944 calibration, 954 test). It is also published as
`JonusNattapong/jev-my-bro-dataset` on Hugging Face. Keep this document as the
acceptance contract for future revisions.

## Required coverage

Each new case keeps the existing four-question contract: action `choice`, review
and prohibition `noul`, and risk `score`. The expanded corpus must include
paraphrase, long context, conflicting context, explicit approval, revoked
approval, indirect wording, Thai/English mixed wording, code/tool-call payloads,
and OOD cases. Coverage must be reported by language, primitive, domain, and
difficulty rather than inferred from the aggregate accuracy.

The target is total cases, not duplicated decision rows. A proposed 5,000-case
minimum is 3,500 train, 500 validation, 500 calibration, and 500 test. Cases
derived from the same scenario family must stay in one split so paraphrases do
not leak across evaluation boundaries.

## Authoring and review gates

1. Author new scenarios by semantic family and perturbation type; do not create
   a Cartesian product of templates.
2. Give every case a stable unique id and explicit metadata for domain,
   difficulty, language, and perturbation type.
3. Review labels independently, including approval/revocation precedence,
   conflicting instructions, and score targets. Risk uses the Score v4 ordinal
   rubric: impact/blast-radius/reversibility are separate from authorization
   and prohibition. Keep probability targets normalized.
4. Run `scripts/validate_dataset.py`, then inspect duplicate and near-duplicate
   families before accepting a split.
5. Run error analysis on the frozen test split before and after expansion. Do
   not use test or calibration cases for authoring feedback.

## Current status

The active 8,508-case snapshot combines the original 1,008 project-authored
bootstrap cases with 7,500 selected and transformed cases from MASSIVE Thai,
BANKING77, and Hermes function calling. Imported labels are marked
`rule_reviewed`, not human-reviewed.

Current Score v4 work uses distance-aware Gaussian ordinal targets (`sigma=0.75`)
and direct RPS training/evaluation. The 1,008 bootstrap subset now covers risk
levels 0–4; the expanded dataset still has sparse level-1 coverage and needs
human review before production authorization use. See [`SCORE_V4.md`](SCORE_V4.md)
and [`DATASET_REVIEW_RUBRIC.md`](DATASET_REVIEW_RUBRIC.md).

Current verification commands:

```bash
python scripts/validate_dataset.py --root data/hf_expanded
python scripts/audit_hf_dataset.py --root data/hf_expanded
```

The current published snapshot reports 2,508 scenario families and zero audit
flags. The Dataset Card at [`../data/hf_expanded/README.md`](../data/hf_expanded/README.md)
is the canonical human-readable description of source composition, licenses,
transformations, split statistics, limitations, and Hugging Face loading
instructions.
