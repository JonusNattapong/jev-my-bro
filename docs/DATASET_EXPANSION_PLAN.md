# Dataset expansion plan

The checked-in dataset currently contains 1,008 cases (576 train, 144 validation,
144 calibration, 144 test). This plan defines the acceptance gates for expanding
it to 5,000-10,000 cases without turning the corpus into template permutations.

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
   conflicting instructions, and score targets. Keep probability targets
   normalized.
4. Run `scripts/validate_dataset.py`, then inspect duplicate and near-duplicate
   families before accepting a split.
5. Run error analysis on the frozen test split before and after expansion. Do
   not use test or calibration cases for authoring feedback.

This file is a specification only. No synthetic cases are added until they have
been authored and reviewed; the current 1,008-case source of truth remains
unchanged.
