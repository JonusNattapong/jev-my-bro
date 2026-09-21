# jev-my-bro typed decision dataset

Static dataset authored for this repository. There is intentionally no dataset generator checked into the project.

Each JSONL row is one operation-governance case containing four Laya-compatible typed questions: action (choice), needs_review (noul), prohibited (noul), and risk (score).

Targets are probability distributions rather than only hard labels. The calibration split is reserved only for fitting temperatures after training, and the test split remains untouched until final evaluation.

Summary:
{
  "train": {
    "cases": 576,
    "decisions": 2304,
    "languages": {
      "en": 384,
      "th": 192
    },
    "action_labels": {
      "execute": 192,
      "ask_user": 192,
      "reject": 192
    }
  },
  "validation": {
    "cases": 144,
    "decisions": 576,
    "languages": {
      "th": 48,
      "en": 96
    },
    "action_labels": {
      "execute": 48,
      "ask_user": 48,
      "reject": 48
    }
  },
  "calibration": {
    "cases": 144,
    "decisions": 576,
    "languages": {
      "en": 96,
      "th": 48
    },
    "action_labels": {
      "execute": 48,
      "ask_user": 48,
      "reject": 48
    }
  },
  "test": {
    "cases": 144,
    "decisions": 576,
    "languages": {
      "en": 96,
      "th": 48
    },
    "action_labels": {
      "execute": 48,
      "ask_user": 48,
      "reject": 48
    }
  }
}

The English cases come from the repository's original static v0.1 bootstrap set. Thai cases are independent static examples and include lexical traps where words such as API_KEY, backdoor, deletion, authentication, and production can appear in legitimate read-only or test contexts.

This is bootstrap data for model development, not a production authorization policy. Add reviewed real operational decisions, ambiguous cases, out-of-distribution cases, and outcome feedback before using it as an enforcement gate.
