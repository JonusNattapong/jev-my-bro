from __future__ import annotations

import pytest

from jevbro.evaluate import macro_recall


def test_macro_recall_uses_only_supported_levels() -> None:
    assert macro_recall(
        {
            "0": {"support": 2, "recall": 1.0},
            "1": {"support": 2, "recall": 0.5},
            "2": {"support": 0, "recall": None},
        }
    ) == pytest.approx(0.75)


def test_selected_and_argmax_reports_can_be_compared_independently() -> None:
    selected = {
        "0": {"support": 2, "recall": 1.0},
        "1": {"support": 2, "recall": 0.0},
    }
    argmax = {
        "0": {"support": 2, "recall": 0.5},
        "1": {"support": 2, "recall": 1.0},
    }
    assert macro_recall(selected) == pytest.approx(0.5)
    assert macro_recall(argmax) == pytest.approx(0.75)
