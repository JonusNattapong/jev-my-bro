from pathlib import Path
from unittest.mock import MagicMock

from jevbro.core import JevCore
from jevbro.feedback import FeedbackStore


def test_cache_hit_and_miss(tmp_path: Path) -> None:
    fake_agent = MagicMock()
    fake_agent.predict.return_value = {
        "answers": {
            "action": {"choice": "execute", "confidence": 0.95},
            "needs_review": {"noul": 0.1},
            "prohibited": {"noul": 0.05},
            "risk": {"score": 1.0},
        },
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }
    store = FeedbackStore(tmp_path / "test_feedback.sqlite3")
    core = JevCore(fake_agent, store, model_name="test-model", cache_size=2)

    # First call: Cache miss
    res1 = core.decide("pytest tests/test_rules.py")
    assert res1["cache_hit"] is False
    assert core.cache_hits == 0
    assert core.cache_misses == 1
    assert fake_agent.predict.call_count == 1

    # Second call: Exact same command -> Cache hit!
    res2 = core.decide("pytest tests/test_rules.py")
    assert res2["cache_hit"] is True
    assert core.cache_hits == 1
    assert core.cache_misses == 1
    assert fake_agent.predict.call_count == 1

    # Third call: Whitespace-normalized -> Cache hit!
    res3 = core.decide("   pytest   tests/test_rules.py   \n")
    assert res3["cache_hit"] is True
    assert core.cache_hits == 2
    assert core.cache_misses == 1
    assert fake_agent.predict.call_count == 1

    # Check cache statistics
    stats = core.cache_stats()
    assert stats["size"] == 1
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.6667


def test_cache_size_limit(tmp_path: Path) -> None:
    fake_agent = MagicMock()
    fake_agent.predict.return_value = {
        "answers": {
            "action": {"choice": "execute", "confidence": 0.95},
            "needs_review": {"noul": 0.1},
            "prohibited": {"noul": 0.05},
            "risk": {"score": 1.0},
        },
        "usage": {},
    }
    store = FeedbackStore(tmp_path / "test_feedback.sqlite3")
    core = JevCore(fake_agent, store, model_name="test-model", cache_size=2)

    # Cache size management - verify it respects limit
    core.decide("x1")
    core.decide("x2")
    assert len(core.cache) <= 2

    core.decide("x3")
    assert len(core.cache) <= 2

    # Stats should reflect operations
    stats = core.cache_stats()
    assert stats["size"] <= 2
    assert stats["max_size"] == 2
