"""Assert the runtime asks exactly what the training corpus asked.

A prompt change silently invalidates every published metric, so this is checked against the
data itself rather than against a copy of the expected string.
"""

import json
from pathlib import Path

from jevbro.questions import default_questions

ROOT = Path(__file__).parents[1]


def _corpus_question_block() -> dict:
    blocks = set()
    for split in ("train", "validation", "calibration", "test"):
        path = ROOT / "data" / "th_curated_1200" / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                blocks.add(json.dumps(json.loads(line)["questions"], sort_keys=True))
    assert len(blocks) == 1, f"corpus must carry one question block, found {len(blocks)}"
    return json.loads(blocks.pop())


def test_runtime_questions_match_the_training_corpus_exactly() -> None:
    """The prompt is model input, so runtime and training must not drift apart."""
    assert default_questions("en") == _corpus_question_block()


def test_question_language_does_not_change_the_prompt() -> None:
    """`language` describes the request, not the question; a translated prompt is out of distribution."""
    assert default_questions("th") == default_questions("en")
    assert default_questions() == default_questions("th")
