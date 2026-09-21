from jevbro.questions import default_questions, detect_question_language


def test_default_questions_cover_all_primitives() -> None:
    questions = default_questions("en")
    assert set(questions) == {"action", "needs_review", "prohibited", "risk"}
    assert {q["type"] for q in questions.values()} == {"choice", "noul", "score"}


def test_language_detection_for_default_schema() -> None:
    assert detect_question_language("Please inspect git status") == "en"
    assert detect_question_language("ช่วยตรวจ git status ให้หน่อย") == "th"
