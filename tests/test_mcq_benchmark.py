from mcq_benchmark import MCQBenchmark


# ── build_prompt ──

def test_build_prompt_includes_question_and_all_choices():
    q = {"prompt": "What is 2+2?", "choices": {"A": "3", "B": "4", "C": "5", "D": "6"}}
    prompt = MCQBenchmark.build_prompt(q)
    assert "What is 2+2?" in prompt
    assert "A. 3" in prompt
    assert "B. 4" in prompt
    assert "C. 5" in prompt
    assert "D. 6" in prompt


# ── parse_answer ──

def test_parse_answer_bare_letter():
    assert MCQBenchmark.parse_answer("B", {"A", "B", "C", "D"}) == "B"


def test_parse_answer_lowercase_letter():
    assert MCQBenchmark.parse_answer("b", {"A", "B", "C", "D"}) == "B"


def test_parse_answer_letter_with_punctuation():
    assert MCQBenchmark.parse_answer("B.", {"A", "B", "C", "D"}) == "B"
    assert MCQBenchmark.parse_answer("(B)", {"A", "B", "C", "D"}) == "B"


def test_parse_answer_reasoning_before_letter():
    text = "Gold's chemical symbol comes from Latin aurum, so the answer is B."
    assert MCQBenchmark.parse_answer(text, {"A", "B", "C", "D"}) == "B"


def test_parse_answer_skips_stray_letters_not_in_valid_choices():
    # "A" appears in "As" but isn't a standalone word; the real answer is C.
    text = "As an assistant, I'd say the correct choice is C."
    assert MCQBenchmark.parse_answer(text, {"A", "B", "C", "D"}) == "C"


def test_parse_answer_returns_none_when_no_valid_letter_found():
    assert MCQBenchmark.parse_answer("I'm not sure.", {"A", "B", "C", "D"}) is None


def test_parse_answer_returns_none_for_empty_response():
    assert MCQBenchmark.parse_answer("", {"A", "B", "C", "D"}) is None
    assert MCQBenchmark.parse_answer(None, {"A", "B", "C", "D"}) is None


def test_parse_answer_ignores_letter_outside_valid_choices_for_this_question():
    # Question only has A/B/C — a stray standalone "D" elsewhere shouldn't match.
    text = "D is not an option here, so I'll go with A."
    assert MCQBenchmark.parse_answer(text, {"A", "B", "C"}) == "A"


# ── score ──

def _questions():
    return [
        {"id": "q1", "category": "science", "answer": "B"},
        {"id": "q2", "category": "science", "answer": "A"},
        {"id": "q3", "category": "history", "answer": "C"},
    ]


def test_score_all_correct():
    answers = {"q1": "B", "q2": "A", "q3": "C"}
    result = MCQBenchmark.score(_questions(), answers)
    assert result["correct"] == 3
    assert result["total"] == 3
    assert result["accuracy_pct"] == 100.0
    assert result["incorrect"] == []


def test_score_partial_and_category_breakdown():
    answers = {"q1": "B", "q2": "D", "q3": None}
    result = MCQBenchmark.score(_questions(), answers)
    assert result["correct"] == 1
    assert result["answered"] == 2  # q3 unanswered (None)
    assert result["accuracy_pct"] == round(100 / 3, 1)
    assert result["by_category"]["science"] == {"correct": 1, "total": 2, "accuracy_pct": 50.0}
    assert result["by_category"]["history"] == {"correct": 0, "total": 1, "accuracy_pct": 0.0}


def test_score_incorrect_list_has_expected_entries():
    answers = {"q1": "B", "q2": "D", "q3": None}
    result = MCQBenchmark.score(_questions(), answers)
    ids = {entry["id"] for entry in result["incorrect"]}
    assert ids == {"q2", "q3"}
    q2_entry = next(e for e in result["incorrect"] if e["id"] == "q2")
    assert q2_entry == {"id": "q2", "category": "science", "given": "D", "expected": "A"}


def test_score_missing_answer_counts_as_incorrect_and_unanswered():
    answers = {}
    result = MCQBenchmark.score(_questions(), answers)
    assert result["correct"] == 0
    assert result["answered"] == 0
    assert len(result["incorrect"]) == 3


# ── load_questions against the real dataset ──

def test_load_questions_returns_well_formed_dataset():
    questions = MCQBenchmark.load_questions()
    assert len(questions) > 0
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))  # unique ids
    for q in questions:
        assert q["answer"] in q["choices"]
