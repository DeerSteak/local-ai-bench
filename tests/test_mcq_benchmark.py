import pytest

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
    assert MCQBenchmark.parse_answer(r"\boxed{B}", {"A", "B", "C", "D"}) == "B"


def test_parse_answer_reasoning_before_letter():
    text = "Gold's chemical symbol comes from Latin aurum, so the answer is B."
    assert MCQBenchmark.parse_answer(text, {"A", "B", "C", "D"}) == "B"


def test_parse_answer_explicit_statement_forms():
    choices = {"A", "B", "C", "D"}
    assert MCQBenchmark.parse_answer("Final answer: D", choices) == "D"
    assert MCQBenchmark.parse_answer("I choose A", choices) == "A"
    assert MCQBenchmark.parse_answer("C is correct", choices) == "C"


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


def test_parse_answer_takes_last_letter_when_multiple_are_mentioned():
    # A model walking through options by letter before concluding — the
    # final stated answer (C) must win over earlier-rejected options (A, B).
    text = "A is wrong because it's silver. B is wrong too. The answer is C."
    assert MCQBenchmark.parse_answer(text, {"A", "B", "C", "D"}) == "C"


def test_parse_answer_explicit_answer_beats_later_rejected_option():
    text = "The answer is C, not D, since gold's symbol is Au, not Go."
    assert MCQBenchmark.parse_answer(text, {"A", "B", "C", "D"}) == "C"


def test_parse_answer_negated_choice_followed_by_correction():
    text = "The answer is not A, it's B."
    assert MCQBenchmark.parse_answer(text, {"A", "B", "C", "D"}) == "B"


@pytest.mark.parametrize("response", [
    "A is not correct, D is.",
    "A is not right, D is.",
    "A isn't correct, D is.",
    "A isn’t right, D is.",
    "A is wrong. D is right.",
    "A is incorrect; D is correct.",
])
def test_parse_answer_rejected_then_affirmed_choice(response):
    assert MCQBenchmark.parse_answer(response, {"A", "B", "C", "D"}) == "D"


@pytest.mark.parametrize(("case", "response", "answer_key", "parsed", "correct"), [
    ("mcq_126/phi4-mini", "C. A 5% decrease", "A", "C", False),
    ("mcq_126/mistral-7b-q4", " C. A 5% decrease", "A", "C", False),
    ("mcq_140/phi4-mini", "B. Raises it to 100 A", "A", "B", False),
    ("mcq_140/phi4-14b", "C. Halves it to 5 A\n\nThis leaves 5 amperes.", "A", "C", False),
    ("mcq_111/phi4-mini", "C. A and 7", "C", "C", True),
    ("mcq_111/phi4-14b", "C\n\nTherefore, you must turn over the cards A and 7.", "C", "C", True),
    ("mcq_116/phi4-14b", "C\n\nA cube has 12 edges, so the result is 12.", "C", "C", True),
])
def test_parse_answer_audited_mcq_verdict_flips(case, response, answer_key, parsed, correct):
    assert case
    assert MCQBenchmark.parse_answer(response, {"A", "B", "C", "D"}) == parsed
    assert (parsed == answer_key) is correct


def test_parse_answer_later_explicit_correction_beats_leading_choice():
    text = "C. 21%. After recalculating, the correct answer is B."
    assert MCQBenchmark.parse_answer(text, {"A", "B", "C", "D"}) == "B"


@pytest.mark.parametrize(("response", "expected"), [
    ("C. Rechecking: the correct answer is B. 42%", "B"),
    ("B. Rechecking. So, the correct answer is: A. 7", "A"),
])
def test_parse_answer_preserves_audited_explicit_self_corrections(response, expected):
    assert MCQBenchmark.parse_answer(response, {"A", "B", "C", "D"}) == expected


def test_parse_answer_single_distinct_fallback_letter_can_repeat():
    text = "Considering C carefully, the evidence still supports C."
    assert MCQBenchmark.parse_answer(text, {"A", "B", "C", "D"}) == "C"


def test_parse_answer_ambiguous_unstructured_letters_returns_none():
    text = "Both C and A may work."
    assert MCQBenchmark.parse_answer(text, {"A", "B", "C", "D"}) is None


def test_parse_answer_does_not_convert_boxed_choice_value_using_answer_key():
    assert MCQBenchmark.parse_answer(r"$\boxed{452}$", {"A", "B", "C", "D"}) is None


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


def test_score_all_list_covers_every_question_including_correct_ones():
    answers = {"q1": "B", "q2": "D", "q3": None}
    result = MCQBenchmark.score(_questions(), answers)
    assert {e["id"] for e in result["all"]} == {"q1", "q2", "q3"}
    q1_entry = next(e for e in result["all"] if e["id"] == "q1")
    assert q1_entry == {"id": "q1", "category": "science", "given": "B", "expected": "B", "correct": True}
    q2_entry = next(e for e in result["all"] if e["id"] == "q2")
    assert q2_entry == {"id": "q2", "category": "science", "given": "D", "expected": "A", "correct": False}


# ── load_questions against the real dataset ──

def test_load_questions_returns_well_formed_dataset():
    questions = MCQBenchmark.load_questions()
    assert len(questions) > 0
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))  # unique ids
    for q in questions:
        assert q["answer"] in q["choices"]
