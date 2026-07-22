import pytest

from math_benchmark import MathBenchmark


# ── build_prompt ──

def test_build_prompt_includes_question_text():
    q = {"prompt": "What is 2+2?"}
    prompt = MathBenchmark.build_prompt(q)
    assert "What is 2+2?" in prompt
    assert "numeric answer" in prompt.lower()


# ── parse_answer ──

def test_parse_answer_bare_integer():
    assert MathBenchmark.parse_answer("936") == 936.0


def test_parse_answer_bare_decimal():
    assert MathBenchmark.parse_answer("7.7") == 7.7


def test_parse_answer_negative_number():
    assert MathBenchmark.parse_answer("-4") == -4.0


def test_parse_answer_number_with_thousands_commas():
    assert MathBenchmark.parse_answer("1,234") == 1234.0


def test_parse_answer_number_with_percent_sign():
    assert MathBenchmark.parse_answer("25%") == 25.0


def test_parse_answer_takes_last_number_after_reasoning():
    text = "347 + 589 = 936, so the answer is 936."
    assert MathBenchmark.parse_answer(text) == 936.0


def test_parse_answer_takes_last_number_when_intermediate_steps_shown():
    # Final stated answer (7) should win over the intermediate values (10, 3).
    text = "First, 10 - 3 = 7."
    assert MathBenchmark.parse_answer(text) == 7.0


def test_parse_answer_prefers_conclusion_result_over_trailing_context_number():
    text = "Therefore, 100! has 48 trailing zeros when written in base 12."
    assert MathBenchmark.parse_answer(text) == 48.0


def test_parse_answer_math_053_ignores_mid_sentence_so():
    text = (
        "To find the required score, calculate the total needed for an average of 82.\n\n"
        "There are 5 tests, so the total score needed is 82 * 5 = 410.\n\n"
        "The first 4 tests total 72 + 84 + 91 + 78 = 325.\n\n"
        "The required score on the 5th test is 410 - 325 = 85."
    )
    assert MathBenchmark.parse_answer(text) == 85.0


def test_parse_answer_math_072_uses_same_clause_expression_result():
    text = (
        "There are 2 choices for the first letter and 4 choices for each remaining letter. "
        "So the total number of strings is 2 * 4 * 4 * 4 * 4 = 2 * 4^4 = 2 * 256 = 512."
    )
    assert MathBenchmark.parse_answer(text) == 512.0


@pytest.mark.parametrize(("text", "expected"), [
    (
        "6\n\nHere's the reasoning:\n\n2. So, the distance is 10 - (12/2) = 10 - 6 = 4.\n"
        "5. The radius is 10, so the perpendicular distance is 10 - 2 = 6.",
        6.0,
    ),
    (
        "3.6\n\nHere's the reasoning:\n\nThe variance uses n = 20 and p = 0.3, "
        "so the variance is 20 * 0.3 * (1 - 0.3) = 3.6.",
        3.6,
    ),
])
def test_parse_answer_preserves_fresh_run_leading_numeric_answers(text, expected):
    assert MathBenchmark.parse_answer(text) == expected


def test_parse_answer_same_clause_result_beats_later_context_number():
    text = "So the total is 5 * 1 = 5. That is within the limit of 10 items."
    assert MathBenchmark.parse_answer(text) == 5.0


def test_parse_answer_accepts_completed_percentage_conclusion():
    text = "Therefore, the result is 25%. A later note mentions 10 items."
    assert MathBenchmark.parse_answer(text) == 25.0


def test_parse_answer_accepts_percentage_result_after_equals():
    text = "So the result is 50 / 2 = 25% . A later note mentions 10 items."
    assert MathBenchmark.parse_answer(text) == 25.0


def test_parse_answer_later_conclusion_overrides_leading_numeric_answer():
    text = "6\n\nAfter rechecking the calculation. Therefore, the result is 8."
    assert MathBenchmark.parse_answer(text) == 8.0


def test_parse_answer_leading_number_requires_line_boundary():
    text = "72 students were surveyed. The final result is 91."
    assert MathBenchmark.parse_answer(text) == 91.0
    assert MathBenchmark._LEADING_NUMBER_RE.match(text) is None


@pytest.mark.parametrize(("text", "expected"), [
    ("5\n\nThe initial guess was wrong. Solving gives x = 8.", 8.0),
    ("35\n\nThe recurrence eventually gives a_5 = 202.", 202.0),
    ("75.0000\n\nThe unfinished recalculation times out at 0.", 0.0),
])
def test_parse_answer_uncorroborated_leading_number_does_not_override_fallback(text, expected):
    assert MathBenchmark.parse_answer(text) == expected


def test_parse_answer_leading_number_can_be_corroborated_by_final_equality():
    text = (
        "42\n\nHere is my reasoning: We start with 100 items, remove 58, giving "
        "100 - 58 = 42 remaining items in the warehouse (out of an original 200 stock)."
    )
    assert MathBenchmark.parse_answer(text) == 42.0


def test_parse_answer_leading_number_can_match_percentage_equality_result():
    text = "25\n\nThe calculation is 50 / 2 = 25% within a set of 10 categories."
    assert MathBenchmark.parse_answer(text) == 25.0


def test_parse_answer_unfinished_equality_does_not_corroborate_leading_number():
    text = "42\n\nThe unfinished calculation says 100 - 58 = 42 + more, then mentions 200."
    assert MathBenchmark.parse_answer(text) == 200.0


@pytest.mark.parametrize("operator", ["+", "-", "*", "/", "^"])
def test_parse_answer_does_not_accept_first_expression_operand(operator):
    text = f"So the result is 20 {operator} unknown. The final value is 7."
    assert MathBenchmark.parse_answer(text) == 7.0


def test_parse_answer_uses_result_after_equals_not_first_operand():
    text = "So the result is 20 = 19. A later constraint mentions 7."
    assert MathBenchmark.parse_answer(text) == 19.0


def test_parse_answer_does_not_backtrack_to_partial_number():
    text = "So the result is 20 * unknown. The final value is 7."
    assert MathBenchmark.parse_answer(text) == 7.0


def test_parse_answer_rejects_unfinished_result_after_equals():
    text = "So the result is 2 * 3 = 6 + unfinished. The final value is 7."
    assert MathBenchmark.parse_answer(text) == 7.0


def test_parse_answer_expression_clause_stops_at_newline():
    text = "So the result is 2 * 3\nThe final value is 7."
    assert MathBenchmark.parse_answer(text) == 7.0


def test_parse_answer_handles_expression_clause_at_end_of_response():
    assert MathBenchmark.parse_answer("So the result is 2 * 3") == 3.0


def test_parse_answer_leaves_audited_unfinished_math_104_partial_conservative():
    text = "The result is 66.6667%. As a decimal this is 0.6667. Recalculate"
    assert MathBenchmark.parse_answer(text) == 0.6667


def test_parse_answer_does_not_evaluate_expression_only_response():
    assert MathBenchmark.parse_answer("2/e") == 2.0


def test_parse_answer_structured_candidates_follow_response_order():
    text = r"\boxed{48}. Wait, the answer is actually 52."
    assert MathBenchmark.parse_answer(text) == 52.0
    text = r"The answer is 48. Rechecking gives \boxed{52}."
    assert MathBenchmark.parse_answer(text) == 52.0


def test_parse_answer_explicit_and_boxed_values():
    assert MathBenchmark.parse_answer("Final answer: 936") == 936.0
    assert MathBenchmark.parse_answer(r"\boxed{936}") == 936.0


def test_parse_answer_returns_none_when_no_number_found():
    assert MathBenchmark.parse_answer("I'm not sure.") is None


def test_parse_answer_returns_none_for_empty_response():
    assert MathBenchmark.parse_answer("") is None
    assert MathBenchmark.parse_answer(None) is None


def test_parse_answer_returns_none_for_bare_minus_sign():
    assert MathBenchmark.parse_answer("-") is None


def test_to_float_defensively_rejects_non_numeric_input():
    assert MathBenchmark._to_float("not-a-number") is None


# ── score ──

def _questions():
    return [
        {"id": "q1", "category": "arithmetic", "answer": 936, "tolerance": 0},
        {"id": "q2", "category": "arithmetic", "answer": 456, "tolerance": 0},
        {"id": "q3", "category": "probability", "answer": 7.7, "tolerance": 0.1},
    ]


def test_score_all_correct():
    answers = {"q1": 936.0, "q2": 456.0, "q3": 7.7}
    result = MathBenchmark.score(_questions(), answers)
    assert result["correct"] == 3
    assert result["total"] == 3
    assert result["accuracy_pct"] == 100.0
    assert result["incorrect"] == []


def test_score_within_tolerance_counts_as_correct():
    # q3's tolerance is 0.1 — 7.65 is within it, 7.8 is not.
    answers = {"q1": 936.0, "q2": 456.0, "q3": 7.65}
    result = MathBenchmark.score(_questions(), answers)
    assert result["correct"] == 3

    answers2 = {"q1": 936.0, "q2": 456.0, "q3": 7.85}
    result2 = MathBenchmark.score(_questions(), answers2)
    assert result2["correct"] == 2


def test_score_exact_tolerance_zero_requires_exact_match():
    answers = {"q1": 936.5, "q2": 456.0, "q3": 7.7}
    result = MathBenchmark.score(_questions(), answers)
    assert result["correct"] == 2
    assert any(e["id"] == "q1" for e in result["incorrect"])


def test_score_partial_and_category_breakdown():
    answers = {"q1": 936.0, "q2": 100.0, "q3": None}
    result = MathBenchmark.score(_questions(), answers)
    assert result["correct"] == 1
    assert result["answered"] == 2  # q3 unanswered (None)
    assert result["accuracy_pct"] == round(100 / 3, 1)
    assert result["by_category"]["arithmetic"] == {"correct": 1, "total": 2, "accuracy_pct": 50.0}
    assert result["by_category"]["probability"] == {"correct": 0, "total": 1, "accuracy_pct": 0.0}


def test_score_incorrect_list_has_expected_entries():
    answers = {"q1": 936.0, "q2": 100.0, "q3": None}
    result = MathBenchmark.score(_questions(), answers)
    ids = {entry["id"] for entry in result["incorrect"]}
    assert ids == {"q2", "q3"}
    q2_entry = next(e for e in result["incorrect"] if e["id"] == "q2")
    assert q2_entry == {"id": "q2", "category": "arithmetic", "given": 100.0, "expected": 456}


def test_score_missing_answer_counts_as_incorrect_and_unanswered():
    answers = {}
    result = MathBenchmark.score(_questions(), answers)
    assert result["correct"] == 0
    assert result["answered"] == 0
    assert len(result["incorrect"]) == 3


def test_score_all_list_covers_every_question_including_correct_ones():
    answers = {"q1": 936.0, "q2": 100.0, "q3": None}
    result = MathBenchmark.score(_questions(), answers)
    assert {e["id"] for e in result["all"]} == {"q1", "q2", "q3"}
    q1_entry = next(e for e in result["all"] if e["id"] == "q1")
    assert q1_entry["correct"] is True
    assert q1_entry["given"] == 936.0
    q2_entry = next(e for e in result["all"] if e["id"] == "q2")
    assert q2_entry == {"id": "q2", "category": "arithmetic", "given": 100.0, "expected": 456, "correct": False}


def test_score_defaults_tolerance_to_zero_when_absent():
    questions = [{"id": "q1", "category": "arithmetic", "answer": 5}]
    result = MathBenchmark.score(questions, {"q1": 5.0})
    assert result["correct"] == 1
    result2 = MathBenchmark.score(questions, {"q1": 5.4})
    assert result2["correct"] == 0


# ── load_questions against the real dataset ──

def test_load_questions_returns_well_formed_dataset():
    questions = MathBenchmark.load_questions()
    assert len(questions) > 0
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))  # unique ids
    for q in questions:
        assert isinstance(q["answer"], (int, float))
        assert isinstance(q["tolerance"], (int, float))
        assert q["tolerance"] >= 0
