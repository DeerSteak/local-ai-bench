from scripts.workloads.accuracy_scoring import score_question_bank


def test_score_question_bank_aggregates_categories_and_extra_groups():
    questions = [
        {"id": "a", "category": "logic", "difficulty": "easy", "answer": "A"},
        {"id": "b", "category": "logic", "difficulty": "hard", "answer": "B"},
    ]

    result = score_question_bank(
        questions, {"a": "A", "b": None},
        lambda question, given: (
            given is not None, given == question["answer"],
            {"id": question["id"], "category": question["category"], "given": given},
        ),
        extra_groups=(("by_difficulty", "difficulty"),),
    )

    assert (result["correct"], result["answered"], result["accuracy_pct"]) == (1, 1, 50.0)
    assert result["by_category"]["logic"]["accuracy_pct"] == 50.0
    assert result["by_difficulty"]["easy"]["accuracy_pct"] == 100.0
    assert result["incorrect"] == [{"id": "b", "category": "logic", "given": None}]


def test_score_question_bank_handles_an_empty_bank():
    result = score_question_bank([], {}, lambda _question, _given: (False, False, {}))
    assert result["accuracy_pct"] == 0.0
    assert result["by_category"] == {}
