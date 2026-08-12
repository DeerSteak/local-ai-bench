import json

import pytest

from scripts.workloads.accuracy_scoring import score_question_bank
from scripts.workloads.code_benchmark import CodeBenchmark
from scripts.workloads.math_benchmark import MathBenchmark
from scripts.workloads.mcq_benchmark import MCQBenchmark
from scripts.workloads.tool_benchmark import ToolBenchmark


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


def test_score_question_bank_preserves_incorrect_entry_shape():
    question = {"id": "q2", "category": "science", "answer": "A"}
    entry = {"id": "q2", "category": "science", "given": "C", "expected": "A"}

    result = score_question_bank(
        [question], {"q2": "C"}, lambda _question, _given: (True, False, entry),
    )

    assert result["incorrect"] == [entry]
    assert "correct" not in result["incorrect"][0]
    assert result["all"] == [{**entry, "correct": False}]


def test_score_question_bank_does_not_mutate_scorer_entry():
    entry = {"id": "q1", "category": "science"}

    score_question_bank(
        [{"id": "q1", "category": "science"}], {"q1": "A"},
        lambda _question, _given: (True, True, entry),
    )

    assert entry == {"id": "q1", "category": "science"}


@pytest.mark.parametrize(("question", "extra_groups", "missing"), [
    ({"category": "logic"}, (), "id"),
    ({"id": "q1"}, (), "category"),
    ({"id": "q1", "category": "logic"}, (("by_difficulty", "difficulty"),), "difficulty"),
])
def test_score_question_bank_rejects_missing_required_fields(question, extra_groups, missing):
    with pytest.raises(ValueError, match=rf"question 0.*{missing}"):
        score_question_bank(
            [question], {}, lambda *_args: (False, False, {}), extra_groups=extra_groups,
        )


def test_score_question_bank_rejects_duplicate_question_ids_before_evaluation():
    questions = [
        {"id": "duplicate", "category": "first"},
        {"id": "duplicate", "category": "second"},
    ]
    evaluated = []
    with pytest.raises(ValueError, match="duplicate question id: duplicate"):
        score_question_bank(
            questions, {"duplicate": "A"},
            lambda question, _given: evaluated.append(question) or (True, True, {}),
        )
    assert evaluated == []


@pytest.mark.parametrize("loader", [
    MCQBenchmark.load_questions, MathBenchmark.load_questions,
    CodeBenchmark.load_questions, ToolBenchmark.load_questions,
])
def test_accuracy_loaders_reject_duplicate_ids_before_inference(loader, tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([
        {"id": "duplicate", "category": "first"},
        {"id": "duplicate", "category": "second"},
    ]))
    with pytest.raises(ValueError, match="duplicate question id: duplicate"):
        loader(path)
