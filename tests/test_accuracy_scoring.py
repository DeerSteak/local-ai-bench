import json

import pytest

from scripts.workloads.accuracy_scoring import score_question_bank, validate_question_bank
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


@pytest.mark.parametrize(("question", "missing"), [
    ({"category": "logic"}, "id"),
    ({"id": "q1"}, "category"),
])
def test_score_question_bank_rejects_missing_required_fields(question, missing):
    with pytest.raises(ValueError, match=rf"question 0.*{missing}"):
        score_question_bank(
            [question], {}, lambda *_args: (False, False, {}),
        )


def test_score_question_bank_skips_missing_extra_group_after_measurement():
    result = score_question_bank(
        [{"id": "q1", "category": "logic"}], {},
        lambda *_args: (False, False, {"id": "q1"}),
        extra_groups=(("by_difficulty", "difficulty"),),
    )
    assert result["by_difficulty"] == {}


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
    question = {
        "id": "duplicate", "category": "first", "prompt": "p", "answer": "A",
        "choices": {"A": "a"}, "tools": [], "expected": {}, "function_name": "f",
        "visible_tests": [], "hidden_tests": [],
    }
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([question, {**question, "category": "second"}]))
    with pytest.raises(ValueError, match="duplicate question id: duplicate"):
        loader(path)


@pytest.mark.parametrize(("loader", "question", "missing"), [
    (MCQBenchmark.load_questions, {"id": "q", "category": "c"}, "answer, choices, prompt"),
    (MathBenchmark.load_questions, {"id": "q", "category": "c"}, "answer, prompt"),
    (ToolBenchmark.load_questions, {"id": "q", "category": "c"}, "expected, prompt, tools"),
    (CodeBenchmark.load_questions, {"id": "q", "category": "c"},
     "hidden_tests, prompt, visible_tests"),
])
def test_accuracy_loaders_reject_missing_workload_fields(loader, question, missing, tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([question]))
    with pytest.raises(ValueError, match=missing):
        loader(path)


@pytest.mark.parametrize("payload", [{}, [None]])
def test_validate_question_bank_rejects_invalid_json_shapes(payload):
    with pytest.raises(ValueError):
        validate_question_bank(payload)


def test_code_loader_requires_a_callable_name(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([{
        "id": "q", "category": "c", "prompt": "p",
        "visible_tests": [], "hidden_tests": [],
    }]))
    with pytest.raises(ValueError, match="function_name or class_name"):
        CodeBenchmark.load_questions(path)
