"""Shared question-bank aggregation for accuracy workloads."""

from collections.abc import Callable
from typing import Any


def validate_question_bank(questions: list[dict], required_fields=()) -> list[dict]:
    if not isinstance(questions, list):
        raise ValueError("question bank must be a JSON array")
    required = {"id", "category", *required_fields}
    seen_ids = set()
    for index, question in enumerate(questions):
        if not isinstance(question, dict):
            raise ValueError(f"question {index} must be a JSON object")
        missing = sorted(required - question.keys())
        if missing:
            raise ValueError(f"question {index} is missing required fields: {', '.join(missing)}")
        question_id = question["id"]
        if question_id in seen_ids:
            raise ValueError(f"duplicate question id: {question_id}")
        seen_ids.add(question_id)
    return questions


def score_question_bank(questions: list[dict], answers: dict,
                        evaluate: Callable[[dict, Any], tuple[bool, bool, dict]],
                        extra_groups: tuple[tuple[str, str], ...] = ()) -> dict:
    validate_question_bank(questions)
    by_category: dict[str, dict] = {}
    extra = {result_key: {} for result_key, _ in extra_groups}
    incorrect = []
    all_results = []
    correct = 0
    answered = 0
    for question in questions:
        given = answers.get(question["id"])
        is_answered, is_correct, entry = evaluate(question, given)
        category = by_category.setdefault(question["category"], {"correct": 0, "total": 0})
        category["total"] += 1
        answered += int(is_answered)
        correct += int(is_correct)
        category["correct"] += int(is_correct)
        all_results.append({**entry, "correct": is_correct})
        if not is_correct:
            incorrect.append(entry)
        for result_key, question_key in extra_groups:
            if question_key not in question:
                continue
            value = question[question_key]
            group = extra[result_key].setdefault(value, {"correct": 0, "total": 0})
            group["total"] += 1
            group["correct"] += int(is_correct)
    for breakdown in (by_category, *extra.values()):
        for group in breakdown.values():
            group["accuracy_pct"] = round(100 * group["correct"] / group["total"], 1) \
                if group["total"] else 0.0
    total = len(questions)
    return {
        "correct": correct, "total": total, "answered": answered,
        "accuracy_pct": round(100 * correct / total, 1) if total else 0.0,
        "by_category": by_category, **extra, "incorrect": incorrect, "all": all_results,
    }
