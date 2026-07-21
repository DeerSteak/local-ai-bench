from collections import Counter
import json

import pytest

from tool_benchmark import ToolBenchmark


# ── evaluate_question: call cases ──

def _q_call():
    return {
        "id": "tool_x", "category": "single_tool_call",
        "prompt": "Weather in Paris?",
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        "expected": {"call": True, "name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}},
    }


def test_evaluate_correct_call():
    calls = [{"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}]
    assert ToolBenchmark.evaluate_question(_q_call(), calls)["correct"] is True


def test_evaluate_wrong_tool_name():
    calls = [{"name": "get_time", "arguments": {"location": "Paris", "unit": "celsius"}}]
    assert ToolBenchmark.evaluate_question(_q_call(), calls)["correct"] is False


def test_evaluate_missing_required_argument():
    calls = [{"name": "get_weather", "arguments": {"location": "Paris"}}]
    assert ToolBenchmark.evaluate_question(_q_call(), calls)["correct"] is False


def test_evaluate_wrong_argument_value():
    calls = [{"name": "get_weather", "arguments": {"location": "London", "unit": "celsius"}}]
    assert ToolBenchmark.evaluate_question(_q_call(), calls)["correct"] is False


def test_evaluate_extra_argument_still_correct():
    # Loose match: extra keys beyond the expected set don't fail the match.
    calls = [{"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius", "verbose": True}}]
    assert ToolBenchmark.evaluate_question(_q_call(), calls)["correct"] is True


def test_evaluate_extra_argument_fails_in_strict_mode():
    question = _q_call()
    question["expected"]["strict_arguments"] = True
    calls = [{"name": "get_weather", "arguments": {
        "location": "Paris", "unit": "celsius", "verbose": True,
    }}]
    assert ToolBenchmark.evaluate_question(question, calls)["correct"] is False


def test_evaluate_multiple_calls_fails_even_when_first_is_correct():
    calls = [
        {"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}},
        {"name": "get_time", "arguments": {"city": "Paris"}},
    ]
    assert ToolBenchmark.evaluate_question(_q_call(), calls)["correct"] is False


def test_evaluate_numeric_string_coercion():
    q = {"expected": {"call": True, "name": "set_timer", "arguments": {"minutes": 10}}}
    calls = [{"name": "set_timer", "arguments": {"minutes": "10"}}]
    assert ToolBenchmark.evaluate_question(q, calls)["correct"] is True


def test_evaluate_float_string_coercion():
    q = {"expected": {"call": True, "name": "calculate_tip", "arguments": {"percent": 20.0}}}
    calls = [{"name": "calculate_tip", "arguments": {"percent": "20"}}]
    assert ToolBenchmark.evaluate_question(q, calls)["correct"] is True


def test_evaluate_incorrectly_declined_when_should_call():
    assert ToolBenchmark.evaluate_question(_q_call(), [])["correct"] is False
    assert ToolBenchmark.evaluate_question(_q_call(), None)["correct"] is False


def test_evaluate_incomplete_call_with_empty_expected_arguments_is_wrong():
    # Regression for tool_070/tool_089 (real bank questions with expected
    # arguments == {}): a completed-but-unparseable-JSON call must not be
    # scored correct just because its coerced-to-{} arguments happen to
    # match an empty expectation — the incomplete flag has to override it.
    q = {
        "id": "tool_z", "category": "single_tool_call",
        "prompt": "Show me the latest headlines.",
        "tools": [{"type": "function", "function": {"name": "get_news"}}],
        "expected": {"call": True, "name": "get_news", "arguments": {}, "strict_arguments": True},
    }
    calls = [{"name": "get_news", "arguments": {}, "incomplete": True}]
    assert ToolBenchmark.evaluate_question(q, calls)["correct"] is False


@pytest.mark.parametrize("bogus_arguments", [None, [], False, 0, ""])
def test_evaluate_non_dict_arguments_with_empty_expected_arguments_is_wrong(bogus_arguments):
    # Regression: `first.get("arguments") or {}` used Python truthiness, so
    # any falsy-but-valid-JSON value (null, [], false, 0, "") was silently
    # coerced to {} before ever reaching _args_match's isinstance(dict)
    # guard — scoring correct against an empty-expected-arguments question
    # even though the model never actually produced an empty object.
    q = {
        "id": "tool_z", "category": "single_tool_call",
        "prompt": "Show me the latest headlines.",
        "tools": [{"type": "function", "function": {"name": "get_news"}}],
        "expected": {"call": True, "name": "get_news", "arguments": {}, "strict_arguments": True},
    }
    calls = [{"name": "get_news", "arguments": bogus_arguments}]
    assert ToolBenchmark.evaluate_question(q, calls)["correct"] is False


def test_evaluate_nested_object_numeric_string_coercion():
    q = {"expected": {"call": True, "name": "add_invoice_items", "arguments": {
        "items": {"quantity": 2, "unit_price": 1.5},
    }}}
    calls = [{"name": "add_invoice_items", "arguments": {
        "items": {"quantity": "2", "unit_price": "1.5"},
    }}]
    assert ToolBenchmark.evaluate_question(q, calls)["correct"] is True


def test_evaluate_list_of_nested_objects_coercion():
    q = {"expected": {"call": True, "name": "add_invoice_items", "arguments": {
        "items": [{"quantity": 2, "unit_price": 1.5}, {"quantity": 1, "unit_price": 4.25}],
    }}}
    calls = [{"name": "add_invoice_items", "arguments": {
        "items": [{"quantity": "2", "unit_price": "1.5"}, {"quantity": "1", "unit_price": "4.25"}],
    }}]
    assert ToolBenchmark.evaluate_question(q, calls)["correct"] is True


def test_evaluate_nested_object_wrong_value_fails():
    q = {"expected": {"call": True, "name": "add_invoice_items", "arguments": {
        "items": {"quantity": 2, "unit_price": 1.5},
    }}}
    calls = [{"name": "add_invoice_items", "arguments": {
        "items": {"quantity": 3, "unit_price": 1.5},
    }}]
    assert ToolBenchmark.evaluate_question(q, calls)["correct"] is False


def test_evaluate_list_length_mismatch_fails():
    q = {"expected": {"call": True, "name": "add_invoice_items", "arguments": {
        "items": [{"quantity": 2}, {"quantity": 1}],
    }}}
    calls = [{"name": "add_invoice_items", "arguments": {"items": [{"quantity": 2}]}}]
    assert ToolBenchmark.evaluate_question(q, calls)["correct"] is False


def test_evaluate_nested_object_extra_key_fails_in_strict_mode():
    q = {"expected": {"call": True, "name": "add_invoice_items", "arguments": {
        "items": {"quantity": 2},
    }, "strict_arguments": True}}
    calls = [{"name": "add_invoice_items", "arguments": {
        "items": {"quantity": 2, "note": "gift"},
    }}]
    assert ToolBenchmark.evaluate_question(q, calls)["correct"] is False


def test_evaluate_boolean_does_not_match_numeric():
    q = {"expected": {"call": True, "name": "set_light", "arguments": {"on": True}}}
    calls = [{"name": "set_light", "arguments": {"on": 1}}]
    assert ToolBenchmark.evaluate_question(q, calls)["correct"] is False


def test_evaluate_numeric_does_not_match_boolean():
    q = {"expected": {"call": True, "name": "set_count", "arguments": {"count": 1}}}
    calls = [{"name": "set_count", "arguments": {"count": True}}]
    assert ToolBenchmark.evaluate_question(q, calls)["correct"] is False


def test_evaluate_boolean_matches_boolean():
    q = {"expected": {"call": True, "name": "set_light", "arguments": {"on": True}}}
    calls = [{"name": "set_light", "arguments": {"on": True}}]
    assert ToolBenchmark.evaluate_question(q, calls)["correct"] is True


def test_evaluate_unordered_list_matches_any_permutation():
    q = {"expected": {"call": True, "name": "search", "arguments": {
        "labels": ["bug", "regression"],
    }, "unordered_keys": ["labels"]}}
    calls = [{"name": "search", "arguments": {"labels": ["regression", "bug"]}}]
    assert ToolBenchmark.evaluate_question(q, calls)["correct"] is True


def test_evaluate_unordered_list_still_requires_same_elements():
    q = {"expected": {"call": True, "name": "search", "arguments": {
        "labels": ["bug", "regression"],
    }, "unordered_keys": ["labels"]}}
    calls = [{"name": "search", "arguments": {"labels": ["bug", "feature"]}}]
    assert ToolBenchmark.evaluate_question(q, calls)["correct"] is False


def test_evaluate_list_without_unordered_key_stays_positional():
    q = {"expected": {"call": True, "name": "plan_route", "arguments": {
        "waypoints": ["Boulder", "Vail"],
    }}}
    calls = [{"name": "plan_route", "arguments": {"waypoints": ["Vail", "Boulder"]}}]
    assert ToolBenchmark.evaluate_question(q, calls)["correct"] is False


def test_evaluate_unordered_list_of_objects():
    q = {"expected": {"call": True, "name": "add_invoice_items", "arguments": {
        "items": [{"description": "pens", "quantity": 2}, {"description": "notebook", "quantity": 1}],
    }, "unordered_keys": ["items"]}}
    calls = [{"name": "add_invoice_items", "arguments": {
        "items": [{"description": "notebook", "quantity": 1}, {"description": "pens", "quantity": 2}],
    }}]
    assert ToolBenchmark.evaluate_question(q, calls)["correct"] is True


def test_evaluate_unordered_overlapping_objects_finds_one_to_one_match():
    q = {"expected": {"call": True, "name": "update", "arguments": {
        "items": [{"x": 1}, {"x": 1, "y": 2}],
    }, "unordered_keys": ["items"]}}
    calls = [{"name": "update", "arguments": {
        "items": [{"x": 1, "y": 2}, {"x": 1}],
    }}]
    assert ToolBenchmark.evaluate_question(q, calls)["correct"] is True


# ── evaluate_question: decline cases ──

def _q_decline():
    return {"id": "tool_y", "category": "decline_no_fit", "expected": {"call": False}}


def test_evaluate_correctly_declined():
    assert ToolBenchmark.evaluate_question(_q_decline(), [])["correct"] is True
    assert ToolBenchmark.evaluate_question(_q_decline(), None)["correct"] is True


def test_evaluate_incorrectly_called_when_should_decline():
    calls = [{"name": "get_weather", "arguments": {"location": "Paris"}}]
    assert ToolBenchmark.evaluate_question(_q_decline(), calls)["correct"] is False


# ── rescore_partial_fn ──

def test_rescore_partial_parses_tool_call_json():
    q = _q_call()
    partial = '[{"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}]'
    assert ToolBenchmark.rescore_partial_fn(q, partial)["correct"] is True


def test_rescore_partial_unparseable_is_decline():
    q = _q_call()
    # Won't parse -> [] -> a decline, which is wrong for a call question.
    assert ToolBenchmark.rescore_partial_fn(q, "get_weat")["correct"] is False
    # For a decline question, unparseable partial text is a correct decline.
    assert ToolBenchmark.rescore_partial_fn(_q_decline(), "garbage")["correct"] is True


def test_rescore_partial_non_list_json_is_decline():
    # Valid JSON but not a list of calls -> treated as a decline.
    assert ToolBenchmark.rescore_partial_fn(_q_decline(), '{"name": "x"}')["correct"] is True


def test_rescore_partial_incomplete_call_is_not_a_decline_or_correct_call():
    partial = json.dumps([{"name": "get_weather", "arguments": {}, "incomplete": True}])
    assert ToolBenchmark.rescore_partial_fn(_q_decline(), partial)["correct"] is False
    assert ToolBenchmark.rescore_partial_fn(_q_call(), partial)["correct"] is False


# ── score ──

def _questions():
    return [
        {"id": "q1", "category": "single_tool_call", "expected": {"call": True, "name": "a", "arguments": {}}},
        {"id": "q2", "category": "single_tool_call", "expected": {"call": True, "name": "b", "arguments": {}}},
        {"id": "q3", "category": "decline_no_fit", "expected": {"call": False}},
    ]


def test_score_all_correct():
    answers = {"q1": {"correct": True}, "q2": {"correct": True}, "q3": {"correct": True}}
    result = ToolBenchmark.score(_questions(), answers)
    assert result["correct"] == 3
    assert result["total"] == 3
    assert result["accuracy_pct"] == 100.0
    assert result["incorrect"] == []


def test_score_partial_and_category_breakdown():
    answers = {"q1": {"correct": True}, "q2": {"correct": False}, "q3": None}
    result = ToolBenchmark.score(_questions(), answers)
    assert result["correct"] == 1
    assert result["answered"] == 2  # q3 unanswered (None)
    assert result["by_category"]["single_tool_call"] == {"correct": 1, "total": 2, "accuracy_pct": 50.0}
    assert result["by_category"]["decline_no_fit"] == {"correct": 0, "total": 1, "accuracy_pct": 0.0}


def test_score_incorrect_list_has_expected_entries():
    answers = {"q1": {"correct": True}, "q2": {"correct": False}, "q3": None}
    result = ToolBenchmark.score(_questions(), answers)
    ids = {entry["id"] for entry in result["incorrect"]}
    assert ids == {"q2", "q3"}
    q2_entry = next(e for e in result["incorrect"] if e["id"] == "q2")
    assert q2_entry == {"id": "q2", "category": "single_tool_call"}


def test_score_missing_answer_counts_incorrect_and_unanswered():
    result = ToolBenchmark.score(_questions(), {})
    assert result["correct"] == 0
    assert result["answered"] == 0
    assert len(result["incorrect"]) == 3


# ── load_questions against the real dataset ──

def test_load_questions_returns_well_formed_dataset():
    questions = ToolBenchmark.load_questions()
    assert len(questions) == 100
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))  # unique ids
    categories = Counter(q["category"] for q in questions)
    assert len(categories) == 20
    assert set(categories.values()) == {5}
    for q in questions:
        assert q["tools"] and isinstance(q["tools"], list)
        expected = q["expected"]
        assert "call" in expected
        if expected["call"]:
            assert expected["name"] and "arguments" in expected
            names = {t["function"]["name"] for t in q["tools"]}
            assert expected["name"] in names  # expected tool is actually offered
            if expected.get("strict_arguments"):
                assert set(expected["arguments"]) <= set(next(
                    t["function"]["parameters"]["properties"]
                    for t in q["tools"] if t["function"]["name"] == expected["name"]
                ))
        else:
            assert "name" not in expected  # a decline case names no tool
