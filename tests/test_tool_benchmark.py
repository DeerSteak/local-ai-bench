from collections import Counter
import json

import pytest

import config
from tool_benchmark import ToolBenchmark


# ── evaluate_question: call cases ──

def _q_call():
    return {
        "id": "tool_x", "category": "single_tool_call",
        "prompt": "Weather in Paris?",
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        "expected": {"call": True, "name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}},
    }


def _real_question(question_id):
    return next(q for q in ToolBenchmark.load_questions() if q["id"] == question_id)


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


@pytest.mark.parametrize(("case", "question_id", "key", "given"), [
    ("llama3.2-3b-q4/tool_010", "tool_010", "title", "Dentist Appointment"),
    ("llama3.1-8b-q4/tool_010", "tool_010", "title", "Dentist appointment"),
    ("llama3.2-3b-q4/tool_012", "tool_012", "message", "The meeting is moved to 3pm."),
    ("llama3.1-8b-q4/tool_012", "tool_012", "message", "The meeting is moved to 3pm."),
    ("llama3.1-8b-q4/tool_024", "tool_024", "note", "Call mom"),
    ("llama3.2-3b-q4/tool_098", "tool_098", "body", "The review is complete."),
    ("llama3.1-8b-q4/tool_098", "tool_098", "body", "The review is complete."),
])
def test_evaluate_accepts_all_seven_audited_free_text_calls(case, question_id, key, given):
    question = _real_question(question_id)
    arguments = dict(question["expected"]["arguments"])
    arguments[key] = given
    calls = [{"name": question["expected"]["name"], "arguments": arguments}]
    assert case
    assert ToolBenchmark.evaluate_question(question, calls)["correct"] is True


def test_evaluate_free_text_stays_exact_without_configuration():
    q = {"expected": {"call": True, "name": "send", "arguments": {"body": "Ready"}}}
    calls = [{"name": "send", "arguments": {"body": "ready."}}]
    assert ToolBenchmark.evaluate_question(q, calls)["correct"] is False


def test_evaluate_normalized_free_text_still_requires_same_content():
    q = {"expected": {"call": True, "name": "send", "arguments": {"body": "review is complete"},
                      "normalized_string_keys": ["body"]}}
    calls = [{"name": "send", "arguments": {"body": "review is incomplete."}}]
    assert ToolBenchmark.evaluate_question(q, calls)["correct"] is False


def test_evaluate_normalized_free_text_preserves_internal_punctuation():
    q = {"expected": {"call": True, "name": "book", "arguments": {"name": "Luigi's"},
                      "normalized_string_keys": ["name"]}}
    calls = [{"name": "book", "arguments": {"name": "Luigis"}}]
    assert ToolBenchmark.evaluate_question(q, calls)["correct"] is False


def test_evaluate_normalized_key_applies_recursively_without_relaxing_siblings():
    q = {"expected": {"call": True, "name": "send", "arguments": {
        "payload": {"body": "Ready", "status": "OPEN"},
    }, "normalized_string_keys": ["body"]}}
    calls = [{"name": "send", "arguments": {
        "payload": {"body": "ready.", "status": "OPEN"},
    }}]
    assert ToolBenchmark.evaluate_question(q, calls)["correct"] is True
    calls = [{"name": "send", "arguments": {
        "payload": {"body": "ready.", "status": "open"},
    }}]
    assert ToolBenchmark.evaluate_question(q, calls)["correct"] is False


def test_evaluate_normalization_does_not_coerce_string_boolean():
    q = {"expected": {"call": True, "name": "set_light", "arguments": {"on": True},
                      "normalized_string_keys": ["on"]}}
    calls = [{"name": "set_light", "arguments": {"on": "true"}}]
    assert ToolBenchmark.evaluate_question(q, calls)["correct"] is False


@pytest.mark.parametrize(("question_id", "key", "given"), [
    ("tool_004", "on", "true"),
    ("tool_046", "enabled", "false"),
    ("tool_048", "muted", "true"),
    ("tool_049", "subscribed", "false"),
])
def test_evaluate_keeps_audited_string_booleans_wrong(question_id, key, given):
    question = _real_question(question_id)
    arguments = dict(question["expected"]["arguments"])
    arguments[key] = given
    calls = [{"name": question["expected"]["name"], "arguments": arguments}]
    assert ToolBenchmark.evaluate_question(question, calls)["correct"] is False


@pytest.mark.parametrize(("question_id", "key", "given"), [
    ("tool_011", "restaurant", "Luigi"),
    ("tool_087", "destination", "archive/slides.pptx"),
    ("tool_056", "seconds", 30),
    ("tool_059", "inches", 77),
])
def test_evaluate_keeps_audited_identifier_and_wrong_values_wrong(question_id, key, given):
    question = _real_question(question_id)
    arguments = dict(question["expected"]["arguments"])
    arguments[key] = given
    calls = [{"name": question["expected"]["name"], "arguments": arguments}]
    assert ToolBenchmark.evaluate_question(question, calls)["correct"] is False


@pytest.mark.parametrize(("question_id", "key"), [
    ("tool_061", "address"),
    ("tool_065", "items"),
])
def test_evaluate_keeps_audited_json_encoded_structures_wrong(question_id, key):
    question = _real_question(question_id)
    arguments = dict(question["expected"]["arguments"])
    arguments[key] = json.dumps(arguments[key])
    calls = [{"name": question["expected"]["name"], "arguments": arguments}]
    assert ToolBenchmark.evaluate_question(question, calls)["correct"] is False


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


# ── _ask ──

class _FakeChatToolsEngine:
    """Minimal stub of just the chat_tools() surface _ask() calls."""

    def __init__(self, response_text: str, tool_calls: list):
        self._response_text = response_text
        self._tool_calls = tool_calls
        self.kwargs = None

    def chat_tools(self, tag, messages, tools, timeout=None, num_ctx=None,
                   num_predict=None, check_loop=None, token_budget=None):
        self.kwargs = {
            "num_predict": num_predict,
            "token_budget": token_budget,
        }
        return None, None, None, None, self._response_text, self._tool_calls, False


def test_ask_raw_response_is_tool_calls_json_when_a_call_was_made():
    engine = _FakeChatToolsEngine("", [{"name": "get_weather", "arguments": {"location": "Paris"}}])
    _, raw, budget_nudged = ToolBenchmark._ask(engine, "tag", _q_call())
    assert raw == json.dumps([{"name": "get_weather", "arguments": {"location": "Paris"}}])
    assert budget_nudged is False
    assert engine.kwargs == {
        "num_predict": -1,
        "token_budget": config.ACC_TOKEN_BUDGET,
    }


def test_ask_raw_response_preserves_prose_on_decline():
    # No tool call was made — the model just answered in prose. Losing that
    # text would make a declined-tool sidecar entry uninformative.
    q = {"id": "tool_y", "category": "decline_no_fit", "prompt": "Play a song for me.",
         "tools": [{"type": "function", "function": {"name": "get_weather"}}],
         "expected": {"call": False}}
    engine = _FakeChatToolsEngine("I can't help with that request.", [])
    _, raw, _budget_nudged = ToolBenchmark._ask(engine, "tag", q)
    assert raw == "I can't help with that request."


def test_ask_raw_response_keeps_both_prose_and_tool_calls_when_model_emits_both():
    # A model can narrate ("Sure, checking the weather...") *and* call a tool
    # in the same turn — neither should be discarded from the sidecar.
    tool_calls = [{"name": "get_weather", "arguments": {"location": "Paris"}}]
    engine = _FakeChatToolsEngine("Sure, let me check that for you.", tool_calls)
    _, raw, _budget_nudged = ToolBenchmark._ask(engine, "tag", _q_call())
    assert json.loads(raw) == {"tool_calls": tool_calls, "text": "Sure, let me check that for you."}


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


def test_score_all_list_covers_every_question_including_correct_ones():
    answers = {"q1": {"correct": True}, "q2": {"correct": False}, "q3": None}
    result = ToolBenchmark.score(_questions(), answers)
    assert {e["id"] for e in result["all"]} == {"q1", "q2", "q3"}
    q1_entry = next(e for e in result["all"] if e["id"] == "q1")
    assert q1_entry == {"id": "q1", "category": "single_tool_call", "correct": True}
    q3_entry = next(e for e in result["all"] if e["id"] == "q3")
    assert q3_entry == {"id": "q3", "category": "decline_no_fit", "correct": False}


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
            normalized_keys = expected.get("normalized_string_keys", [])
            assert len(normalized_keys) == len(set(normalized_keys))
            for key in normalized_keys:
                values = []

                def collect(value):
                    if isinstance(value, dict):
                        values.extend(v for k, v in value.items() if k == key)
                        for child in value.values():
                            collect(child)
                    elif isinstance(value, list):
                        for child in value:
                            collect(child)

                collect(expected["arguments"])
                assert values and all(isinstance(value, str) for value in values)
        else:
            assert "name" not in expected  # a decline case names no tool


def test_real_bank_declares_only_reviewed_free_text_normalization():
    configured = {
        q["id"]: q["expected"]["normalized_string_keys"]
        for q in ToolBenchmark.load_questions()
        if q["expected"].get("normalized_string_keys")
    }
    assert configured == {
        "tool_010": ["title"],
        "tool_012": ["message"],
        "tool_024": ["note"],
        "tool_098": ["body"],
    }
