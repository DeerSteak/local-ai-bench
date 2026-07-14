from code_benchmark import CodeBenchmark


# ── build_prompt ──

def test_build_prompt_includes_question_text_and_function_name():
    q = {"prompt": "Write a function sum_two(a, b) ...", "function_name": "sum_two"}
    prompt = CodeBenchmark.build_prompt(q)
    assert "Write a function sum_two(a, b) ..." in prompt
    assert "sum_two" in prompt


def test_build_prompt_dispatches_to_class_definition_for_stateful_question():
    q = {"prompt": "Implement a class Stack ...", "class_name": "Stack"}
    prompt = CodeBenchmark.build_prompt(q)
    assert "Implement a class Stack ..." in prompt
    assert "class definition for Stack" in prompt
    assert "function definition" not in prompt


def test_build_prompt_without_visible_tests_key_has_no_examples_block():
    q = {"prompt": "Write a function sum_two(a, b) ...", "function_name": "sum_two"}
    assert "Examples:" not in CodeBenchmark.build_prompt(q)


def test_build_prompt_renders_visible_tests_as_worked_examples():
    q = {
        "prompt": "Write a function sum_two(a, b) ...",
        "function_name": "sum_two",
        "visible_tests": [{"args": [2, 3], "expected": 5}, {"args": [-1, 1], "expected": 0}],
        "hidden_tests": [{"args": [100, 200], "expected": 300}],
    }
    prompt = CodeBenchmark.build_prompt(q)
    assert "Examples:" in prompt
    assert "sum_two(2, 3) == 5" in prompt
    assert "sum_two(-1, 1) == 0" in prompt


def test_build_prompt_never_leaks_hidden_tests():
    q = {
        "prompt": "Write a function sum_two(a, b) ...",
        "function_name": "sum_two",
        "visible_tests": [{"args": [2, 3], "expected": 5}],
        "hidden_tests": [{"args": [999, 999], "expected": 1998}],
    }
    prompt = CodeBenchmark.build_prompt(q)
    assert "999" not in prompt
    assert "1998" not in prompt


def test_build_prompt_renders_stateful_visible_tests_as_worked_examples():
    q = {
        "prompt": "Implement a class Stack ...",
        "class_name": "Stack",
        "visible_tests": [
            {"ops": [["push", [1]], ["pop", []]], "expected": [None, 1]},
        ],
        "hidden_tests": [
            {"init": [], "ops": [["push", [999]]], "expected": [None]},
        ],
    }
    prompt = CodeBenchmark.build_prompt(q)
    assert "Examples:" in prompt
    assert "obj = Stack()" in prompt
    assert "obj.push(1)" in prompt
    assert "obj.pop()" in prompt
    assert "999" not in prompt


def test_build_prompt_stateful_example_uses_init_args():
    q = {
        "prompt": "Implement a class LRUCache(capacity) ...",
        "class_name": "LRUCache",
        "visible_tests": [
            {"init": [2], "ops": [["put", [1, 1]], ["get", [1]]], "expected": [None, 1]},
        ],
        "hidden_tests": [{"init": [2], "ops": [["get", [9]]], "expected": [-1]}],
    }
    prompt = CodeBenchmark.build_prompt(q)
    assert "obj = LRUCache(2)" in prompt
    assert "obj.put(1, 1)" in prompt
    assert "obj.get(1)" in prompt


# ── extract_code ──

def test_extract_code_pulls_fenced_python_block():
    text = "Sure, here you go:\n```python\ndef f(x):\n    return x\n```\nHope that helps!"
    assert CodeBenchmark.extract_code(text) == "def f(x):\n    return x"


def test_extract_code_pulls_bare_fenced_block():
    text = "```\ndef f(x):\n    return x\n```"
    assert CodeBenchmark.extract_code(text) == "def f(x):\n    return x"


def test_extract_code_falls_back_to_whole_reply_when_unfenced():
    text = "def f(x):\n    return x"
    assert CodeBenchmark.extract_code(text) == "def f(x):\n    return x"


def test_extract_code_returns_empty_for_empty_response():
    assert CodeBenchmark.extract_code("") == ""
    assert CodeBenchmark.extract_code(None) == ""


def test_extract_code_takes_last_block_when_multiple_are_present():
    # A reasoning model sketching a draft before its real answer — the draft
    # (first block) must not be graded instead of the final answer (last).
    text = (
        "Let me sketch this out first:\n"
        "```python\ndef f(x):\n    pass  # TODO\n```\n"
        "Now here's the real implementation:\n"
        "```python\ndef f(x):\n    return x * 2\n```"
    )
    assert CodeBenchmark.extract_code(text) == "def f(x):\n    return x * 2"


# ── _values_close ──

def test_values_close_exact_match_for_ints_and_strings():
    assert CodeBenchmark._values_close(5, 5) is True
    assert CodeBenchmark._values_close("abc", "abc") is True
    assert CodeBenchmark._values_close(5, 6) is False
    assert CodeBenchmark._values_close("abc", "abd") is False


def test_values_close_tolerates_float_rounding_difference():
    # The classic floating-point example: 0.1 + 0.2 != 0.3 under exact ==.
    assert CodeBenchmark._values_close(0.1 + 0.2, 0.3) is True


def test_values_close_rejects_float_difference_beyond_tolerance():
    assert CodeBenchmark._values_close(1.5, 1.6) is False


def test_values_close_int_and_float_compare_by_value():
    assert CodeBenchmark._values_close(2, 2.0) is True


def test_values_close_recurses_into_lists_with_float_tolerance():
    got = [None, None, 0.1 + 0.2, 2]
    expected = [None, None, 0.3, 2]
    assert CodeBenchmark._values_close(got, expected) is True


def test_values_close_list_length_mismatch_fails():
    assert CodeBenchmark._values_close([1, 2], [1, 2, 3]) is False


def test_values_close_type_mismatch_is_not_accidentally_close():
    assert CodeBenchmark._values_close("3.5", 3.5) is False
    assert CodeBenchmark._values_close(None, 0.0) is False


# ── execute_tests ──

def test_execute_tests_all_pass():
    code = "def sum_two(a, b):\n    return a + b"
    tests = [{"args": [2, 3], "expected": 5}, {"args": [-1, 1], "expected": 0}]
    results = CodeBenchmark.execute_tests(code, "sum_two", tests)
    assert results == [
        {"passed": True, "got": 5, "error": None},
        {"passed": True, "got": 0, "error": None},
    ]


def test_execute_tests_wrong_answer_fails_that_test_only():
    code = "def sum_two(a, b):\n    return a - b"
    tests = [{"args": [2, 3], "expected": 5}]
    results = CodeBenchmark.execute_tests(code, "sum_two", tests)
    assert results == [{"passed": False, "got": -1, "error": None}]


def test_execute_tests_runtime_error_in_one_call_does_not_abort_others():
    code = "def divide(a, b):\n    return a / b"
    tests = [{"args": [10, 0], "expected": None}, {"args": [10, 2], "expected": 5.0}]
    results = CodeBenchmark.execute_tests(code, "divide", tests)
    assert results[0]["passed"] is False
    assert results[0]["error"] is not None
    assert results[1] == {"passed": True, "got": 5.0, "error": None}


def test_execute_tests_syntax_error_fails_every_test_with_error():
    code = "def f(a, b:\n    return a + b"
    tests = [{"args": [1, 2], "expected": 3}, {"args": [3, 4], "expected": 7}]
    results = CodeBenchmark.execute_tests(code, "f", tests)
    assert len(results) == 2
    assert all(r["passed"] is False and r["error"] for r in results)


def test_execute_tests_infinite_loop_times_out_instead_of_hanging():
    code = "def f(a, b):\n    while True:\n        pass"
    tests = [{"args": [1, 2], "expected": 3}]
    results = CodeBenchmark.execute_tests(code, "f", tests, timeout=1)
    assert results == [{"passed": False, "got": None, "error": "timeout"}]


def test_execute_tests_missing_function_name_fails_with_error():
    code = "def some_other_name(a, b):\n    return a + b"
    tests = [{"args": [1, 2], "expected": 3}]
    results = CodeBenchmark.execute_tests(code, "sum_two", tests)
    assert results == [{"passed": False, "got": None, "error": "name 'sum_two' is not defined"}]


# ── execute_stateful_tests ──

_STACK_CODE = (
    "class Stack:\n"
    "    def __init__(self):\n"
    "        self._items = []\n"
    "    def push(self, x):\n"
    "        self._items.append(x)\n"
    "    def pop(self):\n"
    "        return self._items.pop() if self._items else None\n"
    "    def peek(self):\n"
    "        return self._items[-1] if self._items else None\n"
    "    def is_empty(self):\n"
    "        return len(self._items) == 0\n"
)


def test_execute_stateful_tests_all_pass():
    tests = [
        {"ops": [["push", [1]], ["push", [2]], ["pop", []], ["peek", []]], "expected": [None, None, 2, 1]},
        {"ops": [["is_empty", []]], "expected": [True]},
    ]
    results = CodeBenchmark.execute_stateful_tests(_STACK_CODE, "Stack", tests)
    assert results == [
        {"passed": True, "got": [None, None, 2, 1], "error": None},
        {"passed": True, "got": [True], "error": None},
    ]


def test_execute_stateful_tests_fresh_instance_per_test():
    # A stray push in one scenario must not leak into the next.
    tests = [
        {"ops": [["push", [1]], ["push", [2]]], "expected": [None, None]},
        {"ops": [["is_empty", []]], "expected": [True]},
    ]
    results = CodeBenchmark.execute_stateful_tests(_STACK_CODE, "Stack", tests)
    assert results[1] == {"passed": True, "got": [True], "error": None}


def test_execute_stateful_tests_respects_init_args():
    code = (
        "class Box:\n"
        "    def __init__(self, cap):\n"
        "        self.cap = cap\n"
        "    def get_cap(self):\n"
        "        return self.cap\n"
    )
    tests = [{"init": [5], "ops": [["get_cap", []]], "expected": [5]}]
    results = CodeBenchmark.execute_stateful_tests(code, "Box", tests)
    assert results == [{"passed": True, "got": [5], "error": None}]


def test_execute_stateful_tests_defaults_init_to_no_args():
    tests = [{"ops": [["is_empty", []]], "expected": [True]}]  # no "init" key
    results = CodeBenchmark.execute_stateful_tests(_STACK_CODE, "Stack", tests)
    assert results == [{"passed": True, "got": [True], "error": None}]


def test_execute_stateful_tests_wrong_output_fails_that_test_only():
    tests = [{"ops": [["push", [1]], ["pop", []]], "expected": [None, 999]}]
    results = CodeBenchmark.execute_stateful_tests(_STACK_CODE, "Stack", tests)
    assert results == [{"passed": False, "got": [None, 1], "error": None}]


def test_execute_stateful_tests_exception_mid_sequence_fails_that_scenario_only():
    tests = [
        {"ops": [["nonexistent_method", []]], "expected": [None]},
        {"ops": [["is_empty", []]], "expected": [True]},
    ]
    results = CodeBenchmark.execute_stateful_tests(_STACK_CODE, "Stack", tests)
    assert results[0]["passed"] is False
    assert results[0]["error"] is not None
    assert results[1] == {"passed": True, "got": [True], "error": None}


def test_execute_stateful_tests_missing_class_name_fails_with_error():
    tests = [{"ops": [["push", [1]]], "expected": [None]}]
    results = CodeBenchmark.execute_stateful_tests(_STACK_CODE, "NoSuchClass", tests)
    assert results == [{"passed": False, "got": None, "error": "name 'NoSuchClass' is not defined"}]


def test_execute_stateful_tests_infinite_loop_times_out_instead_of_hanging():
    code = (
        "class Loopy:\n"
        "    def __init__(self):\n"
        "        pass\n"
        "    def go(self):\n"
        "        while True:\n"
        "            pass\n"
    )
    tests = [{"ops": [["go", []]], "expected": [None]}]
    results = CodeBenchmark.execute_stateful_tests(code, "Loopy", tests, timeout=1)
    assert results == [{"passed": False, "got": None, "error": "timeout"}]


# ── evaluate_question ──

def _question():
    return {
        "id": "code_001", "category": "arithmetic", "function_name": "sum_two",
        "prompt": "...",
        "visible_tests": [{"args": [2, 3], "expected": 5}],
        "hidden_tests": [{"args": [100, 200], "expected": 300}],
    }


def test_evaluate_question_all_tests_pass_is_correct():
    code = "def sum_two(a, b):\n    return a + b"
    result = CodeBenchmark.evaluate_question(_question(), code)
    assert result == {"correct": True, "tests_passed": 2, "tests_total": 2, "error": None}


def test_evaluate_question_partial_pass_is_not_correct():
    code = "def sum_two(a, b):\n    return 5"  # only passes the first (hardcoded) test
    result = CodeBenchmark.evaluate_question(_question(), code)
    assert result["correct"] is False
    assert result["tests_passed"] == 1
    assert result["tests_total"] == 2


def test_evaluate_question_no_code_short_circuits_without_running():
    result = CodeBenchmark.evaluate_question(_question(), "")
    assert result == {"correct": False, "tests_passed": 0, "tests_total": 2, "error": "no code found"}
    result_none = CodeBenchmark.evaluate_question(_question(), None)
    assert result_none["error"] == "no code found"


def _stateful_question():
    return {
        "id": "code_026", "category": "stateful", "class_name": "Stack",
        "prompt": "...",
        "visible_tests": [{"ops": [["push", [1]], ["pop", []]], "expected": [None, 1]}],
        "hidden_tests": [{"ops": [["is_empty", []]], "expected": [True]}],
    }


def test_evaluate_question_dispatches_to_stateful_execution_for_class_name_question():
    result = CodeBenchmark.evaluate_question(_stateful_question(), _STACK_CODE)
    assert result == {"correct": True, "tests_passed": 2, "tests_total": 2, "error": None}


def test_evaluate_question_stateful_partial_pass_is_not_correct():
    # A Stack that never actually stores anything: push does nothing, pop is always None.
    code = (
        "class Stack:\n"
        "    def __init__(self):\n"
        "        pass\n"
        "    def push(self, x):\n"
        "        pass\n"
        "    def pop(self):\n"
        "        return None\n"
        "    def is_empty(self):\n"
        "        return True\n"
    )
    result = CodeBenchmark.evaluate_question(_stateful_question(), code)
    assert result["correct"] is False
    assert result["tests_passed"] == 1  # only the is_empty hidden test passes
    assert result["tests_total"] == 2


# ── score ──

def _questions():
    return [
        {"id": "q1", "category": "arithmetic", "visible_tests": [{"args": [1], "expected": 1}], "hidden_tests": []},
        {"id": "q2", "category": "arithmetic", "visible_tests": [{"args": [1], "expected": 1}], "hidden_tests": []},
        {"id": "q3", "category": "string", "visible_tests": [{"args": [1], "expected": 1}], "hidden_tests": []},
    ]


def test_score_all_correct():
    answers = {
        "q1": {"correct": True, "tests_passed": 1, "tests_total": 1, "error": None},
        "q2": {"correct": True, "tests_passed": 1, "tests_total": 1, "error": None},
        "q3": {"correct": True, "tests_passed": 1, "tests_total": 1, "error": None},
    }
    result = CodeBenchmark.score(_questions(), answers)
    assert result["correct"] == 3
    assert result["total"] == 3
    assert result["accuracy_pct"] == 100.0
    assert result["incorrect"] == []


def test_score_partial_and_category_breakdown():
    answers = {
        "q1": {"correct": True, "tests_passed": 1, "tests_total": 1, "error": None},
        "q2": {"correct": False, "tests_passed": 0, "tests_total": 1, "error": "AssertionError"},
        "q3": None,
    }
    result = CodeBenchmark.score(_questions(), answers)
    assert result["correct"] == 1
    assert result["answered"] == 2  # q3 unanswered (None)
    assert result["accuracy_pct"] == round(100 / 3, 1)
    assert result["by_category"]["arithmetic"] == {"correct": 1, "total": 2, "accuracy_pct": 50.0}
    assert result["by_category"]["string"] == {"correct": 0, "total": 1, "accuracy_pct": 0.0}


def test_score_incorrect_list_has_expected_entries():
    answers = {
        "q1": {"correct": True, "tests_passed": 1, "tests_total": 1, "error": None},
        "q2": {"correct": False, "tests_passed": 0, "tests_total": 1, "error": "AssertionError"},
        "q3": None,
    }
    result = CodeBenchmark.score(_questions(), answers)
    ids = {entry["id"] for entry in result["incorrect"]}
    assert ids == {"q2", "q3"}
    q2_entry = next(e for e in result["incorrect"] if e["id"] == "q2")
    assert q2_entry == {"id": "q2", "category": "arithmetic", "tests_passed": 0, "tests_total": 1, "error": "AssertionError"}
    q3_entry = next(e for e in result["incorrect"] if e["id"] == "q3")
    assert q3_entry == {"id": "q3", "category": "string", "tests_passed": 0, "tests_total": 1, "error": "unanswered"}


def test_score_missing_answer_counts_as_incorrect_and_unanswered():
    result = CodeBenchmark.score(_questions(), {})
    assert result["correct"] == 0
    assert result["answered"] == 0
    assert len(result["incorrect"]) == 3


# ── load_questions against the real dataset ──

def test_load_questions_returns_well_formed_dataset():
    questions = CodeBenchmark.load_questions()
    assert len(questions) > 0
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))  # unique ids
    for q in questions:
        assert bool(q.get("function_name")) != bool(q.get("class_name"))  # exactly one
        assert len(q["visible_tests"]) > 0
        assert len(q["hidden_tests"]) > 0
        for test in q["visible_tests"] + q["hidden_tests"]:
            if "class_name" in q:
                assert "ops" in test and "expected" in test
            else:
                assert "args" in test and "expected" in test


def test_load_questions_dataset_has_stateful_problems():
    questions = CodeBenchmark.load_questions()
    stateful = [q for q in questions if "class_name" in q]
    assert len(stateful) > 0
    assert {q["category"] for q in stateful} == {"stateful"}


def test_build_prompt_never_reads_hidden_tests_key():
    # Stronger than a substring scan (which small shared values like 0/1/True
    # could pass by coincidence): build_prompt must not even look up
    # "hidden_tests", proven by deleting the key and confirming no KeyError,
    # across every real question in both problem shapes.
    for q in CodeBenchmark.load_questions():
        q_without_hidden = {k: v for k, v in q.items() if k != "hidden_tests"}
        CodeBenchmark.build_prompt(q_without_hidden)  # raises if it were accessed
