from code_benchmark import CodeBenchmark


# ── build_prompt ──

def test_build_prompt_includes_question_text_and_function_name():
    q = {"prompt": "Write a function sum_two(a, b) ...", "function_name": "sum_two"}
    prompt = CodeBenchmark.build_prompt(q)
    assert "Write a function sum_two(a, b) ..." in prompt
    assert "sum_two" in prompt


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
        assert q["function_name"]
        assert len(q["visible_tests"]) > 0
        assert len(q["hidden_tests"]) > 0
        for test in q["visible_tests"] + q["hidden_tests"]:
            assert "args" in test and "expected" in test
