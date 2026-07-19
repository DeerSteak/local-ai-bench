from benchmark import ACCURACY_TESTS, CONCURRENCY_TESTS, expand_tests


def test_acc_expands_to_accuracy_tests():
    assert expand_tests(["acc"]) == ACCURACY_TESTS


def test_conc_expands_to_concurrency_tests():
    assert expand_tests(["conc"]) == CONCURRENCY_TESTS


def test_conc_and_acc_together_preserve_order():
    assert expand_tests(["conc", "acc"]) == CONCURRENCY_TESTS + ACCURACY_TESTS


def test_conc_and_explicit_member_does_not_duplicate():
    assert expand_tests(["conc", "conc_tool"]) == CONCURRENCY_TESTS
    assert expand_tests(["conc_tool", "conc"]) == CONCURRENCY_TESTS


def test_non_acc_tests_pass_through_unchanged():
    assert expand_tests(["llm", "conv"]) == ["llm", "conv"]


def test_acc_mixed_with_other_tests_preserves_order():
    assert expand_tests(["llm", "acc", "img"]) == ["llm"] + ACCURACY_TESTS + ["img"]


def test_acc_and_explicit_member_does_not_duplicate():
    assert expand_tests(["acc", "mcq"]) == ACCURACY_TESTS
    assert expand_tests(["mcq", "acc"]) == ACCURACY_TESTS


def test_repeated_plain_test_does_not_duplicate():
    assert expand_tests(["llm", "llm"]) == ["llm"]
