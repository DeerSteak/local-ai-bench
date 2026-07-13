from benchmark import ACCURACY_TESTS, expand_tests


def test_acc_expands_to_accuracy_tests():
    assert expand_tests(["acc"]) == ACCURACY_TESTS


def test_non_acc_tests_pass_through_unchanged():
    assert expand_tests(["llm", "conv"]) == ["llm", "conv"]


def test_acc_mixed_with_other_tests_preserves_order():
    assert expand_tests(["llm", "acc", "img"]) == ["llm"] + ACCURACY_TESTS + ["img"]


def test_acc_and_explicit_member_does_not_duplicate():
    assert expand_tests(["acc", "mcq"]) == ACCURACY_TESTS
    assert expand_tests(["mcq", "acc"]) == ACCURACY_TESTS


def test_repeated_plain_test_does_not_duplicate():
    assert expand_tests(["llm", "llm"]) == ["llm"]
