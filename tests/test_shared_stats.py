from shared import Shared


def test_mean_empty_is_zero():
    assert Shared.mean([]) == 0


def test_mean_basic():
    assert Shared.mean([1, 2, 3]) == 2


def test_stdev_needs_at_least_two_values():
    assert Shared.stdev([]) == 0
    assert Shared.stdev([5]) == 0


def test_stdev_basic():
    assert Shared.stdev([1, 1, 1]) == 0
    assert Shared.stdev([1, 2, 3]) > 0


def test_build_prompt_for_context_reaches_target_length():
    target_tokens = 500
    prompt = Shared.build_prompt_for_context(target_tokens)
    assert len(prompt) == target_tokens * 4


def test_build_prompt_for_context_small_target_still_includes_base_prompt():
    # Even a tiny target can't be shorter than the nonce prefix; just confirm
    # it doesn't crash and returns a non-empty, truncated string.
    prompt = Shared.build_prompt_for_context(1)
    assert isinstance(prompt, str)
    assert len(prompt) == 4


def test_build_prompt_for_context_nonce_differs_between_calls():
    p1 = Shared.build_prompt_for_context(2000)
    p2 = Shared.build_prompt_for_context(2000)
    assert p1 != p2
