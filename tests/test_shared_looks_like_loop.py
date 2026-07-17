from shared import Shared


def test_looks_like_loop_detects_verbatim_repetition():
    block = "the answer must be b because the rule only applies to vowels here"
    text = " ".join([block] * 4)
    assert Shared.looks_like_loop(text) is True


def test_looks_like_loop_false_on_normal_prose():
    text = (
        "The roots of x^3 - 6x^2 + 11x - 6 = 0 are r=1, s=2 and t=3. "
        "So we have 1/r + 1/s + 1/t = (r*s + r*t + s*t) / (r*s*t) = 11/6. "
        "The answer is: 11/6"
    )
    assert Shared.looks_like_loop(text) is False


def test_looks_like_loop_false_on_short_text():
    assert Shared.looks_like_loop("B") is False
    assert Shared.looks_like_loop("") is False


def test_looks_like_loop_false_below_min_repeats():
    block = "the answer must be b because the rule only applies to vowels here"
    text = " ".join([block] * 2)
    assert Shared.looks_like_loop(text) is False


def test_looks_like_loop_respects_custom_thresholds():
    block = "short phrase repeats"
    text = " ".join([block] * 2)
    assert Shared.looks_like_loop(text, ngram_words=3, min_repeats=2) is True
    assert Shared.looks_like_loop(text, ngram_words=3, min_repeats=3) is False


def test_looks_like_loop_detects_paraphrased_hedging():
    # Mirrors the math_124 pattern: re-deriving the same result and
    # repeatedly apologizing/self-correcting without ever landing on an
    # answer. No single 12-word run repeats verbatim, so only the hedging
    # signal should catch this.
    text = (
        "Calculating this gives us a slope of 4. "
        "However, there seems to have been a mistake in my calculation, let me reconsider. "
        "Recomputing with the same numbers gives a slope of 4 again. "
        "Apologies for the confusion, there seems to have been an error above, let me reconsider. "
        "Let me reconsider once more with fresh eyes. "
        "Apologies again, there seems to have been a further miscalculation."
    )
    assert Shared._has_repeated_verbatim_ngram(text) is False
    assert Shared._has_repeated_hedging_phrase(text) is True
    assert Shared.looks_like_loop(text) is True


def test_looks_like_loop_false_on_single_hedge():
    text = (
        "Let me reconsider this problem. The total is 42, which matches the "
        "expected form of the answer. Final answer: 42."
    )
    assert Shared.looks_like_loop(text) is False


def test_looks_like_loop_false_on_verbose_but_correct_cot():
    # A capable model saying "wait," / "actually," 4-5 times while genuinely
    # reasoning toward a correct answer isn't stuck — it shouldn't trip the
    # high-threshold hedge phrases (default threshold is 5, so 4 repeats of
    # each stays under it).
    text = (
        "Let's compute 17 * 23. Actually, let me break it down: 17*20=340. "
        "Wait, plus 17*3=51. Actually, 340+51=391. "
        "Wait, let me double check that addition. Actually, yes 391 is right. "
        "Wait, one more sanity check: 23*17 should equal 391. Confirmed. "
        "The answer is 391."
    )
    assert Shared._has_repeated_verbatim_ngram(text) is False
    assert Shared._has_repeated_hedging_phrase(text) is False
    assert Shared.looks_like_loop(text) is False


def test_looks_like_loop_true_on_high_repeat_hedging_with_no_answer():
    # Same short filler phrases, but 6+ repeats and never landing on an
    # answer — still a loop even though each individual phrase is common CoT
    # filler.
    text = " ".join(["Wait, that's not quite it. Actually, let me look again."] * 6)
    assert Shared._has_repeated_hedging_phrase(text) is True
    assert Shared.looks_like_loop(text) is True
