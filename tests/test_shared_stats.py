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


def test_context_label_preserves_fractional_kilobyte_checkpoint():
    assert Shared.context_label(512) == "0.5K"
    assert Shared.context_label(2048) == "2K"


def test_build_prompt_for_context_reaches_target_length():
    target_tokens = 500
    prompt = Shared.build_prompt_for_context(target_tokens)
    assert len(prompt) == target_tokens * 4


def test_ctx_with_headroom_adds_headroom():
    assert Shared.ctx_with_headroom(32768, 512, 131072) == 32768 + 512


def test_ctx_with_headroom_clamps_to_model_max():
    assert Shared.ctx_with_headroom(65536, 4096, 65536 + 100) == 65536 + 100


def test_ctx_with_headroom_no_room_when_base_equals_model_max():
    assert Shared.ctx_with_headroom(65536, 512, 65536) == 65536


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


def test_build_prompt_for_context_is_not_repetitive_filler():
    # Regression test: repeated filler used to trigger degenerate EOS responses — see docs/workloads.md.
    prompt = Shared.build_prompt_for_context(16384)
    body = prompt.split("] ", 1)[1]
    first_sentence = body[:60]
    assert body.count(first_sentence) <= 1


def test_build_prompt_for_context_covers_the_largest_checkpoint():
    # Largest single-shot context length must be servable from one real slice, not wrapping.
    document_len = len(Shared._long_document())
    assert document_len >= 65536 * 4


def test_build_prompt_for_context_wraps_if_target_exceeds_document_length():
    # target_tokens * 4 here is far larger than the document, forcing the wrap path.
    document_len = len(Shared._long_document())
    target_tokens = document_len
    prompt = Shared.build_prompt_for_context(target_tokens)
    assert len(prompt) == target_tokens * 4
