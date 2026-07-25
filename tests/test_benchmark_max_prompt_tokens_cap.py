import pytest

from benchmark import apply_max_prompt_tokens_cap


def test_no_cap_returns_inputs_unchanged():
    context_lengths, llamabench_pp, llamabenchconc_pp = apply_max_prompt_tokens_cap(
        None, [512, 2048, 8192], [512, 2048, 4096], 8192,
    )
    assert context_lengths == [512, 2048, 8192]
    assert llamabench_pp == [512, 2048, 4096]
    assert llamabenchconc_pp == 8192


def test_cap_drops_entries_above_the_limit():
    context_lengths, llamabench_pp, llamabenchconc_pp = apply_max_prompt_tokens_cap(
        4096, [512, 2048, 8192, 32768], [512, 2048, 4096, 8192, 16384], 8192,
    )
    assert context_lengths == [512, 2048]
    assert llamabench_pp == [512, 2048, 4096]
    assert llamabenchconc_pp == 4096


def test_cap_above_llamabenchconc_pp_leaves_it_unchanged():
    _, _, llamabenchconc_pp = apply_max_prompt_tokens_cap(
        65536, [512, 2048], [512, 2048], 8192,
    )
    assert llamabenchconc_pp == 8192


def test_cap_exactly_on_a_boundary_value_is_kept():
    context_lengths, llamabench_pp, _ = apply_max_prompt_tokens_cap(
        2048, [512, 2048, 8192], [512, 2048, 4096], 8192,
    )
    assert context_lengths == [512, 2048]
    assert llamabench_pp == [512, 2048]


def test_cap_below_every_context_length_raises():
    with pytest.raises(ValueError):
        apply_max_prompt_tokens_cap(256, [512, 2048], [512, 2048], 8192)


def test_cap_below_every_llamabench_pp_but_above_smallest_context_length_raises():
    with pytest.raises(ValueError):
        apply_max_prompt_tokens_cap(256, [128, 2048], [512, 2048], 8192)
