import pytest

from scripts.app.benchmark import apply_max_prompt_tokens_cap


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


@pytest.mark.parametrize("cap, expected", [
    (32768, [8192, 16384, 32768]),
    (49152, [8192, 16384, 32768, 49152]),
    (65536, [8192, 16384, 32768, 65536]),
    (81920, [8192, 16384, 32768, 65536, 81920]),
    (98304, [8192, 16384, 32768, 65536, 98304]),
    (131072, [8192, 16384, 32768, 65536, 131072]),
    (140000, [8192, 16384, 32768, 65536, 131072]),
])
def test_large_llamabench_caps_keep_coarse_depths_and_endpoint(cap, expected):
    from scripts.app.benchmark import select_llamabench_prompt_sizes
    from scripts.runtime import config

    original = list(config.LLAMABENCH_PP)
    assert select_llamabench_prompt_sizes(original, cap) == expected
    assert original == config.LLAMABENCH_PP


def test_small_caps_and_uncapped_sweeps_keep_dense_depths():
    from scripts.app.benchmark import select_llamabench_prompt_sizes
    from scripts.runtime import config

    sizes = config.LLAMABENCH_PP
    assert select_llamabench_prompt_sizes(sizes, None) == sizes
    assert select_llamabench_prompt_sizes(sizes, 16384) == [512, 2048, 4096, 8192, 16384]
    assert select_llamabench_prompt_sizes([512, 2048], 131072) == [512, 2048]
    assert select_llamabench_prompt_sizes([], 131072) == []
    assert select_llamabench_prompt_sizes([512], 256) == []


def test_sparse_policy_preserves_the_deepest_configured_size_below_an_arbitrary_cap():
    from scripts.app.benchmark import select_llamabench_prompt_sizes

    assert select_llamabench_prompt_sizes([512, 8192, 32768, 49152, 65536], 60000) == [
        8192, 32768, 49152,
    ]
