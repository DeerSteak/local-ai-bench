from scripts.app.benchmark import apply_tg_tokens_override


def test_no_override_returns_defaults_unchanged():
    llamabench_tg, llamabenchconc_tg = apply_tg_tokens_override(None, [128, 512], [128, 512])
    assert llamabench_tg == [128, 512]
    assert llamabenchconc_tg == [128, 512]


def test_override_replaces_both_lists_with_the_selection():
    llamabench_tg, llamabenchconc_tg = apply_tg_tokens_override([1024], [128, 512], [128, 512])
    assert llamabench_tg == [1024]
    assert llamabenchconc_tg == [1024]


def test_override_sorts_and_dedupes():
    llamabench_tg, llamabenchconc_tg = apply_tg_tokens_override(
        [512, 128, 512, 1024], [128, 512], [128, 512],
    )
    assert llamabench_tg == [128, 512, 1024]
    assert llamabenchconc_tg == [128, 512, 1024]


def test_returned_lists_are_independent_copies():
    llamabench_tg, llamabenchconc_tg = apply_tg_tokens_override([128], [128, 512], [128, 512])
    llamabench_tg.append(9999)
    assert llamabenchconc_tg == [128]
