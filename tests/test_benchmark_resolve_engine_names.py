from benchmark import resolve_engine_names


def test_single_engine_passes_through():
    assert resolve_engine_names("llamacpp", ["llamacpp"]) == ["llamacpp"]


def test_all_expands_to_every_available_engine():
    assert resolve_engine_names("all", ["llamacpp", "mlx"]) == ["llamacpp", "mlx"]


def test_all_with_one_registered_engine_is_a_no_op():
    assert resolve_engine_names("all", ["llamacpp"]) == ["llamacpp"]


def test_does_not_mutate_available_list():
    available = ["llamacpp", "mlx"]
    result = resolve_engine_names("all", available)
    result.append("extra")
    assert available == ["llamacpp", "mlx"]
