from scripts.app.benchmark import resolve_engine_names


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


def test_a_comma_list_runs_exactly_those_engines():
    assert resolve_engine_names("llamacpp,vllm", ["llamacpp", "vllm"]) == ["llamacpp", "vllm"]
    assert resolve_engine_names("vllm", ["llamacpp", "vllm"]) == ["vllm"]


def test_comma_list_order_follows_the_registry_not_the_typing():
    """Two runs written differently must execute in the same order, so results files
    land in a predictable sequence."""
    assert resolve_engine_names("vllm,llamacpp", ["llamacpp", "vllm"]) == ["llamacpp", "vllm"]
    assert resolve_engine_names(" vllm , llamacpp ", ["llamacpp", "vllm"]) == ["llamacpp", "vllm"]


def test_an_unknown_engine_in_a_list_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="mlx"):
        resolve_engine_names("llamacpp,mlx", ["llamacpp", "vllm"])
    with pytest.raises(ValueError):
        resolve_engine_names("", ["llamacpp"])


def test_duplicates_collapse():
    assert resolve_engine_names("vllm,vllm", ["llamacpp", "vllm"]) == ["vllm"]
