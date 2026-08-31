from scripts.app.benchmark import engines_requiring_install_probe, resolve_engine_names


def test_single_engine_passes_through():
    assert resolve_engine_names("llamacpp", ["llamacpp"]) == ["llamacpp"]


def test_all_expands_to_every_available_engine():
    assert resolve_engine_names("all", ["llamacpp", "mlx"]) == ["llamacpp", "mlx"]


def test_all_expands_only_to_installed_engines_when_inventory_is_known():
    assert resolve_engine_names(
        "all", ["llamacpp", "llamacpp-vulkan", "vllm"],
        installed=["llamacpp", "vllm"],
    ) == ["llamacpp", "vllm"]


def test_only_all_engine_selection_probes_the_registered_inventory():
    available = ["llamacpp", "llamacpp-vulkan", "vllm"]
    assert engines_requiring_install_probe("all", available) == available
    assert engines_requiring_install_probe("llamacpp", available) == []
    assert engines_requiring_install_probe("llamacpp-vulkan,vllm", available) == []


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


def test_explicit_uninstalled_engine_and_empty_all_are_rejected():
    import pytest
    with pytest.raises(ValueError, match="llamacpp-vulkan is not installed"):
        resolve_engine_names(
            "llamacpp-vulkan", ["llamacpp", "llamacpp-vulkan"],
            installed=["llamacpp"],
        )
    with pytest.raises(ValueError, match="No installed inference engines"):
        resolve_engine_names("all", ["llamacpp-vulkan"], installed=[])


def test_duplicates_collapse():
    assert resolve_engine_names("vllm,vllm", ["llamacpp", "vllm"]) == ["vllm"]
