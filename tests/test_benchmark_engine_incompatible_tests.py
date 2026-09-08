from scripts.app.benchmark import (
    build_engine_execution_profiles, engine_incompatible_tests, engine_pass_tests,
    engine_version_applies,
)


def test_llamacpp_pass_drops_vllmbench_only():
    dropped = engine_incompatible_tests(["llm", "llamabench", "vllmbench", "emb"], "llamacpp")
    assert dropped == ["vllmbench"]


def test_vllm_pass_drops_both_llamabench_stages():
    dropped = engine_incompatible_tests(
        ["llm", "llamabench", "llamabenchconc", "vllmbench"], "vllm")
    assert dropped == ["llamabench", "llamabenchconc"]


def test_neither_native_test_selected_drops_nothing():
    assert engine_incompatible_tests(["llm", "conv", "emb", "img"], "llamacpp") == []
    assert engine_incompatible_tests(["llm", "conv", "emb", "img"], "vllm") == []


def test_engine_own_native_tests_are_kept():
    assert engine_incompatible_tests(["llamabench", "llamabenchconc"], "llamacpp") == []
    assert engine_incompatible_tests(["vllmbench"], "vllm") == []


def test_vulkan_llamacpp_keeps_llamacpp_native_tests():
    assert engine_incompatible_tests(
        ["llamabench", "llamabenchconc", "vllmbench"], "llamacpp-vulkan",
    ) == ["vllmbench"]


def test_unknown_engine_drops_every_others_native_test():
    """A future third engine has no ENGINE_NATIVE_TESTS entry of its own, so both
    llama.cpp's and vLLM's native tests are foreign to it."""
    dropped = engine_incompatible_tests(["llamabench", "vllmbench", "llm"], "mlx")
    assert set(dropped) == {"llamabench", "vllmbench"}


def test_preserves_selection_order():
    dropped = engine_incompatible_tests(["vllmbench", "llm", "llamabench"], "vllm")
    assert dropped == ["llamabench"]


def test_vllm_only_selection_omits_the_empty_llamacpp_pass():
    assert engine_pass_tests(["vllmbench"], "llamacpp", include_images=True) == []
    assert engine_pass_tests(["vllmbench"], "vllm", include_images=False) == ["vllmbench"]


def test_later_engine_pass_omits_images_and_foreign_native_workloads():
    assert engine_pass_tests(
        ["img", "llamabench", "llamabenchconc", "llm"], "vllm", include_images=False,
    ) == ["llm"]


def test_engine_version_applies_only_to_engine_backed_workloads():
    assert engine_version_applies(["llm", "img"])
    assert engine_version_applies(["vllmbench"])
    assert engine_version_applies(["emb"])
    assert not engine_version_applies(["img"])


def test_execution_profiles_are_built_once_for_each_nonempty_engine_pass():
    llamacpp = object()
    vllm = object()
    calls = []

    def build(engine, tests, **kwargs):
        calls.append((engine, tests, kwargs))
        return {"engine_support": {"runtime_version": "1.0"}}

    profiles = build_engine_execution_profiles(
        [{"name": "llamacpp", "engine": llamacpp}, {"name": "vllm", "engine": vllm}],
        ["img", "llm"], cpu_only=False, hardware_profile={"backend": "cuda"},
        profile_builder=build,
    )
    assert set(profiles) == {"llamacpp", "vllm"}
    assert calls[0][1] == ["img", "llm"]
    assert calls[1][1] == ["llm"]
    assert all(call[2]["hardware_profile"] == {"backend": "cuda"} for call in calls)
