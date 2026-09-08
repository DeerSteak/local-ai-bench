from scripts.runtime import config
from scripts.runtime.engines.llamacpp_vulkan import LlamaCppVulkanEngine


def test_vulkan_engine_has_distinct_identity_and_shared_llamacpp_family():
    engine = LlamaCppVulkanEngine()
    assert engine.name == "llamacpp-vulkan"
    assert engine.family == "llamacpp"


def test_vulkan_engine_uses_separate_runtime_and_shared_model_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LLAMACPP_VULKAN_DIR", tmp_path / "runtime")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    assert LlamaCppVulkanEngine._runtime_dir() == tmp_path / "runtime"
    assert LlamaCppVulkanEngine._models_dir() == tmp_path / "models" / "llamacpp"


def test_vulkan_engine_never_falls_back_to_native_path_binary(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LLAMACPP_VULKAN_DIR", tmp_path / "missing")
    monkeypatch.setattr(config, "SETUP_CONFIG_PATH", tmp_path / "missing-config.json")
    monkeypatch.setattr(
        "scripts.runtime.engines.llamacpp.shutil.which",
        lambda _name: "/native/llama-server",
    )
    assert LlamaCppVulkanEngine.tool_path("llama-server") is None


def test_vulkan_engine_installation_requires_vulkan_backend(monkeypatch):
    monkeypatch.setattr(
        LlamaCppVulkanEngine, "_binary_path", classmethod(lambda _cls: "/vulkan/llama-server"),
    )
    monkeypatch.setattr(
        "scripts.runtime.engines.llamacpp.probe_llamacpp_backend", lambda _binary: "cuda",
    )
    assert not LlamaCppVulkanEngine().is_installed()

    monkeypatch.setattr(
        "scripts.runtime.engines.llamacpp.probe_llamacpp_backend", lambda _binary: "vulkan",
    )
    assert LlamaCppVulkanEngine().is_installed()


def test_vulkan_runtime_backend_records_actual_probe_without_oneapi(monkeypatch):
    monkeypatch.setattr(
        LlamaCppVulkanEngine, "_binary_path", classmethod(lambda _cls: "/vulkan/llama-server"),
    )
    monkeypatch.setattr(
        "scripts.runtime.engines.llamacpp.probe_llamacpp_backend", lambda *_args, **_kwargs: "vulkan",
    )
    monkeypatch.setattr(
        "scripts.runtime.engines.llamacpp.oneapi_environment",
        lambda: (_ for _ in ()).throw(AssertionError("Vulkan must not source oneAPI")),
    )
    engine = LlamaCppVulkanEngine()
    assert engine.runtime_backend("xpu") == "vulkan"
    assert engine.process_environment() is None
