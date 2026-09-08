import json

from scripts.app.engine_management import (
    collect_engine_statuses, engine_diagnostics_text, engine_status_lines, inspection_placeholder,
    vllm_update_support,
)
from scripts.setup.runtime_status import EngineStatus


def test_engine_status_lines_include_identity_components_and_warnings():
    status = EngineStatus(
        "vllm", "app_managed", "/app/vllm", "0.26.0", "cuda", "ready",
        {"torch": "2.11.0", "wsl": True, "kernel": None}, ("Pinned memory disabled",),
    )
    lines = engine_status_lines(status)
    assert ("Ownership", "App managed") in lines
    assert ("Torch", "2.11.0") in lines
    assert ("Kernel", "None") not in lines
    assert ("Warning", "Pinned memory disabled") in lines


def test_engine_status_lines_collapse_missing_runtime_to_not_installed():
    status = EngineStatus("llamacpp-vulkan", "missing", "", None, "", "unavailable")

    assert engine_status_lines(status) == [("Status", "Not installed")]


def test_inspection_placeholder_populates_every_visible_identity_field():
    lines = dict(engine_status_lines(inspection_placeholder("llamacpp")))
    assert lines == {
        "Ownership": "Inspecting…",
        "Location": "Inspecting…",
        "Version": "Inspecting…",
        "Backend": "Inspecting…",
        "Health": "Inspecting…",
        "Install Type": "Inspecting…",
    }


def test_engine_statuses_include_independent_vulkan_runtime_on_linux(monkeypatch, tmp_path):
    class Engine:
        def __init__(self, name):
            self.name = name

        def runtime_location(self):
            return str(tmp_path / self.name)

        def runtime_backend(self, _hardware_backend):
            return "vulkan" if self.name == "llamacpp-vulkan" else "cuda"

        def runtime_launcher(self):
            return None

        def external_server_url(self):
            return None

    monkeypatch.setattr("scripts.app.engine_management.platform.system", lambda: "Linux")
    monkeypatch.setattr("scripts.app.engine_management.platform.release", lambda: "test")
    monkeypatch.setattr(
        "scripts.app.engine_management.build_llamacpp_status",
        lambda location, managed, backend: EngineStatus(
            "llamacpp", "app_managed", location, "1", backend, "ready",
        ),
    )
    monkeypatch.setattr(
        "scripts.app.engine_management.build_vllm_status",
        lambda *_args, **_kwargs: EngineStatus("vllm", "missing", "", "", "", "missing"),
    )

    statuses = collect_engine_statuses(Engine, "cuda")

    assert [status.engine for status in statuses] == ["llamacpp", "llamacpp-vulkan", "vllm"]
    assert statuses[1].backend == "vulkan"


def test_missing_vulkan_runtime_on_macos_does_not_inherit_metal_backend(monkeypatch, tmp_path):
    class Engine:
        def __init__(self, name):
            self.name = name

        def runtime_location(self):
            return None if self.name == "llamacpp-vulkan" else str(tmp_path / self.name)

        def runtime_backend(self, _hardware_backend):
            if self.name == "llamacpp-vulkan":
                raise AssertionError("missing Vulkan runtime must not be probed")
            return "metal"

        def runtime_launcher(self):
            return None

        def external_server_url(self):
            return None

    monkeypatch.setattr("scripts.app.engine_management.platform.system", lambda: "Darwin")
    monkeypatch.setattr("scripts.app.engine_management.platform.release", lambda: "test")
    monkeypatch.setattr(
        "scripts.app.engine_management.build_vllm_status",
        lambda *_args, **_kwargs: EngineStatus("vllm", "missing", "", None, "", "missing"),
    )

    statuses = collect_engine_statuses(Engine, "metal")
    vulkan = next(status for status in statuses if status.engine == "llamacpp-vulkan")

    assert vulkan.ownership == "missing"
    assert vulkan.backend == ""
    assert engine_status_lines(vulkan) == [("Status", "Not installed")]


def test_engine_diagnostics_are_stable_machine_readable_json():
    statuses = [EngineStatus("llamacpp", "system_managed", "/bin/llama", "1", "cpu", "ready")]
    payload = json.loads(engine_diagnostics_text(statuses))
    assert payload["engines"]["llamacpp"]["ownership"] == "system_managed"
    assert payload["engines"]["llamacpp"]["version"] == "1"
    assert payload["imported_models"] == []


def test_vllm_update_support_matches_managed_backend():
    cuda = EngineStatus("vllm", "app_managed", "/app/vllm", "1", "cuda", "ready")
    rocm = EngineStatus("vllm", "app_managed", "/app/vllm", "1", "rocm", "ready")
    cuda_support = vllm_update_support(cuda, {}, "x86_64")
    rocm_support = vllm_update_support(rocm, {}, "x86_64")
    assert cuda_support is not None and cuda_support.method == "cuda_wheel"
    assert rocm_support is not None and rocm_support.method == "rocm_wheel"


def test_vllm_update_support_detects_dgx_spark_from_recorded_gpu():
    status = EngineStatus("vllm", "app_managed", "/app/vllm", "1", "cuda", "ready")
    setup = {"gpu": {"devices": [{"name": "NVIDIA GB10 Superchip", "backend": "cuda"}]}}
    support = vllm_update_support(status, setup, "aarch64")
    assert support is not None
    assert support.method == "cu130_wheel"


def test_vllm_update_support_rejects_non_managed_and_unknown_backends():
    system = EngineStatus("vllm", "system_managed", "/usr/bin/vllm", "1", "cuda", "ready")
    cpu = EngineStatus("vllm", "app_managed", "/app/vllm", "1", "cpu", "ready")
    assert vllm_update_support(system, {}, "x86_64") is None
    assert vllm_update_support(cpu, {}, "x86_64") is None
