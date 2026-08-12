import json

from scripts.app.engine_management import (
    engine_diagnostics_text, engine_status_lines, inspection_placeholder,
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


def test_inspection_placeholder_populates_every_visible_identity_field():
    lines = dict(engine_status_lines(inspection_placeholder("llamacpp")))
    assert lines == {
        "Ownership": "Inspecting…",
        "Location": "Inspecting…",
        "Version": "Inspecting…",
        "Backend": "Inspecting…",
        "Health": "Inspecting…",
    }


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
    assert support.method == "nightly_cu130"


def test_vllm_update_support_rejects_non_managed_and_unknown_backends():
    system = EngineStatus("vllm", "system_managed", "/usr/bin/vllm", "1", "cuda", "ready")
    cpu = EngineStatus("vllm", "app_managed", "/app/vllm", "1", "cpu", "ready")
    assert vllm_update_support(system, {}, "x86_64") is None
    assert vllm_update_support(cpu, {}, "x86_64") is None
