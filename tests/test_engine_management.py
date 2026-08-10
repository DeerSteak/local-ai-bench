import json

from scripts.app.engine_management import engine_diagnostics_text, engine_status_lines
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


def test_engine_diagnostics_are_stable_machine_readable_json():
    statuses = [EngineStatus("llamacpp", "system_managed", "/bin/llama", "1", "cpu", "ready")]
    payload = json.loads(engine_diagnostics_text(statuses))
    assert payload["engines"]["llamacpp"]["ownership"] == "system_managed"
    assert payload["engines"]["llamacpp"]["version"] == "1"
    assert payload["imported_models"] == []
