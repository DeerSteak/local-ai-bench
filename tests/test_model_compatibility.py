import json
from pathlib import Path
from types import SimpleNamespace

from scripts.setup.model_compatibility import (
    architecture_from_config, architecture_from_gguf, inspect_llamacpp_model,
    inspect_vllm_model, llamacpp_probe_command, probe_llamacpp_load,
    probe_vllm_architecture,
)


def test_architecture_from_config_reads_first_declared_architecture(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"architectures": ["MuseGlimmerForCausalLM"]}), encoding="utf-8")
    assert architecture_from_config(path) == "MuseGlimmerForCausalLM"
    path.write_text("{}", encoding="utf-8")
    assert architecture_from_config(path) is None


def test_architecture_from_gguf_reads_general_architecture():
    class Field:
        def contents(self): return b"muse-glimmer"

    class Reader:
        fields = {"general.architecture": Field()}

    assert architecture_from_gguf(Path("model.gguf"), lambda _path: Reader()) == "muse-glimmer"


def test_vllm_architecture_probe_reports_registry_verdict(tmp_path):
    supported = SimpleNamespace(stdout="supported\n", stderr="", returncode=0)
    unsupported = SimpleNamespace(stdout="unsupported\n", stderr="", returncode=0)
    assert probe_vllm_architecture(tmp_path / "python", "Muse", run=lambda *_a, **_k: supported)[0] == "supported"
    assert probe_vllm_architecture(tmp_path / "python", "Muse", run=lambda *_a, **_k: unsupported)[0] == "unsupported"


def test_vllm_architecture_probe_handles_missing_inputs_without_execution():
    fail = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not execute"))
    assert probe_vllm_architecture(None, "Muse", run=fail)[0] == "unavailable"
    assert probe_vllm_architecture(Path("python"), None, run=fail)[0] == "unknown"


def test_model_inspections_return_engine_specific_status(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"architectures": ["Muse"]}), encoding="utf-8")
    result = SimpleNamespace(stdout="unsupported", stderr="", returncode=0)
    vllm = inspect_vllm_model("muse", config, tmp_path / "python", run=lambda *_a, **_k: result)
    assert vllm.status == "unsupported" and vllm.architecture == "Muse"

    class Field:
        def contents(self): return "muse"
    class Reader:
        fields = {"general.architecture": Field()}
    import scripts.setup.model_compatibility as module
    original = module.architecture_from_gguf
    module.architecture_from_gguf = lambda _path: original(_path, lambda _value: Reader())
    try:
        llama = inspect_llamacpp_model("muse", tmp_path / "model.gguf")
    finally:
        module.architecture_from_gguf = original
    assert llama.status == "load_probe_required" and llama.architecture == "muse"


def test_llamacpp_probe_command_is_bounded_cpu_only_and_loopback(tmp_path):
    command = llamacpp_probe_command("/bin/llama-server", tmp_path / "model.gguf", 4123)
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "4123"
    assert command[command.index("-ngl") + 1] == "0"
    assert command[command.index("-c") + 1] == "512"


def test_llamacpp_load_probe_reports_health_and_terminates(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.setup.model_compatibility.architecture_from_gguf", lambda _path: "muse")

    class Process:
        def __init__(self): self.terminated = False
        def poll(self): return None
        def terminate(self): self.terminated = True
        def wait(self, timeout): return 0

    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *_args): return False

    process = Process()
    result = probe_llamacpp_load(
        "muse", tmp_path / "model.gguf", "/bin/llama-server",
        popen=lambda *args, **kwargs: process, open_fn=lambda *args, **kwargs: Response(),
        port_factory=lambda: 4123,
    )
    assert result.status == "supported"
    assert result.detail == "Model loaded successfully CPU-only; GPU compatibility was not tested."
    assert process.terminated


def test_llamacpp_load_probe_reports_early_process_failure(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.setup.model_compatibility.architecture_from_gguf", lambda _path: "muse")

    class Process:
        def poll(self): return 1

    result = probe_llamacpp_load(
        "muse", tmp_path / "model.gguf", "/bin/llama-server",
        popen=lambda *args, **kwargs: Process(), port_factory=lambda: 4123,
    )
    assert result.status == "load_failed"


def test_llamacpp_load_probe_times_out_and_kills_process(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.setup.model_compatibility.architecture_from_gguf", lambda _path: None)

    class Process:
        def __init__(self): self.killed = False
        def poll(self): return None
        def terminate(self): pass
        def wait(self, timeout):
            if not self.killed:
                raise __import__("subprocess").TimeoutExpired("probe", timeout)
            return 0
        def kill(self): self.killed = True

    process = Process()
    ticks = iter((0.0, 0.0, 2.0))
    result = probe_llamacpp_load(
        "muse", tmp_path / "model.gguf", "/bin/llama-server", timeout=1,
        popen=lambda *args, **kwargs: process,
        open_fn=lambda *args, **kwargs: (_ for _ in ()).throw(OSError()),
        monotonic=lambda: next(ticks), sleep=lambda _seconds: None,
        port_factory=lambda: 4123,
    )
    assert result.status == "timed_out"
    assert process.killed


def test_llamacpp_load_probe_honors_cancellation(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.setup.model_compatibility.architecture_from_gguf", lambda _path: "muse")

    class Process:
        def __init__(self): self.terminated = False
        def poll(self): return None
        def terminate(self): self.terminated = True
        def wait(self, timeout): return 0

    process = Process()
    class Control:
        cancelled = True
        def track_process(self, tracked): tracked.terminate()
        def clear_process(self, _tracked): pass

    result = probe_llamacpp_load(
        "muse", tmp_path / "model.gguf", "/bin/llama-server",
        popen=lambda *args, **kwargs: process, control=Control(),
        port_factory=lambda: 4123,
    )
    assert result.status == "cancelled"
    assert process.terminated
