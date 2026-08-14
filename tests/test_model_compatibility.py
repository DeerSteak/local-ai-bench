import json
from pathlib import Path
from types import SimpleNamespace

from scripts.setup.model_compatibility import (
    CompatibilityCheck, architecture_from_config, architecture_from_gguf,
    chat_template_check, context_capacity_check, declared_context_length,
    gguf_metadata, inspect_llamacpp_model,
    inspect_vllm_model, llamacpp_probe_command, probe_llamacpp_load,
    preflight_verdict, probe_vllm_architecture, tool_support_check,
    weight_completeness_check,
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


def test_gguf_metadata_reports_unreadable_input():
    def fail(_path):
        raise ValueError("malformed metadata")

    metadata, error = gguf_metadata(Path("model.gguf"), fail)
    assert metadata == {}
    assert error == "malformed metadata"


def test_chat_template_check_distinguishes_embedded_absent_and_unreadable():
    embedded = chat_template_check({"tokenizer.chat_template": b"{{ messages }}"})
    absent = chat_template_check({})
    unreadable = chat_template_check({}, "bad header")
    assert (embedded.status, embedded.evidence["source"]) == ("passed", "tokenizer.chat_template")
    assert (absent.status, absent.severity) == ("absent", "warning")
    assert (unreadable.status, unreadable.severity) == ("unreadable", "warning")


def test_context_capacity_check_uses_declared_and_engine_limits():
    metadata = {
        "muse.context_length": 8192,
        "muse.rope.scaling.original_context_length": 2048,
    }
    assert declared_context_length(metadata) == 8192
    assert context_capacity_check(8192, 4096, 16384).status == "passed"
    assert context_capacity_check(8192, 12288, 16384).status == "exceeded"
    assert context_capacity_check(16384, 12288, 8192).status == "exceeded"
    assert context_capacity_check(None, 4096, 8192).status == "unknown"


def test_tool_support_check_blocks_only_the_tool_workload():
    blocked = tool_support_check(False, True)
    assert blocked.severity == "workload_blocking"
    assert blocked.scope == "tool"
    assert tool_support_check(False, False).status == "not_applicable"
    assert tool_support_check(True, True).status == "passed"


def test_weight_completeness_rejects_missing_and_empty_declared_files(tmp_path):
    present = tmp_path / "part-1.gguf"
    empty = tmp_path / "part-2.gguf"
    present.write_bytes(b"weights")
    empty.touch()
    result = weight_completeness_check([present, empty, tmp_path / "part-3.gguf"])
    assert (result.status, result.severity) == ("incomplete", "hard_failure")
    assert result.evidence["empty"] == [str(empty)]
    assert weight_completeness_check([present]).status == "passed"
    assert weight_completeness_check([]).status == "incomplete"


def test_preflight_verdict_force_all_bypasses_warnings_not_hard_failures():
    warning = CompatibilityCheck("template", "absent", "warning", "missing")
    hard = CompatibilityCheck("weights", "incomplete", "hard_failure", "missing")
    limited = CompatibilityCheck("tools", "unsupported", "workload_blocking", "unsupported")
    assert preflight_verdict((warning,), force_all=False) == "warning"
    assert preflight_verdict((warning,), force_all=True) == "passed"
    assert preflight_verdict((hard,), force_all=True) == "excluded"
    assert preflight_verdict((limited,), force_all=True) == "workload_limited"


def test_model_compatibility_serializes_passing_checks():
    check = CompatibilityCheck("weights", "passed", "info", "complete")
    report = __import__(
        "scripts.setup.model_compatibility", fromlist=["ModelCompatibility"]
    ).ModelCompatibility("llamacpp", "muse", "muse", "passed", "ok", (check,))
    assert report.to_dict()["checks"] == [check.to_dict()]


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
