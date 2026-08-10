import json
from pathlib import Path
from types import SimpleNamespace

from scripts.setup.model_compatibility import (
    architecture_from_config, architecture_from_gguf, inspect_llamacpp_model,
    inspect_vllm_model, probe_vllm_architecture,
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
