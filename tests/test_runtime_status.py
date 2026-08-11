import json
from pathlib import Path
from types import SimpleNamespace

from scripts.setup.runtime_status import (
    build_llamacpp_status, build_vllm_status, parse_vllm_environment,
    probe_vllm_environment, runtime_python,
)


def test_runtime_python_uses_executable_sibling(tmp_path):
    executable = tmp_path / "bin" / "vllm"
    python = executable.with_name("python")
    assert runtime_python(executable, exists_fn=lambda path: path == python) == python
    assert runtime_python(executable, exists_fn=lambda _path: False) is None


def test_vllm_environment_parser_accepts_only_known_scalar_fields():
    payload = json.dumps({
        "vllm": "0.26.0", "torch": "2.11.0", "cuda_runtime": "13.0",
        "cuda_available": True, "python": "3.12.13", "extra": ["ignored"],
    })
    assert parse_vllm_environment(payload) == {
        "vllm": "0.26.0", "torch": "2.11.0", "cuda_runtime": "13.0",
        "cuda_available": True, "python": "3.12.13",
    }
    assert parse_vllm_environment("not json") == {}


def test_probe_reports_import_failure_without_raising(tmp_path):
    result = SimpleNamespace(stdout="", stderr="vllm import failed", returncode=1)
    components, warning = probe_vllm_environment(
        tmp_path / "python", run=lambda *_args, **_kwargs: result,
    )
    assert components == {}
    assert warning == "vllm import failed"


def test_llamacpp_status_combines_identity_backend_and_health(tmp_path):
    result = SimpleNamespace(stdout="version: 6527\n", stderr="", returncode=0)
    status = build_llamacpp_status(
        tmp_path / "llama.cpp" / "bin" / "llama-server", tmp_path / "llama.cpp", "metal",
        run=lambda *_args, **_kwargs: result,
    )
    assert status.managed and status.version == "6527"
    assert status.backend == "metal" and status.health == "ready"


def test_vllm_status_reports_dependency_stack_and_wsl_policy(tmp_path):
    executable = tmp_path / "vllm-env" / "bin" / "vllm"
    python = executable.with_name("python")
    python.parent.mkdir(parents=True)
    python.touch()

    def run(command, **_kwargs):
        if command[-1] == "--version":
            return SimpleNamespace(stdout="vllm 0.26.0", stderr="", returncode=0)
        assert command[0] == str(python)
        return SimpleNamespace(stdout=json.dumps({
            "vllm": "0.26.0", "torch": "2.11.0", "cuda_runtime": "13.0",
            "rocm_runtime": None, "cuda_available": True, "python": "3.12.13",
        }), stderr="", returncode=0)

    status = build_vllm_status(
        executable, tmp_path / "vllm-env", "cuda", is_wsl=True,
        env={"VLLM_WSL2_ENABLE_PIN_MEMORY": "1"}, run=run,
    )
    assert status.managed and status.health == "ready"
    assert status.components["torch"] == "2.11.0"
    assert status.components["wsl_pin_memory"] == "1"


def test_external_and_launcher_vllm_status_do_not_probe_local_python(tmp_path):
    fail = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not execute"))
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def read(self, _size): return b'{"version":"0.14.0"}'
    external = build_vllm_status(
        None, tmp_path / "vllm-env", "cuda", server_url="http://localhost:8000", run=fail,
        open_fn=lambda *_args, **_kwargs: Response(),
    )
    launcher = build_vllm_status(
        None, tmp_path / "vllm-env", "rocm", launcher="/usr/bin/vllm-launch", run=fail,
    )
    assert external.ownership == "external_server" and external.health == "ready"
    assert external.version == "0.14.0"
    assert launcher.ownership == "platform_launcher" and launcher.health == "ready"


def test_managed_wsl_status_reports_injected_pin_memory_default(tmp_path):
    executable = tmp_path / "vllm-env" / "bin" / "vllm"
    python = executable.with_name("python")
    python.parent.mkdir(parents=True)
    python.touch()

    def run(command, **_kwargs):
        if command[-1] == "--version":
            return SimpleNamespace(stdout="vllm 0.26.0", stderr="", returncode=0)
        return SimpleNamespace(stdout=json.dumps({"vllm": "0.26.0"}), stderr="", returncode=0)

    status = build_vllm_status(
        executable, tmp_path / "vllm-env", "cuda", is_wsl=True, env={}, run=run,
    )
    assert status.components["wsl_pin_memory"] == "1"


def test_external_vllm_status_uses_health_when_version_is_unavailable(tmp_path):
    class Response:
        status = 200
        def __init__(self, payload): self.payload = payload
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def read(self, _size): return self.payload

    def healthy(request, **_kwargs):
        return Response(b"not json" if request.full_url.endswith("/version") else b"")

    ready = build_vllm_status(
        None, tmp_path, "cuda", server_url="http://external", open_fn=healthy, env={},
    )
    offline = build_vllm_status(
        None, tmp_path, "cuda", server_url="http://external",
        open_fn=lambda *_a, **_k: (_ for _ in ()).throw(OSError()), env={},
    )
    assert ready.health == "ready" and ready.version is None
    assert offline.health == "unavailable"
    assert any("/health" in warning for warning in offline.warnings)
