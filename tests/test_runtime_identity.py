from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.setup.runtime_identity import (
    engine_runtime_version, inspect_runtime, parse_runtime_version, runtime_ownership,
)


@pytest.mark.parametrize("output,expected", [
    ("vllm 0.26.0", "0.26.0"),
    ("version: 6527 (abcdef)", "6527"),
    ("llama-server version 1.2.3", "1.2.3"),
    ("0.25.1rc1", "0.25.1rc1"),
    ("unknown build", None),
])
def test_parse_runtime_version_handles_engine_formats(output, expected):
    assert parse_runtime_version(output) == expected


def test_runtime_ownership_distinguishes_managed_system_external_and_missing(tmp_path):
    managed = tmp_path / "runtime"
    executable = managed / "bin" / "vllm"
    assert runtime_ownership(executable, managed) == "app_managed"
    assert runtime_ownership(Path("/usr/bin/vllm"), managed) == "system_managed"
    assert runtime_ownership("http://localhost:8000", managed) == "external_server"
    assert runtime_ownership(None, managed) == "missing"


def test_inspect_runtime_reports_version_without_mutating_runtime(tmp_path):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="vllm 0.26.0\n", stderr="", returncode=0)

    identity = inspect_runtime("vllm", tmp_path / "vllm-env" / "bin" / "vllm",
                               tmp_path / "vllm-env", run=run)

    assert identity.managed and identity.version == "0.26.0"
    assert calls[0][0][-1] == "--version"
    assert calls[0][1]["timeout"] == 15


def test_inspect_runtime_does_not_execute_external_server(tmp_path):
    identity = inspect_runtime(
        "vllm", "http://localhost:8000", tmp_path,
        run=lambda *_args, **_kwargs: pytest.fail("external server must not be executed"),
    )

    assert identity.ownership == "external_server"
    assert identity.version is None


def test_engine_runtime_version_uses_the_engine_runtime_descriptor():
    engine = type("Engine", (), {"runtime_location": lambda self: "/runtime/llama-server"})()
    version = engine_runtime_version(
        "llamacpp", engine,
        run=lambda *args, **kwargs: SimpleNamespace(
            stdout="version: 7000 (abcdef)", stderr="", returncode=0,
        ),
    )
    assert version == "7000"


def test_engine_runtime_version_is_absent_without_a_local_executable():
    engine = type("Engine", (), {"runtime_location": lambda self: None})()
    assert engine_runtime_version("vllm", engine) is None
