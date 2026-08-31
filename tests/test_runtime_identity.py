from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.setup.runtime_identity import (
    RuntimeIdentity, engine_runtime_version, inspect_runtime, managed_distribution_version,
    parse_llamacpp_commit, parse_runtime_version,
    probe_vllm_server_health, probe_vllm_server_version, runtime_ownership,
    source_commit_version,
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


def test_parse_llamacpp_commit_reads_the_embedded_source_revision():
    assert parse_llamacpp_commit("version: 1 (A1B2C3D4)") == "a1b2c3d4"
    assert parse_llamacpp_commit("version: 7000") is None


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
    environment = {"LD_LIBRARY_PATH": "/opt/intel/oneapi/compiler/lib"}
    engine = type("Engine", (), {
        "runtime_location": lambda self: "/runtime/llama-server",
        "process_environment": lambda self: environment,
    })()
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout="version: 7000 (abcdef)", stderr="", returncode=0)

    version = engine_runtime_version(
        "llamacpp", engine, run=run,
    )
    assert version == "7000"
    assert calls[0][1]["env"] is environment


def test_managed_vllm_version_is_read_without_starting_the_cli(monkeypatch, tmp_path):
    metadata = tmp_path / "lib" / "python3.12" / "site-packages" / "vllm-0.27.1.dist-info" / "METADATA"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("Name: vllm\nVersion: 0.27.1+cu130\n", encoding="utf-8")
    monkeypatch.setattr("scripts.setup.runtime_identity.config.VLLM_VENV", tmp_path)
    engine = type("Engine", (), {"runtime_location": lambda self: tmp_path / "bin" / "vllm"})()

    assert managed_distribution_version(tmp_path, "vllm") == "0.27.1+cu130"
    assert engine_runtime_version(
        "vllm", engine, run=lambda *_args, **_kwargs: pytest.fail("CLI must not start"),
    ) == "0.27.1+cu130"


def test_source_build_version_uses_utc_commit_date_and_short_hash(tmp_path):
    (tmp_path / ".git").mkdir()
    identity = inspect_runtime(
        "llamacpp", tmp_path / "build" / "bin" / "llama-server", tmp_path,
        run=lambda *_args, **_kwargs: SimpleNamespace(
            stdout="version: 1 (a1b2c3d4e5f6)", stderr="", returncode=0,
        ),
    )
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="2026.08.11\n", stderr="", returncode=0)

    assert source_commit_version(identity, tmp_path, run=run) == "2026.08.11-a1b2c3d"
    assert calls[0][0][-1] == "a1b2c3d4e5f6"
    assert calls[1][0][-1] == "a1b2c3d4e5f6"
    assert calls[1][1]["env"]["TZ"] == "UTC"


def test_source_build_version_uses_exact_release_tag(tmp_path):
    (tmp_path / ".git").mkdir()
    identity = RuntimeIdentity(
        "llamacpp", "app_managed", "/runtime", "10362", "version: 10362 (a1b2c3d4)",
    )

    result = source_commit_version(
        identity, tmp_path,
        run=lambda *_args, **_kwargs: SimpleNamespace(
            stdout="b10362\n", stderr="", returncode=0,
        ),
    )

    assert result == "10362"


def test_source_build_version_does_not_inherit_head_tag_without_binary_commit(tmp_path):
    (tmp_path / ".git").mkdir()
    identity = RuntimeIdentity(
        "llamacpp", "app_managed", "/runtime", "0.3.0", "version: 0.3.0",
    )

    result = source_commit_version(
        identity, tmp_path,
        run=lambda *_args, **_kwargs: SimpleNamespace(
            stdout="b10615\n", stderr="", returncode=0,
        ),
    )

    assert result == "0.3.0"


def test_source_build_version_falls_back_when_git_metadata_is_unavailable(tmp_path):
    identity = inspect_runtime(
        "llamacpp", tmp_path / "llama-server", tmp_path,
        run=lambda *_args, **_kwargs: SimpleNamespace(
            stdout="version: 1 (a1b2c3d4)", stderr="", returncode=0,
        ),
    )
    assert source_commit_version(identity, tmp_path) == "1"


def test_engine_runtime_version_resolves_managed_llamacpp_source_commit(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    executable = tmp_path / "build" / "bin" / "llama-server"
    monkeypatch.setattr("scripts.setup.runtime_identity.config.LLAMACPP_DIR", tmp_path)
    engine = type("Engine", (), {"runtime_location": lambda self: executable})()

    def run(command, **_kwargs):
        if command[0] == "git":
            return SimpleNamespace(stdout="2026.08.11\n", stderr="", returncode=0)
        return SimpleNamespace(stdout="version: 1 (a1b2c3d4)", stderr="", returncode=0)

    assert engine_runtime_version("llamacpp", engine, run=run) == "2026.08.11-a1b2c3d"


def test_engine_runtime_version_uses_the_vulkan_managed_source_root(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    executable = tmp_path / "build" / "bin" / "llama-server"
    monkeypatch.setattr("scripts.setup.runtime_identity.config.LLAMACPP_VULKAN_DIR", tmp_path)
    engine = type("Engine", (), {"runtime_location": lambda self: executable})()

    def run(command, **_kwargs):
        if command[0] == "git":
            return SimpleNamespace(stdout="2026.08.12\n", stderr="", returncode=0)
        return SimpleNamespace(stdout="version: 2 (b2c3d4e5)", stderr="", returncode=0)

    assert engine_runtime_version(
        "llamacpp-vulkan", engine, run=run,
    ) == "2026.08.12-b2c3d4e"


def test_source_build_version_rejects_malformed_git_date(tmp_path):
    (tmp_path / ".git").mkdir()
    identity = RuntimeIdentity(
        "llamacpp", "app_managed", "/runtime", "1", "version: 1 (a1b2c3d4)",
    )
    run = lambda *_args, **_kwargs: SimpleNamespace(stdout="yesterday\n", stderr="", returncode=0)
    assert source_commit_version(identity, tmp_path, run=run) == "1"


def test_engine_runtime_version_is_absent_without_a_local_executable():
    engine = type("Engine", (), {"runtime_location": lambda self: None})()
    assert engine_runtime_version("vllm", engine) is None


def test_engine_runtime_version_probes_an_external_vllm_server(monkeypatch):
    engine = type("Engine", (), {"external_server_url": lambda self: "http://external:8000"})()
    monkeypatch.setattr(
        "scripts.setup.runtime_identity.probe_vllm_server_version",
        lambda url: "0.14.0" if url == "http://external:8000" else None,
    )
    assert engine_runtime_version("vllm", engine) == "0.14.0"


class VersionResponse:
    def __init__(self, payload, status=200): self.payload, self.status = payload, status
    def __enter__(self): return self
    def __exit__(self, *_args): return False
    def read(self, _size): return self.payload


def test_probe_vllm_server_version_reads_official_version_endpoint():
    seen = {}

    def open_version(request, timeout):
        seen["url"], seen["auth"], seen["timeout"] = (
            request.full_url, request.get_header("Authorization"), timeout,
        )
        return VersionResponse(b'{"version":"0.14.0"}')

    version = probe_vllm_server_version(
        "http://external:8000/", open_fn=open_version, env={"VLLM_API_KEY": "secret"},
    )
    assert version == "0.14.0"
    assert seen == {
        "url": "http://external:8000/version", "auth": "Bearer secret", "timeout": 3,
    }


def test_probe_vllm_server_version_tolerates_absent_or_invalid_endpoint():
    assert probe_vllm_server_version(
        "http://external", open_fn=lambda *_a, **_k: VersionResponse(b"not json"), env={},
    ) is None
    assert probe_vllm_server_version(
        "http://external", open_fn=lambda *_a, **_k: (_ for _ in ()).throw(OSError()), env={},
    ) is None


def test_probe_vllm_server_health_checks_status_and_authentication():
    seen = {}

    def open_health(request, timeout):
        seen["url"], seen["auth"], seen["timeout"] = (
            request.full_url, request.get_header("Authorization"), timeout,
        )
        return VersionResponse(b"", 204)

    assert probe_vllm_server_health(
        "http://external:8000/", open_fn=open_health, env={"VLLM_API_KEY": "secret"},
    )
    assert seen == {
        "url": "http://external:8000/health", "auth": "Bearer secret", "timeout": 3,
    }
    assert not probe_vllm_server_health(
        "http://external", open_fn=lambda *_a, **_k: (_ for _ in ()).throw(OSError()), env={},
    )
