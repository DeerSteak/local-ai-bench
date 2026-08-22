from pathlib import Path
from types import SimpleNamespace

from scripts.runtime.llamacpp_tools import (
    CUDA_BIN_DIRS, cuda_architecture, find_llamacpp_tool, find_nvcc,
    llamacpp_backend_error, llamacpp_backend_mismatch, probe_llamacpp_backend,
)


def test_compute_capability_becomes_a_cmake_architecture():
    assert cuda_architecture("8.9") == "89"
    assert cuda_architecture("12.0") == "120"
    assert cuda_architecture(" 7.5 ") == "75"


def test_unreadable_compute_capability_yields_no_architecture():
    for value in (None, "", "native", "No CUDA devices found.", "8", "8.9.1", "x.y"):
        assert cuda_architecture(value) is None


def test_llamacpp_backend_probe_uses_the_explicit_binary_and_both_output_streams():
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="Available devices:\n", stderr="HIP0: AMD")

    assert probe_llamacpp_backend(
        "/managed/llama-server", env={"RUNTIME": "ready"}, run=run,
    ) == "rocm"
    assert calls[0][0] == ["/managed/llama-server", "--list-devices"]
    assert calls[0][1]["env"] == {"RUNTIME": "ready"}


def test_llamacpp_backend_probe_does_not_guess_after_a_failed_command():
    run = lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr="failed")
    assert probe_llamacpp_backend("llama-server", run=run) is None


def test_required_backend_rejects_wrong_or_unverifiable_builds():
    assert llamacpp_backend_mismatch("llama-server", "cpu", "cuda")
    assert llamacpp_backend_mismatch("llama-server", None, "cuda")
    assert not llamacpp_backend_mismatch("llama-server", "cuda", "cuda")
    assert not llamacpp_backend_mismatch(None, None, "cuda")


def test_backend_error_uses_the_shared_probe_contract():
    calls = []

    def probe(binary, **kwargs):
        calls.append((binary, kwargs))
        return "cpu"

    error = llamacpp_backend_error(
        "llama-server", "rocm", env={"HIP": "1"}, run="runner",
        probe=probe, context="managed update",
    )
    assert error == "managed update requires rocm, but installed llama.cpp exposes cpu"
    assert calls == [("llama-server", {"env": {"HIP": "1"}, "run": "runner"})]


def test_nvcc_on_path_is_preferred_over_a_toolkit_directory():
    assert find_nvcc(which_fn=lambda _name: "/usr/bin/nvcc",
                     exists_fn=lambda _path: True) == "/usr/bin/nvcc"


def test_nvcc_is_found_in_the_toolkit_directory_when_not_on_path():
    expected = str(Path(CUDA_BIN_DIRS[0]) / "nvcc")
    found = find_nvcc(which_fn=lambda _name: None,
                      exists_fn=lambda path: str(path) == expected)
    assert found == expected


def test_nvcc_is_absent_when_neither_path_nor_toolkit_has_it():
    assert find_nvcc(which_fn=lambda _name: None, exists_fn=lambda _path: False) is None


def test_system_path_wins_over_an_incomplete_project_toolset(tmp_path):
    vendored = tmp_path / "vendor"
    vendored.mkdir()
    (vendored / "llama-server").write_text("vendored")
    assert find_llamacpp_tool(
        "llama-server", vendored_dir=vendored, platform_name="Linux",
        which_fn=lambda _: "/usr/local/bin/llama-server",
    ) == "/usr/local/bin/llama-server"


def test_complete_project_toolset_wins_as_one_coherent_runtime(tmp_path):
    vendored = tmp_path / "vendor" / "build" / "bin"
    vendored.mkdir(parents=True)
    for name in ("llama-server", "llama-bench", "llama-batched-bench"):
        (vendored / name).write_text(name)
    for name in ("llama-server", "llama-bench", "llama-batched-bench"):
        assert find_llamacpp_tool(
            name, vendored_dir=tmp_path / "vendor", platform_name="Linux",
            which_fn=lambda requested: f"/usr/bin/{requested}",
        ) == str(vendored / name)


def test_managed_tools_from_different_directories_are_not_mixed(tmp_path):
    managed = tmp_path / "vendor"
    for directory, name in (
        ("old", "llama-server"),
        ("new", "llama-bench"),
        ("new", "llama-batched-bench"),
    ):
        path = managed / directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    assert find_llamacpp_tool(
        "llama-bench", vendored_dir=managed, platform_name="Linux",
        which_fn=lambda requested: f"/usr/bin/{requested}",
    ) == "/usr/bin/llama-bench"


def test_project_binary_is_a_fallback_when_system_tool_is_missing(tmp_path):
    binary = tmp_path / "vendor" / "build" / "bin" / "llama-bench"
    binary.parent.mkdir(parents=True)
    binary.write_text("vendored")
    assert find_llamacpp_tool(
        "llama-bench", vendored_dir=tmp_path / "vendor", platform_name="Linux",
        which_fn=lambda _: None,
    ) == str(binary)


def test_macos_managed_binary_wins_over_homebrew_or_path(tmp_path):
    managed = tmp_path / "managed" / "bin"
    managed.mkdir(parents=True)
    for name in ("llama-server", "llama-bench", "llama-batched-bench"):
        (managed / name).touch()
    binary = managed / "llama-server"
    assert find_llamacpp_tool(
        "llama-server", vendored_dir=tmp_path / "managed", platform_name="Darwin",
        which_fn=lambda _: "/opt/homebrew/bin/llama-server",
    ) == str(binary)


def test_source_directory_is_not_mistaken_for_a_binary(tmp_path):
    (tmp_path / "tools" / "llama-batched-bench").mkdir(parents=True)
    assert find_llamacpp_tool(
        "llama-batched-bench", vendored_dir=tmp_path, platform_name="Linux",
        which_fn=lambda _: None,
    ) is None


def test_missing_system_and_vendored_tool_returns_none(tmp_path):
    assert find_llamacpp_tool(
        "llama-server", vendored_dir=tmp_path / "missing", platform_name="Linux",
        which_fn=lambda _: None,
    ) is None


def test_windows_vendored_fallback_uses_exe_suffix(tmp_path):
    binary = tmp_path / "llama-server.exe"
    binary.write_text("vendored")
    assert find_llamacpp_tool(
        "llama-server", vendored_dir=tmp_path, platform_name="Windows",
        which_fn=lambda _: None,
    ) == str(binary)


def test_valid_configured_tool_wins_over_vendored_copy(tmp_path, monkeypatch):
    configured = tmp_path / "configured" / "llama-server"
    configured.parent.mkdir()
    configured.touch()
    vendored = tmp_path / "vendored" / "llama-server"
    vendored.parent.mkdir()
    vendored.touch()
    monkeypatch.setattr(
        "scripts.runtime.llamacpp_tools.load_setup_config",
        lambda path: {"schema_version": 1, "llama_cpp": {"llama-server": str(configured)}},
    )
    assert find_llamacpp_tool(
        "llama-server", vendored_dir=vendored.parent,
        platform_name="Linux", which_fn=lambda _: None,
    ) == str(configured)
