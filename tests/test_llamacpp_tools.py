from pathlib import Path

from scripts.runtime.llamacpp_tools import (
    CUDA_BIN_DIRS, cuda_architecture, find_llamacpp_tool, find_nvcc,
)


def test_compute_capability_becomes_a_cmake_architecture():
    assert cuda_architecture("8.9") == "89"
    assert cuda_architecture("12.0") == "120"
    assert cuda_architecture(" 7.5 ") == "75"


def test_unreadable_compute_capability_yields_no_architecture():
    for value in (None, "", "native", "No CUDA devices found.", "8", "8.9.1", "x.y"):
        assert cuda_architecture(value) is None


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


def test_system_path_wins_over_project_vendored_binary(tmp_path):
    vendored = tmp_path / "vendor"
    vendored.mkdir()
    (vendored / "llama-server").write_text("vendored")
    assert find_llamacpp_tool(
        "llama-server", vendored_dir=vendored, platform_name="Linux",
        which_fn=lambda _: "/usr/local/bin/llama-server",
    ) == "/usr/local/bin/llama-server"


def test_project_binary_is_a_fallback_when_system_tool_is_missing(tmp_path):
    binary = tmp_path / "vendor" / "build" / "bin" / "llama-bench"
    binary.parent.mkdir(parents=True)
    binary.write_text("vendored")
    assert find_llamacpp_tool(
        "llama-bench", vendored_dir=tmp_path / "vendor", platform_name="Linux",
        which_fn=lambda _: None,
    ) == str(binary)


def test_macos_managed_binary_wins_over_homebrew_or_path(tmp_path):
    binary = tmp_path / "managed" / "bin" / "llama-server"
    binary.parent.mkdir(parents=True)
    binary.touch()
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
