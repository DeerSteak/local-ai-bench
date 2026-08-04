from llamacpp_tools import find_llamacpp_tool


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
        "llamacpp_tools.load_setup_config",
        lambda path: {"schema_version": 1, "llama_cpp": {"llama-server": str(configured)}},
    )
    assert find_llamacpp_tool(
        "llama-server", vendored_dir=vendored.parent,
        platform_name="Linux", which_fn=lambda _: None,
    ) == str(configured)
