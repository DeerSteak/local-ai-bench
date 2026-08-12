from pathlib import Path

from scripts.setup import llamacpp_install


def _log(_message):
    pass


def test_find_tool_uses_runtime_directory(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        llamacpp_install, "find_llamacpp_tool",
        lambda name, **kwargs: calls.append((name, kwargs)) or Path("/tool"),
    )

    assert llamacpp_install.find_tool("llama-server", tmp_path, "Linux") == Path("/tool")
    assert calls[0][1]["vendored_dir"] == tmp_path
    assert calls[0][1]["platform_name"] == "Linux"


def test_linux_install_requires_build_tools(monkeypatch, tmp_path):
    failures = []
    monkeypatch.setattr(llamacpp_install.shutil, "which", lambda _name: None)

    result = llamacpp_install.install(
        tmp_path / "runtime", tmp_path, "Linux", nvidia=False, rocm=False,
        compute_capability=None, max_cuda_version=None,
        info=_log, warn=_log, fail=failures.append, ok=_log,
    )

    assert result is False
    assert failures == ["git and cmake are required to build llama.cpp from source"]


def test_unknown_platform_is_not_installed(tmp_path):
    assert not llamacpp_install.install(
        tmp_path / "runtime", tmp_path, "Haiku", nvidia=False, rocm=False,
        compute_capability=None, max_cuda_version=None,
        info=_log, warn=_log, fail=_log, ok=_log,
    )
