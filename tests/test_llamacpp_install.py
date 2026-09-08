from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_find_tools_discovers_the_complete_runtime_toolset(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        llamacpp_install, "find_tool",
        lambda name, runtime_dir, platform_name: calls.append(
            (name, runtime_dir, platform_name)
        ) or f"/{name}",
    )
    assert llamacpp_install.find_tools(tmp_path, "Linux") == {
        "llama-server": "/llama-server",
        "llama-bench": "/llama-bench",
        "llama-batched-bench": "/llama-batched-bench",
    }
    assert [call[0] for call in calls] == [
        "llama-server", "llama-bench", "llama-batched-bench",
    ]


def test_managed_toolset_requires_all_three_tools(tmp_path):
    directory = tmp_path / "build" / "bin"
    directory.mkdir(parents=True)
    for name in ("llama-server", "llama-bench", "llama-batched-bench"):
        (directory / name).touch()
    assert llamacpp_install.managed_toolset_ready(tmp_path, "Linux")
    (directory / "llama-batched-bench").unlink()
    assert not llamacpp_install.managed_toolset_ready(tmp_path, "Linux")


def test_qualification_rebuilds_existing_runtime_with_wrong_or_unverifiable_backend():
    assert llamacpp_install.qualification_backend_mismatch("llama-server", "cpu", "rocm")
    assert llamacpp_install.qualification_backend_mismatch("llama-server", None, "rocm")
    assert not llamacpp_install.qualification_backend_mismatch("llama-server", "rocm", "rocm")
    assert not llamacpp_install.qualification_backend_mismatch(None, None, "rocm")
    assert not llamacpp_install.qualification_backend_mismatch("llama-server", "cpu", None)


def test_qualification_rejects_an_installed_runtime_that_does_not_expose_required_backend():
    assert llamacpp_install.qualification_backend_error(
        "llama-server", "xpu", probe=lambda _binary: "xpu",
    ) is None
    assert llamacpp_install.qualification_backend_error(
        "llama-server", "xpu", probe=lambda _binary: None,
    ) == "qualification requires xpu, but installed llama.cpp exposes no backend"
    assert llamacpp_install.qualification_backend_error(
        None, "xpu", probe=lambda _binary: "vulkan",
    ) is None


def test_unknown_platform_is_not_installed(tmp_path):
    assert not llamacpp_install.install(
        tmp_path / "runtime", tmp_path, "Haiku", nvidia=False, rocm=False, intel_xpu=False,
        compute_capability=None, max_cuda_version=None,
        info=_log, warn=_log, fail=_log, ok=_log,
    )


def test_macos_install_resolves_requested_release(monkeypatch, tmp_path):
    requested = []
    monkeypatch.setattr(llamacpp_install.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        llamacpp_install, "fetch_llamacpp_release_tag",
        lambda tag: requested.append(tag) or {"tag_name": tag},
    )

    def update(_runtime, _system, _machine, _backend, *, release_fetcher, **kwargs):
        assert release_fetcher()["tag_name"] == "b7000"
        return SimpleNamespace(success=True, detail="installed")

    monkeypatch.setattr(llamacpp_install, "install_managed_llamacpp", update)
    assert llamacpp_install.install(
        tmp_path / "runtime", tmp_path, "Darwin", nvidia=False, rocm=False, intel_xpu=False,
        compute_capability=None, max_cuda_version=None, version="b7000",
        info=_log, warn=_log, fail=_log, ok=_log,
    )
    assert requested == ["b7000"]


def test_macos_rejects_managed_vulkan_runtime(tmp_path):
    failures = []
    assert not llamacpp_install.install(
        tmp_path / "runtime", tmp_path, "Darwin",
        nvidia=False, rocm=False, intel_xpu=False,
        compute_capability=None, max_cuda_version=None, vulkan=True,
        info=_log, warn=_log, fail=failures.append, ok=_log,
    )
    assert failures == [
        "The managed Vulkan llama.cpp runtime is available only on Windows and Linux",
    ]


def _windows_release(*assets):
    return {"tag_name": "b9999", "assets": list(assets)}


def _asset(name, size=1024):
    return {"name": name, "size": size, "browser_download_url": f"https://example/{name}"}


def _install_windows(monkeypatch, tmp_path, release, *, intel_xpu=False, vulkan=False,
                     **callbacks):
    monkeypatch.setattr(llamacpp_install, "fetch_llamacpp_release", lambda: release)
    logs = {name: [] for name in ("info", "warn", "fail", "ok")}
    logs.update(callbacks)
    result = llamacpp_install.install_windows(
        tmp_path / "runtime", tmp_path / "downloads", "12.8",
        intel_xpu=intel_xpu, vulkan=vulkan,
        info=logs["info"].append, warn=logs["warn"].append,
        fail=logs["fail"].append, ok=logs["ok"].append,
    )
    return result, logs


def test_windows_install_falls_back_to_vulkan_without_cuda_pair(monkeypatch, tmp_path):
    release = _windows_release(_asset("llama-win-vulkan-x64.zip"))

    def download(_url, archive, **_kwargs):
        archive.parent.mkdir(exist_ok=True)
        archive.touch()

    def extract(_archive, runtime):
        runtime.mkdir(exist_ok=True)
        (runtime / "llama-server.exe").touch()

    monkeypatch.setattr(llamacpp_install, "download_file", download)
    monkeypatch.setattr(llamacpp_install, "safe_extract_zip", extract)
    result, logs = _install_windows(monkeypatch, tmp_path, release)
    assert result is True
    assert any("Vulkan" in message for message in logs["ok"])
    assert len(logs["warn"]) == 2


def test_windows_explicit_vulkan_ignores_available_cuda_package(monkeypatch, tmp_path):
    release = _windows_release(
        _asset("llama-b9999-bin-win-cuda-12.8-x64.zip"),
        _asset("cudart-llama-bin-win-cuda-12.8-x64.zip"),
        _asset("llama-win-vulkan-x64.zip"),
    )
    selected = []

    def download(_url, archive, **_kwargs):
        selected.append(archive.name)
        archive.parent.mkdir(exist_ok=True)
        archive.touch()

    def extract(_archive, runtime):
        runtime.mkdir(exist_ok=True)
        for name in ("llama-server", "llama-bench", "llama-batched-bench"):
            (runtime / f"{name}.exe").touch()

    monkeypatch.setattr(llamacpp_install, "download_file", download)
    monkeypatch.setattr(llamacpp_install, "safe_extract_zip", extract)
    result, _logs = _install_windows(
        monkeypatch, tmp_path, release, vulkan=True,
    )
    assert result is True
    assert selected == ["llama-win-vulkan-x64.zip"]


def test_windows_install_fails_without_vulkan_fallback(monkeypatch, tmp_path):
    result, logs = _install_windows(
        monkeypatch, tmp_path, _windows_release(_asset("source.zip")),
    )
    assert result is False
    assert logs["fail"] == ["No Windows Vulkan build found in the latest llama.cpp release"]


def test_windows_intel_install_uses_sycl_and_never_falls_back_to_vulkan(monkeypatch, tmp_path):
    release = _windows_release(
        _asset("llama-win-sycl-x64.zip"), _asset("llama-win-vulkan-x64.zip"),
    )

    def download(_url, archive, **_kwargs):
        archive.parent.mkdir(exist_ok=True)
        archive.touch()

    def extract(archive, runtime):
        assert "sycl" in archive.name
        runtime.mkdir(exist_ok=True)
        for name in ("llama-server", "llama-bench", "llama-batched-bench"):
            (runtime / f"{name}.exe").touch()

    monkeypatch.setattr(llamacpp_install, "download_file", download)
    monkeypatch.setattr(llamacpp_install, "safe_extract_zip", extract)
    result, logs = _install_windows(
        monkeypatch, tmp_path, release, intel_xpu=True,
    )
    assert result is True
    assert any("SYCL" in message for message in logs["ok"])


def test_windows_intel_install_fails_instead_of_using_vulkan_without_sycl(
        monkeypatch, tmp_path):
    result, logs = _install_windows(
        monkeypatch, tmp_path,
        _windows_release(_asset("llama-win-vulkan-x64.zip")), intel_xpu=True,
    )
    assert result is False
    assert logs["fail"] == ["No Windows SYCL build found in the latest llama.cpp release"]


def test_windows_intel_install_replaces_existing_vulkan_runtime(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "ggml-vulkan.dll").touch()
    calls = []
    monkeypatch.setattr(
        llamacpp_install, "update_windows_llamacpp",
        lambda target, max_cuda, **kwargs: calls.append((target, max_cuda, kwargs))
        or SimpleNamespace(success=True, detail="replaced"),
    )
    result, logs = _install_windows(
        monkeypatch, tmp_path,
        _windows_release(_asset("llama-win-sycl-x64.zip")), intel_xpu=True,
    )
    assert result is True
    assert calls[0][0:2] == (runtime, "12.8")
    assert calls[0][2]["intel_xpu"] is True
    assert calls[0][2]["release_fetcher"]()["tag_name"] == "b9999"
    assert any("replaced" in message for message in logs["ok"])


def test_windows_vulkan_update_preserves_explicit_backend(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    calls = []
    monkeypatch.setattr(
        llamacpp_install, "update_windows_llamacpp",
        lambda target, max_cuda, **kwargs: calls.append(kwargs)
        or SimpleNamespace(success=True, detail="replaced"),
    )
    result, _logs = _install_windows(
        monkeypatch, tmp_path, _windows_release(_asset("llama-win-vulkan-x64.zip")),
        vulkan=True,
    )
    assert result is True
    assert calls[0]["vulkan"] is True


def test_windows_install_reports_release_fetch_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        llamacpp_install, "fetch_llamacpp_release",
        lambda: (_ for _ in ()).throw(OSError("offline")),
    )
    failures = []
    result = llamacpp_install.install_windows(
        tmp_path / "runtime", tmp_path, None,
        info=_log, warn=_log, fail=failures.append, ok=_log,
    )
    assert result is False
    assert failures == ["Could not fetch llama.cpp release info: offline"]


@pytest.mark.parametrize("failure_stage", ["download", "extract"])
def test_windows_install_cleans_all_archives_after_failure(monkeypatch, tmp_path, failure_stage):
    assets = [
        _asset("llama-b9999-bin-win-cuda-12.8-x64.zip"),
        _asset("cudart-llama-bin-win-cuda-12.8-x64.zip"),
    ]

    def download(_url, archive, **_kwargs):
        archive.parent.mkdir(exist_ok=True)
        archive.touch()
        if failure_stage == "download" and archive.name.startswith("cudart-"):
            raise OSError("download failed")

    def extract(_archive, _runtime):
        if failure_stage == "extract":
            raise ValueError("unsafe archive")

    monkeypatch.setattr(llamacpp_install, "download_file", download)
    monkeypatch.setattr(llamacpp_install, "safe_extract_zip", extract)
    result, _logs = _install_windows(monkeypatch, tmp_path, _windows_release(*assets))
    assert result is False
    assert not list((tmp_path / "downloads").glob("*.zip"))


def test_windows_install_requires_server_after_extraction(monkeypatch, tmp_path):
    asset = _asset("llama-win-vulkan-x64.zip")

    def download(_url, archive, **_kwargs):
        archive.parent.mkdir(exist_ok=True)
        archive.touch()

    monkeypatch.setattr(llamacpp_install, "download_file", download)
    monkeypatch.setattr(llamacpp_install, "safe_extract_zip", lambda *_args: None)
    result, logs = _install_windows(monkeypatch, tmp_path, _windows_release(asset))
    assert result is False
    assert "llama-server.exe wasn't found" in logs["fail"][0]


def test_installed_toolset_error_rejects_missing_managed_tools():
    assert "incomplete" in (llamacpp_install.installed_toolset_error(None, "cuda") or "")


@pytest.mark.parametrize("error", [None, "setup requires cuda, but installed llama.cpp exposes cpu"])
def test_installed_toolset_error_retains_backend_validation(monkeypatch, error):
    calls = []
    monkeypatch.setattr(
        llamacpp_install, "llamacpp_backend_error",
        lambda *args, **kwargs: calls.append((args, kwargs)) or error,
    )
    assert llamacpp_install.installed_toolset_error("/managed/server", "cuda", env={"CUDA": "1"}) == error
    assert calls == [(("/managed/server", "cuda"), {"env": {"CUDA": "1"}, "context": "setup"})]


@pytest.mark.parametrize("platform_name", ["Linux", "Darwin", "Windows"])
def test_install_refuses_to_modify_a_symlinked_runtime(tmp_path, symlink_or_skip, platform_name):
    external = tmp_path / "external"
    external.mkdir()
    marker = external / "keep"
    marker.write_text("existing runtime")
    runtime = tmp_path / "llama.cpp"
    symlink_or_skip(runtime, external, directory=True)
    failures = []
    assert not llamacpp_install.install(
        runtime, tmp_path, platform_name, nvidia=False, rocm=False, intel_xpu=False,
        compute_capability=None, max_cuda_version=None,
        info=_log, warn=_log, fail=failures.append, ok=_log,
    )
    assert "Remove the symlink" in failures[0]
    assert marker.read_text() == "existing runtime"
    assert runtime.is_symlink()


@pytest.mark.parametrize('system,nvidia,rocm,xpu,vulkan,backend', [
    ('Linux', True, False, False, False, 'cuda'),
    ('Linux', False, True, False, False, 'rocm'),
    ('Linux', False, False, True, False, 'xpu'),
    ('Linux', False, False, False, False, 'cpu'),
    ('Linux', True, True, True, True, 'vulkan'),
    ('Windows', False, False, True, False, 'xpu'),
    ('Windows', True, False, False, False, 'cuda'),
    ('Darwin', False, False, False, False, 'metal'),
])
def test_setup_uses_shared_release_installer(monkeypatch, tmp_path, system, nvidia, rocm, xpu, vulkan, backend):
    calls = []
    monkeypatch.setattr(llamacpp_install, 'rocm_version', lambda: (10, 0))
    monkeypatch.setattr(llamacpp_install, 'install_managed_llamacpp', lambda *a, **k:
                        calls.append((a, k)) or SimpleNamespace(success=True, detail='installed'))
    assert llamacpp_install.install(tmp_path/'runtime', tmp_path, system, nvidia=nvidia,
        rocm=rocm, intel_xpu=xpu, compute_capability=None, max_cuda_version='13.0', vulkan=vulkan,
        info=_log, warn=_log, fail=_log, ok=_log)
    assert calls[0][0][1] == system
    assert calls[0][0][3] == backend
    assert calls[0][1]['max_cuda_version'] == '13.0'
