from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from scripts.setup.runtime_update import (
    detect_nvidia_compute_capability, detect_nvidia_max_cuda_version,
    homebrew_llamacpp_prefix, llamacpp_cmake_flags, rebuild_managed_llamacpp,
    RuntimeUpdateControl, select_windows_llamacpp_assets, update_homebrew_llamacpp,
    update_windows_llamacpp,
    update_managed_vllm, validate_vllm_environment, vllm_executable,
)
from scripts.setup.vllm_install import VllmSupport


SUPPORT = VllmSupport("supported", "cuda_wheel", "supported")


def test_runtime_update_control_prevents_commands_after_cancellation():
    control = RuntimeUpdateControl()
    control.cancel()
    result = control.run(["never-run"])
    assert result.returncode == -1
    assert result.stderr == "update cancelled"


def test_runtime_update_control_streams_utf8_output(capsys):
    output = []
    control = RuntimeUpdateControl(output.append)

    result = control.run([
        sys.executable, "-c", "print('Downloading — 50% ✓')",
    ])

    assert result.returncode == 0
    assert result.stdout == "Downloading — 50% ✓\n"
    assert output == ["Downloading — 50% ✓\n"]
    assert capsys.readouterr().out == "Downloading — 50% ✓\n"


def test_vllm_executable_uses_platform_venv_layout():
    assert vllm_executable(Path("/runtime"), "posix") == Path("/runtime/bin/vllm")
    assert vllm_executable(Path("C:/runtime"), "nt") == Path("C:/runtime/Scripts/vllm.exe")


def test_validate_vllm_environment_requires_executable(tmp_path):
    result = validate_vllm_environment(tmp_path, run=lambda *args, **kwargs: None)
    assert not result.success
    assert "missing" in result.detail


def test_validate_vllm_environment_captures_version(tmp_path):
    executable = vllm_executable(tmp_path)
    executable.parent.mkdir()
    executable.touch()
    result = validate_vllm_environment(
        tmp_path,
        run=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="vllm 0.10.0\n", stderr=""),
    )
    assert result.success
    assert result.version == "vllm 0.10.0"


def test_update_managed_vllm_recreates_venv_at_final_path_after_staging(tmp_path):
    target = tmp_path / "vllm-env"
    target.mkdir()
    (target / "old").write_text("old", encoding="utf-8")

    installed_at = []

    def installer(_support, **kwargs):
        destination = kwargs["venv_dir"]
        installed_at.append(destination)
        executable = vllm_executable(destination)
        executable.parent.mkdir(parents=True)
        executable.touch()
        return True

    result = update_managed_vllm(
        SUPPORT, target, installer=installer,
        run=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="vllm 1.0", stderr=""),
        token_factory=lambda: "test",
    )

    assert result.success
    assert result.version == "vllm 1.0"
    assert vllm_executable(target).is_file()
    assert not (target / "old").exists()
    assert not (tmp_path / ".vllm-env-backup-test").exists()
    assert installed_at == [tmp_path / ".vllm-env-update-test", target]


def test_update_managed_vllm_preserves_target_when_staging_fails(tmp_path):
    target = tmp_path / "vllm-env"
    target.mkdir()
    marker = target / "old"
    marker.touch()

    result = update_managed_vllm(
        SUPPORT, target, installer=lambda *args, **kwargs: False,
        token_factory=lambda: "test",
    )

    assert not result.success
    assert marker.exists()


def test_update_managed_vllm_reports_cancellation_and_preserves_target(tmp_path):
    target = tmp_path / "vllm-env"
    target.mkdir()
    marker = target / "old"
    marker.touch()
    control = RuntimeUpdateControl()

    def installer(*args, **kwargs):
        control.cancel()
        return False

    result = update_managed_vllm(
        SUPPORT, target, installer=installer, control=control,
        token_factory=lambda: "test",
    )
    assert not result.success
    assert "cancelled" in result.detail
    assert marker.exists()


def test_update_managed_vllm_stops_before_validation_when_cancelled(tmp_path):
    target = tmp_path / "vllm-env"
    target.mkdir()
    marker = target / "old"
    marker.touch()
    control = RuntimeUpdateControl()

    def installer(_support, **kwargs):
        executable = vllm_executable(kwargs["venv_dir"])
        executable.parent.mkdir(parents=True)
        executable.touch()
        control.cancel()
        return True

    result = update_managed_vllm(
        SUPPORT, target, installer=installer, control=control,
        token_factory=lambda: "test",
    )
    assert not result.success and "cancelled" in result.detail
    assert marker.exists()


def test_update_managed_vllm_reports_cancellation_during_validation(tmp_path):
    target = tmp_path / "vllm-env"
    target.mkdir()
    marker = target / "old"
    marker.touch()

    class Control(RuntimeUpdateControl):
        def run(self, command, **_kwargs):
            self.cancel()
            return subprocess.CompletedProcess(command, -1, "", "update cancelled")

    def installer(_support, **kwargs):
        executable = vllm_executable(kwargs["venv_dir"])
        executable.parent.mkdir(parents=True)
        executable.touch()
        return True

    result = update_managed_vllm(
        SUPPORT, target, installer=installer, control=Control(),
        token_factory=lambda: "test",
    )
    assert not result.success and "cancelled" in result.detail
    assert marker.exists()


def test_update_managed_vllm_rolls_back_failed_final_install(tmp_path):
    target = tmp_path / "vllm-env"
    target.mkdir()
    marker = target / "old"
    marker.touch()
    installs = []

    def installer(_support, **kwargs):
        installs.append(kwargs["venv_dir"])
        if len(installs) == 2:
            return False
        executable = vllm_executable(kwargs["venv_dir"])
        executable.parent.mkdir(parents=True)
        executable.touch()
        return True

    result = update_managed_vllm(
        SUPPORT, target, installer=installer,
        run=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="vllm 1.0", stderr=""),
        token_factory=lambda: "test",
    )

    assert not result.success
    assert "prior environment was preserved" in result.detail
    assert marker.exists()
    assert installs == [tmp_path / ".vllm-env-update-test", target]


def test_update_managed_vllm_reports_retained_backup_after_success(tmp_path):
    target = tmp_path / "vllm-env"
    target.mkdir()

    def installer(_support, **kwargs):
        executable = vllm_executable(kwargs["venv_dir"])
        executable.parent.mkdir(parents=True)
        executable.touch()
        return True

    backup = tmp_path / ".vllm-env-backup-test"

    def remove(path):
        if Path(path) == backup:
            raise OSError("busy")
        __import__("shutil").rmtree(path)

    result = update_managed_vllm(
        SUPPORT, target, installer=installer,
        run=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="vllm 1.0", stderr=""),
        remove=remove,
        token_factory=lambda: "test",
    )

    assert result.success
    assert "backup remains" in result.detail
    assert vllm_executable(target).is_file()
    assert backup.exists()


def test_update_managed_vllm_rejects_unmanaged_or_unsupported_target(tmp_path):
    missing = update_managed_vllm(SUPPORT, tmp_path / "missing")
    unsupported = update_managed_vllm(
        VllmSupport("unsupported", None, "unsupported here"), tmp_path,
    )
    assert "does not exist" in missing.detail
    assert unsupported.detail == "unsupported here"


def test_detect_nvidia_compute_capability_reads_first_gpu():
    result = detect_nvidia_compute_capability(
        run=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="8.9\n8.9\n"),
    )
    assert result == "8.9"


def test_llamacpp_cmake_flags_match_backend():
    assert llamacpp_cmake_flags("cuda", nvcc="/cuda/nvcc", compute_capability="8.9") == [
        "-DGGML_CUDA=ON", "-DCMAKE_CUDA_COMPILER=/cuda/nvcc",
        "-DCMAKE_CUDA_ARCHITECTURES=89",
    ]
    assert llamacpp_cmake_flags("rocm") == ["-DGGML_HIP=ON"]
    assert llamacpp_cmake_flags("cpu") == []


def test_homebrew_llamacpp_prefix_rejects_missing_formula():
    result = homebrew_llamacpp_prefix(
        run=lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    assert result is None


def test_update_homebrew_llamacpp_upgrades_and_validates_all_tools(tmp_path):
    prefix = tmp_path / "Cellar" / "llama.cpp" / "1"
    for name in ("llama-server", "llama-bench", "llama-batched-bench"):
        tool = prefix / "bin" / name
        tool.parent.mkdir(parents=True, exist_ok=True)
        tool.touch()
    commands = []

    def run(command, **kwargs):
        commands.append((command, kwargs))
        if command == ["brew", "--prefix", "llama.cpp"]:
            return SimpleNamespace(returncode=0, stdout=str(prefix), stderr="")
        if "--version" in command:
            return SimpleNamespace(returncode=0, stdout="version: 7001", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = update_homebrew_llamacpp(prefix / "bin" / "llama-server", run=run)

    assert result.success and result.version == "version: 7001"
    assert [entry[0] for entry in commands[1:3]] == [
        ["brew", "update"], ["brew", "upgrade", "llama.cpp"],
    ]
    assert commands[1][1]["env"]["NONINTERACTIVE"] == "1"


def test_update_homebrew_llamacpp_rejects_unrelated_system_binary(tmp_path):
    prefix = tmp_path / "homebrew" / "llama.cpp"
    result = update_homebrew_llamacpp(
        tmp_path / "usr" / "bin" / "llama-server",
        run=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=str(prefix), stderr=""),
    )
    assert not result.success
    assert "not owned by Homebrew" in result.detail


def test_detect_nvidia_max_cuda_version_reads_driver_header():
    result = detect_nvidia_max_cuda_version(
        run=lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="NVIDIA-SMI 580.1 CUDA Version: 13.0",
        ),
    )
    assert result == "13.0"


def test_select_windows_assets_prefers_compatible_cuda_pair():
    assets = [
        {"name": "llama-b1-bin-win-cuda-12.4-x64.zip"},
        {"name": "cudart-llama-bin-win-cuda-12.4-x64.zip"},
        {"name": "llama-b1-bin-win-vulkan-x64.zip"},
    ]
    selected = select_windows_llamacpp_assets({"assets": assets}, "12.4")
    assert [asset["name"] for asset in selected] == [
        "llama-b1-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
    ]


def test_update_windows_llamacpp_stages_validates_and_swaps(tmp_path):
    target = tmp_path / "llama.cpp"
    target.mkdir()
    (target / "old").touch()
    asset = {
        "name": "llama-b1-bin-win-vulkan-x64.zip", "browser_download_url": "https://x/a.zip",
        "size": 10,
    }

    def downloader(_url, destination, **_kwargs):
        destination.touch()
        return destination

    def extractor(_archive, destination):
        for name in ("llama-server", "llama-bench", "llama-batched-bench"):
            tool = destination / "bin" / f"{name}.exe"
            tool.parent.mkdir(parents=True, exist_ok=True)
            tool.touch()

    result = update_windows_llamacpp(
        target, None, release_fetcher=lambda: {"assets": [asset]},
        downloader=downloader, extractor=extractor,
        run=lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="version: 7002", stderr="",
        ), token_factory=lambda: "test",
    )

    assert result.success and result.version == "version: 7002"
    assert (target / "bin" / "llama-server.exe").is_file()
    assert not (target / "old").exists()
    assert not (tmp_path / ".llama.cpp-downloads-test").exists()


def test_update_windows_llamacpp_preserves_target_when_validation_fails(tmp_path):
    target = tmp_path / "llama.cpp"
    target.mkdir()
    marker = target / "old"
    marker.touch()
    asset = {"name": "llama-win-vulkan-x64.zip", "browser_download_url": "https://x/a", "size": 1}

    result = update_windows_llamacpp(
        target, None, release_fetcher=lambda: {"assets": [asset]},
        downloader=lambda _url, destination, **_kwargs: destination,
        extractor=lambda _archive, destination: destination.mkdir(exist_ok=True),
        token_factory=lambda: "test",
    )
    assert not result.success
    assert marker.exists()


def test_rebuild_managed_llamacpp_builds_all_tools_before_swap(tmp_path, monkeypatch):
    target = tmp_path / "llama.cpp"
    target.mkdir()
    (target / "old").touch()
    commands = []
    monkeypatch.setattr("scripts.setup.runtime_update.find_nvcc", lambda: "/cuda/nvcc")

    def run(command, **kwargs):
        commands.append(command)
        if command[:3] == ["git", "clone", "--depth"]:
            staged = Path(command[-1])
            for name in ("llama-server", "llama-bench", "llama-batched-bench"):
                tool = staged / "build" / "bin" / name
                tool.parent.mkdir(parents=True, exist_ok=True)
                tool.touch()
        if "--version" in command:
            return SimpleNamespace(returncode=0, stdout="version: 7000", stderr="")
        if command[:1] == ["nvidia-smi"]:
            return SimpleNamespace(returncode=0, stdout="8.9\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = rebuild_managed_llamacpp(
        target, "cuda", run=run, os_name="posix", token_factory=lambda: "test",
    )

    assert result.success
    configure = next(command for command in commands if command[:2] == ["cmake", "-B"])
    build = next(command for command in commands if command[:2] == ["cmake", "--build"])
    assert "-DBUILD_SHARED_LIBS=OFF" in configure
    assert "-DCMAKE_CUDA_ARCHITECTURES=89" in configure
    assert all(name in build for name in ("llama-server", "llama-bench", "llama-batched-bench"))
    assert not (target / "old").exists()


def test_rebuild_managed_llamacpp_preserves_checkout_on_build_failure(tmp_path):
    target = tmp_path / "llama.cpp"
    target.mkdir()
    marker = target / "old"
    marker.touch()

    result = rebuild_managed_llamacpp(
        target, "cpu",
        run=lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="failed"),
        token_factory=lambda: "test",
    )

    assert not result.success
    assert marker.exists()


def test_rebuild_managed_llamacpp_rolls_back_when_final_path_validation_fails(tmp_path):
    target = tmp_path / "llama.cpp"
    target.mkdir()
    marker = target / "old"
    marker.touch()
    version_calls = 0

    def run(command, **kwargs):
        nonlocal version_calls
        if command[:3] == ["git", "clone", "--depth"]:
            staged = Path(command[-1])
            for name in ("llama-server", "llama-bench", "llama-batched-bench"):
                tool = staged / "build" / "bin" / name
                tool.parent.mkdir(parents=True, exist_ok=True)
                tool.touch()
        if "--version" in command:
            version_calls += 1
            if version_calls == 2:
                return SimpleNamespace(returncode=127, stdout="", stderr="missing shared library")
            return SimpleNamespace(returncode=0, stdout="version: 7000", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = rebuild_managed_llamacpp(
        target, "cpu", run=run, os_name="posix", token_factory=lambda: "test",
    )

    assert not result.success
    assert "prior checkout was preserved" in result.detail
    assert marker.exists()
    assert not (target / "build" / "bin" / "llama-server").exists()
