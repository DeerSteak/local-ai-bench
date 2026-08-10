from pathlib import Path
from types import SimpleNamespace

from scripts.setup.runtime_update import (
    update_managed_vllm, validate_vllm_environment, vllm_executable,
)
from scripts.setup.vllm_install import VllmSupport


SUPPORT = VllmSupport("supported", "cuda_wheel", "supported")


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


def test_update_managed_vllm_swaps_only_after_validation(tmp_path):
    target = tmp_path / "vllm-env"
    target.mkdir()
    (target / "old").write_text("old", encoding="utf-8")

    def installer(_support, **kwargs):
        executable = vllm_executable(kwargs["venv_dir"])
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


def test_update_managed_vllm_rolls_back_failed_swap(tmp_path):
    target = tmp_path / "vllm-env"
    target.mkdir()
    marker = target / "old"
    marker.touch()
    calls = []

    def installer(_support, **kwargs):
        executable = vllm_executable(kwargs["venv_dir"])
        executable.parent.mkdir(parents=True)
        executable.touch()
        return True

    def replace(source, destination):
        calls.append((Path(source), Path(destination)))
        if len(calls) == 2:
            raise OSError("swap failed")
        Path(source).replace(destination)

    result = update_managed_vllm(
        SUPPORT, target, installer=installer, replace=replace,
        run=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="vllm 1.0", stderr=""),
        token_factory=lambda: "test",
    )

    assert not result.success
    assert "prior environment was preserved" in result.detail
    assert marker.exists()
    assert len(calls) == 3


def test_update_managed_vllm_reports_retained_backup_after_success(tmp_path):
    target = tmp_path / "vllm-env"
    target.mkdir()

    def installer(_support, **kwargs):
        executable = vllm_executable(kwargs["venv_dir"])
        executable.parent.mkdir(parents=True)
        executable.touch()
        return True

    result = update_managed_vllm(
        SUPPORT, target, installer=installer,
        run=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="vllm 1.0", stderr=""),
        remove=lambda _path: (_ for _ in ()).throw(OSError("busy")),
        token_factory=lambda: "test",
    )

    assert result.success
    assert "backup remains" in result.detail
    assert vllm_executable(target).is_file()
    assert (tmp_path / ".vllm-env-backup-test").exists()


def test_update_managed_vllm_rejects_unmanaged_or_unsupported_target(tmp_path):
    missing = update_managed_vllm(SUPPORT, tmp_path / "missing")
    unsupported = update_managed_vllm(
        VllmSupport("unsupported", None, "unsupported here"), tmp_path,
    )
    assert "does not exist" in missing.detail
    assert unsupported.detail == "unsupported here"
