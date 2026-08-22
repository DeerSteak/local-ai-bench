from pathlib import Path

from scripts.setup.setup_check import accessible_file, llamacpp_backend_rebuild_warning


def test_accessible_file_accepts_a_regular_file(tmp_path):
    path = tmp_path / "python"
    path.touch()
    assert accessible_file(path)


def test_accessible_file_rejects_missing_and_permission_denied_paths(tmp_path, monkeypatch):
    assert not accessible_file(tmp_path / "missing")

    def denied(_path):
        raise PermissionError("mounted path is not traversable")

    monkeypatch.setattr(Path, "is_file", denied)
    assert not accessible_file(tmp_path / "inaccessible")


def test_backend_rebuild_warning_distinguishes_setup_from_qualification():
    assert llamacpp_backend_rebuild_warning(
        "vulkan", "cuda", qualification=False,
    ) == "llama-server exposes vulkan, but setup requires cuda — it will be rebuilt"
    assert llamacpp_backend_rebuild_warning(
        None, "xpu", qualification=True,
    ) == ("llama-server exposes no detectable backend, but qualification requires xpu "
          "— it will be rebuilt")
