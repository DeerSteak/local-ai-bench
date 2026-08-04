from pathlib import Path

import pytest

from maintenance import build_uninstall_plan, execute_uninstall_plan, installation_health


def repo(tmp_path):
    (tmp_path / "README.md").write_text("local ai bench", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "requirements.txt").write_text("requests", encoding="utf-8")
    (tmp_path / "setup.sh").write_text("", encoding="utf-8")
    (tmp_path / "run_bench.sh").write_text("", encoding="utf-8")
    return tmp_path


def test_default_uninstall_preserves_models_results_config_and_credentials(tmp_path):
    root = repo(tmp_path)
    for name in ("bench-env", "ComfyUI", "llama.cpp", "models", "results"):
        (root / name).mkdir()
    for name in ("local_ai_bench_config.json", "hf.txt"):
        (root / name).write_text("private", encoding="utf-8")
    plan = build_uninstall_plan(root)
    actions = {(Path(item.path).name, item.kind) for item in plan}
    assert ("bench-env", "remove") in actions
    assert ("models", "preserve") in actions
    assert ("results", "preserve") in actions
    assert ("hf.txt", "preserve") in actions


def test_optional_data_removal_is_previewed_and_executed_exactly(tmp_path):
    root = repo(tmp_path)
    for name in ("bench-env", "models", "results"):
        (root / name).mkdir()
        (root / name / "content").write_text("x", encoding="utf-8")
    (root / "hf.txt").write_text("token", encoding="utf-8")
    plan = build_uninstall_plan(
        root, remove_models=True, remove_results=True, remove_credentials=True,
    )
    removed = execute_uninstall_plan(root, plan, "REMOVE LOCAL AI BENCH COMPONENTS")
    assert set(map(Path, removed)) == {root / "bench-env", root / "ComfyUI", root / "llama.cpp", root / "models", root / "results", root / "hf.txt"}
    assert not any((root / name).exists() for name in ("bench-env", "models", "results", "hf.txt"))
    assert (root / "README.md").is_file()


def test_uninstall_rejects_wrong_confirmation_and_injected_target(tmp_path):
    root = repo(tmp_path)
    plan = build_uninstall_plan(root)
    with pytest.raises(ValueError, match="confirmation"):
        execute_uninstall_plan(root, plan, "yes")
    outside = tmp_path.parent / "outside"
    injected = (*plan, type(plan[0])("remove", str(outside), False))
    with pytest.raises(ValueError, match="outside"):
        execute_uninstall_plan(root, injected, "REMOVE LOCAL AI BENCH COMPONENTS")


def test_uninstall_rejects_symlinked_managed_directory(tmp_path):
    root = repo(tmp_path)
    outside = tmp_path / "outside-models"
    outside.mkdir()
    (root / "models").symlink_to(outside, target_is_directory=True)
    plan = build_uninstall_plan(root, remove_models=True)
    with pytest.raises(ValueError, match="symbolic link"):
        execute_uninstall_plan(root, plan, "REMOVE LOCAL AI BENCH COMPONENTS")
    assert outside.is_dir()


def test_health_check_is_read_only_and_reports_missing_environment(tmp_path):
    root = repo(tmp_path)
    assert installation_health(root) == {
        "healthy": False,
        "checks": {"environment": False, "requirements": True, "setup_launcher": True, "benchmark_launcher": True},
    }


def test_maintenance_rejects_unrelated_directory(tmp_path):
    with pytest.raises(ValueError, match="repository root"):
        build_uninstall_plan(tmp_path)
