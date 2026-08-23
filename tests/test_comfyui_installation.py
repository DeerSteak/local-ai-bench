from pathlib import Path

import pytest

from scripts.runtime.comfyui_installation import (
    find_image_asset,
    image_asset_dirs,
    legacy_models_dir_with_assets,
    add_managed_models_to_comfyui,
    checkpoint_names_from_object_info,
    common_comfyui_candidates,
    comfyui_dirs_from_commands,
    find_comfyui_python,
    managed_checkpoints_visible,
    find_comfyui_installation,
    normalize_comfyui_dir,
    resolve_comfyui_setup_choice,
    running_comfyui_dirs,
    write_extra_model_paths,
)


def make_comfyui(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "main.py").touch()
    return path.resolve()


def test_normalize_accepts_manual_installation(tmp_path):
    comfyui = make_comfyui(tmp_path / "ComfyUI")
    assert normalize_comfyui_dir(comfyui) == comfyui


def test_normalize_accepts_windows_portable_root(tmp_path):
    comfyui = make_comfyui(tmp_path / "portable" / "ComfyUI")
    assert normalize_comfyui_dir(tmp_path / "portable") == comfyui


def test_normalize_accepts_main_py_and_portable_launcher(tmp_path):
    comfyui = make_comfyui(tmp_path / "portable" / "ComfyUI")
    launcher = tmp_path / "portable" / "run_nvidia_gpu.bat"
    launcher.touch()
    assert normalize_comfyui_dir(comfyui / "main.py") == comfyui
    assert normalize_comfyui_dir(launcher) == comfyui


def test_normalize_rejects_directory_without_main(tmp_path):
    assert normalize_comfyui_dir(tmp_path) is None


def test_setup_choice_defaults_to_download_and_classifies_paths(tmp_path):
    comfyui = make_comfyui(tmp_path / "ComfyUI")
    assert resolve_comfyui_setup_choice("") == ("download", None)
    assert resolve_comfyui_setup_choice("1") == ("download", None)
    assert resolve_comfyui_setup_choice("2", str(comfyui)) == ("existing", comfyui)
    assert resolve_comfyui_setup_choice("2", str(tmp_path / "missing")) == ("invalid", None)


def test_resolver_precedence_is_explicit_env_saved_common_managed(tmp_path):
    explicit = make_comfyui(tmp_path / "explicit")
    env_dir = make_comfyui(tmp_path / "env")
    saved_dir = make_comfyui(tmp_path / "saved")
    common = make_comfyui(tmp_path / "home" / "ComfyUI")
    managed = make_comfyui(tmp_path / "managed")
    assert find_comfyui_installation(
        explicit=explicit, environ={"COMFYUI_DIR": str(env_dir)},
        saved_path=saved_dir, managed_dir=managed,
        home=tmp_path / "home", platform_name="Linux", process_dirs=[],
    ) == explicit
    assert find_comfyui_installation(
        environ={"COMFYUI_DIR": str(env_dir)}, saved_path=saved_dir,
        managed_dir=managed, home=tmp_path / "home", platform_name="Linux", process_dirs=[],
    ) == env_dir
    assert find_comfyui_installation(
        environ={}, saved_path=saved_dir, managed_dir=managed,
        home=tmp_path / "home", platform_name="Linux", process_dirs=[],
    ) == saved_dir
    assert find_comfyui_installation(
        environ={}, managed_dir=managed,
        home=tmp_path / "home", platform_name="Linux", process_dirs=[],
    ) == common


def test_resolver_skips_stale_saved_path(tmp_path):
    managed = make_comfyui(tmp_path / "managed")
    assert find_comfyui_installation(
        environ={}, saved_path=tmp_path / "missing", managed_dir=managed,
        home=tmp_path / "empty-home", platform_name="Linux", process_dirs=[],
    ) == managed


def test_windows_candidates_include_portable_layouts(tmp_path):
    candidates = common_comfyui_candidates(tmp_path, "Windows")
    assert tmp_path / "ComfyUI_windows_portable" in candidates
    assert tmp_path / "Downloads" / "ComfyUI_windows_portable" in candidates


def test_process_commands_find_absolute_main_py_paths():
    commands = [
        'python "/opt/Comfy UI/ComfyUI/main.py" --listen',
        "python main.py --listen",
        "python /other/script.py",
    ]
    assert comfyui_dirs_from_commands(commands, "Linux") == [Path("/opt/Comfy UI/ComfyUI")]


def test_process_discovery_tolerates_permission_denial(monkeypatch):
    monkeypatch.setattr(
        "scripts.runtime.comfyui_installation.subprocess.check_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )
    assert running_comfyui_dirs("Linux") == []


def test_resolver_prefers_running_process_over_saved_and_common(tmp_path):
    running = make_comfyui(tmp_path / "running")
    saved = make_comfyui(tmp_path / "saved")
    make_comfyui(tmp_path / "home" / "ComfyUI")
    assert find_comfyui_installation(
        environ={}, saved_path=saved, home=tmp_path / "home",
        platform_name="Linux", process_dirs=[running],
    ) == running


def test_find_python_prefers_windows_portable_environment(tmp_path):
    comfyui = make_comfyui(tmp_path / "portable" / "ComfyUI")
    python = tmp_path / "portable" / "python_embeded" / "python.exe"
    python.parent.mkdir()
    python.touch()
    assert find_comfyui_python(comfyui, {}) == str(python)


def test_find_python_supports_desktop_dotvenv_on_windows(tmp_path):
    comfyui = make_comfyui(tmp_path / "ComfyUI")
    python = comfyui / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    assert find_comfyui_python(comfyui, {}) == str(python)


def test_write_extra_model_paths_points_to_managed_models(tmp_path):
    config_path = tmp_path / "models" / "extra_model_paths.yaml"
    models_dir = tmp_path / "models"
    write_extra_model_paths(config_path, models_dir)
    content = config_path.read_text()
    assert f'base_path: "{models_dir.resolve()}"' in content
    assert "checkpoints: checkpoints" in content
    assert "diffusion_models: diffusion_models" in content
    assert "text_encoders: text_encoders" in content


def test_add_managed_models_preserves_existing_config_and_is_idempotent(tmp_path):
    comfyui = make_comfyui(tmp_path / "ComfyUI")
    config_path = comfyui / "extra_model_paths.yaml"
    config_path.write_text("existing:\n  base_path: /models\n")
    models_dir = tmp_path / "bench-models"

    add_managed_models_to_comfyui(comfyui, models_dir)
    add_managed_models_to_comfyui(comfyui, models_dir)

    content = config_path.read_text()
    assert "existing:\n  base_path: /models" in content
    assert content.count("BEGIN local-ai-bench") == 1
    assert f'base_path: "{models_dir.resolve()}"' in content
    assert "diffusion_models: diffusion_models" in content


def test_checkpoint_names_from_object_info_handles_valid_and_malformed_data():
    data = {
        "CheckpointLoaderSimple": {
            "input": {"required": {"ckpt_name": [["one.safetensors", "two.safetensors"]]}},
        },
        "UNETLoader": {
            "input": {"required": {"unet_name": [["z-image.safetensors"]]}},
        },
    }
    assert checkpoint_names_from_object_info(data, "CheckpointLoaderSimple") == {
        "one.safetensors", "two.safetensors",
    }
    assert checkpoint_names_from_object_info(data, "UNETLoader") == {
        "z-image.safetensors",
    }
    with pytest.raises(ValueError, match="unsupported ComfyUI checkpoint loader"):
        checkpoint_names_from_object_info(data, "UnknownLoader")
    assert checkpoint_names_from_object_info({}, "UNETLoader") == set()


def test_managed_checkpoint_visibility_requires_overlap_when_models_exist():
    assert managed_checkpoints_visible({"one.safetensors"}, set())
    assert managed_checkpoints_visible({"one.safetensors"}, {"one.safetensors"})
    assert not managed_checkpoints_visible({"other.safetensors"}, {"one.safetensors"})


# ── pre-4.1 image-model locations ──

def _legacy_tree(tmp_path, subdirs=("checkpoints",), name="sd_xl_base_1.0.safetensors"):
    comfyui = tmp_path / "ComfyUI"
    for subdir in subdirs:
        directory = comfyui / "models" / subdir
        directory.mkdir(parents=True)
        (directory / name).touch()
    return comfyui, tmp_path / "models" / "comfyui"


def test_image_asset_dirs_prefer_the_managed_location(tmp_path):
    comfyui, managed = _legacy_tree(tmp_path)
    assert image_asset_dirs(managed, "checkpoints", comfyui) == [
        managed / "checkpoints", comfyui / "models" / "checkpoints",
    ]


def test_image_asset_dirs_support_diffusion_model_locations(tmp_path):
    comfyui, managed = _legacy_tree(
        tmp_path, subdirs=("diffusion_models",), name="z_image.safetensors",
    )
    assert image_asset_dirs(managed, "diffusion_models", comfyui) == [
        managed / "diffusion_models", comfyui / "models" / "diffusion_models",
    ]


def test_image_asset_dirs_omit_the_legacy_path_without_a_comfyui_dir(tmp_path):
    assert image_asset_dirs(tmp_path, "checkpoints") == [tmp_path / "checkpoints"]


def test_image_asset_dirs_ignore_unknown_subdirs(tmp_path):
    assert image_asset_dirs(tmp_path, "loras", tmp_path / "ComfyUI") == [tmp_path / "loras"]


def test_an_asset_only_in_the_legacy_location_is_found(tmp_path):
    comfyui, managed = _legacy_tree(tmp_path)
    found = find_image_asset("sd_xl_base_1.0.safetensors", managed, "checkpoints", comfyui)
    assert found == comfyui / "models" / "checkpoints" / "sd_xl_base_1.0.safetensors"


def test_the_managed_copy_wins_when_both_exist(tmp_path):
    comfyui, managed = _legacy_tree(tmp_path)
    (managed / "checkpoints").mkdir(parents=True)
    (managed / "checkpoints" / "sd_xl_base_1.0.safetensors").touch()
    found = find_image_asset("sd_xl_base_1.0.safetensors", managed, "checkpoints", comfyui)
    assert found == managed / "checkpoints" / "sd_xl_base_1.0.safetensors"


def test_a_missing_asset_is_none(tmp_path):
    comfyui, managed = _legacy_tree(tmp_path)
    assert find_image_asset("flux1-dev.safetensors", managed, "checkpoints", comfyui) is None


def test_legacy_models_dir_is_reported_only_when_it_holds_assets(tmp_path):
    comfyui, _ = _legacy_tree(tmp_path)
    assert legacy_models_dir_with_assets(comfyui) == comfyui / "models"

    empty = tmp_path / "Empty"
    (empty / "models" / "checkpoints").mkdir(parents=True)
    assert legacy_models_dir_with_assets(empty) is None
    assert legacy_models_dir_with_assets(None) is None


def test_generated_yaml_gains_a_legacy_section_only_when_needed(tmp_path):
    from scripts.runtime.comfyui_installation import write_extra_model_paths
    target = tmp_path / "extra_model_paths.yaml"

    write_extra_model_paths(target, tmp_path / "managed")
    assert "local_ai_bench_legacy" not in target.read_text()

    write_extra_model_paths(target, tmp_path / "managed", tmp_path / "ComfyUI" / "models")
    text = target.read_text()
    assert "local_ai_bench:" in text and "local_ai_bench_legacy:" in text
    assert text.count("checkpoints: checkpoints") == 2
