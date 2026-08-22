from scripts.setup.comfyui_assets import missing_download_size_gb, provision
from scripts.workloads.models import IMAGE_MODELS


def _log(_message):
    pass


def test_provision_downloads_public_checkpoint(monkeypatch, tmp_path):
    downloads = []
    selected = [{"short": "sd15", "checkpoint": "sd15.safetensors", "label": "SD 1.5"}]

    found = provision(
        selected, tmp_path, find_asset=lambda *_args: None,
        download=lambda *args, **kwargs: downloads.append((args, kwargs)) or True,
        load_token=lambda: "", info=_log, warn=_log, fail=_log, ok=_log,
    )

    assert found == ["sd15.safetensors"]
    assert downloads[0][0] == ("Comfy-Org/stable-diffusion-v1-5-archive", "sd15.safetensors")


def test_provision_skips_gated_checkpoint_without_token(tmp_path):
    downloads = []
    selected = [{"short": "flux-dev", "checkpoint": "flux1-dev.safetensors", "label": "Flux"}]

    found = provision(
        selected, tmp_path, find_asset=lambda *_args: None,
        download=lambda *args, **kwargs: downloads.append((args, kwargs)) or True,
        load_token=lambda: "", info=_log, warn=_log, fail=_log, ok=_log,
    )

    assert found == []
    assert downloads == []


def test_flux_checkpoint_provisions_shared_assets(tmp_path):
    downloads = []
    selected = [next(model for model in IMAGE_MODELS if model["short"] == "flux-dev")]

    found = provision(
        selected, tmp_path, find_asset=lambda *_args: None,
        download=lambda *args, **kwargs: downloads.append((args, kwargs)) or True,
        load_token=lambda: "token", info=_log, warn=_log, fail=_log, ok=_log,
    )

    assert found == ["flux1-dev.safetensors"]
    assert [args[1] for args, _kwargs in downloads] == [
        "flux1-dev.safetensors", "t5xxl_fp16.safetensors",
        "clip_l.safetensors", "ae.safetensors",
    ]


def test_z_image_provisions_public_pipeline_in_comfyui_model_folders(tmp_path):
    downloads = []
    selected = [{
        "short": "z-image-turbo", "label": "Z-Image Turbo",
        "checkpoint": "z_image_turbo_bf16.safetensors",
        "checkpoint_folder": "diffusion_models",
        "checkpoint_repo": "Comfy-Org/z_image_turbo",
        "checkpoint_remote": "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
        "support_assets": [
            {"name": "qwen_3_4b.safetensors", "folder": "text_encoders",
             "repo": "Comfy-Org/z_image_turbo",
             "remote": "split_files/text_encoders/qwen_3_4b.safetensors"},
            {"name": "z_image_ae.safetensors", "folder": "vae",
             "repo": "Comfy-Org/z_image_turbo", "remote": "split_files/vae/ae.safetensors"},
        ],
    }]

    found = provision(
        selected, tmp_path, find_asset=lambda *_args: None,
        download=lambda *args, **kwargs: downloads.append((args, kwargs)) or True,
        load_token=lambda: "", info=_log, warn=_log, fail=_log, ok=_log,
    )

    assert found == ["z_image_turbo_bf16.safetensors"]
    assert downloads == [
        (("Comfy-Org/z_image_turbo",
          "split_files/diffusion_models/z_image_turbo_bf16.safetensors"), {
              "token": "", "dest_dir": tmp_path / "diffusion_models",
              "save_as": "z_image_turbo_bf16.safetensors",
          }),
        (("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors"), {
              "token": "", "dest_dir": tmp_path / "text_encoders",
              "save_as": "qwen_3_4b.safetensors",
          }),
        (("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors"), {
              "token": "", "dest_dir": tmp_path / "vae",
              "save_as": "z_image_ae.safetensors",
          }),
    ]


def test_z_image_existing_pipeline_is_not_redownloaded(tmp_path):
    selected = [{
        "short": "z-image-turbo", "label": "Z-Image Turbo", "checkpoint": "z.safetensors",
        "checkpoint_folder": "diffusion_models", "support_assets": [
            {"name": "qwen.safetensors", "folder": "text_encoders", "repo": "repo",
             "remote": "remote/qwen.safetensors"},
        ],
    }]
    paths = {
        ("z.safetensors", "diffusion_models"): tmp_path / "z.safetensors",
        ("qwen.safetensors", "text_encoders"): tmp_path / "qwen.safetensors",
    }
    for path in paths.values():
        path.write_bytes(b"asset")
    downloads = []

    assert provision(
        selected, tmp_path, find_asset=lambda name, folder: paths.get((name, folder)),
        download=lambda *args, **kwargs: downloads.append((args, kwargs)) or True,
        load_token=lambda: "", info=_log, warn=_log, fail=_log, ok=_log,
    ) == ["z.safetensors"]
    assert downloads == []


def test_pipeline_is_not_ready_when_a_support_asset_download_fails(tmp_path):
    selected = [{
        "short": "pipeline", "label": "Pipeline", "checkpoint": "model.safetensors",
        "support_assets": [
            {"name": "encoder.safetensors", "folder": "text_encoders",
             "repo": "owner/model", "remote": "encoder.safetensors"},
        ],
    }]

    found = provision(
        selected, tmp_path, find_asset=lambda *_args: None,
        download=lambda _repo, remote, **_kwargs: remote == "model.safetensors",
        load_token=lambda: "", info=_log, warn=_log, fail=_log, ok=_log,
    )

    assert found == []


def test_missing_download_size_counts_complete_z_image_pipeline_once():
    z_image = next(model for model in IMAGE_MODELS if model["short"] == "z-image-turbo")
    duplicate = dict(z_image)
    assert missing_download_size_gb(
        [z_image, duplicate], lambda *_args: None,
    ) == 12.4 + 8.1 + 0.4


def test_missing_download_size_uses_each_asset_folder_and_skips_existing():
    z_image = next(model for model in IMAGE_MODELS if model["short"] == "z-image-turbo")
    lookups = []

    def find_asset(name, folder):
        lookups.append((name, folder))
        return object() if name == "qwen_3_4b.safetensors" else None

    assert missing_download_size_gb([z_image], find_asset) == 12.4 + 0.4
    assert lookups == [
        ("z_image_turbo_bf16.safetensors", "diffusion_models"),
        ("qwen_3_4b.safetensors", "text_encoders"),
        ("z_image_ae.safetensors", "vae"),
    ]
