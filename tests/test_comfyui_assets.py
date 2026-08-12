from scripts.setup.comfyui_assets import provision


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
    selected = [{"short": "flux-dev", "checkpoint": "flux1-dev.safetensors", "label": "Flux"}]

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
