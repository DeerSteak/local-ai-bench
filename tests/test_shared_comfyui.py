from types import SimpleNamespace

from scripts.runtime import config
from scripts.runtime.shared import Shared
from scripts.workloads.models import IMAGE_MODELS


def test_running_comfyui_without_managed_models_is_not_stopped(monkeypatch, tmp_path):
    checkpoint = IMAGE_MODELS[0]["checkpoint"]
    managed = tmp_path / "checkpoints" / checkpoint
    managed.parent.mkdir(parents=True)
    managed.touch()
    warnings = []

    monkeypatch.setattr(config, "COMFYUI_MODELS_DIR", tmp_path)
    monkeypatch.setattr(Shared, "comfyui_available", lambda: True)
    monkeypatch.setattr(Shared, "warn", warnings.append)
    monkeypatch.setattr(
        "scripts.runtime.shared.requests.get",
        lambda *_args, **_kwargs: SimpleNamespace(json=lambda: {
            "CheckpointLoaderSimple": {
                "input": {"required": {"ckpt_name": [["unmanaged.safetensors"]]}},
            },
        }),
    )

    assert not Shared.ensure_comfyui(tmp_path / "ComfyUI")
    assert any("restart it once" in warning for warning in warnings)
