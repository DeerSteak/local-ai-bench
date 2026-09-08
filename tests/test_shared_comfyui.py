from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from scripts.runtime import config
from scripts.runtime.shared import Shared
from scripts.workloads.models import IMAGE_MODELS


@pytest.fixture
def installation(monkeypatch, tmp_path):
    from scripts.runtime import shared

    checkpoint = IMAGE_MODELS[0]["checkpoint"]
    managed = tmp_path / "checkpoints" / checkpoint
    managed.parent.mkdir(parents=True)
    managed.touch()
    comfyui = tmp_path / "ComfyUI"
    comfyui.mkdir()
    (comfyui / "main.py").touch()
    monkeypatch.setattr(config, "COMFYUI_MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "COMFYUI_EXTRA_MODEL_PATHS", tmp_path / "extra.yaml")
    monkeypatch.setattr(config, "COMFYUI_URL", "http://localhost:8188")
    monkeypatch.setattr(Shared, "_managed_procs", [])
    monkeypatch.setattr(Shared, "_comfyui_log_path", None)
    monkeypatch.setattr(Shared, "find_comfyui_python", lambda _: "selected-python")
    monkeypatch.setattr(shared.time, "sleep", lambda _: None)
    monkeypatch.setattr(shared.tempfile, "NamedTemporaryFile", lambda **_: (tmp_path / "server.log").open("w"))
    proc = Mock(pid=123, returncode=None)
    proc.poll.return_value = None
    launch = Mock(return_value=proc)
    monkeypatch.setattr(shared.subprocess, "Popen", launch)
    return comfyui, checkpoint, launch, proc


def object_info(names):
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
        "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [names]}}},
    })


def test_incompatible_server_launches_separate_instance_and_reuses_it(monkeypatch, installation):
    comfyui, checkpoint, launch, proc = installation
    monkeypatch.setattr(Shared, "comfyui_available", lambda: True)
    addresses = []
    monkeypatch.setattr(Shared, "comfyui_launch_address", lambda isolate: (
        addresses.append(isolate) or ("127.0.0.1", 49123)))
    seen = []

    def get(url, **kwargs):
        seen.append(url)
        return object_info([checkpoint] if ":49123/" in url else ["unmanaged.safetensors"])

    monkeypatch.setattr("scripts.runtime.shared.requests.get", get)
    assert Shared.ensure_comfyui(comfyui)
    assert addresses == [True]
    cmd = launch.call_args.args[0]
    assert cmd[:2] == ["selected-python", str(comfyui / "main.py")]
    assert cmd[2:6] == ["--listen", "127.0.0.1", "--port", "49123"]
    assert cmd[6:8] == ["--extra-model-paths-config", str(config.COMFYUI_EXTRA_MODEL_PATHS)]
    assert str(config.COMFYUI_MODELS_DIR) in config.COMFYUI_EXTRA_MODEL_PATHS.read_text()
    assert config.COMFYUI_URL == "http://127.0.0.1:49123"
    assert Shared._managed_procs == [proc]
    proc.terminate.assert_not_called()
    assert Shared.ensure_comfyui(comfyui)
    launch.assert_called_once()
    assert seen[0].startswith("http://localhost:8188/")
    assert all(url.startswith(config.COMFYUI_URL) for url in seen[1:])


def test_compatible_server_is_reused_without_launch(monkeypatch, installation):
    comfyui, checkpoint, launch, _ = installation
    monkeypatch.setattr(Shared, "comfyui_available", lambda: True)
    monkeypatch.setattr("scripts.runtime.shared.requests.get", lambda *a, **k: object_info([checkpoint]))
    assert Shared.ensure_comfyui(comfyui)
    launch.assert_not_called()
    assert Shared._managed_procs == []
    assert config.COMFYUI_URL == "http://localhost:8188"


@pytest.mark.parametrize("portable", [False, True])
def test_restart_uses_selected_port_and_python_layout(monkeypatch, installation, portable):
    comfyui, checkpoint, launch, _ = installation
    if portable:
        python = comfyui.parent / "python_embeded" / "python.exe"
        python.parent.mkdir()
        python.touch()
        monkeypatch.setattr(Shared, "detect_backend", lambda: "rocm")
    monkeypatch.setattr(config, "COMFYUI_URL", "http://127.0.0.1:49123")
    available = iter([False, True])
    monkeypatch.setattr(Shared, "comfyui_available", lambda: next(available))
    monkeypatch.setattr("scripts.runtime.shared.requests.get", lambda *a, **k: object_info([checkpoint]))
    assert Shared.ensure_comfyui(comfyui)
    cmd = launch.call_args.args[0]
    assert cmd[cmd.index("--port") + 1] == "49123"
    assert ("--windows-standalone-build" in cmd) is portable
    assert launch.call_args.kwargs["cwd"] == str(comfyui.parent if portable else comfyui)
    if portable:
        assert launch.call_args.kwargs["env"]["TRITON_INTERPRET"] == "1"


@pytest.mark.parametrize("failure", ["exit", "models", "spawn", "timeout"])
def test_failed_launch_never_accepts_incompatible_server(monkeypatch, installation, failure):
    comfyui, _, launch, proc = installation
    monkeypatch.setattr(Shared, "comfyui_available", lambda: failure != "timeout")
    monkeypatch.setattr(Shared, "running_comfyui_models_visible", lambda: False)
    monkeypatch.setattr(Shared, "comfyui_launch_address", lambda _: ("127.0.0.1", 49123))
    if failure == "exit":
        proc.poll.return_value = 1
    if failure == "spawn":
        launch.side_effect = OSError("cannot launch")
    assert not Shared.ensure_comfyui(comfyui)
    assert config.COMFYUI_URL == ("http://localhost:8188" if failure == "spawn" else "http://127.0.0.1:49123")


def test_partial_or_unverifiable_checkpoint_list_is_not_reusable(monkeypatch, installation):
    _, checkpoint, _, _ = installation
    second = dict(IMAGE_MODELS[0], checkpoint="second.safetensors")
    (config.COMFYUI_MODELS_DIR / "checkpoints" / second["checkpoint"]).touch()
    monkeypatch.setattr("scripts.runtime.shared.IMAGE_MODELS", [IMAGE_MODELS[0], second])
    monkeypatch.setattr("scripts.runtime.shared.requests.get", lambda *a, **k: object_info([checkpoint]))
    assert not Shared.running_comfyui_models_visible()
    monkeypatch.setattr("scripts.runtime.shared.requests.get", Mock(side_effect=requests.Timeout()))
    assert not Shared.running_comfyui_models_visible()


def test_isolated_launch_selects_loopback_ephemeral_port(monkeypatch):
    listener = Mock()
    listener.getsockname.return_value = ("127.0.0.1", 49123)
    context = Mock()
    context.__enter__ = Mock(return_value=listener)
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr("scripts.runtime.shared.socket.socket", lambda *a: context)
    assert Shared.comfyui_launch_address(True) == ("127.0.0.1", 49123)
    listener.bind.assert_called_once_with(("127.0.0.1", 0))


@pytest.mark.parametrize("url,expected", [
    ("http://localhost:8188", ("localhost", 8188)),
    ("http://127.0.0.1:49123", ("127.0.0.1", 49123)),
    ("http://[::1]:49123", ("::1", 49123)),
])
def test_launch_address_preserves_endpoint(monkeypatch, url, expected):
    monkeypatch.setattr(config, "COMFYUI_URL", url)
    assert Shared.comfyui_launch_address(False) == expected


def test_launch_address_rejects_remote_endpoint(monkeypatch):
    monkeypatch.setattr(config, "COMFYUI_URL", "http://remote:8188")
    with pytest.raises(ValueError, match="non-local"):
        Shared.comfyui_launch_address(False)


def test_incompatible_server_with_missing_installation_does_not_launch(monkeypatch, installation):
    comfyui, _, launch, _ = installation
    (comfyui / "main.py").unlink()
    monkeypatch.setattr(Shared, "comfyui_available", lambda: True)
    monkeypatch.setattr(Shared, "running_comfyui_models_visible", lambda: False)
    assert not Shared.ensure_comfyui(comfyui)
    launch.assert_not_called()
    assert config.COMFYUI_URL == "http://localhost:8188"


def test_port_allocation_failure_is_reported_without_launch(monkeypatch, installation):
    comfyui, _, launch, _ = installation
    monkeypatch.setattr(Shared, "comfyui_available", lambda: True)
    monkeypatch.setattr(Shared, "running_comfyui_models_visible", lambda: False)
    monkeypatch.setattr(Shared, "comfyui_launch_address", Mock(side_effect=OSError("no ports")))
    assert not Shared.ensure_comfyui(comfyui)
    launch.assert_not_called()


def test_every_loader_group_must_be_visible(monkeypatch, installation):
    _, checkpoint, _, _ = installation
    from scripts.workloads.models import image_checkpoint_loader, image_checkpoint_path

    unet = next(model for model in IMAGE_MODELS if image_checkpoint_loader(model) == "UNETLoader")
    asset = image_checkpoint_path(unet, config.COMFYUI_MODELS_DIR)
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.touch()
    monkeypatch.setattr("scripts.runtime.shared.IMAGE_MODELS", [IMAGE_MODELS[0], unet])
    seen = []

    def get(url, **kwargs):
        seen.append(url)
        return object_info([checkpoint])

    monkeypatch.setattr("scripts.runtime.shared.requests.get", get)
    assert not Shared.running_comfyui_models_visible()
    assert seen[-1].endswith("/object_info/UNETLoader")


def test_cleanup_stops_only_owned_instance(monkeypatch, installation):
    comfyui, checkpoint, _, proc = installation
    monkeypatch.setattr(Shared, "_active_engine", None)
    monkeypatch.setattr(Shared, "comfyui_available", lambda: True)
    monkeypatch.setattr(Shared, "running_comfyui_models_visible", Mock(side_effect=[False, True]))
    monkeypatch.setattr(Shared, "comfyui_launch_address", lambda _: ("127.0.0.1", 49123))
    proc.own_process_group = False
    assert Shared.ensure_comfyui(comfyui)
    Shared.shutdown_managed()
    proc.terminate.assert_called_once()
    proc.wait.assert_called_once_with(timeout=10)
    assert Shared._managed_procs == []
