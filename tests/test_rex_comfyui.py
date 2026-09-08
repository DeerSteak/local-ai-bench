import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.runtime import rex_comfyui as rex


@pytest.mark.parametrize("release,kernel,expected", [
    ({"NAME": "AMD Ryzen AI Developer Platform", "VERSION_CODENAME": "rex"}, "linux", True),
    ({"PRETTY_NAME": "AMD Ryzen AI Developer Platform 1 (rex)"}, "6.18.35+rex+2-amd64", True),
    ({"NAME": "Debian GNU/Linux", "VERSION_CODENAME": "rex"}, "6.18.35+rex+2-amd64", False),
    ({"NAME": "AMD Ryzen AI Developer Platform"}, "generic", False),
    ({}, "6.18.35+rex+2-amd64", False),
])
def test_rex_detection_requires_vendor_and_rex_identity(release, kernel, expected):
    assert rex.is_rex_os(release, kernel) is expected


@pytest.fixture
def service(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    root = tmp_path / "config"
    commands = []
    warnings = []
    logs = []
    state = {"mounted": False, "running": True, "busy": False, "restart_error": False,
             "generated_mount": True, "visible": True, "owner": rex.SERVICE}

    def container():
        return {"Config": {"Labels": {"PODMAN_SYSTEMD_UNIT": state["owner"]}},
                "State": {"Running": state["running"]},
                "Mounts": [{"Type": "bind", "Source": str(models),
                            "Destination": rex.MODEL_DESTINATION, "RW": False}]
                if state["mounted"] else []}

    def run(cmd, **kwargs):
        commands.append(cmd)
        output = ""
        error = ""
        if cmd[:2] == ["podman", "inspect"]:
            output = json.dumps([container()])
        elif "SourcePath" in cmd:
            output = rex.SOURCE
        elif "ExecStart" in cmd:
            output = f"podman run -v {models}:{rex.MODEL_DESTINATION}:ro,z" if state["generated_mount"] else "podman run"
        elif "restart" in cmd:
            state["mounted"] = True
            state["running"] = True
            if state["restart_error"]:
                error = "restart failed"
        return SimpleNamespace(returncode=int(bool(error)), stdout=output, stderr=error)

    def get(url, **kwargs):
        assert state["running"], "must not inspect an inactive queue"
        assert url == "http://localhost:8188/queue"
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "queue_running": [1] if state["busy"] else [], "queue_pending": [],
        })

    def prepare():
        return rex.prepare_rex_comfyui(
            models, "http://localhost:8188", detected=True, run=run, get=get,
            visible=lambda: state["visible"], sleep=lambda _: None, config_home=root,
            log=logs.append, warn=warnings.append,
        )

    return SimpleNamespace(models=models, root=root, commands=commands, warnings=warnings,
                           logs=logs, state=state, prepare=prepare, run=run, get=get,
                           dropin=root / "containers/systemd/comfyui@.container.d/90-local-ai-bench-models.conf")


def test_configures_idle_rex_and_reuses_it_without_another_restart(service):
    assert service.prepare() is True
    assert service.dropin.read_text().split("Volume=", 1)[1].strip() == (
        f"{service.models}:{rex.MODEL_DESTINATION}:ro,z")
    assert [cmd for cmd in service.commands if "restart" in cmd] == [
        ["systemctl", "--user", "restart", rex.SERVICE]]
    assert service.prepare() is True
    assert len([cmd for cmd in service.commands if "restart" in cmd]) == 1
    assert not service.warnings


@pytest.mark.parametrize("mounted", [False, True])
def test_busy_service_is_not_restarted_or_used(service, mounted):
    service.state.update(busy=True, mounted=mounted)
    assert service.prepare() is False
    assert not service.dropin.exists()
    assert not any("restart" in cmd for cmd in service.commands)
    assert any("active or queued work" in message for message in service.warnings)


def test_inactive_existing_container_can_be_started(service):
    service.state["running"] = False
    assert service.prepare() is True


@pytest.mark.parametrize("previous", [None, b"[Container]\nVolume=/old:/var/cache/models/comfyui:ro,z\n"])
def test_bad_generated_mount_restores_override_before_any_restart(service, previous):
    if previous is not None:
        service.dropin.parent.mkdir(parents=True)
        service.dropin.write_bytes(previous)
    service.state["generated_mount"] = False
    assert service.prepare() is False
    assert not any("restart" in cmd for cmd in service.commands)
    assert (service.dropin.read_bytes() if service.dropin.exists() else None) == previous


def test_restart_failure_is_reported_and_configuration_is_restored(service):
    service.state["restart_error"] = True
    assert service.prepare() is False
    assert not service.dropin.exists()
    assert any("restart failed" in message for message in service.warnings)


def test_invisible_models_fail_readiness_and_restore_configuration(service):
    service.state["visible"] = False
    assert service.prepare() is False
    assert not service.dropin.exists()
    assert any("within 60 seconds" in message for message in service.warnings)


def test_unknown_container_owner_is_never_modified(service):
    service.state["owner"] = "custom.service"
    assert service.prepare() is False
    assert not service.dropin.exists()
    assert not any("restart" in cmd for cmd in service.commands)


def test_other_platforms_and_custom_endpoints_are_untouched(tmp_path):
    run = Mock(side_effect=AssertionError("must not probe"))
    for detected, url in ((False, "http://localhost:8188"), (True, "http://remote:8188")):
        assert rex.prepare_rex_comfyui(tmp_path, url, detected=detected, run=run,
                                       visible=Mock(), log=Mock(), warn=Mock()) is None


def test_mount_config_preserves_spaces_and_escapes_systemd_specifiers():
    text = rex.model_mount_config(Path("/home/name/models with % marks"))
    assert 'Volume=/home/name/models with %% marks:' in text
    with pytest.raises(ValueError, match="unsupported"):
        rex.model_mount_config(Path("/home/bad\npath"))


def test_os_detection_does_not_need_fastfetch(monkeypatch):
    monkeypatch.setattr(rex.platform, "system", lambda: "Linux")
    monkeypatch.setattr(rex.platform, "release", lambda: "6.18.35+rex+2-amd64")
    monkeypatch.setattr(rex.platform, "freedesktop_os_release", lambda: {
        "NAME": "AMD Ryzen AI Developer Platform", "VERSION_CODENAME": "rex"})
    assert rex.rex_os_detected()
    monkeypatch.setattr(rex.platform, "freedesktop_os_release", Mock(side_effect=OSError()))
    assert not rex.rex_os_detected()


@pytest.mark.parametrize("output", ["null", "[]", "{}", '[{"Mounts":null}]', "not json"])
def test_malformed_container_inspection_is_rejected(output):
    with pytest.raises(ValueError):
        rex.container_from_json(output)


def test_missing_standard_service_keeps_normal_comfyui_behavior(tmp_path):
    assert rex.prepare_rex_comfyui(
        tmp_path, "http://localhost:8188", detected=True,
        run=lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""),
        visible=Mock(), log=Mock(), warn=Mock(),
    ) is None


def test_mount_match_requires_exact_source_destination_and_readonly(tmp_path):
    mount = {"Type": "bind", "Source": str(tmp_path), "Destination": rex.MODEL_DESTINATION, "RW": False}
    assert rex.mount_is_current({"Mounts": [mount]}, tmp_path)
    for change in ({"Source": "/elsewhere"}, {"Destination": "/elsewhere"}, {"RW": True}):
        assert not rex.mount_is_current({"Mounts": [{**mount, **change}]}, tmp_path)


@pytest.mark.parametrize("active,expected", [("inactive", True), ("failed", True), ("active", False)])
def test_missing_container_requires_inactive_service_before_start(service, active, expected):
    def run(cmd, **kwargs):
        if cmd[:2] == ["podman", "inspect"] and not service.state["mounted"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="no container")
        if "ActiveState" in cmd:
            return SimpleNamespace(returncode=0, stdout=active, stderr="")
        return service.run(cmd, **kwargs)

    assert rex.prepare_rex_comfyui(
        service.models, "http://localhost:8188", detected=True, run=run,
        get=Mock(side_effect=AssertionError("no running queue")), visible=lambda: True,
        sleep=lambda _: None, config_home=service.root, log=Mock(), warn=Mock(),
    ) is expected
    assert any("restart" in cmd for cmd in service.commands) is expected


@pytest.mark.parametrize("queue", [None, {}, {"queue_running": [], "queue_pending": None},
                                   {"queue_running": [], "queue_pending": [1]}])
def test_unknown_or_pending_queue_blocks_changes(service, queue):
    assert rex.prepare_rex_comfyui(
        service.models, "http://localhost:8188", detected=True, run=service.run,
        get=lambda *a, **k: SimpleNamespace(raise_for_status=lambda: None, json=lambda: queue),
        visible=lambda: True, config_home=service.root, log=Mock(), warn=Mock(),
    ) is False
    assert not service.dropin.exists()
    assert not any("restart" in cmd for cmd in service.commands)


@pytest.mark.parametrize("character", ["\n", "\r", "\t", ":", "\\", '"'])
def test_mount_path_rejects_ambiguous_quadlet_characters(character):
    with pytest.raises(ValueError, match="unsupported"):
        rex.model_mount_config(Path(f"/models/with{character}character"))
