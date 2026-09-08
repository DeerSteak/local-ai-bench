"""Configure Rex's existing rootless ComfyUI service for benchmark models."""

import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import requests


SERVICE = "comfyui@8188.service"
CONTAINER = "comfyui-8188"
SOURCE = "/etc/containers/systemd/users/comfyui@.container"
MODEL_DESTINATION = "/var/cache/models/comfyui"


def is_rex_os(release: dict[str, str], kernel: str) -> bool:
    name = release.get("NAME", "") + " " + release.get("PRETTY_NAME", "")
    return "AMD Ryzen AI Developer Platform" in name and (
        release.get("VERSION_CODENAME") == "rex" or "+rex+" in kernel)


def rex_os_detected() -> bool:
    if platform.system() != "Linux":
        return False
    try:
        return is_rex_os(platform.freedesktop_os_release(), platform.release())
    except OSError:
        return False


def mount_is_current(container: dict, models_dir: Path) -> bool:
    return any(mount.get("Type") == "bind"
               and mount.get("Source") == str(models_dir)
               and mount.get("Destination") == MODEL_DESTINATION
               and mount.get("RW") is False for mount in container.get("Mounts", []))


def container_from_json(output: str) -> dict:
    values = json.loads(output)
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise ValueError("Unexpected Rex ComfyUI container inspection")
    container = values[0]
    if (not isinstance(container.get("Config"), dict)
            or not isinstance(container["Config"].get("Labels"), dict)
            or not isinstance(container.get("State"), dict)
            or not isinstance(container["State"].get("Running"), bool)
            or not isinstance(container.get("Mounts"), list)
            or not all(isinstance(mount, dict) for mount in container["Mounts"])):
        raise ValueError("Incomplete Rex ComfyUI container inspection")
    if container["Config"]["Labels"].get("PODMAN_SYSTEMD_UNIT") != SERVICE:
        raise ValueError("Existing ComfyUI container is not owned by Rex's service")
    return container


def model_mount_config(models_dir: Path) -> str:
    source = str(models_dir)
    if any(ord(char) < 32 or char in ':\\"' for char in source):
        raise ValueError("ComfyUI model directory contains an unsupported mount-path character")
    volume = f"{source.replace('%', '%%')}:{MODEL_DESTINATION}:ro,z"
    # Quadlet consumes Volume as a raw value and quotes the generated command itself.
    return f"[Container]\nVolume={volume}\n"


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def prepare_rex_comfyui(models_dir: Path, url: str, *, visible, log, warn,
                        detected=None, run=None, get=None, sleep=None,
                        config_home: Path | None = None) -> bool | None:
    """Return None off Rex; a detected Rex service must succeed without host fallback."""
    if not (rex_os_detected() if detected is None else detected):
        return None
    if url.rstrip("/") not in {"http://localhost:8188", "http://127.0.0.1:8188"}:
        return None
    run = run or subprocess.run
    get = get or requests.get
    sleep = sleep or time.sleep
    models_dir = models_dir.resolve()
    config_home = config_home or Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    dropin = config_home / "containers/systemd/comfyui@.container.d/90-local-ai-bench-models.conf"
    previous = None
    changed = False

    def command(*args, timeout=15):
        result = run(list(args), capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"{' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout.strip()

    try:
        source = command("systemctl", "--user", "show", SERVICE, "-p", "SourcePath", "--value")
        if not source:
            return None
        if source != SOURCE:
            raise ValueError("Rex ComfyUI Quadlet was not found at its expected source path")
        log("Detected Rex's ComfyUI Quadlet; using its container Python and AMD runtime")
        inspection = run(["podman", "inspect", CONTAINER], capture_output=True, text=True, timeout=15)
        container = None
        if inspection.returncode == 0:
            container = container_from_json(inspection.stdout)
        else:
            # Inspect failures must not be mistaken for proof that no container exists.
            command("podman", "info", "--format", "json")
            active = command("systemctl", "--user", "show", SERVICE, "-p", "ActiveState", "--value")
            if active not in {"inactive", "failed"}:
                raise ValueError("Rex ComfyUI is active but its container could not be inspected")
        running = container is not None and container.get("State", {}).get("Running") is True
        ready = container is not None and running and mount_is_current(container, models_dir) and visible()
        if not models_dir.is_dir():
            raise ValueError(f"Benchmark model directory does not exist: {models_dir}")
        if running:
            response = get(f"{url.rstrip('/')}/queue", timeout=5)
            response.raise_for_status()
            queue = response.json()
            if not isinstance(queue, dict) or any(not isinstance(queue.get(key), list)
                                                  for key in ("queue_running", "queue_pending")):
                raise ValueError("Cannot verify that Rex ComfyUI's queue is idle")
            if queue["queue_running"] or queue["queue_pending"]:
                raise ValueError("Rex ComfyUI has active or queued work; retry after it finishes")
        if ready:
            log("Rex ComfyUI already has the benchmark model mount")
            return True
        if dropin.is_symlink():
            raise ValueError(f"Refusing to replace a symlinked ComfyUI override: {dropin}")
        previous = dropin.read_bytes() if dropin.exists() else None
        content = model_mount_config(models_dir).encode("utf-8")
        if previous != content:
            _atomic_write(dropin, content)
            changed = True
        command("systemctl", "--user", "daemon-reload")
        launch = command("systemctl", "--user", "show", SERVICE, "-p", "ExecStart", "--value")
        if f"{models_dir}:{MODEL_DESTINATION}:ro,z" not in launch:
            raise ValueError("Generated Rex ComfyUI command did not include the requested read-only model mount")
        log("Restarting idle Rex ComfyUI with the benchmark model mount")
        command("systemctl", "--user", "restart", SERVICE, timeout=120)
        for attempt in range(30):
            inspection = run(["podman", "inspect", CONTAINER], capture_output=True, text=True, timeout=15)
            if inspection.returncode == 0:
                current = container_from_json(inspection.stdout)
                if current["State"]["Running"] and mount_is_current(current, models_dir) and visible():
                    log("Rex ComfyUI is ready and sees the benchmark checkpoints")
                    return True
            if attempt and attempt % 5 == 0:
                log("Waiting for Rex ComfyUI and its benchmark checkpoints ...")
            sleep(2)
        raise RuntimeError("Rex ComfyUI did not expose the mounted benchmark checkpoints within 60 seconds")
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, requests.RequestException) as exc:
        warn(f"Could not prepare Rex ComfyUI: {exc}")
        if changed:
            try:
                if previous is None:
                    dropin.unlink(missing_ok=True)
                else:
                    _atomic_write(dropin, previous)
                command("systemctl", "--user", "daemon-reload")
                warn("Restored the previous override; check the service before restarting it again")
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as rollback_error:
                warn(f"Could not restore the ComfyUI override: {rollback_error}")
        warn(f"Inspect the service log with: journalctl --user -u {SERVICE} -n 60 --no-pager")
        return False
