"""Read-only engine status records shared by the GUI and diagnostics."""

import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from scripts.runtime import config
from scripts.setup.runtime_identity import (
    RuntimeIdentity, inspect_runtime, parse_llamacpp_commit, probe_vllm_server_health,
    probe_vllm_server_version, source_commit_version,
)


VLLM_ENV_PROBE = (
    "import json, platform, torch, vllm; "
    "print(json.dumps({'vllm': vllm.__version__, 'torch': torch.__version__, "
    "'cuda_runtime': torch.version.cuda, 'rocm_runtime': torch.version.hip, "
    "'cuda_available': torch.cuda.is_available(), 'python': platform.python_version()}))"
)


@dataclass(frozen=True)
class EngineStatus:
    engine: str
    ownership: str
    location: str
    version: str | None
    backend: str
    health: str
    components: dict[str, str | bool | None] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def managed(self) -> bool:
        return self.ownership == "app_managed"

    def as_dict(self) -> dict:
        return asdict(self)


def runtime_python(executable: str | Path | None, exists_fn=None) -> Path | None:
    if executable is None:
        return None
    exists_fn = exists_fn or (lambda path: path.is_file())
    path = Path(executable)
    name = "python.exe" if path.suffix.lower() == ".exe" else "python"
    candidate = path.with_name(name)
    return candidate if exists_fn(candidate) else None


def parse_vllm_environment(output: str | None) -> dict[str, str | bool | None]:
    try:
        payload = json.loads((output or "").strip())
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    allowed = {"vllm", "torch", "cuda_runtime", "rocm_runtime", "cuda_available", "python"}
    return {
        key: value for key, value in payload.items()
        if key in allowed and (isinstance(value, (str, bool)) or value is None)
    }


def probe_vllm_environment(python_exe: Path | None, *, run=subprocess.run) -> tuple[dict, str | None]:
    if python_exe is None:
        return {}, "The Python interpreter for this vLLM executable was not found."
    try:
        result = run(
            [str(python_exe), "-c", VLLM_ENV_PROBE], capture_output=True, text=True,
            timeout=config.VLLM_COLD_IMPORT_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {}, str(exc)
    components = parse_vllm_environment(result.stdout)
    if result.returncode != 0 or not components:
        detail = (result.stderr or result.stdout or "vLLM environment probe failed").strip()
        return {}, detail
    return components, None


def build_llamacpp_status(location: str | Path | None, managed_root: Path, backend: str,
                          *, run=subprocess.run) -> EngineStatus:
    identity = inspect_runtime("llamacpp", location, managed_root, run=run)
    version = source_commit_version(identity, managed_root, run=run)
    health = "ready" if version else "unavailable" if not location else "unverified"
    warnings = () if version else ((identity.version_output or "Version unavailable"),)
    adjusted = RuntimeIdentity(
        identity.engine, identity.ownership, identity.location, version, identity.version_output,
    )
    components = {"install_type": llamacpp_install_type(identity, managed_root)}
    if (identity.managed and identity.version and identity.version != "1"
            and parse_llamacpp_commit(identity.version_output)):
        components["build_number"] = identity.version
    return _status(adjusted, backend, health, components, warnings)


def llamacpp_install_type(identity: RuntimeIdentity, managed_root: Path) -> str:
    if identity.managed:
        return "Built from source" if (Path(managed_root) / ".git").is_dir() \
            else "Binary download"
    if identity.ownership == "missing":
        return "Not installed"
    return "System installation"


def build_vllm_status(location: str | Path | None, managed_root: Path, backend: str,
                      *, launcher: str | None = None, server_url: str | None = None,
                      is_wsl: bool = False, env=None, run=subprocess.run,
                      open_fn=None) -> EngineStatus:
    selected = server_url or launcher or location
    reachable = False
    if server_url:
        version = probe_vllm_server_version(
            server_url, env=env, **({"open_fn": open_fn} if open_fn is not None else {}),
        )
        reachable = bool(version) or probe_vllm_server_health(
            server_url, env=env, **({"open_fn": open_fn} if open_fn is not None else {}),
        )
        identity = RuntimeIdentity("vllm", "external_server", server_url, version, "")
    elif launcher:
        identity = RuntimeIdentity("vllm", "platform_launcher", launcher, None, "")
    else:
        identity = inspect_runtime("vllm", location, managed_root, run=run)
    ownership = identity.ownership
    components, warning = ({}, None)
    if location and not launcher and not server_url:
        components, warning = probe_vllm_environment(runtime_python(location), run=run)
    effective_env = os.environ if env is None else env
    wsl_pin_memory = None
    if is_wsl and not launcher and not server_url:
        wsl_pin_memory = effective_env.get("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
    components.update({
        "wsl": is_wsl,
        "wsl_pin_memory": wsl_pin_memory,
        "kernel": platform.release() if is_wsl else None,
    })
    version = components.get("vllm") if isinstance(components.get("vllm"), str) else identity.version
    warnings = tuple(value for value in (warning,) if value)
    if ownership == "external_server":
        health = "ready" if reachable else "unavailable"
        if not reachable:
            warnings = (*warnings, "External vLLM server did not respond to /health.")
    else:
        health = "ready" if (version or ownership == "platform_launcher") else "unverified"
    adjusted = RuntimeIdentity("vllm", ownership, str(selected or ""), version, identity.version_output)
    return _status(adjusted, backend, health, components, warnings)


def _status(identity: RuntimeIdentity, backend: str, health: str, components: dict,
            warnings: tuple[str, ...]) -> EngineStatus:
    return EngineStatus(
        identity.engine, identity.ownership, identity.location, identity.version,
        backend, health, components, warnings,
    )
