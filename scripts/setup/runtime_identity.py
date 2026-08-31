"""Read-only inference-runtime ownership and version inspection."""

import json
import os
import re
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from scripts.runtime import config
from scripts.runtime.engine_identity import engine_family


@dataclass(frozen=True)
class RuntimeIdentity:
    engine: str
    ownership: str
    location: str
    version: str | None
    version_output: str

    @property
    def managed(self) -> bool:
        return self.ownership == "app_managed"


def runtime_ownership(location: str | Path | None, managed_root: Path) -> str:
    if location is None:
        return "missing"
    text = str(location)
    if text.startswith(("http://", "https://")):
        return "external_server"
    path = Path(text).expanduser()
    try:
        path.resolve().relative_to(Path(managed_root).expanduser().resolve())
    except (OSError, ValueError):
        return "system_managed"
    return "app_managed"


def parse_runtime_version(output: str | None) -> str | None:
    text = (output or "").strip()
    patterns = (
        r"(?im)^vllm\s+([0-9]+(?:\.[0-9A-Za-z+-]+)+)\s*$",
        r"(?im)^version\s*:\s*([^\s]+)",
        r"(?i)\bversion\s+v?([0-9]+(?:\.[0-9A-Za-z+-]+)+)",
        r"(?i)^v?([0-9]+(?:\.[0-9A-Za-z+-]+)+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def parse_llamacpp_commit(output: str | None) -> str | None:
    match = re.search(r"\(([0-9a-f]{7,40})\)", output or "", re.IGNORECASE)
    return match.group(1).lower() if match else None


def managed_distribution_version(managed_root: Path, distribution: str) -> str | None:
    site_packages = [managed_root / "Lib" / "site-packages"]
    site_packages.extend(sorted(managed_root.glob("lib/python*/site-packages")))
    for directory in site_packages:
        for metadata in sorted(directory.glob(f"{distribution}-*.dist-info/METADATA")):
            try:
                content = metadata.read_text(encoding="utf-8")
                match = re.search(r"(?im)^Version:\s*([^\s]+)", content)
            except OSError:
                continue
            if match:
                return match.group(1)
    return None


def source_commit_version(identity: RuntimeIdentity, managed_root: Path, *,
                          run=subprocess.run) -> str | None:
    """Prefer a sortable commit identity for a managed llama.cpp source build."""
    if not identity.managed or not (Path(managed_root) / ".git").exists():
        return identity.version
    try:
        commit = parse_llamacpp_commit(identity.version_output)
        if not commit:
            return identity.version
        tag_result = run(
            ["git", "-C", str(managed_root), "describe", "--tags", "--exact-match", commit],
            capture_output=True, text=True, timeout=15,
        )
        tag = tag_result.stdout.strip()
        if tag_result.returncode == 0 and re.fullmatch(r"b\d+", tag):
            return tag[1:]
        result = run(
            ["git", "-C", str(managed_root), "show", "-s", "--format=%cd",
             "--date=format-local:%Y.%m.%d", commit],
            capture_output=True, text=True, timeout=15,
            env={**os.environ, "TZ": "UTC"},
        )
    except (OSError, subprocess.SubprocessError):
        return identity.version
    commit_date = result.stdout.strip()
    if result.returncode != 0 or not re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", commit_date):
        return identity.version
    return f"{commit_date}-{commit[:7]}"


def inspect_runtime(engine: str, location: str | Path | None, managed_root: Path,
                    *, run=subprocess.run, env=None) -> RuntimeIdentity:
    ownership = runtime_ownership(location, managed_root)
    if ownership in {"missing", "external_server"}:
        return RuntimeIdentity(engine, ownership, str(location or ""), None, "")
    try:
        result = run(
            [str(location), "--version"], capture_output=True, text=True, timeout=15,
            env=env,
        )
        output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        output = str(exc)
    return RuntimeIdentity(
        engine, ownership, str(location), parse_runtime_version(output), output,
    )


def engine_runtime_version(engine_name: str, engine, *, run=subprocess.run) -> str | None:
    if engine_name == "vllm":
        server_url = getattr(engine, "external_server_url", lambda: None)()
        if server_url:
            return probe_vllm_server_version(server_url)
    location = getattr(engine, "runtime_location", lambda: None)()
    managed_root = (
        config.LLAMACPP_VULKAN_DIR if engine_name == "llamacpp-vulkan"
        else config.LLAMACPP_DIR if engine_family(engine_name) == "llamacpp"
        else config.VLLM_VENV
    )
    if (engine_name == "vllm" and runtime_ownership(location, managed_root) == "app_managed"):
        version = managed_distribution_version(managed_root, "vllm")
        if version:
            return version
    process_env = getattr(engine, "process_environment", lambda: None)()
    identity = inspect_runtime(engine_name, location, managed_root, run=run, env=process_env)
    if engine_family(engine_name) == "llamacpp":
        return source_commit_version(identity, managed_root, run=run)
    return identity.version


def probe_vllm_server_version(server_url: str, *, open_fn=urllib.request.urlopen,
                              env=None) -> str | None:
    request = _vllm_request(server_url, "/version", env)
    try:
        with open_fn(request, timeout=3) as response:
            payload = response.read(4097)
    except Exception:
        return None
    if len(payload) > 4096:
        return None
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    version = data.get("version") if isinstance(data, dict) else None
    return version.strip() if isinstance(version, str) and version.strip() else None


def probe_vllm_server_health(server_url: str, *, open_fn=urllib.request.urlopen,
                             env=None) -> bool:
    request = _vllm_request(server_url, "/health", env)
    try:
        with open_fn(request, timeout=3) as response:
            status = getattr(response, "status", 200)
            return 200 <= status < 300
    except Exception:
        return False


def _vllm_request(server_url: str, path: str, env) -> urllib.request.Request:
    headers = {}
    token = (os.environ if env is None else env).get("VLLM_API_KEY")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(f"{server_url.rstrip('/')}{path}", headers=headers)
