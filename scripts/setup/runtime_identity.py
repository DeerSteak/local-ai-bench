"""Read-only inference-runtime ownership and version inspection."""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


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


def inspect_runtime(engine: str, location: str | Path | None, managed_root: Path,
                    *, run=subprocess.run) -> RuntimeIdentity:
    ownership = runtime_ownership(location, managed_root)
    if ownership in {"missing", "external_server"}:
        return RuntimeIdentity(engine, ownership, str(location or ""), None, "")
    try:
        result = run(
            [str(location), "--version"], capture_output=True, text=True, timeout=15,
        )
        output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        output = str(exc)
    return RuntimeIdentity(
        engine, ownership, str(location), parse_runtime_version(output), output,
    )
