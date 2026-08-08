"""Pure helpers shared by setup interfaces."""

import os
from pathlib import Path


def toggle_all_models(entries: list[dict]) -> None:
    """Toggle every install entry without changing destructive cleanup choices."""
    model_entries = [entry for entry in entries if entry["kind"] != "cleanup"]
    checked = not all(entry["checked"] for entry in model_entries)
    for entry in model_entries:
        entry["checked"] = checked


def selected_cleanup_names(entries: list[dict], kind: str = "cleanup") -> list[str]:
    """Return names from explicitly selected cleanup entries of one kind."""
    return [
        name
        for entry in entries
        if entry["kind"] == kind and entry["checked"]
        for name in entry["item"]["directory_names"]
    ]


def save_hf_token(path: Path, token: str) -> None:
    """Save a Hugging Face token with private permissions where supported."""
    value = token.strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError("Hugging Face token must be a non-empty single line")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as token_file:
            token_file.write(value + "\n")
    finally:
        if os.name != "nt":
            path.chmod(0o600)


def additional_disk_space_needed(free_gb: float, download_gb: float) -> float:
    """Return the download shortfall in GB, or zero when it fits."""
    return max(0.0, download_gb - free_gb)
