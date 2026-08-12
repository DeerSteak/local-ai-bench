"""Engine-scoped crash-cache persistence and skip policy."""

import json
from datetime import datetime
from pathlib import Path

from scripts.results.result_store import atomic_write_json
from scripts.runtime import config
from scripts.runtime.shared import Shared


def load_crash_cache(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_crash_cache(path: Path, cache: dict) -> None:
    try:
        atomic_write_json(path, cache)
    except Exception as exc:
        Shared.warn(f"Failed to save crash cache to {path}: {exc}")


def crash_cache_paths(root: Path) -> list[Path]:
    return sorted(
        path for path in Path(root).glob(".*_crash_cache.json")
        if path.is_file() or path.is_symlink()
    )


def clear_crash_caches(root: Path) -> tuple[list[Path], dict[Path, str]]:
    removed = []
    failures = {}
    for path in crash_cache_paths(root):
        try:
            path.unlink()
            removed.append(path)
        except OSError as exc:
            failures[path] = str(exc)
    return removed, failures


def check_crash_cache(tag: str, label: str, crash_cache: dict, cache_path: Path,
                      expected_bank_hash: str | None = None, *,
                      engine_name: str) -> dict | None:
    detail = crash_cache.get(engine_name, {}).get(tag)
    if detail is None:
        return None
    if config.RETRY_CRASHED_MODELS:
        Shared.warn(f"{tag}: ignoring its prior crash record for this run")
        return None
    if expected_bank_hash is not None and detail.get("bank_hash") != expected_bank_hash:
        Shared.warn(
            f"{tag}'s recorded crash is for a different question-bank version "
            "— ignoring stale entry and retrying",
        )
        return None
    crashed_at = detail.get("crashed_at", "an earlier run")
    Shared.warn(
        f"{tag} previously crashed {engine_name}'s runner repeatedly on {crashed_at} "
        f"— skipping (delete {cache_path} to retry)",
    )
    return {
        "label": label, "skipped": True, "skip_reason": "known_crash",
        "skip_detail": f"Crashed {engine_name}'s runner repeatedly on {crashed_at}",
    }


def record_crash(tag: str, crash_cache: dict, cache_path: Path, what: str,
                 extra: dict | None = None, *, engine_name: str) -> str:
    crashed_at = datetime.now().isoformat(timespec="seconds")
    crash_cache.setdefault(engine_name, {})[tag] = {
        "crashed_at": crashed_at, **(extra or {}),
    }
    save_crash_cache(cache_path, crash_cache)
    Shared.err(f"{engine_name}'s runner crashed repeatedly {what} — recorded to {cache_path}")
    return crashed_at
