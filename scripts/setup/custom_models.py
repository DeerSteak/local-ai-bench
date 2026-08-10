"""Persistent engine-specific custom model provenance."""

import json
import os
import tempfile
from pathlib import Path

from scripts.runtime import config


SCHEMA_VERSION = 1


def load_custom_models(path: Path = config.CUSTOM_MODELS_PATH) -> list[dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        return []
    models = data.get("models")
    return [entry for entry in models if isinstance(entry, dict)] if isinstance(models, list) else []


def custom_model(engine: str, tag: str, path: Path = config.CUSTOM_MODELS_PATH) -> dict | None:
    return next(
        (entry for entry in load_custom_models(path)
         if entry.get("engine") == engine and entry.get("tag") == tag),
        None,
    )


def save_custom_model(entry: dict, path: Path = config.CUSTOM_MODELS_PATH) -> None:
    models = [
        item for item in load_custom_models(path)
        if (item.get("engine"), item.get("tag")) != (entry.get("engine"), entry.get("tag"))
    ]
    models.append(dict(entry))
    models.sort(key=lambda item: (str(item.get("tag", "")), str(item.get("engine", ""))))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump({"schema_version": SCHEMA_VERSION, "models": models}, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def forget_custom_models(*, engine: str, tag: str | None = None, repo: str | None = None,
                         path: Path = config.CUSTOM_MODELS_PATH) -> int:
    models = load_custom_models(path)
    kept = [entry for entry in models if not (
        entry.get("engine") == engine
        and (tag is None or entry.get("tag") == tag)
        and (repo is None or entry.get("repo") == repo)
    )]
    if len(kept) == len(models):
        return 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump({"schema_version": SCHEMA_VERSION, "models": kept}, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return len(models) - len(kept)
