"""Private run-local paths needed by isolated workload recovery."""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = 1


def local_execution_path(event_path: Path) -> Path:
    return Path(f"{Path(event_path).resolve()}.local.json")


def images_dir_for_result(result_path: Path, results_dir: Path) -> Path:
    stem = Path(result_path).stem
    suffix = stem[len("results_"):] if stem.startswith("results_") else stem
    return (Path(results_dir) / f"images_{suffix}").resolve()


@dataclass(frozen=True)
class LocalExecutionContext:
    job_id: str
    comfyui_dir: Path
    images_dir: Path

    def __post_init__(self):
        if not isinstance(self.job_id, str) or not self.job_id.startswith("job_"):
            raise ValueError("local execution context requires a valid job identity")
        for name in ("comfyui_dir", "images_dir"):
            path = getattr(self, name)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"local execution context requires an absolute {name}")

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION, "job_id": self.job_id,
            "comfyui_dir": str(self.comfyui_dir), "images_dir": str(self.images_dir),
        }

    @classmethod
    def from_dict(cls, value: dict):
        if not isinstance(value, dict) or set(value) != {
            "schema_version", "job_id", "comfyui_dir", "images_dir",
        }:
            raise ValueError("local execution context is incomplete")
        if value["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported local execution context schema")
        if not isinstance(value["comfyui_dir"], str) or not isinstance(value["images_dir"], str):
            raise ValueError("local execution paths must be strings")
        return cls(
            job_id=value["job_id"], comfyui_dir=Path(value["comfyui_dir"]),
            images_dir=Path(value["images_dir"]),
        )


def write_local_execution_context(event_path: Path, context: LocalExecutionContext) -> Path:
    destination = local_execution_path(event_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(context.to_dict(), stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        if os.name != "nt":
            os.chmod(destination, 0o600)
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_local_execution_context(event_path: Path, job_id: str) -> LocalExecutionContext:
    path = local_execution_path(event_path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("private local execution context is missing") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("private local execution context is unreadable") from exc
    context = LocalExecutionContext.from_dict(value)
    if context.job_id != job_id:
        raise ValueError("local execution context belongs to a different job")
    return context
