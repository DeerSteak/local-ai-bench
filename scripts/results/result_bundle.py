"""Portable deterministic result bundles and integrity verification."""

import hashlib
import json
import statistics
import zipfile
from pathlib import Path

from scripts.results.canonical_json import canonical_json_bytes
from scripts.results.result_store import atomic_write_json, validate_json_data
from scripts.results.run_plan import RunPlan
from scripts.results.outbound_metadata import prepare_outbound_result


BUNDLE_SCHEMA_VERSION = 1
MAX_BUNDLE_MEMBER_BYTES = 2 * 1024 ** 3
FIXED_ZIP_TIME = (2020, 1, 1, 0, 0, 0)
BANK_PATHS = {
    name: Path(__file__).resolve().parents[1] / "workloads" / "data" / filename for name, filename in {
        "mcq": "mcq_questions.json", "math": "math_questions.json",
        "reasoning": "reasoning_questions.json", "code": "code_problems.json",
        "tool": "tool_questions.json",
    }.items()
}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _zip_entry(name: str, data: bytes) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    return info, data


def export_result_bundle(result_path: Path, bundle_path: Path,
                         artifacts: list[Path] | None = None, *,
                         system_alias: str | None = None,
                         hardware_alias: str | None = None) -> dict:
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    validate_json_data(result)
    result = prepare_outbound_result(
        result, system_alias=system_alias, hardware_alias=hardware_alias,
    )
    files = {"result.json": canonical_json_bytes(result)}
    artifact_records = []
    for artifact in artifacts or []:
        data = Path(artifact).read_bytes()
        digest = _digest(data)
        name = f"artifacts/{digest}{Path(artifact).suffix.lower()}"
        files.setdefault(name, data)
        artifact_records.append({
            "bundle_path": name, "original_name": Path(artifact).name,
            "sha256": digest, "size": len(data),
        })
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "application_version": result.get("version"),
        "result_schema_version": result.get("run", {}).get("schema_version"),
        "files": {
            name: {"sha256": _digest(data), "size": len(data)}
            for name, data in sorted(files.items())
        },
        "artifacts": artifact_records,
    }
    files["manifest.json"] = canonical_json_bytes(manifest)
    Path(bundle_path).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, "w") as archive:
        for name, data in sorted(files.items()):
            archive.writestr(*_zip_entry(name, data))
    return manifest


def aggregate_reproduction_errors(result: dict) -> list[str]:
    errors = []

    def visit(value, path):
        if isinstance(value, dict):
            samples = value.get("valid_samples")
            if isinstance(samples, list) and samples:
                if value.get("valid_runs") != len(samples):
                    errors.append(f"{path}.valid_runs does not match valid_samples")
                checks = {
                    "tps_mean": ("tokens_per_sec", 2),
                    "client_ttft_mean_sec": ("client_ttft_sec", 3),
                    "ttft_mean_sec": ("client_ttft_sec", 3),
                }
                for aggregate, (sample_key, digits) in checks.items():
                    values = [sample.get(sample_key) for sample in samples]
                    if aggregate in value and all(isinstance(item, (int, float)) for item in values):
                        reproduced = round(statistics.mean(values), digits)
                        if value[aggregate] != reproduced:
                            errors.append(f"{path}.{aggregate} does not match valid_samples")
            runs = value.get("runs")
            if isinstance(runs, list) and runs and all(isinstance(item, (int, float)) for item in runs):
                for aggregate, digits in (
                    ("chunks_per_sec_mean", 1), ("sec_per_image_mean", 2),
                ):
                    if aggregate in value and value[aggregate] != round(statistics.mean(runs), digits):
                        errors.append(f"{path}.{aggregate} does not match runs")
            for key, child in value.items():
                visit(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(result, "$")
    return errors


def methodology_availability_errors(result: dict) -> list[str]:
    errors = []
    for name, expected in result.get("bank_versions", {}).items():
        path = BANK_PATHS.get(name)
        if path is None or not path.is_file():
            errors.append(f"Methodology bank is unavailable: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest()[:12] != expected:
            errors.append(f"Methodology bank version does not match: {name}")
    return errors


def _verify_result_bundle(bundle_path: Path) -> dict:
    with zipfile.ZipFile(bundle_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "manifest.json" not in names or "result.json" not in names:
            raise ValueError("Bundle must contain one manifest.json and one result.json.")
        if any(info.file_size > MAX_BUNDLE_MEMBER_BYTES for info in archive.infolist()):
            raise ValueError("Bundle contains an oversized file.")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            raise ValueError("Unsupported result-bundle schema version.")
        declared = manifest.get("files")
        if not isinstance(declared, dict) or set(names) != set(declared) | {"manifest.json"}:
            raise ValueError("Bundle file inventory does not match its manifest.")
        for name, identity in declared.items():
            data = archive.read(name)
            if identity != {"sha256": _digest(data), "size": len(data)}:
                raise ValueError(f"Bundle integrity check failed for {name}.")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("Bundle artifact inventory is invalid.")
        for record in artifacts:
            if (not isinstance(record, dict)
                    or set(record) != {"bundle_path", "original_name", "sha256", "size"}
                    or record["bundle_path"] not in declared
                    or Path(record["original_name"]).name != record["original_name"]
                    or declared[record["bundle_path"]] != {
                        "sha256": record["sha256"], "size": record["size"],
                    }):
                raise ValueError("Bundle artifact inventory is invalid.")
        result = json.loads(archive.read("result.json"))
    validate_json_data(result)
    plan_value = result.get("run", {}).get("plan")
    if plan_value is not None:
        RunPlan.from_dict(plan_value)
    methodology_errors = methodology_availability_errors(result)
    if methodology_errors:
        raise ValueError(methodology_errors[0])
    aggregate_errors = aggregate_reproduction_errors(result)
    if aggregate_errors:
        raise ValueError(aggregate_errors[0])
    return {"manifest": manifest, "result": result}


def verify_result_bundle(bundle_path: Path) -> dict:
    try:
        return _verify_result_bundle(bundle_path)
    except zipfile.BadZipFile as exc:
        raise ValueError("File is not a valid result bundle.") from exc


def import_result_bundle(bundle_path: Path, result_path: Path,
                         artifact_dir: Path | None = None) -> dict:
    verified = verify_result_bundle(bundle_path)
    atomic_write_json(Path(result_path), verified["result"])
    if artifact_dir is not None:
        with zipfile.ZipFile(bundle_path) as archive:
            for record in verified["manifest"]["artifacts"]:
                destination = Path(artifact_dir) / Path(record["bundle_path"]).name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(record["bundle_path"]))
    return verified["manifest"]
