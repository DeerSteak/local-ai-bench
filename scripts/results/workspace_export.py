"""Deterministic report and bundle exports from one workspace selection."""

import hashlib
import json
from pathlib import Path
import zipfile

from scripts.results.canonical_json import canonical_json_bytes
from scripts.results.decision_report import write_html_report, write_pdf_report
from scripts.results.result_store import validate_json_data
from scripts.results.workspace_selection import validate_workspace_selection


WORKSPACE_BUNDLE_SCHEMA_VERSION = 1
FIXED_ZIP_TIME = (2020, 1, 1, 0, 0, 0)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_workspace_results(selection: dict, candidates: list[Path]) -> list[Path]:
    validate_workspace_selection(selection)
    by_digest = {}
    for candidate in candidates:
        path = Path(candidate).resolve()
        digest = _digest(path.read_bytes())
        if digest in by_digest:
            raise ValueError("workspace result candidates must be distinct")
        by_digest[digest] = path
    missing = [item["name"] for item in selection["results"] if item["sha256"] not in by_digest]
    if missing:
        raise ValueError(f"workspace results are missing or changed: {', '.join(missing)}")
    return [by_digest[item["sha256"]] for item in selection["results"]]


def _selected_report_result(selection: dict, paths: list[Path]) -> tuple[Path, dict]:
    baseline = selection["baseline_sha256"] or selection["results"][0]["sha256"]
    index = next(index for index, item in enumerate(selection["results"])
                 if item["sha256"] == baseline)
    path = paths[index]
    result = json.loads(path.read_text(encoding="utf-8"))
    validate_json_data(result)
    return path, result


def write_workspace_reports(selection: dict, candidates: list[Path], *,
                            html_path: Path | None = None,
                            pdf_path: Path | None = None) -> list[Path]:
    paths = resolve_workspace_results(selection, candidates)
    _, result = _selected_report_result(selection, paths)
    policy = selection.get("acceptance_policy")
    recommendation = selection.get("recommendation")
    written = []
    if html_path is not None:
        written.append(write_html_report(
            result, html_path, policy, recommendation, result, selection,
        ))
    if pdf_path is not None:
        written.append(write_pdf_report(
            result, pdf_path, policy, recommendation, result, selection,
        ))
    if not written:
        raise ValueError("workspace report requires an HTML or PDF output")
    return written


def export_workspace_bundle(selection: dict, candidates: list[Path], bundle_path: Path) -> dict:
    paths = resolve_workspace_results(selection, candidates)
    files = {"workspace_selection.json": canonical_json_bytes(selection)}
    result_records = []
    for index, (identity, path) in enumerate(zip(selection["results"], paths)):
        data = path.read_bytes()
        name = f"results/{index}.json"
        files[name] = data
        result_records.append({**identity, "bundle_path": name, "size": len(data)})
    manifest = {
        "schema_version": WORKSPACE_BUNDLE_SCHEMA_VERSION,
        "files": {name: {"sha256": _digest(data), "size": len(data)}
                  for name, data in sorted(files.items())},
        "results": result_records,
    }
    files["manifest.json"] = canonical_json_bytes(manifest)
    Path(bundle_path).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, "w") as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, data)
    return manifest


def verify_workspace_bundle(bundle_path: Path) -> dict:
    try:
        with zipfile.ZipFile(bundle_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or "manifest.json" not in names \
                    or "workspace_selection.json" not in names:
                raise ValueError("workspace bundle inventory is invalid")
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("schema_version") != WORKSPACE_BUNDLE_SCHEMA_VERSION:
                raise ValueError("unsupported workspace-bundle schema")
            declared = manifest.get("files")
            if not isinstance(declared, dict) or set(names) != set(declared) | {"manifest.json"}:
                raise ValueError("workspace bundle inventory is invalid")
            for name, identity in declared.items():
                data = archive.read(name)
                if identity != {"sha256": _digest(data), "size": len(data)}:
                    raise ValueError(f"workspace bundle integrity check failed for {name}")
            selection = validate_workspace_selection(json.loads(
                archive.read("workspace_selection.json"),
            ))
            records = manifest.get("results")
            if not isinstance(records, list) or len(records) != len(selection["results"]):
                raise ValueError("workspace bundle result inventory is invalid")
            results = []
            for expected, record in zip(selection["results"], records):
                if not isinstance(record, dict) or record.get("name") != expected["name"] \
                        or record.get("sha256") != expected["sha256"]:
                    raise ValueError("workspace bundle result inventory is invalid")
                data = archive.read(record["bundle_path"])
                if _digest(data) != expected["sha256"] or record.get("size") != len(data):
                    raise ValueError("workspace bundle result identity is invalid")
                result = json.loads(data)
                validate_json_data(result)
                results.append(result)
    except zipfile.BadZipFile as exc:
        raise ValueError("file is not a valid workspace bundle") from exc
    return {"manifest": manifest, "selection": selection, "results": results}
