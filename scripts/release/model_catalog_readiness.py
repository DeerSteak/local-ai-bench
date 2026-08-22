"""Verify Milestone 9 source, screen, and catalog-cost evidence."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re

from scripts.release.model_catalog_screen import (
    build_screen_spec, compatibility_screen_errors, load_source_audit,
)
from scripts.runtime import config
from scripts.runtime.shared import Shared


SCHEMA_VERSION = 1
DEFAULT_INCUMBENT_AUDIT = config.SCRIPT_DIR / "docs" / \
    "model-catalog-incumbent-source-audit-v6.json"
DEFAULT_SCREEN_ROOT = config.RESULTS_DIR / "catalog-audit"


def load_incumbent_audit(path: Path = DEFAULT_INCUMBENT_AUDIT) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("incumbents"), list):
        raise ValueError("unsupported incumbent source audit")
    return value


def required_candidate_screens(candidate: dict) -> tuple[str, ...]:
    if candidate.get("status") != "source_ready":
        return ()
    if candidate.get("family") == "image":
        return ("comfyui",)
    return ("llamacpp", "vllm")


def _safe_artifact_path(directory: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return None
    directory = directory.resolve()
    path = (directory / value).resolve()
    return path if path.is_relative_to(directory) else None


def _artifact_errors(directory: Path, records: object, *, label: str) -> list[str]:
    if not isinstance(records, list) or not records:
        return [f"{label} manifest is missing"]
    errors, seen = [], set()
    for record in records:
        if not isinstance(record, dict):
            errors.append(f"{label} manifest has an invalid record")
            continue
        path = _safe_artifact_path(directory, record.get("path"))
        if path is None:
            errors.append(f"{label} path is unsafe")
            continue
        relative = str(path.relative_to(directory.resolve()))
        if relative in seen:
            errors.append(f"{label} path is duplicated: {relative}")
            continue
        seen.add(relative)
        if not path.is_file():
            errors.append(f"{label} file is missing: {relative}")
        elif path.stat().st_size != record.get("size"):
            errors.append(f"{label} size does not match: {relative}")
        elif Shared.file_sha256(path) != record.get("sha256"):
            errors.append(f"{label} digest does not match: {relative}")
    return errors


def _manifest_paths(records: object) -> set[str]:
    return {
        record["path"] for record in records
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    } if isinstance(records, list) else set()


def validate_screen_report(path: Path, report: dict, candidate: dict) -> list[str]:
    errors = []
    engine = report.get("engine")
    screen_engine = engine if engine in {"llamacpp", "vllm"} else "llamacpp"
    try:
        expected = build_screen_spec(candidate, screen_engine, path.parent)
    except ValueError as exc:
        return [str(exc)]
    if report.get("schema_version") != 1:
        errors.append("unsupported screen report schema")
    if report.get("candidate") != candidate.get("id"):
        errors.append("candidate identity does not match")
    if candidate.get("family") == "image":
        if engine not in {"llamacpp", "vllm"}:
            errors.append("image screen engine is invalid")
    elif engine not in required_candidate_screens(candidate):
        errors.append("screen engine is not required for this candidate")
    if report.get("repo") != expected.repo or report.get("revision") != expected.revision:
        errors.append("source identity does not match the pinned audit")
    if tuple(report.get("files") or ()) != expected.files:
        errors.append("artifact files do not match the pinned audit")
    if report.get("status") != "passed" or report.get("errors") not in ([], None):
        errors.append("screen report did not pass")
    evidence_artifacts = report.get("evidence_artifacts")
    errors.extend(_artifact_errors(path.parent, evidence_artifacts,
                                   label="screen evidence"))
    if _manifest_paths(evidence_artifacts) != {
            "result.json", "result.events.sqlite3", "initial.log", "resume.log"}:
        errors.append("screen evidence manifest does not list the required files")
    if candidate.get("family") == "image":
        errors.extend(_artifact_errors(path.parent, report.get("image_artifacts"),
                                       label="generated image"))
    result_path = _safe_artifact_path(path.parent, report.get("result"))
    if report.get("result") != "result.json":
        errors.append("result path does not match the evidence manifest")
    if result_path is None:
        errors.append("result path is unsafe")
        return errors
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        errors.append("result is missing or unreadable")
        return errors
    if candidate.get("family") == "image":
        if not isinstance(report.get("comfyui_revision"), str) \
                or re.fullmatch(r"[0-9a-f]{40}", report["comfyui_revision"]) is None:
            errors.append("ComfyUI revision is missing")
        resolutions = result.get("images", {}).get(expected.tag, {}).get("resolutions", {})
        expected_images = {
            f"images_result/{expected.tag}_{resolution}.png" for resolution in resolutions
        } if isinstance(resolutions, dict) else set()
        if _manifest_paths(report.get("image_artifacts")) != expected_images:
            errors.append("generated image manifest does not match measured resolutions")
    elif not isinstance(result.get("engine_version"), str) \
            or not result["engine_version"].strip():
        errors.append("runtime version is missing")
    profile = result.get("profile")
    if not isinstance(profile, dict) or any(
            profile.get(key) in (None, "") for key in ("os", "arch", "ram_gb", "hardware_backend")):
        errors.append("hardware profile is incomplete")
    errors.extend(compatibility_screen_errors(result, replace(expected, output_path=result_path)))
    return list(dict.fromkeys(errors))


def discover_screen_reports(root: Path) -> tuple[list[tuple[Path, dict]], list[str]]:
    reports, errors = [], []
    for path in sorted(Path(root).rglob("screen-report.json")) if Path(root).exists() else ():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append(f"unreadable screen report: {path}")
            continue
        if not isinstance(value, dict):
            errors.append(f"invalid screen report: {path}")
            continue
        reports.append((path, value))
    return reports, errors


def _source_artifact_bytes(source: object) -> int | None:
    if not isinstance(source, dict) or not isinstance(source.get("artifact"), dict):
        return None
    artifact = source["artifact"]
    if isinstance(artifact.get("size"), int):
        return artifact["size"]
    if isinstance(artifact.get("weight_size"), int):
        return artifact["weight_size"]
    files = artifact.get("files")
    if isinstance(files, list) and all(isinstance(item, dict) for item in files):
        sizes = [item.get("size") for item in files]
        return sum(sizes) if sizes and all(isinstance(size, int) for size in sizes) else None
    return None


def incumbent_catalog_cost(incumbents: list[dict]) -> dict:
    totals = {"llamacpp_bytes": 0, "vllm_bytes": 0, "comfyui_bytes": 0}
    unknown = {key: [] for key in totals}
    roles = {
        "llamacpp_bytes": "llamacpp", "vllm_bytes": "vllm", "comfyui_bytes": "comfyui",
    }
    for incumbent in incumbents:
        for total, role in roles.items():
            source = incumbent.get("sources", {}).get(role)
            if source is None:
                continue
            size = _source_artifact_bytes(source)
            if size is None:
                unknown[total].append(incumbent.get("id"))
            else:
                totals[total] += size
    return {**totals, "unknown": unknown}


def build_readiness(candidate_audit: dict, incumbent_audit: dict,
                    screen_reports: list[tuple[Path, dict]],
                    discovery_errors: list[str] | None = None) -> dict:
    candidates = candidate_audit["candidates"]
    by_id = {candidate["id"]: candidate for candidate in candidates}
    found: dict[tuple[str, str], list[tuple[Path, dict]]] = {}
    orphaned = []
    for path, report in screen_reports:
        candidate = by_id.get(report.get("candidate"))
        if candidate is None:
            orphaned.append(str(path))
            continue
        requirement = "comfyui" if candidate["family"] == "image" \
            else str(report.get("engine") or "")
        found.setdefault((candidate["id"], requirement), []).append((path, report))
    candidate_records, blockers = [], list(discovery_errors or [])
    for candidate in candidates:
        screens = []
        for requirement in required_candidate_screens(candidate):
            matches = found.get((candidate["id"], requirement), [])
            if len(matches) != 1:
                errors = ["screen report is missing"] if not matches else [
                    "multiple screen reports match the same requirement",
                ]
            else:
                path, report = matches[0]
                errors = validate_screen_report(path, report, candidate)
            status = "passed" if not errors else "missing_or_invalid"
            screens.append({"requirement": requirement, "status": status, "errors": errors})
            if errors:
                blockers.append(f"{candidate['id']}/{requirement}: " + "; ".join(errors))
        candidate_records.append({
            "id": candidate["id"], "family": candidate["family"],
            "source_status": candidate["status"], "source_reasons": candidate.get("reasons", []),
            "screens": screens,
        })
    if orphaned:
        blockers.extend(f"orphaned screen report: {path}" for path in orphaned)
    source_review = [
        {"id": item["id"], "reasons": item.get("source_reasons", [])}
        for item in incumbent_audit["incumbents"] if item.get("source_status") != "source_ready"
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_for_decisions" if not blockers else "awaiting_evidence",
        "catalog_cost": incumbent_catalog_cost(incumbent_audit["incumbents"]),
        "incumbent_source_review": source_review,
        "candidates": candidate_records,
        "blockers": blockers,
    }


def main(argv=None) -> int:  # pragma: no cover - command entrypoint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-audit", type=Path)
    parser.add_argument("--incumbent-audit", type=Path, default=DEFAULT_INCUMBENT_AUDIT)
    parser.add_argument("--screen-root", type=Path, default=DEFAULT_SCREEN_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    candidate_audit = load_source_audit(args.candidate_audit) if args.candidate_audit \
        else load_source_audit()
    incumbent_audit = load_incumbent_audit(args.incumbent_audit)
    reports, discovery_errors = discover_screen_reports(args.screen_root)
    readiness = build_readiness(candidate_audit, incumbent_audit, reports, discovery_errors)
    rendered = json.dumps(readiness, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return int(args.check and readiness["status"] != "ready_for_decisions")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
