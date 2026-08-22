"""Metadata-only source audit for every model currently shipped by the catalog."""

import argparse
import json
from pathlib import Path

from scripts.release.model_catalog_audit import (
    _base_models, _configuration_metadata, _files, _license, _default_json_reader,
)
from scripts.release.model_catalog_inventory import (
    build_incumbent_inventory, load_incumbent_register,
)


AUDIT_SCHEMA_VERSION = 1


def repository_record(repo: str, *, api, read_json=_default_json_reader) -> dict:
    info = api.model_info(repo, revision="main", files_metadata=True)
    revision = str(getattr(info, "sha", None) or "main")
    files = _files(info)
    try:
        configuration = _configuration_metadata(
            repo, revision, {record["name"] for record in files}, read_json,
        )
        configuration_error = None
    except Exception as exc:
        configuration = None
        configuration_error = type(exc).__name__
    return {
        "repo": repo, "revision": revision,
        "private": bool(getattr(info, "private", False)),
        "gated": getattr(info, "gated", False) or False,
        "license": _license(info), "base_models": _base_models(info),
        "configuration": configuration, "configuration_error": configuration_error,
        "files": files,
    }


def selected_file_records(repository: dict, names: list[str]) -> list[dict]:
    available = {record["name"]: record for record in repository["files"]}
    return [available.get(name, {"name": name, "size": None, "sha256": None}) for name in names]


def snapshot_artifact(repository: dict) -> dict:
    weights = [
        record for record in repository["files"]
        if record["name"].endswith((".safetensors", ".bin"))
    ]
    return {
        "weight_file_count": len(weights),
        "weight_size": sum(record["size"] for record in weights
                           if isinstance(record["size"], int)),
        "has_config": any(record["name"] == "config.json" for record in repository["files"]),
    }


def incumbent_source_status(record: dict) -> tuple[str, list[str]]:
    reasons = []
    upstream = record["sources"]["upstream"]
    if upstream["private"]:
        reasons.append("upstream repository is private")
    if upstream["gated"]:
        reasons.append("upstream repository requires access approval")
    if not upstream["license"]:
        reasons.append("upstream license is not declared")
    elif upstream["license"] != "apache-2.0":
        reasons.append(f"upstream {upstream['license']} license requires review")
    if upstream["configuration"] is None:
        reasons.append("upstream configuration could not be inspected")
    for role, source in record["sources"].items():
        if role == "upstream":
            continue
        if source["private"] or source["gated"]:
            reasons.append(f"selected {role} repository is not publicly accessible")
        if not source["license"]:
            reasons.append(f"selected {role} license is not declared")
        if (role in {"llamacpp", "vllm"} and source["repo"] != upstream["repo"]
                and upstream["repo"] not in source["base_models"]):
            reasons.append(f"selected {role} provenance does not identify the upstream repository")
        artifact = source["artifact"]
        if "files" in artifact and any(
                file["size"] is None for file in artifact["files"]):
            reasons.append(f"selected {role} artifact is unresolved")
        if "weight_file_count" in artifact and (
                artifact["weight_file_count"] < 1 or not artifact["has_config"]):
            reasons.append(f"selected {role} snapshot is incomplete")
    return ("source_ready" if not reasons else "review_required", reasons)


def build_incumbent_source_audit(inventory: dict, *, api,
                                 read_json=_default_json_reader) -> dict:
    cache = {}

    def repository(repo):
        if repo not in cache:
            cache[repo] = repository_record(repo, api=api, read_json=read_json)
        return cache[repo]

    audited = []
    for incumbent in inventory["incumbents"]:
        upstream = repository(incumbent["upstream"])
        sources = {"upstream": {key: value for key, value in upstream.items() if key != "files"}}
        for role, selected in incumbent["selected_artifacts"].items():
            selected_repo = repository(selected["repo"])
            artifact = (
                {"files": selected_file_records(selected_repo, selected["files"])}
                if "files" in selected else snapshot_artifact(selected_repo)
            )
            sources[role] = {
                key: selected_repo[key]
                for key in ("repo", "revision", "private", "gated", "license", "base_models")
            } | {"artifact": artifact}
        record = {**incumbent, "sources": sources}
        status, reasons = incumbent_source_status(record)
        audited.append({**record, "source_status": status, "source_reasons": reasons})
    return {"schema_version": AUDIT_SCHEMA_VERSION, "incumbents": audited}


def main(argv=None) -> int:  # pragma: no cover - network command
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    from huggingface_hub import HfApi

    register = load_incumbent_register(args.register) if args.register else load_incumbent_register()
    audit = build_incumbent_source_audit(build_incumbent_inventory(register), api=HfApi())
    rendered = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 1 if any(record["source_status"] != "source_ready"
                    for record in audit["incumbents"]) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
