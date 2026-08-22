"""Metadata-only source audit for the Version 6 model candidates."""

import argparse
import json
from pathlib import Path

from scripts.setup.model_import import inspect_repository, preferred_variant


AUDIT_SCHEMA_VERSION = 1
DEFAULT_CANDIDATES = Path(__file__).with_name("model_catalog_candidates.json")


def load_candidate_register(path: Path = DEFAULT_CANDIDATES) -> list[dict]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("candidates"), list):
        raise ValueError("unsupported model-candidate register")
    candidates = value["candidates"]
    ids = [candidate.get("id") for candidate in candidates]
    if any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in ids):
        raise ValueError("every model candidate requires an id")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate model-candidate id")
    for candidate in candidates:
        if candidate.get("family") not in {"llm", "embedding", "image"}:
            raise ValueError(f"invalid candidate family: {candidate.get('id')}")
        sources = candidate.get("sources")
        if not isinstance(sources, dict) or not isinstance(sources.get("upstream"), str):
            raise ValueError(f"candidate requires an upstream source: {candidate.get('id')}")
        if candidate["family"] != "image" and not isinstance(sources.get("gguf"), str):
            raise ValueError(f"candidate requires a GGUF source: {candidate.get('id')}")
    return candidates


def _license(info) -> str | None:
    card_data = getattr(info, "card_data", None)
    value = getattr(card_data, "license", None)
    return value if isinstance(value, str) and value else None


def _base_models(info) -> list[str]:
    card_data = getattr(info, "card_data", None)
    value = getattr(card_data, "base_model", None)
    if isinstance(value, str):
        return [value]
    return sorted(item for item in (value or []) if isinstance(item, str))


def _files(info) -> list[dict]:
    records = []
    for item in getattr(info, "siblings", None) or []:
        name = getattr(item, "rfilename", None)
        if not isinstance(name, str):
            continue
        size = getattr(item, "size", None)
        lfs = getattr(item, "lfs", None)
        if not isinstance(size, int):
            size = getattr(lfs, "size", None)
        records.append({"name": name, "size": size if isinstance(size, int) else None})
    return sorted(records, key=lambda record: record["name"])


def audit_repository(repo: str, role: str, *, api, inspect_fn=inspect_repository) -> dict:
    info = api.model_info(repo, revision="main", files_metadata=True)
    revision = str(getattr(info, "sha", None) or "main")
    inspection = inspect_fn(repo, revision=revision, api=api)
    record = {
        "repo": repo,
        "revision": revision,
        "gated": getattr(info, "gated", False) or False,
        "private": bool(getattr(info, "private", False)),
        "license": _license(info),
        "base_models": _base_models(info),
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "library_name": getattr(info, "library_name", None),
    }
    if role == "upstream":
        variant = inspection.vllm_variant
        if variant:
            record["artifact"] = {
                "kind": "safetensors",
                "files": list(variant.files),
                "support_files": list(variant.support_files),
                "size": variant.size,
            }
        elif any(file["name"].endswith(".safetensors") for file in _files(info)):
            record["artifact"] = {
                "kind": "pipeline",
                "files": [file for file in _files(info) if file["name"].endswith(".safetensors")],
            }
        else:
            record["artifact"] = None
    else:
        variant = preferred_variant(inspection.llama_variants)
        record["artifact"] = ({
            "kind": "gguf",
            "label": variant.label,
            "files": list(variant.files),
            "size": variant.size,
        } if variant else None)
    return record


def source_status(candidate: dict, sources: dict) -> tuple[str, list[str]]:
    reasons = []
    upstream = sources["upstream"]
    if upstream["private"]:
        reasons.append("upstream repository is private")
    if upstream["gated"]:
        reasons.append("upstream repository requires access approval")
    if not upstream["license"]:
        reasons.append("upstream license is not declared")
    elif upstream["license"] != "apache-2.0":
        reasons.append(f"upstream {upstream['license']} license requires review")
    if upstream["artifact"] is None:
        reasons.append("upstream artifact could not be resolved")
    if candidate["family"] != "image":
        gguf = sources["gguf"]
        if gguf["private"] or gguf["gated"]:
            reasons.append("GGUF repository is not publicly accessible")
        if not gguf["license"]:
            reasons.append("GGUF license is not declared")
        elif upstream["license"] and gguf["license"] != upstream["license"]:
            reasons.append("GGUF and upstream licenses do not match")
        if upstream["repo"] not in gguf["base_models"]:
            reasons.append("GGUF provenance does not identify the selected upstream repository")
        if gguf["artifact"] is None:
            reasons.append("GGUF artifact could not be resolved")
    elif upstream["artifact"] is not None:
        reasons.append("complete ComfyUI pipeline artifact selection remains pending")
    return ("source_ready" if not reasons else "blocked", reasons)


def build_source_audit(candidates: list[dict], *, api,
                       inspect_fn=inspect_repository) -> dict:
    audited = []
    for candidate in candidates:
        sources = {
            role: audit_repository(repo, role, api=api, inspect_fn=inspect_fn)
            for role, repo in candidate["sources"].items()
        }
        status, reasons = source_status(candidate, sources)
        audited.append({**candidate, "sources": sources, "status": status, "reasons": reasons})
    return {"schema_version": AUDIT_SCHEMA_VERSION, "candidates": audited}


def main(argv=None) -> int:  # pragma: no cover - network CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    from huggingface_hub import HfApi

    audit = build_source_audit(load_candidate_register(args.candidates), api=HfApi())
    rendered = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 1 if any(item["status"] != "source_ready" for item in audit["candidates"]) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
