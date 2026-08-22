"""Metadata-only source audit for the Version 6 model candidates."""

import argparse
import json
from pathlib import Path
import re

from scripts.setup.model_import import inspect_repository, preferred_variant


AUDIT_SCHEMA_VERSION = 2
CANDIDATE_SCHEMA_VERSION = 2
VLLM_4BIT_METHODS = {"awq", "bitsandbytes", "compressed-tensors", "gptq"}
DEFAULT_CANDIDATES = Path(__file__).with_name("model_catalog_candidates.json")


def load_candidate_register(path: Path = DEFAULT_CANDIDATES) -> list[dict]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != CANDIDATE_SCHEMA_VERSION \
            or not isinstance(value.get("candidates"), list):
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
        if not isinstance(candidate.get("role"), str) or not candidate["role"]:
            raise ValueError(f"candidate requires a measurable role: {candidate.get('id')}")
        incumbents = candidate.get("incumbents")
        if not isinstance(incumbents, list) or not incumbents \
                or any(not isinstance(item, str) or not item for item in incumbents) \
                or len(incumbents) != len(set(incumbents)):
            raise ValueError(f"candidate requires unique incumbent comparisons: {candidate.get('id')}")
        sources = candidate.get("sources")
        if not isinstance(sources, dict) or not isinstance(sources.get("upstream"), str):
            raise ValueError(f"candidate requires an upstream source: {candidate.get('id')}")
        if candidate["family"] != "image" and not isinstance(sources.get("gguf"), str):
            raise ValueError(f"candidate requires a GGUF source: {candidate.get('id')}")
        if candidate["family"] == "llm" and not isinstance(sources.get("vllm"), str):
            raise ValueError(f"LLM candidate requires a vLLM source: {candidate.get('id')}")
        if candidate.get("gguf_provenance") not in {None, "publisher_exact_variant"}:
            raise ValueError(f"invalid GGUF provenance: {candidate.get('id')}")
        pipeline = sources.get("pipeline")
        if pipeline is not None and (
                not isinstance(pipeline, list) or not pipeline
                or any(not isinstance(source, dict)
                       or not isinstance(source.get("repo"), str)
                       or not isinstance(source.get("files"), list)
                       or not source["files"]
                       or any(not isinstance(name, str) or not name for name in source["files"])
                       for source in pipeline)):
            raise ValueError(f"invalid pipeline source: {candidate.get('id')}")
        if candidate["family"] == "image" and not pipeline:
            raise ValueError(f"image candidate requires pipeline sources: {candidate.get('id')}")
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
        sha256 = getattr(lfs, "sha256", None)
        records.append({
            "name": name,
            "size": size if isinstance(size, int) else None,
            "sha256": sha256 if isinstance(sha256, str) else None,
        })
    return sorted(records, key=lambda record: record["name"])


def _default_json_reader(repo: str, revision: str, filename: str) -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo, revision=revision, filename=filename)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_object(value) -> dict:
    return value if isinstance(value, dict) else {}


def _is_q4_gguf_artifact(artifact: dict) -> bool:
    return re.search(r"(?:^|[-_.])Q4(?:[-_.]|$)", str(artifact.get("label", "")).upper()) \
        is not None


def _quantization_metadata(config: dict) -> dict | None:
    quantization = _json_object(config.get("quantization_config"))
    if not quantization:
        return None
    method = quantization.get("quant_method") or quantization.get("method")
    bits = quantization.get("bits")
    if bits is None and quantization.get("load_in_4bit") is True:
        bits = 4
    groups = _json_object(quantization.get("config_groups"))
    group_bits = {
        _json_object(_json_object(group).get("weights")).get("num_bits")
        for group in groups.values()
    }
    group_bits.discard(None)
    if bits is None and len(group_bits) == 1:
        bits = group_bits.pop()
    return {
        "method": method,
        "bits": bits,
        "format": quantization.get("format") or quantization.get("checkpoint_format"),
    }


def _configuration_metadata(repo: str, revision: str, files: set[str], read_json) -> dict:
    config = _json_object(
        read_json(repo, revision, "config.json") if "config.json" in files else {}
    )
    text_config = _json_object(config.get("text_config"))
    tokenizer = _json_object(
        read_json(repo, revision, "tokenizer_config.json")
        if "tokenizer_config.json" in files else {}
    )
    generation = _json_object(
        read_json(repo, revision, "generation_config.json")
        if "generation_config.json" in files else {}
    )
    pipeline = _json_object(
        read_json(repo, revision, "model_index.json")
        if "model_index.json" in files else {}
    )
    sampling_keys = {
        "temperature", "top_k", "top_p", "min_p", "repetition_penalty",
        "presence_penalty", "frequency_penalty", "do_sample",
    }
    return {
        "model_type": config.get("model_type") or text_config.get("model_type"),
        "architectures": config.get("architectures") or text_config.get("architectures") or [],
        "context_tokens": (
            config.get("max_position_embeddings")
            or text_config.get("max_position_embeddings")
        ),
        "dtype": config.get("dtype") or config.get("torch_dtype")
        or text_config.get("dtype") or text_config.get("torch_dtype"),
        "hidden_size": config.get("hidden_size") or text_config.get("hidden_size"),
        "num_hidden_layers": (
            config.get("num_hidden_layers") or text_config.get("num_hidden_layers")
        ),
        "num_experts": config.get("num_experts") or text_config.get("num_experts"),
        "num_experts_per_token": (
            config.get("num_experts_per_tok") or text_config.get("num_experts_per_tok")
            or config.get("num_selected_experts") or text_config.get("num_selected_experts")
        ),
        "chat_template": (
            "chat_template.jinja" if "chat_template.jinja" in files
            else "tokenizer_config.json" if tokenizer.get("chat_template") else None
        ),
        "publisher_sampling": {
            key: generation[key] for key in sorted(sampling_keys) if key in generation
        },
        "quantization": _quantization_metadata(config),
        "pipeline_class": pipeline.get("_class_name"),
    }


def audit_repository(repo: str, role: str, *, api, inspect_fn=inspect_repository,
                     read_json=_default_json_reader) -> dict:
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
        "custom_code": "custom_code" in (getattr(info, "tags", None) or []),
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
    }
    file_records = _files(info)
    try:
        record["configuration"] = _configuration_metadata(
            repo, revision, {file["name"] for file in file_records}, read_json,
        )
    except Exception as exc:
        record["configuration"] = None
        record["configuration_error"] = type(exc).__name__
    if role in {"upstream", "vllm"}:
        variant = inspection.vllm_variant
        if variant:
            record["artifact"] = {
                "kind": "safetensors",
                "files": list(variant.files),
                "support_files": list(variant.support_files),
                "size": variant.size,
            }
        elif any(file["name"].endswith(".safetensors") for file in file_records):
            record["artifact"] = {
                "kind": "pipeline",
                "files": [file for file in file_records if file["name"].endswith(".safetensors")],
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


def audit_pipeline_source(source: dict, *, api) -> dict:
    repo = source["repo"]
    info = api.model_info(repo, revision="main", files_metadata=True)
    available = {record["name"]: record for record in _files(info)}
    files = [
        available.get(name, {"name": name, "size": None, "sha256": None})
        for name in source["files"]
    ]
    return {
        "repo": repo,
        "revision": str(getattr(info, "sha", None) or "main"),
        "gated": getattr(info, "gated", False) or False,
        "private": bool(getattr(info, "private", False)),
        "license": _license(info),
        "files": files,
    }


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
    if upstream.get("configuration") is None:
        reasons.append("upstream configuration could not be inspected")
    elif candidate["family"] == "llm" and not upstream["configuration"]["chat_template"]:
        reasons.append("upstream LLM has no chat template")
    if upstream.get("custom_code"):
        reasons.append("upstream requires custom code review")
    if candidate["family"] != "image":
        gguf = sources["gguf"]
        if gguf["private"] or gguf["gated"]:
            reasons.append("GGUF repository is not publicly accessible")
        if not gguf["license"]:
            reasons.append("GGUF license is not declared")
        elif upstream["license"] and gguf["license"] != upstream["license"]:
            reasons.append("GGUF and upstream licenses do not match")
        publisher_exact = (
            candidate.get("gguf_provenance") == "publisher_exact_variant"
            and gguf["repo"] == f"{upstream['repo']}-GGUF"
            and gguf["repo"].split("/", 1)[0] == upstream["repo"].split("/", 1)[0]
        )
        if upstream["repo"] not in gguf["base_models"] and not publisher_exact:
            reasons.append("GGUF provenance does not identify the selected upstream repository")
        if gguf["artifact"] is None:
            reasons.append("GGUF artifact could not be resolved")
        elif candidate["family"] == "llm" and not _is_q4_gguf_artifact(gguf["artifact"]):
            reasons.append("GGUF artifact is not a 4-bit Q4 variant")
        if candidate["family"] == "llm":
            vllm = sources["vllm"]
            if vllm["private"] or vllm["gated"]:
                reasons.append("vLLM repository is not publicly accessible")
            if not vllm["license"]:
                reasons.append("vLLM license is not declared")
            elif upstream["license"] and vllm["license"] != upstream["license"]:
                reasons.append("vLLM and upstream licenses do not match")
            if upstream["repo"] not in vllm["base_models"]:
                reasons.append("vLLM provenance does not identify the selected upstream repository")
            if vllm["artifact"] is None:
                reasons.append("vLLM artifact could not be resolved")
            configuration = vllm.get("configuration")
            quantization = configuration.get("quantization") if configuration else None
            if not quantization or quantization.get("method") not in VLLM_4BIT_METHODS \
                    or quantization.get("bits") != 4:
                reasons.append("vLLM artifact is not a supported 4-bit quantization")
    else:
        pipeline = sources.get("pipeline") or []
        if not pipeline:
            reasons.append("complete ComfyUI pipeline artifact selection remains pending")
        for dependency in pipeline:
            if dependency["private"] or dependency["gated"]:
                reasons.append(f"pipeline repository is not public: {dependency['repo']}")
            if not dependency["license"]:
                reasons.append(f"pipeline license is not declared: {dependency['repo']}")
            elif dependency["license"] != "apache-2.0":
                reasons.append(
                    f"pipeline {dependency['license']} license requires review: {dependency['repo']}"
                )
            if any(file["size"] is None or file["sha256"] is None for file in dependency["files"]):
                reasons.append(f"pipeline artifact is unresolved: {dependency['repo']}")
    return ("source_ready" if not reasons else "blocked", reasons)


def build_source_audit(candidates: list[dict], *, api,
                       inspect_fn=inspect_repository, read_json=_default_json_reader) -> dict:
    audited = []
    for candidate in candidates:
        sources: dict = {
            role: audit_repository(
                repo, role, api=api, inspect_fn=inspect_fn, read_json=read_json,
            )
            for role, repo in candidate["sources"].items() if role != "pipeline"
        }
        if candidate["sources"].get("pipeline"):
            sources["pipeline"] = [
                audit_pipeline_source(source, api=api)
                for source in candidate["sources"]["pipeline"]
            ]
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
