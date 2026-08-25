"""Representative smallest-model workload coverage for platform qualification."""

import json
from pathlib import Path

from scripts.workloads.models import EMBED_MODELS, IMAGE_MODELS, qualification_llm_model


SMALLEST_EMBEDDING_MODEL = EMBED_MODELS[0]["tag"]
SMALLEST_IMAGE_MODEL = IMAGE_MODELS[0]["short"]


RESULT_SECTIONS = {
    "llm": "llm", "conv": "llm_conversation", "emb": "embeddings",
    "mcq": "mcq", "math": "math", "reasoning": "reasoning", "code": "code",
    "tool": "tool", "conc_tool": "concurrency_tool", "conc_chat": "concurrency_chat",
    "sustained": "sustained", "llamabench": "llamabench",
    "llamabenchconc": "llamabenchconc", "vllmbench": "vllmbench", "img": "images",
}
EVIDENCE_MARKERS = {
    "llm": {"valid_runs"}, "conv": {"valid_runs", "generated_tokens"},
    "emb": {"valid_runs", "n_runs"},
    "mcq": {"answered"}, "math": {"answered"}, "reasoning": {"answered"},
    "code": {"answered"}, "tool": {"answered"},
    "conc_tool": {"valid_runs"}, "conc_chat": {"valid_runs"},
    "sustained": {"series"}, "llamabench": {"completed_cases"},
    "llamabenchconc": {"entries"},
    "vllmbench": {"latency_entries", "throughput_entries"},
    "img": {"valid_runs", "n_runs"},
}
DIAGNOSTIC_KEYS = {
    "error", "skip_reason", "invalid_runs", "timed_out", "timed_out_at", "crashed",
}


def qualification_workloads(engine: str) -> list[str]:
    shared = [
        "llm", "conv", "emb", "mcq", "math", "reasoning", "code", "tool",
        "conc_tool", "conc_chat", "sustained",
    ]
    if engine == "llamacpp":
        return [*shared, "llamabench", "llamabenchconc", "img"]
    if engine == "vllm":
        return [*shared, "vllmbench"]
    raise ValueError(f"unsupported qualification engine: {engine}")


def _has_evidence(value, markers: set[str]) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in markers and (
                    isinstance(child, (int, float)) and child > 0
                    or isinstance(child, list) and bool(child)):
                return True
            if _has_evidence(child, markers):
                return True
    elif isinstance(value, list):
        return any(_has_evidence(child, markers) for child in value)
    return False


def _incomplete_evidence(value, *, scoring_detail: bool = False) -> bool:
    if isinstance(value, dict):
        if not scoring_detail and any(
                value.get(key) for key in ("error", "timed_out", "crashed", "skipped")):
            return True
        pairs = (
            ("requested_cases", "completed_cases"),
            ("requested_runs", "valid_runs"),
            ("total", "answered"),
        )
        for requested_key, completed_key in pairs:
            if requested_key in value and completed_key in value \
                    and value[requested_key] != value[completed_key]:
                return True
        if "requested_runs" not in value and "n_runs" in value and "valid_runs" in value \
                and value["n_runs"] != value["valid_runs"]:
            return True
        return any(
            _incomplete_evidence(child, scoring_detail=scoring_detail or key == "incorrect")
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_incomplete_evidence(child, scoring_detail=scoring_detail) for child in value)
    return False


def workload_failure_details(result: dict, workloads: list[str], limit: int = 4000) -> list[str]:
    details = []

    def collect(value, path, *, scoring_detail=False):
        if isinstance(value, dict):
            for key in (() if scoring_detail else DIAGNOSTIC_KEYS):
                if key in value and value[key] not in (None, False, "", []):
                    rendered = json.dumps(value[key], ensure_ascii=False) \
                        if not isinstance(value[key], str) else value[key]
                    details.append(f"{path}.{key}: {rendered}")
            for requested_key, completed_key in (
                ("requested_cases", "completed_cases"),
                ("requested_runs", "valid_runs"),
                ("total", "answered"),
            ):
                if requested_key in value and completed_key in value \
                        and value[requested_key] != value[completed_key]:
                    details.append(
                        f"{path}: {completed_key}={value[completed_key]}, "
                        f"{requested_key}={value[requested_key]}"
                    )
            if "requested_runs" not in value and "n_runs" in value and "valid_runs" in value \
                    and value["n_runs"] != value["valid_runs"]:
                details.append(
                    f"{path}: valid_runs={value['valid_runs']}, n_runs={value['n_runs']}"
                )
            for key, child in value.items():
                collect(child, f"{path}.{key}", scoring_detail=scoring_detail or key == "incorrect")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                collect(child, f"{path}[{index}]", scoring_detail=scoring_detail)

    for workload in workloads:
        section = RESULT_SECTIONS[workload]
        collect(result.get(section), section)
    return [detail if len(detail) <= limit else detail[:limit] + "... [truncated]"
            for detail in dict.fromkeys(details)]


def workload_coverage_errors(result: dict, workloads: list[str]) -> list[str]:
    errors = []
    if result.get("run", {}).get("status") != "complete":
        errors.append("qualification result run is not complete")
    if not isinstance(result.get("engine_version"), str) or not result["engine_version"].strip():
        errors.append("qualification result does not identify the engine version")
    for workload in workloads:
        section = RESULT_SECTIONS[workload]
        value = result.get(section)
        if not isinstance(value, dict) or not _has_evidence(value, EVIDENCE_MARKERS[workload]):
            errors.append(f"{workload} produced no measured result evidence")
        elif _incomplete_evidence(value):
            errors.append(f"{workload} did not complete all requested qualification evidence")
    return errors


def qualification_arguments(engine: str, model: str, result: Path) -> list[str]:
    workloads = qualification_workloads(engine)
    command = [
        "--ui", "none", "--engine", engine,
        "--tests", *workloads, "--llm-models", model,
        "--embedding-models", SMALLEST_EMBEDDING_MODEL,
        "--runs", "1", "--warmup", "0", "--sample", "1",
        "--max-prompt-tokens", "2048", "--sustained-duration", "120",
        "--out", str(result),
    ]
    if engine == "llamacpp":
        command += ["--image-models", SMALLEST_IMAGE_MODEL]
    return command
