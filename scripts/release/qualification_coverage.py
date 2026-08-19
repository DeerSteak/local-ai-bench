"""Representative smallest-model workload coverage for platform qualification."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.workloads.models import EMBED_MODELS, IMAGE_MODELS, LLM_MODELS


SMALLEST_LLM_MODEL = LLM_MODELS[0]["tag"]
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
    "llamabenchconc": {"entries"}, "vllmbench": {"entries"}, "img": {"valid_runs", "n_runs"},
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


def workload_coverage_errors(result: dict, workloads: list[str]) -> list[str]:
    errors = []
    if result.get("run", {}).get("status") != "complete":
        errors.append("qualification result run is not complete")
    for workload in workloads:
        section = RESULT_SECTIONS[workload]
        value = result.get(section)
        if not isinstance(value, dict) or not _has_evidence(value, EVIDENCE_MARKERS[workload]):
            errors.append(f"{workload} produced no measured result evidence")
    return errors


def qualification_command(engine: str, model: str, result: Path,
                          comfyui: Path | None = None) -> list[str]:
    workloads = qualification_workloads(engine)
    command = [
        sys.executable, "-m", "scripts.app.benchmark", "--engine", engine,
        "--tests", *workloads, "--llm-models", model,
        "--embedding-models", SMALLEST_EMBEDDING_MODEL,
        "--runs", "1", "--warmup", "0", "--sample", "1",
        "--max-prompt-tokens", "2048", "--sustained-duration", "120",
        "--out", str(result),
    ]
    if engine == "llamacpp":
        if comfyui is None:
            raise ValueError("llama.cpp qualification requires its managed ComfyUI path")
        command += ["--image-models", SMALLEST_IMAGE_MODEL, "--comfyui", str(comfyui)]
    else:
        command.append("--ack-experimental-engine")
    return command


def run_qualification_coverage(engine: str, model: str, result: Path,
                               comfyui: Path | None = None) -> None:  # pragma: no cover
    result = Path(result)
    if result.is_file():
        existing = json.loads(result.read_text(encoding="utf-8"))
        if not workload_coverage_errors(existing, qualification_workloads(engine)):
            return
    completed = subprocess.run(qualification_command(engine, model, result, comfyui))
    if completed.returncode:
        raise RuntimeError(f"qualification workloads failed with code {completed.returncode}")
    data = json.loads(Path(result).read_text(encoding="utf-8"))
    errors = workload_coverage_errors(data, qualification_workloads(engine))
    if errors:
        raise ValueError("; ".join(errors))


def main(argv=None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run and verify qualification workloads")
    parser.add_argument("--engine", required=True, choices=("llamacpp", "vllm"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--comfyui", type=Path)
    args = parser.parse_args(argv)
    try:
        run_qualification_coverage(args.engine, args.model, args.result, args.comfyui)
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
