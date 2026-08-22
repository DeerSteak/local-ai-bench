"""Unattended import, interrupt, resume, and evidence gate for catalog candidates."""

import argparse
import json
import platform
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from scripts.app.benchmark_gui_process import launch_controlled_process
from scripts.results.result_store import atomic_write_json
from scripts.results.canonical_json import sha256_json
from scripts.results.llm_event_stage import event_store_path
from scripts.results.local_execution_context import images_dir_for_result
from scripts.runtime import config
from scripts.runtime.log_redaction import redact_log_text
from scripts.runtime.shared import Shared
from scripts.runtime.pause_control import write_pause_state
from scripts.runtime.sampling import baseline_sampling_profile, publisher_sampling_profile
from scripts.setup.custom_models import custom_model
from scripts.setup.model_download import (
    custom_model_artifacts_present, download_hf_files, import_model, load_hf_token,
)
from scripts.setup.model_import import ImportVariant, inspect_repository
from scripts.setup.runtime_identity import repository_revision
from scripts.workloads.llm_conversation_benchmark import LLMConversationBenchmark


DEFAULT_AUDIT = config.SCRIPT_DIR / "docs" / "model-catalog-source-audit-v6.json"
DEFAULT_OUTPUT_ROOT = config.RESULTS_DIR / "catalog-audit"
SCREEN_SCHEMA_VERSION = 1
MIN_SCREEN_GENERATED_TOKENS = LLMConversationBenchmark.CONV_STEP_MIN


@dataclass(frozen=True)
class ScreenSpec:
    candidate_id: str
    candidate_name: str
    engine: str
    tag: str
    family: str
    repo: str
    revision: str
    files: tuple[str, ...]
    context_tokens: int | None
    output_path: Path
    command: tuple[str, ...]
    sampling_profile: dict
    pipeline_sources: tuple[dict, ...] = ()
    image_model: dict | None = None


def load_source_audit(path: Path = DEFAULT_AUDIT) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 2 or not isinstance(value.get("candidates"), list):
        raise ValueError("unsupported model source audit")
    return value


def candidate_record(audit: dict, candidate_id: str) -> dict:
    matches = [item for item in audit["candidates"] if item.get("id") == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"unknown model candidate: {candidate_id}")
    return matches[0]


def build_screen_spec(candidate: dict, engine: str, output_root: Path,
                      *, python_executable=sys.executable,
                      publisher_sampling: bool = False) -> ScreenSpec:
    if candidate.get("status") != "source_ready":
        detail = "; ".join(candidate.get("reasons") or []) or "source audit is incomplete"
        raise ValueError(f"candidate is not source-ready: {detail}")
    family = candidate["family"]
    if engine not in {"llamacpp", "vllm"}:
        raise ValueError("screen engine must be llamacpp or vllm")
    raw_pipeline = candidate["sources"].get("pipeline") or []
    pipeline_sources: tuple[dict, ...] = tuple(
        dependency for dependency in raw_pipeline if isinstance(dependency, dict)
    )
    if family == "image":
        source = candidate["sources"]["upstream"]
        files = tuple(
            file["name"] for dependency in pipeline_sources for file in dependency["files"]
        )
        if candidate["id"] != "z-image-turbo" or not files:
            raise ValueError("image candidate needs a supported fixed ComfyUI workflow")
    else:
        role = "gguf" if engine == "llamacpp" else (
            "vllm" if family == "llm" else "upstream"
        )
        source = candidate["sources"].get(role)
        artifact = source.get("artifact") if isinstance(source, dict) else None
        if not isinstance(artifact, dict) or not artifact.get("files"):
            raise ValueError(f"candidate has no {engine} artifact")
        files = tuple(artifact["files"])
    tag = candidate["id"] if family == "image" else f"audit-{candidate['id']}"
    revision = source["revision"]
    output_dir = Path(output_root) / candidate["id"] / engine / revision[:12]
    if publisher_sampling:
        output_dir /= "publisher"
    output_path = output_dir / "result.json"
    configuration = candidate["sources"]["upstream"].get("configuration") or {}
    context_tokens = configuration.get("context_tokens")
    if publisher_sampling:
        sampling = publisher_sampling_profile(
            engine, name=candidate["id"], repo=candidate["sources"]["upstream"]["repo"],
            revision=candidate["sources"]["upstream"]["revision"],
            controls=configuration.get("publisher_sampling") or {},
        )
    else:
        sampling = baseline_sampling_profile(engine)
    image_model = None
    if family == "llm":
        context_cap = min(int(context_tokens or 32768), 32768)
        command = [
            python_executable, "-m", "scripts.app.benchmark",
            "--engine", engine, "--tests", "llm", "conv",
            "--llm-models", tag, "--runs", "1", "--warmup", "1",
            "--max-prompt-tokens", str(context_cap), "--force-all",
            "--out", str(output_path),
        ]
    elif family == "embedding":
        command = [
            python_executable, "-m", "scripts.app.benchmark",
            "--engine", engine, "--tests", "emb", "--embedding-models", tag,
            "--runs", "1", "--warmup", "1", "--out", str(output_path),
        ]
    else:
        image_model = {
            "audit_candidate": True,
            "artifact_digest": sha256_json(list(pipeline_sources)),
            "label": candidate["name"], "short": candidate["id"], "tier": "medium",
            "checkpoint": "z_image_turbo_bf16.safetensors",
            "checkpoint_folder": "diffusion_models", "workflow": "z_image",
            "steps": 8, "cfg": 1.0, "sampler": "res_multistep", "scheduler": "simple",
        }
        command = [
            python_executable, "-m", "scripts.app.benchmark",
            "--engine", engine, "--tests", "img", "--image-models", candidate["id"],
            "--audit-image-model", str(output_path.with_name("image-model.json")),
            "--runs", "1", "--warmup", "1", "--out", str(output_path),
        ]
    if publisher_sampling:
        command.extend((
            "--publisher-sampling-profile",
            str(output_path.with_name("publisher-sampling.json")),
        ))
    return ScreenSpec(
        candidate["id"], candidate["name"], engine, tag, family,
        source["repo"], revision, files,
        context_tokens if isinstance(context_tokens, int) else None,
        output_path, tuple(command), sampling, pipeline_sources, image_model,
    )


def select_exact_variant(inspection, engine: str, files: tuple[str, ...]) -> ImportVariant:
    variants = inspection.llama_variants if engine == "llamacpp" else (
        (inspection.vllm_variant,) if inspection.vllm_variant else ()
    )
    match = next((variant for variant in variants if variant.files == files), None)
    if match is None:
        raise ValueError("pinned candidate artifact no longer matches repository metadata")
    return match


def candidate_import_matches(existing: dict, spec: ScreenSpec) -> bool:
    if existing.get("repo") != spec.repo or existing.get("revision") != spec.revision:
        return False
    if spec.engine == "vllm":
        return existing.get("format") == "safetensors"
    expected = tuple(Path(name).name for name in spec.files)
    return existing.get("format") == "gguf" and tuple(existing.get("files") or ()) == expected


def pipeline_asset_target(file_name: str, models_root: Path) -> Path:
    parts = Path(file_name).parts
    if len(parts) != 3 or parts[0] != "split_files" \
            or parts[1] not in {"diffusion_models", "text_encoders", "vae"}:
        raise ValueError("candidate pipeline artifact has an invalid path")
    return Path(models_root) / parts[1] / parts[2]


def interrupt_ready(result: dict, spec: ScreenSpec) -> bool:
    section_name = "llm" if spec.family == "llm" else "embeddings"
    if spec.family == "image":
        model = result.get("images", {}).get(spec.tag, {})
        return any(
            isinstance(case, dict) and case.get("n_runs", 0) > 0
            for case in model.get("resolutions", {}).values()
        ) if isinstance(model, dict) else False
    model = result.get(section_name, {}).get(spec.tag, {})
    if spec.family == "embedding":
        return isinstance(model, dict) and model.get("valid_runs", 0) > 0
    return any(
        isinstance(case, dict) and case.get("valid_runs", 0) > 0
        for case in model.values()
    ) if isinstance(model, dict) else False


def compatibility_screen_errors(result: dict, spec: ScreenSpec) -> list[str]:
    errors = []
    run = result.get("run", {})
    if run.get("status") != "complete":
        errors.append("run did not complete")
    expected_stages = (
        ("llm", "conv") if spec.family == "llm"
        else ("emb",) if spec.family == "embedding" else ("img",)
    )
    for stage in expected_stages:
        if run.get("stages", {}).get(stage, {}).get("status") != "complete":
            errors.append(f"{stage} stage did not complete")
    if not any(item.get("status") == "interrupted" for item in run.get("recovery_history", [])):
        errors.append("interrupt/resume evidence is missing")
    settings = run.get("plan", {}).get("effective_config", {})
    expected_methodology = (
        "publisher-v1" if spec.sampling_profile["profile"].startswith(
            "publisher-recommended-v1:") else "neutral-v2"
    )
    if settings.get("methodology_profile") != expected_methodology:
        errors.append(f"{expected_methodology} methodology is missing")
    if spec.family == "llm" and settings.get("sampling_profile") != spec.sampling_profile:
        errors.append("resolved sampler identity is missing or incorrect")
    preflight = result.get("preflight", {}).get("models", {}).get(spec.tag, {})
    checks = {check.get("name"): check for check in preflight.get("checks", [])}
    if preflight.get("status") == "excluded":
        errors.append("runtime preflight excluded the model")
    if spec.family == "llm" and checks.get("formatting_probe", {}).get("status") != "passed":
        errors.append("chat formatting probe did not pass")
    if spec.family == "embedding":
        model = result.get("embeddings", {}).get(spec.tag, {})
        if model.get("valid_runs", 0) < 1:
            errors.append("embedding measurement is missing")
        return errors
    if spec.family == "image":
        model = result.get("images", {}).get(spec.tag, {})
        resolutions = model.get("resolutions", {}) if isinstance(model, dict) else {}
        if not resolutions or any(case.get("n_runs", 0) < 1 for case in resolutions.values()):
            errors.append("image measurement evidence is missing")
        return errors
    deepest = min(spec.context_tokens or 32768, 32768)
    labels = ("2K", f"{deepest / 1024:g}K")
    sections = (("llm", "single-shot"), ("llm_conversation", "conversation"))
    for section_name, label in sections:
        model = result.get(section_name, {}).get(spec.tag, {})
        for context in dict.fromkeys(labels):
            if not isinstance(model.get(context), dict) or model[context].get("valid_runs", 0) < 1:
                errors.append(f"{label} {context} evidence is missing")
        for context, case in model.items():
            if not isinstance(case, dict) or case.get("valid_runs", 0) < 1:
                errors.append(f"{label} {context} has no valid measurement")
                continue
            samples = case.get("valid_samples") or []
            if not any(
                isinstance(sample, dict)
                and isinstance(sample.get("generated_tokens"), int)
                and not isinstance(sample["generated_tokens"], bool)
                and sample["generated_tokens"] >= MIN_SCREEN_GENERATED_TOKENS
                for sample in samples
            ):
                errors.append(
                    f"{label} {context} generated fewer than "
                    f"{MIN_SCREEN_GENERATED_TOKENS} measurable tokens"
                )
    return errors


def screen_image_artifacts(result: dict, spec: ScreenSpec) -> tuple[list[dict], list[str]]:
    if spec.family != "image":
        return [], []
    model = result.get("images", {}).get(spec.tag, {})
    resolutions = model.get("resolutions", {}) if isinstance(model, dict) else {}
    records, errors = [], []
    for resolution in sorted(resolutions):
        path = images_dir_for_result(spec.output_path) / f"{spec.tag}_{resolution}.png"
        if not path.is_file() or path.stat().st_size < 1:
            errors.append(f"generated image is missing: {resolution}")
            continue
        records.append({
            "resolution": resolution,
            "path": str(path.relative_to(spec.output_path.parent.resolve())),
            "size": path.stat().st_size, "sha256": Shared.file_sha256(path),
        })
    return records, errors


def screen_evidence_artifacts(spec: ScreenSpec) -> tuple[list[dict], list[str]]:
    paths = (
        spec.output_path,
        event_store_path(spec.output_path),
        spec.output_path.with_name("initial.log"),
        spec.output_path.with_name("resume.log"),
    )
    records, errors = [], []
    for path in paths:
        if not path.is_file() or path.stat().st_size < 1:
            errors.append(f"screen evidence is missing: {path.name}")
            continue
        records.append({
            "path": str(path.relative_to(spec.output_path.parent)),
            "size": path.stat().st_size,
            "sha256": Shared.file_sha256(path),
        })
    return records, errors


def ensure_candidate_import(spec: ScreenSpec) -> str:  # pragma: no cover - real downloads
    if spec.family == "image":
        token = load_hf_token()
        downloaded = False
        for dependency in spec.pipeline_sources:
            for file in dependency["files"]:
                target = pipeline_asset_target(file["name"], config.COMFYUI_MODELS_DIR)
                destination = target.parent
                expected = file.get("sha256")
                if target.is_file():
                    if not expected or Shared.file_sha256(target) != expected:
                        raise ValueError(f"candidate pipeline digest mismatch: {target.name}")
                    continue
                if not download_hf_files(
                        dependency["repo"], file["name"], destination,
                        revision=dependency["revision"], token=token, save_as=target.name):
                    raise RuntimeError(f"candidate pipeline download failed: {target.name}")
                if not expected or Shared.file_sha256(target) != expected:
                    raise ValueError(f"candidate pipeline digest mismatch: {target.name}")
                downloaded = True
        return "downloaded" if downloaded else "reused"
    existing = custom_model(spec.engine, spec.tag)
    vllm_cache = None
    if spec.engine == "vllm":
        from scripts.runtime.engines.vllm import VllmEngine
        vllm_cache = VllmEngine().cache_home()
    if existing is not None:
        if (not candidate_import_matches(existing, spec)
                or not custom_model_artifacts_present(existing, vllm_cache=vllm_cache)):
            raise ValueError("existing audit tag does not match the pinned candidate artifact")
        return "reused"
    token = load_hf_token()
    inspection = inspect_repository(spec.repo, revision=spec.revision, token=token)
    if inspection.revision != spec.revision:
        raise ValueError("repository revision changed during candidate import")
    variant = select_exact_variant(inspection, spec.engine, spec.files)
    import_model(
        inspection=inspection, engine=spec.engine, variant=variant,
        tag=spec.tag, label=f"{spec.candidate_name} audit", vllm_cache=vllm_cache,
        token=token,
    )
    return "downloaded"


def _read_result(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _relay_output(process, log_path: Path) -> threading.Thread:
    def relay():
        with log_path.open("w", encoding="utf-8") as log:
            for line in process.stdout or ():
                output = f"{redact_log_text(line.rstrip())}\n"
                sys.stdout.write(output)
                log.write(output)
    thread = threading.Thread(target=relay, daemon=True)
    thread.start()
    return thread


def _interrupt_signal() -> signal.Signals | int:
    if platform.system() == "Windows":
        return getattr(signal, "CTRL_BREAK_EVENT", signal.SIGINT)
    return signal.SIGINT


def _wait_after_interrupt(process) -> None:  # pragma: no cover - real process tree
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired as exc:
        from scripts.runtime.process_tree import stop_process_tree
        stop_process_tree(process, interrupt=False)
        raise RuntimeError("benchmark did not stop within 120 seconds") from exc


def _run_new_screen(spec: ScreenSpec, timeout: int) -> None:  # pragma: no cover - real benchmark
    creationflags = (
        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        if platform.system() == "Windows" else 0
    )
    process, pause_path = launch_controlled_process(
        list(spec.command), creationflags=creationflags,
    )
    reader = _relay_output(process, spec.output_path.with_name("initial.log"))
    deadline = time.monotonic() + timeout
    interrupted = False
    while process.poll() is None and time.monotonic() < deadline:
        result = _read_result(spec.output_path)
        if result is not None and interrupt_ready(result, spec):
            write_pause_state(pause_path, "paused")
            time.sleep(0.5)
            process.send_signal(_interrupt_signal())
            interrupted = True
            break
        time.sleep(0.25)
    if not interrupted:
        if process.poll() is None:
            process.send_signal(_interrupt_signal())
        _wait_after_interrupt(process)
        reader.join(timeout=5)
        raise RuntimeError("screen could not interrupt after a valid committed case")
    _wait_after_interrupt(process)
    reader.join(timeout=5)
    stopped = _read_result(spec.output_path)
    if stopped is None or stopped.get("run", {}).get("status") != "interrupted":
        raise RuntimeError("initial screen did not checkpoint an interrupted result")


def _resume_screen(spec: ScreenSpec) -> int:  # pragma: no cover - real benchmark
    command = [
        sys.executable, "-m", "scripts.results.recovery_executor", str(spec.output_path),
    ]
    with spec.output_path.with_name("resume.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=config.SCRIPT_DIR, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        )
        for line in process.stdout or ():
            output = f"{redact_log_text(line.rstrip())}\n"
            sys.stdout.write(output)
            log.write(output)
        return process.wait()


def execute_screen(spec: ScreenSpec, timeout: int) -> dict:  # pragma: no cover - real benchmark
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    if spec.image_model is not None:
        atomic_write_json(spec.output_path.with_name("image-model.json"), {
            "schema_version": 1, "model": spec.image_model,
        })
    if spec.sampling_profile["profile"].startswith("publisher-recommended-v1:"):
        atomic_write_json(spec.output_path.with_name("publisher-sampling.json"), {
            "schema_version": 1,
            "name": spec.sampling_profile["profile"].split(":", 1)[1],
            "source": {
                "repo": spec.sampling_profile["source"]["repo"],
                "revision": spec.sampling_profile["source"]["revision"],
            },
            "controls": spec.sampling_profile["publisher_controls"],
        })
    import_status = ensure_candidate_import(spec)
    current = _read_result(spec.output_path)
    if current is None:
        _run_new_screen(spec, timeout)
        if _resume_screen(spec) != 0:
            raise RuntimeError("candidate recovery process failed")
        result = _read_result(spec.output_path)
        if result is None:
            raise RuntimeError("candidate screen produced no result")
    elif current.get("run", {}).get("status") == "complete":
        errors = compatibility_screen_errors(current, spec)
        if errors:
            raise RuntimeError("existing screen is incomplete: " + "; ".join(errors))
        result = current
    else:
        if _resume_screen(spec) != 0:
            raise RuntimeError("candidate recovery process failed")
        result = _read_result(spec.output_path)
        if result is None:
            raise RuntimeError("candidate screen produced no result")
    errors = compatibility_screen_errors(result, spec)
    comfyui_revision = repository_revision(config.COMFYUI_DIR) if spec.family == "image" else None
    if spec.family == "image" and comfyui_revision is None:
        errors.append("ComfyUI revision is unavailable")
    evidence_artifacts, evidence_errors = screen_evidence_artifacts(spec)
    errors.extend(evidence_errors)
    image_artifacts, artifact_errors = screen_image_artifacts(result, spec)
    errors.extend(artifact_errors)
    report = {
        "schema_version": SCREEN_SCHEMA_VERSION,
        "candidate": spec.candidate_id,
        "engine": spec.engine,
        "repo": spec.repo,
        "revision": spec.revision,
        "files": list(spec.files),
        "import": import_status,
        "comfyui_revision": comfyui_revision,
        "result": spec.output_path.name,
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "evidence_artifacts": evidence_artifacts,
        "image_artifacts": image_artifacts,
    }
    atomic_write_json(spec.output_path.with_name("screen-report.json"), report)
    if errors:
        raise RuntimeError("candidate screen failed: " + "; ".join(errors))
    return result


def main(argv=None) -> int:  # pragma: no cover - command entrypoint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--candidate")
    parser.add_argument("--engine", choices=("llamacpp", "vllm"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--interrupt-timeout", type=int, default=7200)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--publisher-sampling", action="store_true")
    args = parser.parse_args(argv)
    audit = load_source_audit(args.audit)
    if args.list:
        for candidate in audit["candidates"]:
            detail = "; ".join(candidate.get("reasons") or [])
            print(f"{candidate['id']}\t{candidate['family']}\t{candidate['status']}\t{detail}")
        return 0
    if not args.candidate or not args.engine:
        parser.error("--candidate and --engine are required unless --list is used")
    if args.interrupt_timeout < 120:
        parser.error("--interrupt-timeout must be at least 120 seconds")
    spec = build_screen_spec(
        candidate_record(audit, args.candidate), args.engine, args.output_root,
        publisher_sampling=args.publisher_sampling,
    )
    if not args.execute:
        print(json.dumps({
            **spec.__dict__, "output_path": str(spec.output_path),
            "command": list(spec.command),
        }, indent=2))
        return 0
    execute_screen(spec, args.interrupt_timeout)
    print(f"Candidate compatibility screen passed: {spec.output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
