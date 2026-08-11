#!/usr/bin/env python3
"""Cross-platform LLM benchmark suite — CLI entry point.
See docs/workloads.md for what each test measures, docs/cli-reference.md for flags."""

import argparse
import fnmatch
import json
import platform
import re
import signal
import sys
from datetime import datetime
from pathlib import Path

from scripts.runtime import config
from scripts.app.benchmark_options import TEST_CHOICES, TG_TOKEN_CHOICES, TIER_CHOICES, option_value_errors
from scripts.app.progress_events import PROGRESS_PREFIX, set_progress_engine
from scripts.runtime.log_redaction import redact_log_text
from scripts.runtime.comfyui_installation import find_comfyui_installation, normalize_comfyui_dir
from scripts.workloads.conversation_selection import conv_skip_entry
from scripts.runtime.shared import Shared
from scripts.runtime.engines import get_engine, engine_names as registered_engine_names
from scripts.runtime.engines.vllm import VllmEngine
from scripts.results.event_store import EventStore
from scripts.workloads.llm_prefill_benchmark import LLMPrefillBenchmark
from scripts.runtime.llamacpp_tools import find_llamacpp_tool
from scripts.results.llm_event_stage import LLMEventStage, event_store_path, export_llm_section
from scripts.results.native_bench_event_stage import NativeBenchEventStage, export_native_bench_section
from scripts.workloads.embedding_benchmark import EmbeddingBenchmark
from scripts.workloads.image_benchmark import ImageBenchmark
from scripts.workloads.mcq_benchmark import MCQBenchmark
from scripts.workloads.math_benchmark import MathBenchmark
from scripts.workloads.methodology_profile import resolve_methodology_profile
from scripts.runtime.network_policy import apply_offline_mode
from scripts.runtime.pause_control import apply_pause_evidence
from scripts.workloads.reasoning_benchmark import ReasoningBenchmark
from scripts.workloads.code_benchmark import CodeBenchmark
from scripts.workloads.tool_benchmark import ToolBenchmark
from scripts.workloads.llamabench_benchmark import LlamaBenchBenchmark
from scripts.workloads.llamabench_concurrency_benchmark import LlamaBenchConcurrencyBenchmark
from scripts.workloads.vllm_benchmark import VllmBenchBenchmark
from scripts.workloads.models import IMAGE_MODELS, LLM_MODELS_XSMALL, LLM_MODELS_SMALL, LLM_MODELS_MEDIUM, LLM_MODELS_LARGE, LLM_MODELS, EMBED_MODELS
from scripts.setup.model_inventory import build_model_inventory, format_model_inventory, sanitize_tag_to_short
from scripts.app.orchestration import (
    LifecycleCoordinator, RunContext, RunPaths, StageDefinition,
    StageExecutionError, execute_stages, execute_with_final_cleanup,
    ordered_stage_keys, select_stages,
)
from scripts.results.result_store import (ResultStore, atomic_write_json, build_run_manifest, finish_run,
                          finish_active_stage, model_identity)
from scripts.results.run_plan import RunPlan, load_run_plan
from scripts.results.resume_policy import build_engine_resume_identity
from scripts.results.result_history import ETA_MATCH_KEYS, estimate_matching_plan_seconds
from typing import Callable, Protocol

from scripts.runtime.runner_supervisor import RunnerSpec, RunnerSupervisor
from scripts.setup.setup_config import (
    available_gpu_split_modes, configured_comfyui_dir, load_setup_config,
)
from scripts.setup.runtime_identity import engine_runtime_version


def relay_runner_log(text: str) -> None:
    """Relay a runner's line as-is. The runner already stamped it; stamping again would
    report when the parent got round to printing, not when the event happened."""
    if text.startswith(PROGRESS_PREFIX):
        sys.stdout.write(text if text.endswith("\n") else f"{text}\n")
        sys.stdout.flush()
        return
    sys.stdout.write(f"{redact_log_text(text.rstrip())}\n")
    sys.stdout.flush()


def checkpoint_terminal_exception(results: dict, exc: BaseException, checkpoint) -> None:
    run = results["run"]
    if run["status"] == "running":
        reason = "invalid_numeric_value" if isinstance(exc, ValueError) \
            and "non-finite numeric value" in str(exc) else (
                f"stage_{exc.phase}_failed" if isinstance(exc, StageExecutionError)
                else type(exc).__name__
            )
        finish_active_stage(run, "failed", reason)
        finish_run(run, "failed", reason)
        checkpoint("run failed")
    elif run["status"] == "interrupted":
        finish_active_stage(run, "interrupted", run.get("reason", "signal"))
        checkpoint("run interrupted")


class _RunnerLike(Protocol):
    """The only two methods run_supervised_stage calls on a supervisor."""
    def run(self, on_event, /) -> int | None: ...
    def cancel(self) -> None: ...


def run_supervised_stage(plan: RunPlan, event_path: Path, stage_name: str, save_fn,
                         supervisor_factory: Callable[..., _RunnerLike] = RunnerSupervisor,
                         resume_identity=None,
                         resume=False, selected_case_ids=None) -> dict:
    event_path = Path(event_path).resolve()
    if stage_name == "llamabench":
        journal = NativeBenchEventStage(
            event_path, plan, lambda _: None, resume_identity=resume_identity, resume=resume,
        )
        project = lambda: export_native_bench_section(event_path, plan.job_id)
    else:
        model_family = "concurrency" if stage_name in {"conc_tool", "conc_chat"} else "llm"
        journal = LLMEventStage(
            event_path, plan, lambda _: None, stage_name=stage_name,
            model_family=model_family, resume_identity=resume_identity, resume=resume,
            selected_case_ids=selected_case_ids,
        )
        project = lambda: export_llm_section(
            event_path, plan.job_id, stage_name, model_family,
        )
    journal.close()
    supervisor = supervisor_factory(RunnerSpec(plan.job_id, stage_name, event_path))
    terminal = []

    def on_runner_event(event):
        if event["kind"] == "event":
            save_fn(project())
        elif event["kind"] == "terminal":
            terminal.append(event["status"])
        elif event["kind"] == "log":
            relay_runner_log(event["text"])

    try:
        return_code = supervisor.run(on_runner_event)
    finally:
        supervisor.cancel()
    section = project()
    save_fn(section)
    if return_code or terminal != ["complete"]:
        raise RuntimeError(f"{stage_name} runner failed with exit code {return_code}")
    return section


def run_supervised_llm(plan: RunPlan, event_path: Path, save_fn,
                       supervisor_factory: Callable[..., _RunnerLike] = RunnerSupervisor,
                       resume_identity=None) -> dict:
    return run_supervised_stage(
        plan, event_path, "llm", save_fn, supervisor_factory, resume_identity,
    )


# Tier selection is cumulative: --maxtier caps at that tier and includes
# everything below it.
TIER_MODELS = {
    "xsmall": LLM_MODELS_XSMALL,
    "small":  LLM_MODELS_XSMALL + LLM_MODELS_SMALL,
    "medium": LLM_MODELS_XSMALL + LLM_MODELS_SMALL + LLM_MODELS_MEDIUM,
    "large":  LLM_MODELS,
}
TIER_LABELS = {
    "xsmall": "extra-small only (≤4GB)",
    "small":  "small and below (≤16GB)",
    "medium": "medium and below (≤32GB)",
    "large":  "large and below — all tiers (32GB+)",
}
TIER_ORDER = ["xsmall", "small", "medium", "large"]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def apply_quick_preset(args) -> None:
    """Apply the fixed CLI smoke-test scope while preserving runtime/path options."""
    if not args.quick:
        return
    args.tests = ["llm"]
    args.warmup = 0
    args.runs = 1
    args.max_prompt_tokens = 2048
    args.maxtier = "xsmall"
    args.llm_models = [LLM_MODELS_XSMALL[0]["tag"]]


def format_duration_estimate(seconds: float | None) -> str:
    if seconds is None:
        return "unavailable — no exact completed local plan match"
    minutes = max(1, round(seconds / 60))
    return f"about {minutes // 60}h {minutes % 60}m" if minutes >= 60 else f"about {minutes}m"


def eta_match_config(args) -> dict:
    """Runtime-shaping settings required for a historical ETA match."""
    values = {
        "runs": config.N_RUNS, "warmup_runs": args.warmup,
        "run_timeout_seconds": config.RUN_TIMEOUT,
        "accuracy_timeout_seconds": config.ACC_TIMEOUT,
        "accuracy_token_budget": config.ACC_TOKEN_BUDGET,
        "cpu_only": args.cpu_only, "force_all": args.force_all,
        "max_prompt_tokens": args.max_prompt_tokens,
        "context_lengths": config.CONTEXT_LENGTHS,
        "llamabench_pp": config.LLAMABENCH_PP,
        "llamabench_tg": config.LLAMABENCH_TG,
        "sample_size": args.sample,
        "concurrency_tool_levels": config.CONCURRENCY_TOOL_LEVELS,
        "concurrency_chat_levels": config.CONCURRENCY_CHAT_LEVELS,
        "concurrency_tool_context": config.CONCURRENCY_TOOL_CONTEXT,
        "concurrency_chat_context": config.CONCURRENCY_CHAT_CONTEXT,
        "concurrency_chat_soft_exit_floor": config.CONCURRENCY_CHAT_MIN_LEVEL_BEFORE_SOFT_EXIT,
    }
    return {key: values[key] for key in ETA_MATCH_KEYS}


def format_resolved_plan(engine: str, tests: list[str], models: dict[str, list[dict]],
                         estimate_seconds: float | None, *, runs: int, warmups: int,
                         max_prompt_tokens: int | None, sample_size: int | None) -> str:
    family_for = {
        "emb": "embeddings", "img": "images", "conc_tool": "concurrency",
        "conc_chat": "concurrency",
    }
    lines = [f"Engine: {engine}", f"Workloads: {', '.join(tests)}"]
    for test in tests:
        family = family_for.get(test, "llm")
        labels = [label for model in models.get(family, [])
                  if (label := str(model.get("label") or model.get("short") or ""))]
        if test == "llm":
            cases = f"contexts {', '.join(map(str, config.CONTEXT_LENGTHS))}"
        elif test == "conv":
            from scripts.workloads.llm_conversation_benchmark import LLMConversationBenchmark
            cap = max_prompt_tokens or max(LLMConversationBenchmark.CONV_CHECKPOINTS)
            checkpoints = [value for value in LLMConversationBenchmark.CONV_CHECKPOINTS if value <= cap]
            cases = f"checkpoints {', '.join(map(str, checkpoints))}"
        elif test == "llamabench":
            cases = f"pp {config.LLAMABENCH_PP}; tg {config.LLAMABENCH_TG}"
        elif test == "llamabenchconc":
            cases = f"pp {config.LLAMABENCH_CONC_PP}; tg {config.LLAMABENCH_CONC_TG}; concurrency {config.LLAMABENCH_CONC_NPL}"
        elif test == "vllmbench":
            cases = f"input {config.VLLMBENCH_INPUT}; output {config.VLLMBENCH_OUTPUT}"
        elif test == "conc_tool":
            cases = f"levels {config.CONCURRENCY_TOOL_LEVELS}"
        elif test == "conc_chat":
            cases = f"levels {config.CONCURRENCY_CHAT_LEVELS}"
        elif test == "img":
            cases = "; ".join(
                f"{model['short']}={model.get('resolutions', config.IMAGE_RESOLUTIONS)}"
                for model in models[family]
            )
        elif test in ACCURACY_TESTS:
            cases = "full question bank" if sample_size is None else f"{sample_size} sampled questions"
        else:
            cases = "one document"
        lines.append(f"  {test}: {', '.join(labels) or '(no models)'} — {cases}")
    lines.append(f"Runs: {runs} measured + {warmups} warmup")
    lines.append(f"Estimated duration: {format_duration_estimate(estimate_seconds)}")
    return "\n".join(lines)


def format_dry_run_output(plans: list[str]) -> str:
    """Join resolved engine passes or explain that selection resolved to no work."""
    return "\n\n".join(plans) if plans else "No workloads resolved for the selected engine pass(es)."


def select_tier(maxtier: str | None, image_models: list) -> tuple[list, str, list]:
    """Resolve --maxtier into (llm_models, tier_label, image_models) — see
    the `--maxtier` row in docs/cli-reference.md."""
    if maxtier:
        llm_models = TIER_MODELS[maxtier]
        tier_label = TIER_LABELS[maxtier]
        max_idx = TIER_ORDER.index(maxtier)
        image_models = [m for m in image_models if TIER_ORDER.index(m["tier"]) <= max_idx]
    else:
        llm_models = LLM_MODELS
        tier_label = "all (extra-small + small + medium + large)"
    return llm_models, tier_label, image_models


def apply_max_prompt_tokens_cap(max_tokens: int | None, context_lengths: list[int],
                                llamabench_pp: list[int], llamabenchconc_pp: int,
                                ) -> tuple[list[int], list[int], int]:
    """Caps 'llm'/'llamabench'/'llamabenchconc' prompt depths to max_tokens — see --max-prompt-tokens.
    Raises ValueError if the cap excludes every depth from a list-based sweep."""
    if max_tokens is None:
        return list(context_lengths), list(llamabench_pp), llamabenchconc_pp
    capped_context_lengths = [c for c in context_lengths if c <= max_tokens]
    capped_llamabench_pp = [c for c in llamabench_pp if c <= max_tokens]
    if not capped_context_lengths or not capped_llamabench_pp:
        raise ValueError(
            f"--max-prompt-tokens {max_tokens} is below the smallest tested depth "
            f"({min(context_lengths[0], llamabench_pp[0])} tokens) — nothing would be tested"
        )
    return capped_context_lengths, capped_llamabench_pp, min(llamabenchconc_pp, max_tokens)


def apply_tg_tokens_override(tg_tokens: list[int] | None, default_llamabench_tg: list[int],
                             default_llamabenchconc_tg: list[int]) -> tuple[list[int], list[int]]:
    """Overrides the tg (generation-size) sweep for 'llamabench'/'llamabenchconc' — see --tg-tokens."""
    if tg_tokens is None:
        return list(default_llamabench_tg), list(default_llamabenchconc_tg)
    selected = sorted(set(tg_tokens))
    return selected, list(selected)


def filter_models_by_pattern(models: list, patterns: list[str] | None, key: str = "tag") -> list:
    """Filter models by exact/wildcard match on `key`. Case-sensitive
    (`fnmatchcase`) so behavior is identical across platforms."""
    if not patterns:
        return models
    return [m for m in models if any(fnmatch.fnmatchcase(m[key], p) for p in patterns)]


def resolve_custom_models(patterns: list[str], catalog: list[dict], installed_tags: list[str],
                          known_catalog: list[dict] = LLM_MODELS + EMBED_MODELS) -> list[dict]:
    """Resolve patterns against catalog entries and installed custom models."""
    known_catalog_tags = {m["tag"] for m in known_catalog}
    resolved = list(filter_models_by_pattern(catalog, patterns))
    seen = {m["tag"] for m in resolved}

    for pattern in patterns:
        for tag in installed_tags:
            if tag in seen or tag in known_catalog_tags:
                continue
            if fnmatch.fnmatchcase(tag, pattern):
                resolved.append({"tag": tag, "label": f"{tag} (custom)", "short": sanitize_tag_to_short(tag)})
                seen.add(tag)

    return resolved


def downloaded_models(catalog: list[dict], installed_tags: list[str]) -> list[dict]:
    """Filter `catalog` to entries actually downloaded locally, preserving
    order — see docs/workloads.md#concurrency."""
    installed = set(installed_tags)
    return [m for m in catalog if m["tag"] in installed]


def fork_provenance(source_path: Path, plan: RunPlan, output_path: Path) -> dict:
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path == source_path:
        raise ValueError("fork output must differ from its source result")
    if output_path.exists() or output_path.with_suffix(".events.sqlite3").exists():
        raise ValueError("fork output or event journal already exists")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_plan = load_run_plan(source_path)
    if source_plan.plan_id != plan.plan_id:
        raise ValueError("fork configuration no longer matches the saved plan")
    return {
        "run_id": source.get("run", {}).get("run_id"),
        "job_id": source_plan.job_id, "plan_id": source_plan.plan_id,
    }


def finish_event_job(path: Path, plan: RunPlan, state: str, reason=None) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    journal = EventStore(path)
    try:
        journal.finish_job(plan.job_id, state, reason)
    finally:
        journal.close()
    return True


def interruption_exit_code(sig) -> int:
    return 128 + int(sig)


def resolve_model_scopes(tier_models: list[dict], installed_tags: list[str],
                         patterns: list[str] | None, concurrency_enabled: bool
                         ) -> tuple[list[dict], list[dict]]:
    """Resolve normal and concurrency model scopes — concurrency ignores the
    tier cap (see docs/workloads.md#concurrency) but still honors --models."""
    run_models = (
        resolve_custom_models(patterns, tier_models, installed_tags)
        if patterns else tier_models
    )
    concurrency_models = []
    if concurrency_enabled:
        concurrency_models = downloaded_models(LLM_MODELS, installed_tags)
        if patterns:
            concurrency_models = resolve_custom_models(
                patterns, concurrency_models, installed_tags,
            )
    return run_models, concurrency_models


ACCURACY_TESTS = ["mcq", "math", "reasoning", "code", "tool"]
CONCURRENCY_TESTS = ["conc_tool", "conc_chat"]
LLM_TESTS = ["llm", "conv", *ACCURACY_TESTS, "llamabench", "llamabenchconc", "vllmbench"]


def engine_version_applies(tests: list[str]) -> bool:
    return bool(set(tests) & (set(LLM_TESTS) | set(CONCURRENCY_TESTS) | {"emb"}))

# Tests that shell out to one engine's own native benchmark binary rather than going
# through InferenceEngine, so they can never run under a different engine — see docs/engines.md.
ENGINE_NATIVE_TESTS = {
    "llamacpp": ("llamabench", "llamabenchconc"),
    "vllm": ("vllmbench",),
}


def engine_incompatible_tests(tests: list[str], engine_name: str) -> list[str]:
    """Selected tests that are native to a *different* engine than `engine_name` and
    would just warn and produce nothing if scheduled — see ENGINE_NATIVE_TESTS."""
    native_here = set(ENGINE_NATIVE_TESTS.get(engine_name, ()))
    other_engines_native = {
        test for name, native_tests in ENGINE_NATIVE_TESTS.items()
        if name != engine_name for test in native_tests
    }
    return [t for t in tests if t in other_engines_native and t not in native_here]


def engine_pass_tests(tests: list[str], engine_name: str, *, include_images: bool) -> list[str]:
    """Workloads that produce results in one engine pass."""
    incompatible = set(engine_incompatible_tests(tests, engine_name))
    return [
        test for test in tests
        if test not in incompatible and (include_images or test != "img")
    ]


def selected_plan_models(tests: list[str], llm_models: list[dict],
                         concurrency_models: list[dict], embedding_models: list[dict],
                         image_models: list[dict]) -> dict[str, list[dict]]:
    selected = set(tests)
    return {
        "llm": model_identity(llm_models) if selected & set(LLM_TESTS) else [],
        "concurrency": (model_identity(concurrency_models)
                        if selected & set(CONCURRENCY_TESTS) else []),
        "embeddings": model_identity(embedding_models) if "emb" in selected else [],
        "images": model_identity(image_models) if "img" in selected else [],
    }


def resolve_catalog_scopes(image_models: list[dict], embedding_patterns: list[str] | None,
                           image_patterns: list[str] | None) -> tuple[list[dict], list[dict]]:
    """Resolve the engine-independent embedding and image model scopes."""
    embedding_models = filter_models_by_pattern(EMBED_MODELS, embedding_patterns)
    image_models = filter_models_by_pattern(image_models, image_patterns, key="short")
    return embedding_models, image_models


def validate_catalog_scopes(tests: list[str], embedding_patterns: list[str] | None,
                            image_patterns: list[str] | None, embedding_models: list[dict],
                            image_models: list[dict]) -> list[str]:
    """Return selector errors for engine-independent workload scopes."""
    errors = []
    if "emb" in tests and embedding_patterns and not embedding_models:
        errors.append(
            f"--embedding-models {' '.join(embedding_patterns)} matched no embedding models"
        )
    if "img" in tests and image_patterns and not image_models:
        errors.append(f"--image-models {' '.join(image_patterns)} matched no image models")
    return errors


def validate_engine_scopes(tests: list[str], engine_name: str, llm_patterns: list[str] | None,
                           llm_models: list[dict], concurrency_models: list[dict],
                           tier_label: str) -> list[str]:
    """Return selector errors for one engine's LLM-backed workload scopes."""
    if not llm_patterns:
        return []
    errors = []
    if any(test in tests for test in LLM_TESTS) and not llm_models:
        errors.append(
            f"--llm-models {' '.join(llm_patterns)} matched no LLM models in the "
            f"selected tier ({tier_label}) or installed for {engine_name}"
        )
    if any(test in tests for test in CONCURRENCY_TESTS) and not concurrency_models:
        errors.append(
            f"--llm-models {' '.join(llm_patterns)} matched no downloaded concurrency "
            f"models for {engine_name}"
        )
    return errors


def resolve_engine_scopes(engine_names: list[str], engine_factory, tier_models: list[dict],
                          tier_label: str, llm_patterns: list[str] | None, tests: list[str]
                          ) -> tuple[list[dict], list[str]]:
    """Resolve and validate every engine before benchmark orchestration."""
    concurrency_enabled = any(test in tests for test in CONCURRENCY_TESTS)
    normal_llm_enabled = any(test in tests for test in LLM_TESTS)
    known_tags = [model["tag"] for model in LLM_MODELS + EMBED_MODELS]
    custom_lookup_needed = bool(
        llm_patterns and normal_llm_enabled
        and any(pattern not in known_tags for pattern in llm_patterns)
    )
    inventory_needed = custom_lookup_needed or concurrency_enabled
    scopes = []
    errors = []
    for engine_name in engine_names:
        engine = engine_factory(engine_name)
        engine_tests = engine_pass_tests(tests, engine_name, include_images=True)
        installed_tags = (
            [model["tag"] for model in engine.list_installed_models()]
            if inventory_needed else []
        )
        llm_models, concurrency_models = resolve_model_scopes(
            tier_models, installed_tags, llm_patterns, concurrency_enabled,
        )
        scopes.append({
            "name": engine_name,
            "engine": engine,
            "llm_models": llm_models,
            "concurrency_models": concurrency_models,
        })
        errors.extend(validate_engine_scopes(
            engine_tests, engine_name, llm_patterns, llm_models, concurrency_models, tier_label,
        ))
    return scopes, errors


def sidecar_path(out_path: str, prefix: str) -> Path:
    """Build a results-directory sidecar path from the main output's stem."""
    stem = Path(out_path).stem
    name = prefix + stem[len("results_"):] if stem.startswith("results_") else f"{prefix}{stem}"
    return config.RESULTS_DIR / f"{name}.json"


# --tests shorthand groups, expanded by expand_tests below.
TEST_GROUPS = {
    "acc":  ACCURACY_TESTS,
    "conc": CONCURRENCY_TESTS,
}


def expand_tests(tests: list[str]) -> list[str]:
    """Expand TEST_GROUPS shorthand in --tests, preserving order and de-duplicating."""
    expanded = []
    for t in tests:
        for name in TEST_GROUPS.get(t, [t]):
            if name not in expanded:
                expanded.append(name)
    return expanded


def resolve_engine_names(engine: str, available: list[str]) -> list[str]:
    """Resolve --engine into an ordered engine-name list ("all", or comma-separated),
    always in registry order so a multi-engine run is deterministic."""
    if engine == "all":
        return list(available)
    requested = [name.strip() for name in engine.split(",") if name.strip()]
    unknown = [name for name in requested if name not in available]
    if unknown or not requested:
        raise ValueError(
            f"Unknown inference engine {', '.join(unknown) or engine!r} — "
            f"known engines: {', '.join(available)}, or 'all'")
    return [name for name in available if name in requested]


def add_model_selection_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the public per-family model selector arguments."""
    parser.add_argument(
        "--llm-models", "--models", dest="llm_models", nargs="+", default=None,
        help="Only test these LLM models (llm, conv, mcq, math, reasoning, code, tool, and "
             "concurrency tests) — exact tags or shell-style wildcards, e.g. "
             "'llama*' matches every tag starting with 'llama' (default: every model "
             "in the selected tier). Applied after --maxtier, so it can only narrow "
             "that selection further within the catalog. Patterns also match models "
             "actually downloaded locally, so a model outside our curated catalog can be "
             "tested (see --list-models). Quote wildcards so your shell doesn't expand "
             "them first. --models is retained as a backward-compatible alias.",
    )
    parser.add_argument(
        "--embedding-models", nargs="+", default=None,
        help="Only test these embedding model tags — exact tags or shell-style wildcards "
             "(default: every catalog embedding model). Quote wildcards in a shell.",
    )
    parser.add_argument(
        "--image-models", nargs="+", default=None, metavar="SHORT",
        help="Only test these image model short identifiers from models.py — exact values "
             "or shell-style wildcards (default: every image model allowed by --maxtier). "
             "Applied after --maxtier. Quote wildcards in a shell.",
    )


def main():  # pragma: no cover — CLI entrypoint; orchestrates real llama.cpp/ComfyUI runs
    parser = argparse.ArgumentParser(description="LLM benchmark suite")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a short pipeline smoke test: the smallest xsmall LLM at 512 and 2K, "
             "one measured run, no warmups, and no other workloads. Runtime, engine, "
             "and output-path options still apply.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the fully resolved engines, models, workloads, checkpoints, and an "
             "ETA from exact matching local result history, then exit without benchmarking.",
    )
    parser.add_argument(
        "--tests", nargs="+",
        choices=TEST_CHOICES,
        default=["llm", "conv", "emb", "mcq", "math", "reasoning", "code", "tool", "img"],
        help="Which benchmarks to run (default: all except the concurrency "
             "tests and 'llamabench'). 'acc' is shorthand for every accuracy-style test "
             "('mcq', 'math', 'reasoning', 'code', and 'tool'). 'conc_tool' and 'conc_chat' are "
             "the two concurrency tests (see workloads.md) — opt-in, not "
             "part of the default set, since each takes noticeably longer "
             "per model than one request at a time, and both scope to "
             "whatever LLM models are actually downloaded locally "
             "(ignoring --maxtier — a machine that only downloaded "
             "xsmall/small models tests those; one with medium/large "
             "downloaded tests those too) rather than a fixed model list. "
             "'conc_tool' simulates short-context agentic fan-out: a 1-16 "
             "concurrent-request sweep at a short per-request context, every "
             "level always run (no early exit). 'conc_chat' simulates a chat "
             "server under load: a 1-32 concurrent-request sweep at a long "
             "per-request context, with an early exit once tok/s craters "
             "(disable via --force-all). 'conc' is shorthand for both. "
             "'llamabench' is also opt-in, not part of the default set: it runs "
             "llama.cpp's own llama-bench tool directly (bypassing this project's "
             "HTTP/SSE pipeline) for a pp/tg throughput sweep per model — see "
             "workloads.md#llama-bench. It scopes to the same models as llm/conv "
             "(--maxtier, --llm-models) and overlaps substantially with the llm test. "
             "'llamabenchconc' is opt-in too: it runs llama.cpp's own llama-batched-bench "
             "tool to sweep decode throughput across rising concurrency levels (1 to 16 "
             "parallel sequences) at a fixed prompt depth, complementing 'conc_tool'/"
             "'conc_chat' with a lower-level, HTTP-bypassing cross-check — see "
             "workloads.md#llama-bench-concurrency.",
    )
    parser.add_argument(
        "--warmup", type=int, default=config.WARMUP_RUNS,
        help=f"Warmup runs before measuring (default: {config.WARMUP_RUNS})",
    )
    parser.add_argument(
        "--runs", type=int, default=config.N_RUNS, choices=range(1, 11),
        metavar="[1-10]",
        help=f"Measured runs per checkpoint for single-shot LLM, embeddings, and "
             f"images, averaged (default: {config.N_RUNS}). Also sets how many isolated "
             "llama-bench -r 1 repetitions are aggregated for 'llamabench'. Conversation, accuracy, "
             "and concurrency tests use one measured pass/batch. Total measured time scales roughly in "
             "proportion — e.g. going from 3 to 6 runs roughly doubles it "
             "(warmup time is unaffected; see --warmup).",
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Seconds per engine generation/chat run and warmup before aborting "
             "(default: 300). Image generations use twice this value; embedding "
             "calls use their fixed 120s engine timeout; accuracy questions use --acc-timeout.",
    )
    parser.add_argument(
        "--acc-timeout", type=int, default=None,
        help="Seconds per question before giving up on it, for the accuracy tests "
             f"(mcq, math, reasoning, code, tool) — any partial response is scored normally and the run "
             f"moves on (default: {config.ACC_TIMEOUT})",
    )
    parser.add_argument(
        "--max-prompt-tokens", type=positive_int, default=None, metavar="N",
        help="Cap the deepest prompt-processing size swept by 'llm' (drops entries from "
             f"CONTEXT_LENGTHS {config.CONTEXT_LENGTHS}), 'llamabench' (drops entries from "
             f"LLAMABENCH_PP {config.LLAMABENCH_PP}), and 'llamabenchconc' (clamps its fixed "
             f"prompt depth, default {config.LLAMABENCH_CONC_PP}); also caps 'conv' checkpoints "
             "and growth target to at most N tokens — only "
             "affects whichever of those tests are actually selected via --tests "
             "(default: no cap, run every configured depth).",
    )
    parser.add_argument(
        "--tg-tokens", type=int, nargs="+", default=None, choices=TG_TOKEN_CHOICES, metavar="N",
        help="Which generation (tg) sizes 'llamabench' and 'llamabenchconc' sweep at each "
             f"prompt depth (default: {config.LLAMABENCH_TG}). Only affects whichever of "
             "those two tests are actually selected via --tests.",
    )
    parser.add_argument(
        "--acc-token-budget", type=positive_int, default=None, metavar="N",
        help="Total completion-token budget per accuracy question, split 60/40 "
             f"between the initial and final-answer passes (default: {config.ACC_TOKEN_BUDGET})",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output JSON file (default: results/results_<hostname>_<timestamp>.json)",
    )
    parser.add_argument("--fork-plan", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--comfyui", type=str, default=None,
        help="Path to an existing ComfyUI program or Windows portable root "
             "(default: saved/system installation, then repository-managed ComfyUI)",
    )
    parser.add_argument(
        "--cpu-only", action="store_true",
        help="Force CPU-only inference for every LLM-backed test (llm, conv, "
             "mcq, math, reasoning, code, tool, emb) by restarting the engine with GPU devices "
             "hidden (HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES / "
             "ROCR_VISIBLE_DEVICES set empty). Stops any running engine server "
             "(even one this script didn't start) and restores normal GPU mode "
             "afterward. Useful on GPU backends unstable under one of those "
             "workloads (originally added for embedding batching, but the same "
             "instability can hit LLM/MCQ inference on some backends too). "
             "'llamabench' and 'llamabenchconc' also honor this (passing -ngl 0 straight to "
             "llama-bench/llama-batched-bench) though they don't go through the engine restart above.",
    )
    parser.add_argument(
        "--gpu-split-mode", choices=("layer", "tensor"), default="layer",
        help="llama.cpp multi-GPU execution mode: compatible layer splitting or experimental "
             "tensor parallelism (default: layer). Tensor mode requires supported CUDA GPUs/models "
             "and uses f16 KV cache because llama.cpp does not support quantized KV there.",
    )
    parser.add_argument(
        "--llamacpp-no-repack", action="store_true",
        help="Disable llama.cpp weight repacking with --no-repack/-nr. This can reduce model "
             "startup time and peak loading memory but may reduce CPU inference throughput "
             "(default: false).",
    )
    parser.add_argument(
        "--maxtier", type=str, default=None,
        choices=TIER_CHOICES,
        help="Cap LLM models (single-shot and conversation tests) at this size tier "
             "and below (default: all tiers). xsmall: <6B params. small: adds ≤20B. "
             "medium: adds 26-35B. large: adds 70B+ (i.e. no cap).",
    )
    add_model_selection_arguments(parser)
    parser.add_argument(
        "--list-models", action="store_true",
        help="List every LLM, embedding, custom LLM, and catalog image model installed "
             "locally, then exit without running anything. Uses --comfyui for image "
             "checkpoint discovery.",
    )
    parser.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Dev-only: run 'mcq'/'math'/'reasoning'/'code'/'tool' against a deterministic N-question "
             "subset of each bank instead of the full thing, selected by deterministic "
             "round-robin across categories. Every category is represented when N is "
             "at least that bank's category count. Same N yields the same questions for "
             "a given bank version, and the exact sampled IDs are recorded in the "
             "output JSON under 'sample_ids'. Never use for a result meant to be "
             "compared against a full-bank run or published (default: full bank).",
    )
    parser.add_argument(
        "--force-all", action="store_true",
        help=f"Ignore the {config.SLOW_MODEL_MIN_TPS:.0f} tok/s slow-model cutoff: run every "
             "context length in the LLM prefill test and always run the conversation "
             "test, even for models that would otherwise be marked slow and skipped; "
             "also disable the chat-concurrency soft exit. "
             "Does not override real failures (timeouts, missing data). (default: false)",
    )
    parser.add_argument(
        "--retry-crashed-models", action="store_true",
        help="Run models recorded in workload crash caches instead of skipping them. "
             "Cache files remain intact and repeated crashes are recorded normally. "
             "(default: false)",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Block non-loopback network connections during benchmark execution and propagate "
             "offline/telemetry-disabled environment settings to managed runtimes. Local "
             "llama.cpp and ComfyUI HTTP connections remain available.",
    )
    _engines = registered_engine_names()
    parser.add_argument(
        "--engine", type=str, default=_engines[0],
        help=f"Inference engine to benchmark against (default: {_engines[0]}). "
             "'all' runs the full --tests suite once per registered engine, back "
             "to back (sorted order), writing a separate results file for each "
             "(engine name appended to the filename) so they can be compared "
             "directly. Only llama.cpp is registered today, so this is a no-op "
             "until a second engine (e.g. MLX) is added — kept here so scripts/"
             "docs referencing --engine don't need to change when one is.",
    )
    args = parser.parse_args()
    apply_quick_preset(args)
    config.LLAMACPP_GPU_SPLIT_MODE = args.gpu_split_mode
    config.LLAMACPP_NO_REPACK = args.llamacpp_no_repack
    config.RETRY_CRASHED_MODELS = args.retry_crashed_models
    if args.offline:
        apply_offline_mode()

    option_errors = option_value_errors({
        "--warmup": args.warmup, "--runs": args.runs, "--timeout": args.timeout,
        "--acc-timeout": args.acc_timeout, "--acc-token-budget": args.acc_token_budget,
        "--max-prompt-tokens": args.max_prompt_tokens, "--sample": args.sample,
    })
    if option_errors:
        parser.error(option_errors[0])

    args.tests = expand_tests(args.tests)
    setup_config = load_setup_config(config.SETUP_CONFIG_PATH)
    if args.comfyui and not normalize_comfyui_dir(Path(args.comfyui)):
        parser.error("--comfyui must contain main.py or a ComfyUI/main.py portable layout")
    comfyui_dir = find_comfyui_installation(
        explicit=args.comfyui,
        saved_path=configured_comfyui_dir(setup_config),
        managed_dir=config.COMFYUI_DIR,
    ) or config.COMFYUI_DIR
    try:
        run_engine_names = resolve_engine_names(args.engine, _engines)
    except ValueError as exc:
        parser.error(str(exc))

    if args.list_models:
        any_installed = False
        for engine_name in run_engine_names:
            inventory = build_model_inventory(get_engine(engine_name), config.COMFYUI_MODELS_DIR)
            any_installed = any_installed or any(inventory.values())
            for line_i, line in enumerate(format_model_inventory(inventory, engine_name)):
                Shared.output(line, leading_blank=line_i == 0)
        if not any_installed:
            Shared.warn("No models are installed — run setup to add catalog models")
        sys.exit(0)

    # Apply CLI overrides to shared config
    if args.timeout is not None:
        config.RUN_TIMEOUT = args.timeout
    if args.acc_timeout is not None:
        config.ACC_TIMEOUT = args.acc_timeout
    if args.acc_token_budget is not None:
        config.ACC_TOKEN_BUDGET = args.acc_token_budget
    config.N_RUNS = args.runs
    if args.max_prompt_tokens is not None:
        try:
            config.CONTEXT_LENGTHS, config.LLAMABENCH_PP, config.LLAMABENCH_CONC_PP = apply_max_prompt_tokens_cap(
                args.max_prompt_tokens, config.CONTEXT_LENGTHS, config.LLAMABENCH_PP, config.LLAMABENCH_CONC_PP,
            )
        except ValueError as e:
            Shared.err(str(e))
            sys.exit(2)
    config.LLAMABENCH_TG, config.LLAMABENCH_CONC_TG = apply_tg_tokens_override(
        args.tg_tokens, config.LLAMABENCH_TG, config.LLAMABENCH_CONC_TG,
    )

    tier_models, tier_label, tier_image_models = select_tier(args.maxtier, IMAGE_MODELS)
    embedding_models, image_models = resolve_catalog_scopes(
        tier_image_models, args.embedding_models, args.image_models,
    )
    validation_errors = validate_catalog_scopes(
        args.tests, args.embedding_models, args.image_models, embedding_models, image_models,
    )
    engine_scopes, engine_errors = resolve_engine_scopes(
        run_engine_names, get_engine, tier_models, tier_label, args.llm_models, args.tests,
    )
    validation_errors.extend(engine_errors)
    if validation_errors:
        for error in validation_errors:
            Shared.err(error)
        sys.exit(2)

    if args.dry_run:
        previews = []
        for run_idx, engine_scope in enumerate(engine_scopes):
            include_images = len(engine_scopes) == 1 or run_idx == 0
            tests = engine_pass_tests(args.tests, engine_scope["name"], include_images=include_images)
            if not tests:
                continue
            plan_models = selected_plan_models(
                tests, engine_scope["llm_models"], engine_scope["concurrency_models"],
                embedding_models, image_models,
            )
            estimate = estimate_matching_plan_seconds(
                config.RESULTS_DIR, engine_scope["name"], tests, plan_models,
                eta_match_config(args),
            )
            display_models = {
                "llm": engine_scope["llm_models"],
                "concurrency": engine_scope["concurrency_models"],
                "embeddings": embedding_models, "images": image_models,
            }
            previews.append(format_resolved_plan(
                engine_scope["name"], tests, display_models, estimate,
                runs=args.runs, warmups=args.warmup,
                max_prompt_tokens=args.max_prompt_tokens, sample_size=args.sample,
            ))
        Shared.output(format_dry_run_output(previews))
        return

    hardware_profile = Shared.build_profile()
    _safe = re.sub(r'[\\/:*?"<>|\s]+', '_', hardware_profile['hostname']).strip('_')
    _start_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base_out_path = args.out or str(config.RESULTS_DIR / f"results_{_safe}_{_start_stamp}.json")

    multi_engine = len(run_engine_names) > 1

    for run_idx, engine_scope in enumerate(engine_scopes):
        engine_name = engine_scope["name"]
        engine = engine_scope["engine"]
        llm_models = engine_scope["llm_models"]
        conc_models = engine_scope["concurrency_models"]
        # Held on Shared so shutdown_managed() (called from the signal handler and
        # the finally block) can consult the live engine without threading it in.
        Shared._active_engine = engine

        set_progress_engine(engine_name)
        if multi_engine:
            Shared.section(f"Engine: {engine_name} ({run_idx + 1}/{len(run_engine_names)})")
            _base = Path(base_out_path)
            out_path = str(_base.with_name(f"{_base.stem}_{engine_name}{_base.suffix}"))
        else:
            out_path = base_out_path

        # Image generation doesn't depend on --engine (separate ComfyUI call) — run it once, first pass only.
        include_images = not multi_engine or run_idx == 0
        if not include_images and "img" in args.tests:
            Shared.log("Image generation doesn't depend on --engine — already "
                       f"captured in the {run_engine_names[0]} pass, skipping for {engine_name}")

        native_elsewhere = engine_incompatible_tests(args.tests, engine_name)
        if native_elsewhere:
            scope = ("runs only under its native engine" if len(native_elsewhere) == 1
                     else "run only under their native engines")
            Shared.log(f"{', '.join(native_elsewhere)} {scope} — skipping for {engine_name}")

        tests = engine_pass_tests(args.tests, engine_name, include_images=include_images)
        if not tests:
            Shared.log(f"No selected workloads apply to {engine_name} — skipping this engine pass")
            continue

        engine_backed_tests = [
            t for t in ("llm", "conv", "llamabench", "llamabenchconc", "emb", "mcq", "math", "reasoning", "code", "tool",
                        "conc_tool", "conc_chat") if t in tests
        ]
        hardware_backend = hardware_profile["backend"]
        profile = {
            **hardware_profile,
            "hardware_backend": hardware_backend,
            "backend": (engine.runtime_backend(hardware_backend, cpu_only=args.cpu_only)
                        if engine_backed_tests else hardware_backend),
        }
        runtime_version = (
            engine_runtime_version(engine_name, engine) if engine_version_applies(tests) else None
        )
        if (engine_backed_tests
                and args.gpu_split_mode not in available_gpu_split_modes(setup_config, profile["backend"])):
            parser.error(
                "--gpu-split-mode tensor requires at least two GPUs recorded by setup "
                "and a CUDA or ROCm/HIP llama.cpp runtime; rerun setup or use layer"
            )

        Shared.output(f"{config.BOLD}LLM Benchmark Suite{config.RESET}", leading_blank=True)
        Shared.output(f"  Host:      {profile['hostname']}")
        Shared.output(f"  OS:        {profile['os']}")
        Shared.output(f"  Backend:   {profile['backend']}")
        if profile["backend"] != profile["hardware_backend"]:
            Shared.output(f"  Hardware:  {profile['hardware_backend']}")
        Shared.output(f"  RAM:       {profile['ram_gb']} GB")
        Shared.output(f"  Engine:    {engine_name}")
        if runtime_version:
            Shared.output(f"  Runtime:   {runtime_version}")
        Shared.output(f"  Runs:      {config.N_RUNS} measured + {args.warmup} warmup")
        Shared.output(
            f"  Timeout:   {config.RUN_TIMEOUT}s per run, "
            f"{config.ACC_TIMEOUT}s per accuracy question"
        )
        Shared.output(f"  Accuracy:  {config.ACC_TOKEN_BUDGET} completion tokens (60/40 split)")
        Shared.output(f"  Models:    {tier_label}")
        if args.llm_models:
            Shared.output(f"  --llm-models: {', '.join(m['label'] for m in llm_models)}")
        if args.embedding_models:
            Shared.output(f"  --embedding-models: {', '.join(m['label'] for m in embedding_models)}")
        if args.maxtier or args.image_models:
            Shared.output(f"  Images:    {', '.join(m['label'] for m in image_models) or '(none — tier too small)'}")
        Shared.output(f"  Tests:     {', '.join(tests)}")
        Shared.output(f"  ComfyUI:   {comfyui_dir}")

        # Register cleanup for Ctrl-C and normal exit
        def _cleanup(sig=None, frame=None):
            if sig is not None:
                Shared.output(
                    f"{config.YELLOW}Interrupted — unloading models before exit ...{config.RESET}",
                    leading_blank=True,
                )
                if results.get("run", {}).get("status") == "running":
                    finish_active_stage(results["run"], "interrupted", "signal")
                    finish_run(results["run"], "interrupted", "signal")
                    try:
                        finish_event_job(
                            event_store_path(Path(out_path)), plan, "interrupted", "signal",
                        )
                    except Exception as exc:
                        Shared.err(f"Failed to terminalize interrupted event journal: {exc}")
                    try:
                        apply_pause_evidence(results["run"])
                        atomic_write_json(Path(out_path), results)
                    except Exception as exc:
                        Shared.err(f"Failed to checkpoint interrupted state: {exc}")
            if engine.available():
                engine.unload_all()
            if Shared.comfyui_available():
                ImageBenchmark.comfyui_free_models()
            if Shared._managed_procs:
                Shared.output(
                    f"{config.YELLOW}Cleaning up managed servers ...{config.RESET}",
                    leading_blank=True,
                )
                Shared.shutdown_managed()
            if sig is not None:
                sys.exit(interruption_exit_code(sig))

        stage_order = ordered_stage_keys(tuple(tests))
        vllm_kv_cache_dtype = "auto"
        vllm_launcher_args = []
        if isinstance(engine, VllmEngine):
            vllm_kv_cache_dtype = engine.configure_kv_cache(profile["backend"])
            vllm_launcher_args = engine.launcher_extra_args
        methodology = resolve_methodology_profile(
            engine_name=engine_name, tests=tests, cpu_only=args.cpu_only,
            vllm_kv_cache_dtype=vllm_kv_cache_dtype,
            vllm_launcher_args=vllm_launcher_args,
        )
        effective_config = {
            "runs": config.N_RUNS, "warmup_runs": args.warmup,
            "run_timeout_seconds": config.RUN_TIMEOUT,
            "accuracy_timeout_seconds": config.ACC_TIMEOUT,
            "accuracy_token_budget": config.ACC_TOKEN_BUDGET,
            "cpu_only": args.cpu_only, "force_all": args.force_all,
            "retry_crashed_models": args.retry_crashed_models,
            "gpu_split_mode": args.gpu_split_mode,
            "llamacpp_no_repack": args.llamacpp_no_repack,
            "max_prompt_tokens": args.max_prompt_tokens,
            "context_lengths": config.CONTEXT_LENGTHS,
            "llamabench_pp": config.LLAMABENCH_PP,
            "llamabench_tg": config.LLAMABENCH_TG,
            "concurrency_tool_levels": config.CONCURRENCY_TOOL_LEVELS,
            "concurrency_chat_levels": config.CONCURRENCY_CHAT_LEVELS,
            "concurrency_tool_context": config.CONCURRENCY_TOOL_CONTEXT,
            "concurrency_chat_context": config.CONCURRENCY_CHAT_CONTEXT,
            "concurrency_chat_soft_exit_floor": config.CONCURRENCY_CHAT_MIN_LEVEL_BEFORE_SOFT_EXIT,
            "sample_size": args.sample,
            "offline": args.offline,
            "methodology_profile": methodology["profile"],
            "effective_optimizations": methodology["effective_optimizations"],
        }
        plan_models = selected_plan_models(
            tests, llm_models, conc_models, embedding_models, image_models,
        )
        plan = RunPlan.create(
            application_version=config.VERSION, engine_name=engine_name,
            tests=tests, stage_order=stage_order, models=plan_models,
            effective_config=effective_config,
        )
        plan.validate_for_execution()
        forked_from = (
            fork_provenance(Path(args.fork_plan), plan, Path(out_path))
            if args.fork_plan else None
        )
        journal_stages = set(tests) & {"llm", "conv", "llamabench", "conc_tool", "conc_chat"}
        resume_identity = None
        if journal_stages:
            extra_resume_runtimes = {}
            if "llamabench" in tests:
                llama_bench_path = find_llamacpp_tool("llama-bench")
                if not llama_bench_path:
                    raise ValueError("cannot identify llama-bench runtime for resume")
                extra_resume_runtimes["llama-bench"] = Path(llama_bench_path).resolve()
            model_families = []
            if journal_stages & {"llm", "conv", "llamabench"}:
                model_families.append("llm")
            if journal_stages & {"conc_tool", "conc_chat"}:
                model_families.append("concurrency")
            identity_model_count = len({
                model["tag"] for family in model_families for model in plan.models[family]
                if engine.model_pulled(model["tag"])
            })
            Shared.log(
                f"Verifying resume identity for {identity_model_count} local model artifact(s) ..."
            )
            resume_identity = build_engine_resume_identity(
                plan, engine, model_families=model_families,
                include_engine_runtime=bool(journal_stages - {"llamabench"}),
                extra_runtimes=extra_resume_runtimes,
                digest_cache_path=config.RESUME_DIGEST_CACHE_PATH,
                environment=profile,
            )

        results = {
            "version":         config.VERSION,
            "engine":          engine_name,
            "engine_version":  runtime_version,
            "profile":         profile,
            "accuracy_settings": {
                "timeout_seconds": config.ACC_TIMEOUT,
                "token_budget": config.ACC_TOKEN_BUDGET,
                "first_pass_fraction": config.ACC_FINALIZE_FRACTION,
            },
            # See docs/workloads.md#bank-versioning.
            "bank_versions": {
                "mcq":  Shared.file_hash(MCQBenchmark.MCQ_DATA_PATH),
                "math": Shared.file_hash(MathBenchmark.MATH_DATA_PATH),
                "reasoning": Shared.file_hash(ReasoningBenchmark.REASONING_DATA_PATH),
                "code": Shared.file_hash(CodeBenchmark.CODE_DATA_PATH),
                "tool": Shared.file_hash(ToolBenchmark.TOOL_DATA_PATH),
            },
            "sample_ids": {},  # populated only when --sample is used
            "llm":             {},
            "llm_conversation": {},
            "embeddings":      {},
            "images":          {},
            "mcq":             {},
            "math":            {},
            "reasoning":       {},
            "code":            {},
            "tool":            {},
            "concurrency_tool": {},
            "concurrency_chat": {},
            "llamabench":      {},
            "llamabenchconc":  {},
            "vllmbench":       {},
        }

        results["run"] = build_run_manifest(
            plan=plan, repo_root=config.SCRIPT_DIR,
        )
        if forked_from:
            results["run"]["forked_from"] = forked_from

        store = ResultStore(Path(out_path), results)

        def _checkpoint(label=""):
            apply_pause_evidence(results["run"])
            store.checkpoint()
            if label:
                Shared.log(f"Partial results saved to {out_path} ({label})")

        def make_save(key, stage_key=None):
            def _save(partial):
                store.update_section(key, partial, stage_key or key)
            return _save

        _checkpoint("run started")
        signal.signal(signal.SIGINT,  _cleanup)
        signal.signal(signal.SIGTERM, _cleanup)

        lifecycle = LifecycleCoordinator(
            engine, engine_name, _engines, get_engine, Shared.shutdown_managed,
            Shared.comfyui_available, ImageBenchmark.comfyui_free_models,
        )
        context = RunContext(
            plan, RunPaths(Path(out_path), comfyui_dir), engine, store, lifecycle,
        )

        def run_llm(_context):
            return run_supervised_llm(
                _context.plan, event_store_path(Path(out_path)), make_save("llm"),
                resume_identity=resume_identity,
            )

        def run_conversation(_context):
            return run_supervised_stage(
                _context.plan, event_store_path(Path(out_path)), "conv",
                make_save("llm_conversation", "conv"), resume_identity=resume_identity,
            )

        def release_port_for_runner(_context):
            """A runner is a separate process and cannot stop this one's server, so a
            server left up here would answer its requests instead — see docs/engines.md."""
            if engine.available():
                engine.stop()

        def run_llamabench(_context):
            return run_supervised_stage(
                _context.plan, event_store_path(Path(out_path)), "llamabench",
                make_save("llamabench"), resume_identity=resume_identity,
            )

        def run_llamabench_concurrency(_context):
            return LlamaBenchConcurrencyBenchmark().run(
                engine=engine, models=llm_models, cpu_only=_context.plan.cpu_only,
                save_fn=make_save("llamabenchconc"),
            )

        def run_vllmbench(_context):
            return VllmBenchBenchmark().run(
                engine=engine, models=llm_models, save_fn=make_save("vllmbench"),
            )

        def run_embeddings(_context):
            return EmbeddingBenchmark().run(
                engine=engine, models=embedding_models, warmup_runs=_context.plan.warmup_runs,
                save_fn=make_save("embeddings", "emb"),
            )

        def accuracy_stage(test_name, Bench):
            def runner(_context):
                questions = Bench.load_questions()
                if args.sample is not None:
                    questions = Shared.stratified_sample(questions, args.sample)
                    results["sample_ids"][test_name] = [q["id"] for q in questions]
                answers_path = sidecar_path(_context.paths.output_path, f"answers_{test_name}_")
                section = Bench().run(
                    engine=engine, models=llm_models, questions=questions,
                    warmup_runs=_context.plan.warmup_runs, save_fn=make_save(test_name),
                    answers_path=answers_path,
                )
                Shared.ok(f"Answers saved to: {answers_path}")
                return section
            return StageDefinition(
                test_name, test_name, len(llm_models), runner, requires_engine=True,
            )

        def concurrency_stage(key, section):
            def runner(_context):
                if not conc_models:
                    Shared.warn(f"No downloaded models to test — {key} test will have nothing to run")
                return run_supervised_stage(
                    _context.plan, event_store_path(Path(out_path)), key,
                    make_save(section, key), resume_identity=resume_identity,
                )
            return StageDefinition(key, section, len(conc_models), runner,
                                   prepare=release_port_for_runner)

        def prepare_images(_context):
            lifecycle.restore_gpu()
            lifecycle.stop_engine()

        def run_images(_context):
            if not Shared.ensure_comfyui(_context.paths.comfyui_dir):
                Shared.warn("Image benchmarks will be skipped")
                return {}
            out_stem = _context.paths.output_path.stem
            images_name = ("images_" + out_stem[len("results_"):]
                           if out_stem.startswith("results_") else f"images_{out_stem}")
            return ImageBenchmark().run(
                image_models=image_models, resolutions=config.IMAGE_RESOLUTIONS,
                seed=config.IMAGE_SEED, prompt=config.IMAGE_PROMPT,
                comfyui_dir=_context.paths.comfyui_dir, timeout=config.RUN_TIMEOUT * 2,
                save_fn=make_save("images", "img"),
                images_dir=config.RESULTS_DIR / images_name,
            )

        registry = [
            StageDefinition("llm", "llm", len(llm_models), run_llm,
                            prepare=release_port_for_runner),
            StageDefinition("conv", "llm_conversation", len(llm_models), run_conversation,
                            requires_engine=False, prepare=release_port_for_runner),
            StageDefinition("llamabench", "llamabench", len(llm_models), run_llamabench,
                            prepare=release_port_for_runner),
            StageDefinition("llamabenchconc", "llamabenchconc", len(llm_models),
                            run_llamabench_concurrency, prepare=release_port_for_runner),
            StageDefinition("vllmbench", "vllmbench", len(llm_models), run_vllmbench,
                            prepare=release_port_for_runner),
            StageDefinition("emb", "embeddings", len(embedding_models), run_embeddings,
                            requires_engine=True),
            accuracy_stage("mcq", MCQBenchmark), accuracy_stage("math", MathBenchmark),
            accuracy_stage("reasoning", ReasoningBenchmark),
            accuracy_stage("code", CodeBenchmark), accuracy_stage("tool", ToolBenchmark),
            concurrency_stage("conc_tool", "concurrency_tool"),
            concurrency_stage("conc_chat", "concurrency_chat"),
            StageDefinition("img", "images", len(image_models), run_images,
                            prepare=prepare_images, cleanup=lambda _: Shared.shutdown_managed()),
        ]
        selected_stages = select_stages(registry, context.plan.stage_order)

        def run_selected_stages():
            if engine_backed_tests:
                Shared.section("Starting Servers")
                if not lifecycle.prepare_engine(args.cpu_only):
                    Shared.err("Failed to start the selected inference engine")
            execute_stages(context, selected_stages)
            lifecycle.restore_gpu()

        try:
            execute_with_final_cleanup(run_selected_stages, lifecycle)
        except BaseException as exc:
            terminal = "interrupted" if results["run"].get("status") == "interrupted" \
                or isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
            try:
                finish_event_job(
                    event_store_path(Path(out_path)), plan, terminal, type(exc).__name__,
                )
            except Exception as journal_exc:
                Shared.err(f"Failed to terminalize event journal: {journal_exc}")
            checkpoint_terminal_exception(results, exc, _checkpoint)
            raise

        # ── Save results ───────────────────────────────────────────────────────────
        Shared.section("Saving Results")
        finish_event_job(event_store_path(Path(out_path)), plan, "complete")
        apply_pause_evidence(results["run"])
        store.finish("complete")
        Shared.ok(f"Results saved to: {out_path}")

    Shared.output("  Compare it against other machines in the dashboard:", leading_blank=True)
    dash_hint = "launch_dashboard.bat" if platform.system() == "Windows" else "bash launch_dashboard.sh"
    Shared.output(f"  {dash_hint}")
    Shared.section("Done")
    Shared.ok("All servers shut down. Benchmark complete.")

if __name__ == "__main__":
    main()
