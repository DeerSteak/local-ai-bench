#!/usr/bin/env python3
"""
benchmark.py — Cross-platform LLM benchmark suite.

Tests: llm, conv, img, emb, mcq, math, reasoning, code, tool, conc_tool, conc_chat
(conc_tool/conc_chat both opt-in) — see docs/workloads.md for what each
measures. Servers start/stop
automatically — see docs/how-it-works.md.

Usage:
  python benchmark.py                  # run all tests except conc_tool/conc_chat
  python benchmark.py --tests llm      # run only LLM single-shot tests
  python benchmark.py --help           # full flag reference
"""

import argparse
import fnmatch
import json
import platform
import re
import signal
import sys
from datetime import datetime
from pathlib import Path

import config
from shared import Shared
from engines import get_engine, engine_names as registered_engine_names
from llm_prefill_benchmark import LLMPrefillBenchmark
from llm_conversation_benchmark import LLMConversationBenchmark
from embedding_benchmark import EmbeddingBenchmark
from image_benchmark import ImageBenchmark
from mcq_benchmark import MCQBenchmark
from math_benchmark import MathBenchmark
from reasoning_benchmark import ReasoningBenchmark
from code_benchmark import CodeBenchmark
from tool_benchmark import ToolBenchmark
from concurrency_benchmark import ConcurrencyBenchmark
from models import IMAGE_MODELS, LLM_MODELS_XSMALL, LLM_MODELS_SMALL, LLM_MODELS_MEDIUM, LLM_MODELS_LARGE, LLM_MODELS, EMBED_MODELS
from model_inventory import build_model_inventory, format_model_inventory, sanitize_tag_to_short


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


def select_tier(maxtier: str | None, image_models: list) -> tuple[list, str, list]:
    """Resolve --maxtier into (llm_models, tier_label, image_models), applying
    the same cumulative cap to both LLM tiers and image-model tiers. No cap
    (maxtier=None) means every tier."""
    if maxtier:
        llm_models = TIER_MODELS[maxtier]
        tier_label = TIER_LABELS[maxtier]
        max_idx = TIER_ORDER.index(maxtier)
        image_models = [m for m in image_models if TIER_ORDER.index(m["tier"]) <= max_idx]
    else:
        llm_models = LLM_MODELS
        tier_label = "all (extra-small + small + medium + large)"
    return llm_models, tier_label, image_models


def filter_models_by_pattern(models: list, patterns: list[str] | None, key: str = "tag") -> list:
    """Filter models by exact or shell-style wildcard matches on `key`.
    Case-sensitive (`fnmatchcase`) so behavior is identical across platforms
    (plain `fnmatch` case-normalizes on Windows only)."""
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
    """Filter `catalog` down to entries whose tag is actually downloaded
    locally (per `installed_tags`, from LlamaCppEngine.list_installed_models),
    preserving catalog order. Used by the concurrency tests, which scale to
    whatever's on the machine — small hardware that only downloaded
    xsmall/small models tests those; a machine with medium/large downloaded
    tests those too — rather than a fixed tier cap like --maxtier."""
    installed = set(installed_tags)
    return [m for m in catalog if m["tag"] in installed]


def resolve_model_scopes(tier_models: list[dict], installed_tags: list[str],
                         patterns: list[str] | None, concurrency_enabled: bool
                         ) -> tuple[list[dict], list[dict]]:
    """Resolve normal and concurrency model scopes for one engine's local
    model inventory. Concurrency ignores the tier cap but still honors
    --llm-models/--models; normal workloads retain the selected tier."""
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
LLM_TESTS = ["llm", "conv", *ACCURACY_TESTS]


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
            tests, engine_name, llm_patterns, llm_models, concurrency_models, tier_label,
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
    """Expand shorthand groups (see TEST_GROUPS) in --tests into their
    underlying individual test names, preserving order and de-duplicating so
    e.g. --tests acc mcq doesn't run the MCQ benchmark twice."""
    expanded = []
    for t in tests:
        for name in TEST_GROUPS.get(t, [t]):
            if name not in expanded:
                expanded.append(name)
    return expanded


def resolve_engine_names(engine: str, available: list[str]) -> list[str]:
    """Resolve --engine into the ordered list of engine names to run this
    pass over: "all" expands to every registered engine (sorted, so the run
    order is deterministic across invocations); anything else is a single
    engine name, passed through as-is (argparse's `choices` already rejects
    an unregistered one before this is called)."""
    return list(available) if engine == "all" else [engine]


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


def conv_skip_entry(model: dict, llm_data: dict | None, first_ctx_label: str, force_all: bool) -> dict | None:
    """Decide whether `model` should be skipped from the (expensive) conversation
    test, based on how it did in the single-shot LLM prefill test. Returns a
    skip-result dict (the schema written into results["llm_conversation"]) if
    it should be skipped, or None if it should proceed to the conversation test.
    """
    label = model["label"]

    if not llm_data:
        detail = "no LLM benchmark data (checkpoint skipped or model failed)"
        return {"label": label, "skipped": True,
                "skip_reason": "no_llm_data", "skip_detail": detail}

    if llm_data.get("skipped") or llm_data.get("crashed"):
        detail = llm_data.get("skip_detail") or (
            f"The engine's runner crashed repeatedly during the LLM test "
            f"(at {llm_data['crashed']} context)"
        )
        return {"label": label, "skipped": True,
                "skip_reason": llm_data.get("skip_reason", "known_crash"), "skip_detail": detail}

    if llm_data.get("timed_out") == first_ctx_label:
        detail = f"LLM test timed out at {llm_data['timed_out']} context"
        return {"label": label, "skipped": True,
                "skip_reason": "timed_out", "skip_detail": detail}

    # A timeout at a deeper context doesn't disqualify the model — it passed
    # the first prefill checkpoint, so fall through to the tok/s check below.
    slow_ctx = None if force_all else llm_data.get("slow_tps") or (
        first_ctx_label if isinstance(llm_data.get(first_ctx_label), dict)
        and llm_data[first_ctx_label].get("tps_mean") is not None
        and llm_data[first_ctx_label]["tps_mean"] < config.SLOW_MODEL_MIN_TPS
        else None
    )
    if slow_ctx is not None:
        ctx_data = llm_data.get(slow_ctx)
        detail = (f"{ctx_data['tps_mean']:.1f} tok/s at {slow_ctx} "
                  f"context (below {config.SLOW_MODEL_MIN_TPS:.0f} tok/s cutoff)"
                  if isinstance(ctx_data, dict) and ctx_data.get("tps_mean") is not None
                  else f"below {config.SLOW_MODEL_MIN_TPS:.0f} tok/s cutoff at {slow_ctx} context")
        return {"label": label, "skipped": True,
                "skip_reason": "slow_tps", "skip_detail": detail}

    return None


def main():  # pragma: no cover — CLI entrypoint; orchestrates real llama.cpp/ComfyUI runs
    parser = argparse.ArgumentParser(description="LLM benchmark suite")
    parser.add_argument(
        "--tests", nargs="+",
        choices=["llm", "conv", "emb", "img", "mcq", "math", "reasoning", "code", "tool", "acc",
                 "conc_tool", "conc_chat", "conc"],
        default=["llm", "conv", "emb", "img", "mcq", "math", "reasoning", "code", "tool"],
        help="Which benchmarks to run (default: all except the concurrency "
             "tests). 'acc' is shorthand for every accuracy-style test "
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
             "(disable via --force-all). 'conc' is shorthand for both.",
    )
    parser.add_argument(
        "--warmup", type=int, default=config.WARMUP_RUNS,
        help=f"Warmup runs before measuring (default: {config.WARMUP_RUNS})",
    )
    parser.add_argument(
        "--runs", type=int, default=config.N_RUNS, choices=range(1, 11),
        metavar="[1-10]",
        help=f"Measured runs per checkpoint for single-shot LLM, embeddings, and "
             f"images, averaged (default: {config.N_RUNS}). Conversation, accuracy, "
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
        "--acc-token-budget", type=positive_int, default=None, metavar="N",
        help="Total completion-token budget per accuracy question, split 60/40 "
             f"between the initial and final-answer passes (default: {config.ACC_TOKEN_BUDGET})",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output JSON file (default: results/results_<hostname>_<timestamp>.json)",
    )
    parser.add_argument(
        "--comfyui", type=str, default=None,
        help=f"Path to ComfyUI directory (default: {config.COMFYUI_DIR})",
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
             "instability can hit LLM/MCQ inference on some backends too).",
    )
    parser.add_argument(
        "--maxtier", type=str, default=None,
        choices=["xsmall", "small", "medium", "large"],
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
    _engines = registered_engine_names()
    parser.add_argument(
        "--engine", type=str, default=_engines[0], choices=_engines + ["all"],
        help=f"Inference engine to benchmark against (default: {_engines[0]}). "
             "'all' runs the full --tests suite once per registered engine, back "
             "to back (sorted order), writing a separate results file for each "
             "(engine name appended to the filename) so they can be compared "
             "directly. Only llama.cpp is registered today, so this is a no-op "
             "until a second engine (e.g. MLX) is added — kept here so scripts/"
             "docs referencing --engine don't need to change when one is.",
    )
    args = parser.parse_args()

    args.tests = expand_tests(args.tests)
    comfyui_dir = Path(args.comfyui) if args.comfyui else config.COMFYUI_DIR
    run_engine_names = resolve_engine_names(args.engine, _engines)

    if args.list_models:
        any_installed = False
        for engine_name in run_engine_names:
            inventory = build_model_inventory(get_engine(engine_name), comfyui_dir)
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

        if multi_engine:
            Shared.section(f"Engine: {engine_name} ({run_idx + 1}/{len(run_engine_names)})")
            _base = Path(base_out_path)
            out_path = str(_base.with_name(f"{_base.stem}_{engine_name}{_base.suffix}"))
        else:
            out_path = base_out_path

        # Image generation doesn't depend on --engine (separate ComfyUI call) — run it once, first pass only.
        tests = args.tests
        if multi_engine and run_idx > 0 and "img" in tests:
            Shared.log("Image generation doesn't depend on --engine — already "
                       f"captured in the {run_engine_names[0]} pass, skipping for {engine_name}")
            tests = [t for t in tests if t != "img"]

        engine_backed_tests = [
            t for t in ("llm", "conv", "mcq", "math", "reasoning", "code", "tool", "emb",
                        "conc_tool", "conc_chat") if t in tests
        ]
        hardware_backend = hardware_profile["backend"]
        profile = {
            **hardware_profile,
            "hardware_backend": hardware_backend,
            "backend": (engine.runtime_backend(hardware_backend, cpu_only=args.cpu_only)
                        if engine_backed_tests else hardware_backend),
        }

        Shared.output(f"{config.BOLD}LLM Benchmark Suite{config.RESET}", leading_blank=True)
        Shared.output(f"  Host:      {profile['hostname']}")
        Shared.output(f"  OS:        {profile['os']}")
        Shared.output(f"  Backend:   {profile['backend']}")
        if profile["backend"] != profile["hardware_backend"]:
            Shared.output(f"  Hardware:  {profile['hardware_backend']}")
        Shared.output(f"  RAM:       {profile['ram_gb']} GB")
        Shared.output(f"  Engine:    {engine_name}")
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
                sys.exit(0)

        signal.signal(signal.SIGINT,  _cleanup)
        signal.signal(signal.SIGTERM, _cleanup)

        results = {
            "version":         config.VERSION,
            "engine":          engine_name,
            "profile":         profile,
            "accuracy_settings": {
                "timeout_seconds": config.ACC_TIMEOUT,
                "token_budget": config.ACC_TOKEN_BUDGET,
                "first_pass_fraction": config.ACC_FINALIZE_FRACTION,
            },
            # Fingerprints of the accuracy question banks actually used for this
            # run, so a raw correct count is never compared across bank sizes
            # (e.g. 185 vs. 360 questions) without noticing the version differs.
            "bank_versions": {
                "mcq":  Shared.file_hash(MCQBenchmark.MCQ_DATA_PATH),
                "math": Shared.file_hash(MathBenchmark.MATH_DATA_PATH),
                "reasoning": Shared.file_hash(ReasoningBenchmark.REASONING_DATA_PATH),
                "code": Shared.file_hash(CodeBenchmark.CODE_DATA_PATH),
                "tool": Shared.file_hash(ToolBenchmark.TOOL_DATA_PATH),
            },
            # Populated only when --sample is used, with the exact question IDs
            # drawn from each bank — so a dev-mode run is reproducible/auditable
            # and never mistaken for a full-bank result.
            "sample_ids": {},
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
        }

        def _checkpoint(label=""):
            Path(out_path).write_text(json.dumps(results, indent=2, allow_nan=False))
            if label:
                Shared.log(f"Partial results saved to {out_path} ({label})")

        try:
            # ── LLM-backed tests share one server lifecycle
            llm_tests = engine_backed_tests
            if llm_tests:
                Shared.section("Starting Servers")
                for other_name in _engines:
                    if other_name == engine_name:
                        continue
                    other_engine = get_engine(other_name)
                    if other_engine.available():
                        Shared.log(f"Stopping {other_name} so only one inference "
                                   f"engine runs at a time ...")
                        other_engine.stop()
                if args.cpu_only:
                    Shared.warn("Stopping the engine to relaunch in CPU-only mode "
                                f"(applies to: {', '.join(llm_tests)}) ...")
                    engine.stop()
                    if not engine.start(gpu_visible=False):
                        Shared.err("Failed to start the engine in CPU-only mode — "
                                   f"{', '.join(llm_tests)} tests will be skipped")
                else:
                    engine.ensure_running()

            # ── LLM ───────────────────────────────────────────────────────────────
            if "llm" in tests:
                def _llm_save(partial):
                    results["llm"] = partial
                    _checkpoint()

                results["llm"] = LLMPrefillBenchmark().run(
                    engine=engine,
                    models=llm_models,
                    context_lengths=config.CONTEXT_LENGTHS,
                    warmup_runs=args.warmup,
                    force_all=args.force_all,
                    save_fn=_llm_save,
                )
                _checkpoint("LLM done")

            if "conv" in tests:
                conv_models = llm_models
                llm_conv_skips = {}
                if "llm" in tests:
                    conv_models = []
                    first_ctx_label = Shared.context_label(config.CONTEXT_LENGTHS[0])
                    for model in llm_models:
                        short = model["short"]
                        llm_data = results["llm"].get(short)
                        skip_entry = conv_skip_entry(model, llm_data, first_ctx_label, args.force_all)
                        if skip_entry is not None:
                            Shared.warn(f"{model['label']}: skipping conversation test — {skip_entry['skip_detail']}")
                            llm_conv_skips[short] = skip_entry
                            continue
                        conv_models.append(model)

                def _conv_save(partial):
                    results["llm_conversation"] = partial
                    _checkpoint()

                results["llm_conversation"] = LLMConversationBenchmark().run(
                    engine=engine,
                    models=conv_models,
                    warmup_runs=args.warmup,
                    force_all=args.force_all,
                    save_fn=_conv_save,
                )
                results["llm_conversation"].update(llm_conv_skips)
                _checkpoint("LLM conversation done")

            # ── Accuracy tests (MCQ / Math / Code / Tool) ─────────────────────────
            # Identical wiring for all five — only the test name, benchmark class,
            # and display label vary.
            for test_name, Bench, done_label in (
                ("mcq", MCQBenchmark, "MCQ"), ("math", MathBenchmark, "Math"),
                ("reasoning", ReasoningBenchmark, "Reasoning"),
                ("code", CodeBenchmark, "Code"), ("tool", ToolBenchmark, "Tool"),
            ):
                if test_name not in tests:
                    continue

                def _save(partial, test_name=test_name):
                    results[test_name] = partial
                    _checkpoint()

                questions = Bench.load_questions()
                if args.sample is not None:
                    questions = Shared.stratified_sample(questions, args.sample)
                    results["sample_ids"][test_name] = [q["id"] for q in questions]

                answers_path = sidecar_path(out_path, f"answers_{test_name}_")
                results[test_name] = Bench().run(
                    engine=engine,
                    models=llm_models,
                    questions=questions,
                    warmup_runs=args.warmup,
                    save_fn=_save,
                    answers_path=answers_path,
                )
                _checkpoint(f"{done_label} done")
                Shared.ok(f"Answers saved to: {answers_path}")

            # ── Embeddings ─────────────────────────────────────────────────────────
            if "emb" in tests:
                def _emb_save(partial):
                    results["embeddings"] = partial
                    _checkpoint()

                results["embeddings"] = EmbeddingBenchmark().run(
                    engine=engine,
                    models=embedding_models,
                    warmup_runs=args.warmup,
                    save_fn=_emb_save,
                )
                _checkpoint("embeddings done")

            # ── Concurrency: tool-style (agentic fan-out, no early exit) ───────────
            if "conc_tool" in tests:
                def _conc_tool_save(partial):
                    results["concurrency_tool"] = partial
                    _checkpoint()

                if not conc_models:
                    Shared.warn("No downloaded models to test — "
                                "conc_tool test will have nothing to run")

                results["concurrency_tool"] = ConcurrencyBenchmark().run(
                    engine=engine,
                    models=conc_models,
                    levels=config.CONCURRENCY_TOOL_LEVELS,
                    per_request_context=config.CONCURRENCY_TOOL_CONTEXT,
                    warmup_runs=args.warmup,
                    crash_cache_path=ConcurrencyBenchmark.TOOL_CRASH_CACHE,
                    section_label="Concurrency (Tool)",
                    soft_exit_floor=None,
                    force_all=args.force_all,
                    save_fn=_conc_tool_save,
                )
                _checkpoint("concurrency (tool) done")

            # ── Concurrency: chat-server (many simultaneous users, soft exit) ──────
            if "conc_chat" in tests:
                def _conc_chat_save(partial):
                    results["concurrency_chat"] = partial
                    _checkpoint()

                if not conc_models:
                    Shared.warn("No downloaded models to test — "
                                "conc_chat test will have nothing to run")

                results["concurrency_chat"] = ConcurrencyBenchmark().run(
                    engine=engine,
                    models=conc_models,
                    levels=config.CONCURRENCY_CHAT_LEVELS,
                    per_request_context=config.CONCURRENCY_CHAT_CONTEXT,
                    warmup_runs=args.warmup,
                    crash_cache_path=ConcurrencyBenchmark.CHAT_CRASH_CACHE,
                    section_label="Concurrency (Chat)",
                    soft_exit_floor=config.CONCURRENCY_CHAT_MIN_LEVEL_BEFORE_SOFT_EXIT,
                    force_all=args.force_all,
                    save_fn=_conc_chat_save,
                )
                _checkpoint("concurrency (chat) done")

            # Done with every LLM-backed test — restore normal GPU-enabled mode if
            # this run forced CPU-only, so the machine isn't left in that state
            # (and so image generation, if it runs next, starts from a clean state).
            if llm_tests and args.cpu_only and engine._cpu_only_active:
                Shared.log("Restoring normal (GPU-enabled) engine ...")
                engine.stop()
                engine.start()

            # ── Image generation ───────────────────────────────────────────────────
            if "img" in tests:
                Shared.section("Starting Servers")
                # Kill the whole server, not just unload its models, to free memory the idle process itself still holds.
                if engine.available():
                    Shared.log("Stopping the engine entirely to free memory for ComfyUI ...")
                    engine.stop()
                comfyui_started = Shared.ensure_comfyui(comfyui_dir)
                if not comfyui_started:
                    Shared.warn("Image benchmarks will be skipped")
                else:
                    def _img_save(img_partial):
                        results["images"] = img_partial
                        _checkpoint()

                    # Same hostname+timestamp as the results JSON, so both can be
                    # grabbed together in a file browser — a sibling folder under
                    # results/, not nested inside a shared "images" folder.
                    _out_stem = Path(out_path).stem
                    _images_dirname = (
                        "images_" + _out_stem[len("results_"):]
                        if _out_stem.startswith("results_") else f"images_{_out_stem}"
                    )

                    results["images"] = ImageBenchmark().run(
                        image_models=image_models,
                        resolutions=config.IMAGE_RESOLUTIONS,
                        seed=config.IMAGE_SEED,
                        prompt=config.IMAGE_PROMPT,
                        comfyui_dir=comfyui_dir,
                        timeout=config.RUN_TIMEOUT * 2,
                        save_fn=_img_save,
                        images_dir=config.RESULTS_DIR / _images_dirname,
                    )
                    # Shut down ComfyUI as soon as image tests are done
                    # to free GPU memory before saving results
                    Shared.shutdown_managed()

        finally:
            # Always shut down anything still running, even on error
            Shared.shutdown_managed()

        # ── Save results ───────────────────────────────────────────────────────────
        Shared.section("Saving Results")
        Path(out_path).write_text(json.dumps(results, indent=2, allow_nan=False))
        Shared.ok(f"Results saved to: {out_path}")

    Shared.output("  Compare it against other machines in the dashboard:", leading_blank=True)
    dash_hint = "launch_dashboard.bat" if platform.system() == "Windows" else "bash launch_dashboard.sh"
    Shared.output(f"  {dash_hint}")
    Shared.section("Done")
    Shared.ok("All servers shut down. Benchmark complete.")

if __name__ == "__main__":
    main()
