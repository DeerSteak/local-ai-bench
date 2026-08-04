#!/usr/bin/env python3
"""Cross-platform LLM benchmark suite — CLI entry point.
See docs/workloads.md for what each test measures, docs/cli-reference.md for flags."""

import argparse
import fnmatch
import platform
import re
import signal
import sys
from datetime import datetime
from pathlib import Path

import config
from benchmark_options import TEST_CHOICES, TG_TOKEN_CHOICES, TIER_CHOICES, option_value_errors
from comfyui_installation import find_comfyui_installation, normalize_comfyui_dir
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
from llamabench_benchmark import LlamaBenchBenchmark
from llamabench_concurrency_benchmark import LlamaBenchConcurrencyBenchmark
from models import IMAGE_MODELS, LLM_MODELS_XSMALL, LLM_MODELS_SMALL, LLM_MODELS_MEDIUM, LLM_MODELS_LARGE, LLM_MODELS, EMBED_MODELS
from model_inventory import build_model_inventory, format_model_inventory, sanitize_tag_to_short
from orchestration import (
    LifecycleCoordinator, RunContext, RunPaths, StageDefinition,
    StageExecutionError, execute_stages, execute_with_final_cleanup,
    ordered_stage_keys, select_stages,
)
from result_store import (ResultStore, atomic_write_json, build_run_manifest, finish_run,
                          finish_active_stage, model_identity)
from run_plan import RunPlan
from setup_config import configured_comfyui_dir, load_setup_config


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
LLM_TESTS = ["llm", "conv", *ACCURACY_TESTS, "llamabench", "llamabenchconc"]


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
    """Expand TEST_GROUPS shorthand in --tests, preserving order and de-duplicating."""
    expanded = []
    for t in tests:
        for name in TEST_GROUPS.get(t, [t]):
            if name not in expanded:
                expanded.append(name)
    return expanded


def resolve_engine_names(engine: str, available: list[str]) -> list[str]:
    """Resolve --engine into an ordered engine-name list; "all" expands to
    every registered engine, sorted for a deterministic run order."""
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
    """Conversation-test skip decision from single-shot LLM results — see
    docs/workloads.md's LLM pre-flight section."""
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
    run_engine_names = resolve_engine_names(args.engine, _engines)

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
                if results.get("run", {}).get("status") == "running":
                    finish_active_stage(results["run"], "interrupted", "signal")
                    finish_run(results["run"], "interrupted", "signal")
                    try:
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
                sys.exit(0)

        stage_order = ordered_stage_keys(tuple(tests))
        effective_config = {
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
        }
        plan_models = {
            "llm": model_identity(llm_models),
            "concurrency": model_identity(conc_models),
            "embeddings": model_identity(embedding_models),
            "images": model_identity(image_models),
        }
        plan = RunPlan.create(
            application_version=config.VERSION, engine_name=engine_name,
            tests=tests, stage_order=stage_order, models=plan_models,
            effective_config=effective_config,
        )

        results = {
            "version":         config.VERSION,
            "engine":          engine_name,
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
        }

        results["run"] = build_run_manifest(
            plan=plan, repo_root=config.SCRIPT_DIR,
        )

        store = ResultStore(Path(out_path), results)

        def _checkpoint(label=""):
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
            plan, RunPaths(Path(out_path), comfyui_dir), engine, store, lifecycle, profile,
        )

        def run_llm(_context):
            return LLMPrefillBenchmark().run(
                engine=engine, models=llm_models, context_lengths=config.CONTEXT_LENGTHS,
                warmup_runs=_context.plan.warmup_runs,
                force_all=_context.plan.force_all, save_fn=make_save("llm"),
            )

        def run_conversation(_context):
            conv_models = llm_models
            skips = {}
            if "llm" in _context.plan.tests:
                conv_models = []
                first_ctx = Shared.context_label(config.CONTEXT_LENGTHS[0])
                for model in llm_models:
                    skip = conv_skip_entry(
                        model, results["llm"].get(model["short"]), first_ctx,
                        _context.plan.force_all,
                    )
                    if skip is None:
                        conv_models.append(model)
                    else:
                        Shared.warn(f"{model['label']}: skipping conversation test — {skip['skip_detail']}")
                        skips[model["short"]] = skip
            section = LLMConversationBenchmark().run(
                engine=engine, models=conv_models, warmup_runs=_context.plan.warmup_runs,
                force_all=_context.plan.force_all,
                save_fn=make_save("llm_conversation", "conv"),
                max_prompt_tokens=args.max_prompt_tokens,
            )
            section.update(skips)
            return section

        def stop_for_native(_context):
            if engine.available():
                engine.stop()

        def run_llamabench(_context):
            return LlamaBenchBenchmark().run(
                engine=engine, models=llm_models, reps=config.N_RUNS,
                cpu_only=_context.plan.cpu_only, save_fn=make_save("llamabench"),
            )

        def run_llamabench_concurrency(_context):
            return LlamaBenchConcurrencyBenchmark().run(
                engine=engine, models=llm_models, cpu_only=_context.plan.cpu_only,
                save_fn=make_save("llamabenchconc"),
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

        def concurrency_stage(key, section, levels, per_context, cache, label, floor):
            def runner(_context):
                if not conc_models:
                    Shared.warn(f"No downloaded models to test — {key} test will have nothing to run")
                return ConcurrencyBenchmark().run(
                    engine=engine, models=conc_models, levels=levels,
                    per_request_context=per_context, warmup_runs=_context.plan.warmup_runs,
                    crash_cache_path=cache, section_label=label, soft_exit_floor=floor,
                    force_all=_context.plan.force_all, save_fn=make_save(section, key),
                )
            return StageDefinition(key, section, len(conc_models), runner, requires_engine=True)

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
            StageDefinition("llm", "llm", len(llm_models), run_llm, requires_engine=True),
            StageDefinition("conv", "llm_conversation", len(llm_models), run_conversation,
                            requires_engine=True),
            StageDefinition("llamabench", "llamabench", len(llm_models), run_llamabench,
                            prepare=stop_for_native),
            StageDefinition("llamabenchconc", "llamabenchconc", len(llm_models),
                            run_llamabench_concurrency, prepare=stop_for_native),
            StageDefinition("emb", "embeddings", len(embedding_models), run_embeddings,
                            requires_engine=True),
            accuracy_stage("mcq", MCQBenchmark), accuracy_stage("math", MathBenchmark),
            accuracy_stage("reasoning", ReasoningBenchmark),
            accuracy_stage("code", CodeBenchmark), accuracy_stage("tool", ToolBenchmark),
            concurrency_stage(
                "conc_tool", "concurrency_tool", config.CONCURRENCY_TOOL_LEVELS,
                config.CONCURRENCY_TOOL_CONTEXT, ConcurrencyBenchmark.TOOL_CRASH_CACHE,
                "Concurrency (Tool)", None,
            ),
            concurrency_stage(
                "conc_chat", "concurrency_chat", config.CONCURRENCY_CHAT_LEVELS,
                config.CONCURRENCY_CHAT_CONTEXT, ConcurrencyBenchmark.CHAT_CRASH_CACHE,
                "Concurrency (Chat)", config.CONCURRENCY_CHAT_MIN_LEVEL_BEFORE_SOFT_EXIT,
            ),
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
            checkpoint_terminal_exception(results, exc, _checkpoint)
            raise

        # ── Save results ───────────────────────────────────────────────────────────
        Shared.section("Saving Results")
        store.finish("complete")
        Shared.ok(f"Results saved to: {out_path}")

    Shared.output("  Compare it against other machines in the dashboard:", leading_blank=True)
    dash_hint = "launch_dashboard.bat" if platform.system() == "Windows" else "bash launch_dashboard.sh"
    Shared.output(f"  {dash_hint}")
    Shared.section("Done")
    Shared.ok("All servers shut down. Benchmark complete.")

if __name__ == "__main__":
    main()
