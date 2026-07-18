#!/usr/bin/env python3
"""
benchmark.py — Cross-platform LLM benchmark suite.

Tests:
  1. LLM generation — 12 models across xsmall/small/medium/large tiers via Ollama
     Metrics: time-to-first-token (TTFT), tokens/sec
     Context lengths: 2K, 8K, 32K, 64K
     Models that exceed the warmup timeout are skipped automatically

  1b. LLM conversation — same models, but via a real multi-turn chat
      (/api/chat) instead of one padded single-shot prompt: the model
      explains Plato's Allegory of the Cave in sections, then each turn asks
      for more detail. TTFT/tokens-per-sec at each depth reflect processing a
      new turn against an already-filled context (llama.cpp's slot prefix
      cache), not a cold fill. Expensive, so it always runs a single
      conversation (--runs ignored), grown from scratch toward 96K, sampled at
      0, 2K, 4K, 8K, 16K, 32K, 64K, and 96K (whichever the model's real ceiling
      reaches). The model gets the full 128K window (or its real max) so 96K
      is reached with headroom.

  2. Image generation — SD1.5, SDXL, SD3.5 Large, Flux.1-dev, Flux.2-dev via
     ComfyUI HTTP API
     Metrics: seconds/image at 1024×1024 and 1536×1536 (SD1.5: 512×512, 768×768)
     (models skipped automatically if checkpoint not found)

  3. Embeddings — nomic-embed-text and mxbai-embed-large via Ollama
     Metrics: chunks/sec chunking one real document and embedding it in a
     single call

  4. MCQ accuracy — every LLM model answers the multiple-choice question
     bank (scripts/data/mcq_questions.json) once via Ollama's /api/chat
     Metrics: overall and per-category accuracy (% correct)

  5. Math accuracy — every LLM model answers the math word-problem bank
     (scripts/data/math_questions.json) once via Ollama's /api/chat
     Metrics: overall and per-category accuracy (% correct, within each
     question's own numeric tolerance)

  6. Code accuracy — every LLM model answers the coding-problem bank
     (scripts/data/code_problems.json) once via Ollama's /api/chat, and the
     generated function is run against each problem's test cases in an
     isolated subprocess
     Metrics: overall and per-category accuracy (% of problems where every
     test case passed)

Servers are managed automatically:
  - Ollama: started if not already running, shut down on exit if we started it
  - ComfyUI: started before image tests, shut down cleanly when done

Usage:
  python benchmark.py                  # run all tests
  python benchmark.py --tests llm      # run only LLM single-shot tests
  python benchmark.py --tests llm emb  # run LLM + embeddings
  python benchmark.py --tests conv     # run only LLM conversation tests
  python benchmark.py --comfyui /path/to/ComfyUI  # override ComfyUI path
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
from engines import get_engine
from llm_prefill_benchmark import LLMPrefillBenchmark
from llm_conversation_benchmark import LLMConversationBenchmark
from embedding_benchmark import EmbeddingBenchmark
from image_benchmark import ImageBenchmark
from mcq_benchmark import MCQBenchmark
from math_benchmark import MathBenchmark
from code_benchmark import CodeBenchmark
from models import IMAGE_MODELS, LLM_MODELS_XSMALL, LLM_MODELS_SMALL, LLM_MODELS_MEDIUM, LLM_MODELS_LARGE, LLM_MODELS, EMBED_MODELS


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

# Ollama and llama-server both want the GPU to themselves — running two
# inference engines' worth of loaded weights at once skews every timing
# number either could produce. Whichever engine is about to start, the other
# gets stopped first (see the "Starting Servers" section in main()), even if
# this process didn't start it — mirrors OllamaEngine.stop()/LlamaCppEngine.stop()
# already killing a stray server they didn't launch themselves.
OTHER_ENGINE = {"ollama": "llamacpp", "llamacpp": "ollama"}


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


def filter_models_by_pattern(models: list, patterns: list[str] | None) -> list:
    """Filter `models` down to those whose tag matches any of `patterns` —
    each an exact Ollama tag or a shell-style wildcard (fnmatch), e.g. "llama*".
    Case-sensitive (`fnmatchcase`) so behavior is identical across platforms
    (plain `fnmatch` case-normalizes on Windows only). `patterns=None` or empty
    disables filtering, which is what makes --models optional."""
    if not patterns:
        return models
    return [m for m in models if any(fnmatch.fnmatchcase(m["tag"], p) for p in patterns)]


def strip_implicit_latest(tag: str) -> str:
    """Ollama's /api/tags always reports an explicit ":latest" for models
    pulled without a tag suffix, but models.py stores those same models under
    their bare tag ("phi4-mini", not "phi4-mini:latest"). Normalize before
    comparing an installed tag against the catalog so those aren't
    misidentified as "custom"."""
    return tag[:-len(":latest")] if tag.endswith(":latest") else tag


def sanitize_tag_to_short(tag: str) -> str:
    """Turn a raw Ollama tag ("qwen3.5:4b-instruct") into a filesystem/JSON-key
    -safe "short" identifier ("qwen3.5-4b-instruct"), mirroring the style of
    the hand-picked "short" values in models.py for catalog entries."""
    return re.sub(r'[:/]', '-', tag)


def resolve_custom_models(patterns: list[str], catalog: list[dict], installed_tags: list[str]) -> list[dict]:
    """Extend `filter_models_by_pattern`'s catalog-only matching so a pattern
    that matches nothing in the curated catalog (models.py) can still resolve
    to a model, as long as it's actually pulled in Ollama. Lets someone
    benchmark a self-installed model without adding it to the catalog first; it
    just runs without curated tier/label/params_b metadata.

    Only patterns with zero catalog matches fall through to the installed-tag
    lookup, so an existing wildcard that already matches curated models behaves
    exactly as before.
    """
    catalog_tags = {m["tag"] for m in catalog}
    resolved = list(filter_models_by_pattern(catalog, patterns))
    seen = {m["tag"] for m in resolved}

    for pattern in patterns:
        if any(fnmatch.fnmatchcase(t, pattern) for t in catalog_tags):
            continue  # already satisfied by the catalog match above
        for tag in installed_tags:
            if tag in seen or strip_implicit_latest(tag) in catalog_tags:
                continue
            if fnmatch.fnmatchcase(tag, pattern):
                resolved.append({"tag": tag, "label": f"{tag} (custom)", "short": sanitize_tag_to_short(tag)})
                seen.add(tag)

    return resolved


def sidecar_path(out_path: str, prefix: str) -> Path:
    """Build a sidecar file path alongside the main results JSON, swapping its
    "results_" stem prefix for `prefix` (or prepending `prefix` if the stem
    doesn't start with "results_", e.g. after --out) so hostname and timestamp
    stay identical between the two — same convention as the images_*/ folder
    (see docs/project-structure.md)."""
    stem = Path(out_path).stem
    name = prefix + stem[len("results_"):] if stem.startswith("results_") else f"{prefix}{stem}"
    return config.RESULTS_DIR / f"{name}.json"


ACCURACY_TESTS = ["mcq", "math", "code"]


def expand_tests(tests: list[str]) -> list[str]:
    """Expand shorthand groups (currently just "acc") in --tests into their
    underlying individual test names, preserving order and de-duplicating so
    e.g. --tests acc mcq doesn't run the MCQ benchmark twice."""
    expanded = []
    for t in tests:
        for name in (ACCURACY_TESTS if t == "acc" else [t]):
            if name not in expanded:
                expanded.append(name)
    return expanded


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
            f"Ollama's runner crashed repeatedly during the LLM test "
            f"(at {llm_data['crashed']} context)"
        )
        return {"label": label, "skipped": True,
                "skip_reason": llm_data.get("skip_reason", "known_crash"), "skip_detail": detail}

    if llm_data.get("timed_out") == first_ctx_label:
        detail = f"LLM test timed out at {llm_data['timed_out']} context"
        return {"label": label, "skipped": True,
                "skip_reason": "timed_out", "skip_detail": detail}

    # A timeout at a deeper context (8K/32K/64K) doesn't disqualify the model —
    # it passed the 2K prefill test, so fall through to the tok/s check below.
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


def main():  # pragma: no cover — CLI entrypoint; orchestrates real Ollama/ComfyUI runs
    parser = argparse.ArgumentParser(description="LLM benchmark suite")
    parser.add_argument(
        "--tests", nargs="+",
        choices=["llm", "conv", "emb", "img", "mcq", "math", "code", "acc"],
        default=["llm", "conv", "emb", "img", "mcq", "math", "code"],
        help="Which benchmarks to run (default: all). 'acc' is shorthand for "
             "every accuracy-style test ('mcq', 'math', and 'code').",
    )
    parser.add_argument(
        "--warmup", type=int, default=config.WARMUP_RUNS,
        help=f"Warmup runs before measuring (default: {config.WARMUP_RUNS})",
    )
    parser.add_argument(
        "--runs", type=int, default=config.N_RUNS, choices=range(1, 11),
        metavar="[1-10]",
        help=f"Measured runs per checkpoint, averaged (default: {config.N_RUNS}). "
             "Applies separately to every model, context length, and test mode "
             "that's enabled, so total measured time scales roughly in "
             "proportion — e.g. going from 3 to 6 runs roughly doubles it "
             "(warmup time is unaffected; see --warmup).",
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Seconds per run (warmup and measured) before aborting this model (default: 300)",
    )
    parser.add_argument(
        "--acc-timeout", type=int, default=None,
        help="Seconds per question before giving up on it, for the accuracy tests "
             f"(mcq, math, code) — a timed-out question is scored wrong and the run "
             f"moves on (default: {config.ACC_TIMEOUT})",
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
        help="Force CPU-only inference for every Ollama-backed test (llm, conv, "
             "mcq, math, code, emb) by restarting Ollama with GPU devices hidden "
             "(HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES "
             "set empty). Stops any running Ollama server (even one this script "
             "didn't start) and restores normal GPU mode afterward. Useful on GPU "
             "backends unstable under one of those workloads (originally added "
             "for embedding batching, but the same instability can hit LLM/MCQ "
             "inference on some backends too).",
    )
    parser.add_argument(
        "--maxtier", type=str, default=None,
        choices=["xsmall", "small", "medium", "large"],
        help="Cap LLM models (single-shot and conversation tests) at this size tier "
             "and below (default: all tiers). xsmall: <6B params. small: adds ≤20B. "
             "medium: adds 26-35B. large: adds 70B+ (i.e. no cap).",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Only test these LLM models (llm, conv, mcq, math, and code tests) — exact Ollama "
             "tags or shell-style wildcards, e.g. 'llama*' matches every tag "
             "starting with 'llama' (default: every model in the selected tier). "
             "Applied after --maxtier, so it can only narrow that selection further "
             "within the catalog — but a pattern that matches nothing in the catalog "
             "falls back to matching against models actually pulled in Ollama, so a "
             "model outside our curated catalog can still be tested (see --list-models). "
             "Quote wildcards (e.g. \"llama*\") so your shell doesn't glob-expand them first.",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List every Ollama model actually installed locally, marking which are in "
             "the curated catalog (models.py) vs custom/extra, then exit without running "
             "anything. Useful for finding the exact tag to pass to --models.",
    )
    parser.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Dev-only: run 'mcq'/'math'/'code' against a deterministic N-question "
             "subset of each bank instead of the full thing, stratified so every "
             "category is represented. Same N always yields the same questions for "
             "a given bank version, and the exact sampled IDs are recorded in the "
             "output JSON under 'sample_ids'. Never use for a result meant to be "
             "compared against a full-bank run or published (default: full bank).",
    )
    parser.add_argument(
        "--engine", type=str, default="ollama", choices=["ollama", "llamacpp", "both"],
        help="Inference engine to benchmark against (default: ollama). "
             "'llamacpp' runs llama-server directly against the same models "
             "already pulled via 'ollama pull' — resolved straight from "
             "Ollama's local model store, no separate download — and "
             "requires the llama.cpp 'llama-server' binary on PATH. "
             "'both' runs the full --tests suite once per engine (ollama, "
             "then llamacpp), back to back, writing a separate results file "
             "for each (engine name appended to the filename) so they can "
             "be compared directly.",
    )
    parser.add_argument(
        "--force-all", action="store_true",
        help=f"Ignore the {config.SLOW_MODEL_MIN_TPS:.0f} tok/s slow-model cutoff: run every "
             "context length in the LLM prefill test and always run the conversation "
             "test, even for models that would otherwise be marked slow and skipped. "
             "Does not override real failures (timeouts, missing data). (default: false)",
    )
    args = parser.parse_args()

    # "both" runs the whole suite once per engine below; for the one-off
    # --list-models / --models resolution steps that need *an* engine just to
    # query Ollama's local model store, 'ollama' is authoritative regardless
    # (llamacpp resolves models from that same store — see --engine help).
    engine_names = ["ollama", "llamacpp"] if args.engine == "both" else [args.engine]
    engine = get_engine(engine_names[0])
    # Held on Shared so shutdown_managed() (called from the signal handler and
    # the finally block) can consult the live engine without threading it in.
    Shared._active_engine = engine

    if args.list_models:
        if not engine.ensure_running():
            Shared.err("Could not start Ollama — install from https://ollama.com/download "
                       "or start it manually with: ollama serve")
            sys.exit(1)
        installed = engine.list_installed_models()
        if not installed:
            Shared.warn("Ollama is running but no models are installed — pull one with: ollama pull <tag>")
            sys.exit(0)
        catalog_tags = {m["tag"] for m in LLM_MODELS} | {m["tag"] for m in EMBED_MODELS}
        print(f"\n{config.BOLD}Installed Ollama models{config.RESET}")
        n_catalog = 0
        for m in sorted(installed, key=lambda m: m["tag"]):
            in_catalog = strip_implicit_latest(m["tag"]) in catalog_tags
            n_catalog += in_catalog
            size_gb = f"{m['size'] / 1e9:.1f} GB" if m.get("size") else "? GB"
            print(f"  {m['tag']:<40} {size_gb:>10}   ({'catalog' if in_catalog else 'custom'})")
        print(f"\n  {len(installed)} installed, {n_catalog} in catalog, {len(installed) - n_catalog} custom")
        sys.exit(0)

    args.tests = expand_tests(args.tests)

    # Apply CLI overrides to shared config
    if args.timeout is not None:
        config.RUN_TIMEOUT = args.timeout
    if args.acc_timeout is not None:
        config.ACC_TIMEOUT = args.acc_timeout
    config.N_RUNS = args.runs

    llm_models, tier_label, image_models = select_tier(args.maxtier, IMAGE_MODELS)

    if args.models:
        engine.ensure_running()  # so a custom (non-catalog) tag can resolve against what's actually pulled
        installed_tags = [m["tag"] for m in engine.list_installed_models()]
        llm_models = resolve_custom_models(args.models, llm_models, installed_tags)
        if not llm_models:
            Shared.err(f"--models {' '.join(args.models)} matched no LLM models "
                       f"in the selected tier ({tier_label}) or installed in Ollama — "
                       "llm/conv/mcq/math/code tests will have nothing to run")

    comfyui_dir = Path(args.comfyui) if args.comfyui else config.COMFYUI_DIR

    profile  = Shared.build_profile()
    _safe = re.sub(r'[\\/:*?"<>|\s]+', '_', profile['hostname']).strip('_')
    _start_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base_out_path = args.out or str(config.RESULTS_DIR / f"results_{_safe}_{_start_stamp}.json")
    multi_engine = len(engine_names) > 1

    for run_idx, engine_name in enumerate(engine_names):
        engine = get_engine(engine_name)
        # Held on Shared so shutdown_managed() (called from the signal handler and
        # the finally block) can consult the live engine without threading it in.
        Shared._active_engine = engine

        if multi_engine:
            Shared.section(f"Engine: {engine_name} ({run_idx + 1}/{len(engine_names)})")
            _base = Path(base_out_path)
            out_path = str(_base.with_name(f"{_base.stem}_{engine_name}{_base.suffix}"))
        else:
            out_path = base_out_path

        # Image generation doesn't go through the LLM engine at all (it's a
        # separate ComfyUI HTTP call), so under --engine both it would just
        # duplicate identical, real-cost data on the second pass — run it
        # only once, on the first pass.
        tests = args.tests
        if multi_engine and run_idx > 0 and "img" in tests:
            Shared.log("Image generation doesn't depend on --engine — already "
                       f"captured in the {engine_names[0]} pass, skipping for {engine_name}")
            tests = [t for t in tests if t != "img"]

        print(f"\n{config.BOLD}LLM Benchmark Suite{config.RESET}")
        print(f"  Host:      {profile['hostname']}")
        print(f"  OS:        {profile['os']}")
        print(f"  Backend:   {profile['backend']}")
        print(f"  RAM:       {profile['ram_gb']} GB")
        print(f"  Engine:    {engine_name}")
        print(f"  Runs:      {config.N_RUNS} measured + {args.warmup} warmup")
        print(f"  Timeout:   {config.RUN_TIMEOUT}s per run, {config.ACC_TIMEOUT}s per accuracy question")
        print(f"  Models:    {tier_label}")
        if args.models:
            print(f"  --models:  {', '.join(m['label'] for m in llm_models) or '(none matched)'}")
        if args.maxtier:
            print(f"  Images:    {', '.join(m['label'] for m in image_models) or '(none — tier too small)'}")
        print(f"  Tests:     {', '.join(tests)}")
        print(f"  ComfyUI:   {comfyui_dir}")

        # Register cleanup for Ctrl-C and normal exit
        def _cleanup(sig=None, frame=None):
            if sig is not None:
                print(f"\n{config.YELLOW}Interrupted — unloading models before exit ...{config.RESET}")
            if engine.available():
                engine.unload_all()
            if Shared.comfyui_available():
                ImageBenchmark.comfyui_free_models()
            if Shared._managed_procs:
                print(f"\n{config.YELLOW}Cleaning up managed servers ...{config.RESET}")
                Shared.shutdown_managed()
            if sig is not None:
                sys.exit(0)

        signal.signal(signal.SIGINT,  _cleanup)
        signal.signal(signal.SIGTERM, _cleanup)

        results = {
            "version":         config.VERSION,
            "engine":          engine_name,
            "profile":         profile,
            # Fingerprints of the accuracy question banks actually used for this
            # run, so a raw correct count is never compared across bank sizes
            # (e.g. 185 vs. 360 questions) without noticing the version differs.
            "bank_versions": {
                "mcq":  Shared.file_hash(MCQBenchmark.MCQ_DATA_PATH),
                "math": Shared.file_hash(MathBenchmark.MATH_DATA_PATH),
                "code": Shared.file_hash(CodeBenchmark.CODE_DATA_PATH),
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
            "code":            {},
        }

        def _checkpoint(label=""):
            Path(out_path).write_text(json.dumps(results, indent=2))
            if label:
                Shared.log(f"Partial results saved to {out_path} ({label})")

        try:
            # ── Ollama-backed tests (llm, conv, mcq, math, code, emb) share one server lifecycle
            ollama_tests = [t for t in ("llm", "conv", "mcq", "math", "code", "emb") if t in tests]
            if ollama_tests:
                Shared.section("Starting Servers")
                other_name = OTHER_ENGINE[engine_name]
                other_engine = get_engine(other_name)
                if other_engine.available():
                    Shared.log(f"Stopping {other_name} so only one inference "
                               f"engine runs at a time ...")
                    other_engine.stop()
                if args.cpu_only:
                    Shared.warn("Stopping Ollama to relaunch in CPU-only mode "
                                f"(applies to: {', '.join(ollama_tests)}) ...")
                    engine.stop()
                    if not engine.start(gpu_visible=False):
                        Shared.err("Failed to start Ollama in CPU-only mode — "
                                   f"{', '.join(ollama_tests)} tests will be skipped")
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
                    first_ctx_label = f"{config.CONTEXT_LENGTHS[0] // 1024}K"
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

            # ── Accuracy tests (MCQ / Math / Code) ────────────────────────────────
            # Identical wiring for all three — only the test name, benchmark class,
            # and display label vary.
            for test_name, Bench, done_label in (
                ("mcq", MCQBenchmark, "MCQ"), ("math", MathBenchmark, "Math"), ("code", CodeBenchmark, "Code"),
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
                    models=EMBED_MODELS,
                    warmup_runs=args.warmup,
                    save_fn=_emb_save,
                )
                _checkpoint("embeddings done")

            # Done with every Ollama-backed test — restore normal GPU-enabled Ollama
            # if this run forced CPU-only, so the machine isn't left in that state
            # (and so image generation, if it runs next, starts from a clean state).
            if ollama_tests and args.cpu_only and engine._cpu_only_active:
                Shared.log("Restoring normal (GPU-enabled) Ollama ...")
                engine.stop()
                engine.start()

            # ── Image generation ───────────────────────────────────────────────────
            if "img" in tests:
                Shared.section("Starting Servers")
                # Nothing from Ollama in memory before ComfyUI loads. Image
                # generation is always the last phase, so nothing later needs
                # Ollama — kill the whole server rather than just unloading its
                # models, to free the memory the idle process itself still holds.
                if engine.available():
                    Shared.log("Stopping Ollama entirely to free memory for ComfyUI ...")
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
        Path(out_path).write_text(json.dumps(results, indent=2))
        Shared.ok(f"Results saved to: {out_path}")

    print(f"\n  Compare it against other machines in the dashboard:")
    dash_hint = "launch_dashboard.bat" if platform.system() == "Windows" else "bash launch_dashboard.sh"
    print(f"  {dash_hint}\n")
    Shared.section("Done")
    Shared.ok("All servers shut down. Benchmark complete.")

if __name__ == "__main__":
    main()
