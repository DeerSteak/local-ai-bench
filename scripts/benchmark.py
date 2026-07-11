#!/usr/bin/env python3
"""
benchmark.py — Cross-platform LLM benchmark suite.

Tests:
  1. LLM generation — 10 models across small/medium/large tiers via Ollama
     Metrics: time-to-first-token (TTFT), tokens/sec
     Context lengths: 2K, 8K, 32K, 64K
     Models that exceed the warmup timeout are skipped automatically

  1b. LLM conversation — same models/context depths, but via a real multi-turn
      chat (/api/chat) instead of one padded single-shot prompt: the model
      explains Plato's Allegory of the Cave in sections, then each turn asks
      for more detail on a section. TTFT/tokens-per-sec at each depth reflect
      processing a new turn against an already-filled context (relying on
      llama.cpp's slot prefix cache), not a cold fill from empty.

  2. Image generation — SDXL, SD3.5 Large, Flux.1-dev via ComfyUI HTTP API
     Metrics: seconds/image at 1024×1024 and 1536×1536
     (models skipped automatically if checkpoint not found)

  3. Embeddings — nomic-embed-text and mxbai-embed-large via Ollama
     Metrics: sentences/sec at batch sizes 32, 128, 512

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
import json
import re
import signal
import sys
from datetime import datetime
from pathlib import Path

import config
from shared import Shared
from llm_prefill_benchmark import LLMPrefillBenchmark
from llm_conversation_benchmark import LLMConversationBenchmark
from embedding_benchmark import EmbeddingBenchmark
from image_benchmark import ImageBenchmark
from models import IMAGE_MODELS, LLM_MODELS_XSMALL, LLM_MODELS_SMALL, LLM_MODELS_MEDIUM, LLM_MODELS_LARGE, LLM_MODELS, EMBED_MODELS


def main():
    parser = argparse.ArgumentParser(description="LLM benchmark suite")
    parser.add_argument(
        "--tests", nargs="+",
        choices=["llm", "conv", "emb", "img"],
        default=["llm", "conv", "emb", "img"],
        help="Which benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--warmup", type=int, default=config.WARMUP_RUNS,
        help=f"Warmup runs before measuring (default: {config.WARMUP_RUNS})",
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Seconds per run (warmup and measured) before aborting this model (default: 300)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output JSON file (default: results_<hostname>_<timestamp>.json)",
    )
    parser.add_argument(
        "--comfyui", type=str, default=None,
        help=f"Path to ComfyUI directory (default: {config.COMFYUI_DIR})",
    )
    parser.add_argument(
        "--emb-cpu-only", action="store_true",
        help="Force CPU-only inference for the embedding benchmarks by restarting "
             "Ollama with GPU devices hidden (HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES "
             "/ ROCR_VISIBLE_DEVICES set empty). Stops any running Ollama server "
             "(even one this script didn't start) and restores normal GPU mode "
             "afterward. Useful on GPU backends unstable under embedding batching.",
    )
    parser.add_argument(
        "--maxtier", type=str, default=None,
        choices=["xsmall", "small", "medium", "large"],
        help="Cap LLM models (single-shot and conversation tests) at this size tier "
             "and below (default: all tiers). xsmall: <6B params. small: adds ≤20B. "
             "medium: adds 26-35B. large: adds 70B+ (i.e. no cap).",
    )
    parser.add_argument(
        "--force-all", action="store_true",
        help=f"Ignore the {config.SLOW_MODEL_MIN_TPS:.0f} tok/s slow-model cutoff: run every "
             "context length in the LLM prefill test and always run the conversation "
             "test, even for models that would otherwise be marked slow and skipped. "
             "Does not override real failures (timeouts, missing data). (default: false)",
    )
    args = parser.parse_args()

    # Apply CLI overrides to shared config
    if args.timeout is not None:
        config.RUN_TIMEOUT = args.timeout

    # Select model tier — cumulative: --maxtier caps at that tier and includes everything below it
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
    if args.maxtier:
        llm_models = TIER_MODELS[args.maxtier]
        tier_label = TIER_LABELS[args.maxtier]
        max_idx = TIER_ORDER.index(args.maxtier)
        image_models = [m for m in IMAGE_MODELS if TIER_ORDER.index(m["tier"]) <= max_idx]
    else:
        llm_models = LLM_MODELS
        tier_label = "all (extra-small + small + medium + large)"
        image_models = IMAGE_MODELS

    comfyui_dir = Path(args.comfyui) if args.comfyui else config.COMFYUI_DIR

    profile  = Shared.build_profile()
    _safe = re.sub(r'[\\/:*?"<>|\s]+', '_', profile['hostname']).strip('_')
    _start_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or str(config.RESULTS_DIR / f"results_{_safe}_{_start_stamp}.json")

    print(f"\n{config.BOLD}LLM Benchmark Suite{config.RESET}")
    print(f"  Host:      {profile['hostname']}")
    print(f"  OS:        {profile['os']}")
    print(f"  Backend:   {profile['backend']}")
    print(f"  RAM:       {profile['ram_gb']} GB")
    print(f"  Runs:      {config.N_RUNS} measured + {args.warmup} warmup")
    print(f"  Timeout:   {config.RUN_TIMEOUT}s per run")
    print(f"  Models:    {tier_label}")
    if args.maxtier:
        print(f"  Images:    {', '.join(m['label'] for m in image_models) or '(none — tier too small)'}")
    print(f"  Tests:     {', '.join(args.tests)}")
    print(f"  ComfyUI:   {comfyui_dir}")

    # Register cleanup for Ctrl-C and normal exit
    def _cleanup(sig=None, frame=None):
        if sig is not None:
            print(f"\n{config.YELLOW}Interrupted — unloading models before exit ...{config.RESET}")
        if Shared.ollama_available():
            Shared.unload_all_models()
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
        "profile":         profile,
        "llm":             {},
        "llm_conversation": {},
        "embeddings":      {},
        "images":          {},
    }

    def _checkpoint(label=""):
        Path(out_path).write_text(json.dumps(results, indent=2))
        if label:
            Shared.log(f"Partial results saved to {out_path} ({label})")

    try:
        # ── LLM ───────────────────────────────────────────────────────────────
        if "llm" in args.tests or "conv" in args.tests:
            Shared.section("Starting Servers")
            Shared.ensure_ollama()

        if "llm" in args.tests:
            results["llm"] = LLMPrefillBenchmark().run(
                models=llm_models,
                context_lengths=config.CONTEXT_LENGTHS,
                warmup_runs=args.warmup,
                force_all=args.force_all,
            )
            _checkpoint("LLM done")

        if "conv" in args.tests:
            conv_models = llm_models
            llm_conv_skips = {}
            if "llm" in args.tests:
                conv_models = []
                for model in llm_models:
                    short = model["short"]
                    llm_data = results["llm"].get(short)
                    if not llm_data:
                        detail = "no LLM benchmark data (checkpoint skipped or model failed)"
                        Shared.warn(f"{model['label']}: skipping conversation test — {detail}")
                        llm_conv_skips[short] = {
                            "label": model["label"], "skipped": True,
                            "skip_reason": "no_llm_data", "skip_detail": detail,
                        }
                        continue
                    first_ctx_label = f"{config.CONTEXT_LENGTHS[0] // 1024}K"
                    if llm_data.get("timed_out") == first_ctx_label:
                        detail = f"LLM test timed out at {llm_data['timed_out']} context"
                        Shared.warn(f"{model['label']}: skipping conversation test — {detail}")
                        llm_conv_skips[short] = {
                            "label": model["label"], "skipped": True,
                            "skip_reason": "timed_out", "skip_detail": detail,
                        }
                        continue
                    # A timeout at a deeper context (8K/32K/64K) doesn't disqualify
                    # the model — it passed the 2K prefill test, so fall through to
                    # the tok/s check below just like a model that wasn't timed out.
                    slow_ctx = None if args.force_all else llm_data.get("slow_tps") or (
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
                        Shared.warn(f"{model['label']}: skipping conversation test — {detail}")
                        llm_conv_skips[short] = {
                            "label": model["label"], "skipped": True,
                            "skip_reason": "slow_tps", "skip_detail": detail,
                        }
                        continue
                    conv_models.append(model)

            results["llm_conversation"] = LLMConversationBenchmark().run(
                models=conv_models,
                context_lengths=config.CONTEXT_LENGTHS,
                warmup_runs=args.warmup,
                force_all=args.force_all,
            )
            results["llm_conversation"].update(llm_conv_skips)
            _checkpoint("LLM conversation done")

        # ── Embeddings ─────────────────────────────────────────────────────────
        if "emb" in args.tests:
            if args.emb_cpu_only:
                Shared.section("Embeddings: forcing CPU-only")
                Shared.warn("Stopping Ollama to relaunch in CPU-only mode for embeddings ...")
                Shared.stop_all_ollama()
                cpu_env = {
                    "HIP_VISIBLE_DEVICES": "",
                    "CUDA_VISIBLE_DEVICES": "",
                    "ROCR_VISIBLE_DEVICES": "",
                }
                if not Shared.start_ollama(extra_env=cpu_env):
                    Shared.err("Failed to start Ollama in CPU-only mode — skipping embeddings")
                    results["embeddings"] = {}
                else:
                    Shared._cpu_only_active = True
                    results["embeddings"] = EmbeddingBenchmark().run(
                        models=EMBED_MODELS,
                        warmup_runs=args.warmup,
                    )
                    _checkpoint("embeddings done")
                    Shared.log("Restoring normal (GPU-enabled) Ollama ...")
                    Shared.stop_all_ollama()
                    Shared._cpu_only_active = False
                    Shared.start_ollama()
            else:
                if not Shared.ollama_available():
                    Shared.section("Starting Servers")
                    Shared.ensure_ollama()
                results["embeddings"] = EmbeddingBenchmark().run(
                    models=EMBED_MODELS,
                    warmup_runs=args.warmup,
                )
                _checkpoint("embeddings done")

        # ── Image generation ───────────────────────────────────────────────────
        if "img" in args.tests:
            Shared.section("Starting Servers")
            # Hard guarantee: nothing from Ollama in memory before ComfyUI loads.
            # Image generation is always the last phase (see phase order above),
            # so there's nothing left in this run that needs Ollama afterward —
            # kill the whole server rather than just unloading its models, to
            # free up whatever memory the idle process itself still holds.
            if Shared.ollama_available():
                Shared.log("Stopping Ollama entirely to free memory for ComfyUI ...")
                Shared.stop_all_ollama()
            comfyui_started = Shared.ensure_comfyui(comfyui_dir)
            if not comfyui_started:
                Shared.warn("Image benchmarks will be skipped")
            else:
                def _img_save(img_partial):
                    results["images"] = img_partial
                    _checkpoint()

                results["images"] = ImageBenchmark().run(
                    image_models=image_models,
                    resolutions=config.IMAGE_RESOLUTIONS,
                    seed=config.IMAGE_SEED,
                    prompt=config.IMAGE_PROMPT,
                    comfyui_dir=comfyui_dir,
                    timeout=config.RUN_TIMEOUT * 2,
                    save_fn=_img_save,
                    # Same stem as the results JSON, so each run's images land in
                    # their own folder alongside the numbers they belong to.
                    images_dir=config.RESULTS_IMAGES_DIR / Path(out_path).stem,
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
    print(f"  python launch_dashboard.py\n")
    Shared.section("Done")
    Shared.ok("All servers shut down. Benchmark complete.")

if __name__ == "__main__":
    main()
