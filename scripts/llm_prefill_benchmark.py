"""llm_prefill_benchmark.py — single-shot LLM prefill/decode benchmark."""

import threading
import time

import config
from shared import Shared


class LLMPrefillBenchmark:
    def run(self, models, context_lengths, warmup_runs, force_all=False, save_fn=None):
        results = {}

        if not Shared.ollama_available():
            Shared.err("Ollama server not reachable — skipping LLM benchmarks")
            Shared.err("Start with: ollama serve")
            return results

        for model in models:
            tag   = model["tag"]
            label = model["label"]
            short = model["short"]

            Shared.section(f"LLM: {label}")

            try:
                if not Shared.model_pulled(tag):
                    Shared.warn(f"{tag} not pulled — skipping")
                    Shared.warn(f"Pull with: ollama pull {tag}")
                    continue

                # Warm up model (load into memory), with a timeout so we don't get stuck.
                # Use the largest context this model will actually run so Ollama
                # pre-allocates the full KV cache once — avoiding a reload at max context.
                max_ctx = min(model.get("max_ctx", max(context_lengths)), max(context_lengths))
                Shared.log(f"Warming up {label} at num_ctx={max_ctx} (timeout: {config.RUN_TIMEOUT}s per run) ...")
                warmup_ok = True
                for warmup_i in range(warmup_runs):
                    result_box = [None]   # mutable container so thread can write back
                    exc_box    = [None]

                    def _warmup():
                        try:
                            result_box[0] = Shared.ollama_generate(
                                tag, "Hello.", timeout=config.RUN_TIMEOUT, num_ctx=max_ctx)
                        except Exception as e:
                            exc_box[0] = e

                    t = threading.Thread(target=_warmup, daemon=True)
                    t_start = time.perf_counter()
                    t.start()
                    t.join(timeout=config.RUN_TIMEOUT)

                    if t.is_alive():
                        elapsed = time.perf_counter() - t_start
                        Shared.warn(f"{label}: warmup run {warmup_i+1} did not complete within {elapsed:.0f}s")
                        Shared.warn(f"{label}: model is likely too large for available memory — skipping")
                        warmup_ok = False
                        break
                    elif exc_box[0] is not None:
                        Shared.warn(f"Warmup run {warmup_i+1} failed: {exc_box[0]}")
                        warmup_ok = False
                        break
                    else:
                        Shared.log(f"Warmup run {warmup_i+1}/{warmup_runs} done")

                if not warmup_ok:
                    Shared.unload_model(tag)
                    continue

                results[short] = {}

                model_ctx_lengths = [c for c in context_lengths
                                     if c <= model.get("max_ctx", max(context_lengths))]

                model_timed_out = False
                for ctx_len in model_ctx_lengths:
                    label_ctx = f"{ctx_len // 1024}K"
                    Shared.log(f"Context {label_ctx} — {config.N_RUNS} runs ...")

                    ttfts, tps_list = [], []
                    ctx_timed_out = False

                    for run_i in range(config.N_RUNS):
                        try:
                            prompt = Shared.build_prompt_for_context(ctx_len)
                            ttft, tokens, tps = Shared.ollama_generate(
                                tag, prompt, timeout=config.RUN_TIMEOUT, num_ctx=ctx_len
                            )
                            ttfts.append(ttft)
                            tps_list.append(tps)
                            print(
                                f"    run {run_i+1}/{config.N_RUNS}: "
                                f"TTFT={ttft:.2f}s  "
                                f"TPS={tps:.1f}"
                            )
                        except Exception as e:
                            is_timeout = isinstance(e, TimeoutError) or "timed out" in str(e).lower()
                            if is_timeout:
                                Shared.err(f"Run {run_i+1} timed out — skipping remaining runs and context lengths for {label}")
                                ctx_timed_out = True
                                break
                            Shared.err(f"Run {run_i+1} failed: {e}")

                    if ttfts:
                        results[short][label_ctx] = {
                            "ttft_mean_sec":  round(Shared.mean(ttfts),    3),
                            "ttft_stdev_sec": round(Shared.stdev(ttfts),   3),
                            "tps_mean":       round(Shared.mean(tps_list), 2),
                            "tps_stdev":      round(Shared.stdev(tps_list),2),
                            "n_runs":         len(tps_list),
                            "ttft_runs":      [round(t, 3) for t in ttfts],
                            "tps_runs":       [round(t, 2) for t in tps_list],
                        }
                        Shared.ok(
                            f"Context {label_ctx} done: "
                            f"TTFT={results[short][label_ctx]['ttft_mean_sec']:.2f}s  "
                            f"TPS={results[short][label_ctx]['tps_mean']:.1f}"
                        )

                    if ctx_timed_out:
                        model_timed_out = True
                        results[short]["timed_out"] = label_ctx
                        break

                    is_first_ctx = ctx_len == model_ctx_lengths[0]
                    if Shared.slow_tps_early_exit(results, short, label, label_ctx, is_first_ctx, tps_list, force_all):
                        break

                # Unload model and confirm it's gone before moving on
                if model_timed_out:
                    Shared.warn(f"{label}: timed out — moving to next model")
                Shared.log(f"Unloading {label} ...")
                Shared.unload_model(tag)
                Shared.wait_until_unloaded(tag)
            finally:
                if save_fn:
                    save_fn(results)

        return results
