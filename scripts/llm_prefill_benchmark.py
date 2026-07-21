"""llm_prefill_benchmark.py — single-shot LLM prefill/decode benchmark."""

from pathlib import Path

import config
from shared import Shared


class LLMPrefillBenchmark:
    # Records models that crashed the engine's runner repeatedly (deterministically,
    # not a transient blip) so future runs don't waste time rediscovering the
    # same crash. Delete this file to retry a skipped model.
    LLM_CRASH_CACHE = Path(".llm_crash_cache.json")

    def run(self, engine, models, context_lengths, warmup_runs, force_all=False, save_fn=None):  # pragma: no cover — orchestrates real engine runs
        results = {}

        if not engine.ensure_running():
            Shared.err("Inference engine not reachable — skipping LLM benchmarks")
            return results

        crash_cache = Shared.load_crash_cache(LLMPrefillBenchmark.LLM_CRASH_CACHE)

        for model in models:
            tag   = model["tag"]
            label = model["label"]
            short = model["short"]

            Shared.section(f"LLM ({engine.name}): {label}")

            if not engine.reachable_or_abort():
                break

            try:
                if not engine.model_pulled(tag):
                    Shared.warn(f"{tag} not pulled — skipping")
                    Shared.warn("Download it with: python setup_check.py")
                    continue

                skip_entry = Shared.check_crash_cache(tag, label, crash_cache, LLMPrefillBenchmark.LLM_CRASH_CACHE)
                if skip_entry is not None:
                    results[short] = skip_entry
                    continue

                model_max = engine.max_context_length(tag)
                results[short] = {}

                model_ctx_lengths = [c for c in context_lengths if c <= model_max]

                model_timed_out = False
                for ctx_len in model_ctx_lengths:
                    label_ctx = Shared.context_label(ctx_len)
                    if not engine.warmup(tag, label, ctx_len, warmup_runs,
                                         crash_cache, LLMPrefillBenchmark.LLM_CRASH_CACHE):
                        results[short]["crashed"] = label_ctx
                        results[short]["crashed_at"] = crash_cache.get(tag, {}).get(
                            "crashed_at", "during warmup",
                        )
                        engine.unload(tag)
                        break
                    Shared.log(f"Context {label_ctx} — {config.N_RUNS} runs ...")

                    def _prefill_once(run_i):
                        prompt = Shared.build_prompt_for_context(ctx_len)
                        ttft, tokens, tps = engine.generate(
                            tag, prompt, timeout=config.RUN_TIMEOUT, num_ctx=ctx_len
                        )
                        print(
                            f"    run {run_i+1}/{config.N_RUNS}: "
                            f"TTFT={ttft:.2f}s  "
                            f"TPS={tps:.1f}"
                        )
                        return ttft, tps

                    samples, status, _ = Shared.run_measured_calls(
                        config.N_RUNS, _prefill_once, tag, crash_cache,
                        LLMPrefillBenchmark.LLM_CRASH_CACHE, f"running {label}", engine)
                    ttfts    = [s[0] for s in samples]
                    tps_list = [s[1] for s in samples]

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

                    if status == "timed_out":
                        Shared.err(f"Skipping remaining runs and context lengths for {label}")
                        model_timed_out = True
                        results[short]["timed_out"] = label_ctx
                        break

                    if status == "crashed":
                        crashed_at = crash_cache.get(tag, {}).get("crashed_at", "an earlier run")
                        results[short]["crashed"] = label_ctx
                        results[short]["crashed_at"] = crashed_at
                        break

                    is_first_ctx = ctx_len == model_ctx_lengths[0]
                    if Shared.slow_tps_early_exit(results, short, label, label_ctx, is_first_ctx, tps_list, force_all):
                        break

                if model_timed_out:
                    Shared.warn(f"{label}: timed out — moving to next model")
                Shared.log(f"Unloading {label} ...")
                engine.unload(tag)
                engine.wait_until_unloaded(tag)
            finally:
                if save_fn:
                    save_fn(results)

        return results
