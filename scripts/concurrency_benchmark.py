"""
concurrency_benchmark.py — per-request latency and aggregate throughput at
increasing simultaneous request counts. See docs/workloads.md#concurrency.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config
from shared import Shared


class ConcurrencyBenchmark:
    # A crashed batch is recorded here (delete to retry); a plain
    # prepare_concurrency ceiling is not — see docs/workloads.md#concurrency.
    CONCURRENCY_CRASH_CACHE = Path(".concurrency_crash_cache.json")

    @staticmethod
    def should_stop_escalating(level: int, mean_tps: float, force_all: bool) -> bool:
        """True if the sweep shouldn't climb past `level` for this model."""
        if force_all:
            return False
        if level < config.CONCURRENCY_MIN_LEVEL_BEFORE_SOFT_EXIT:
            return False
        return mean_tps < config.SLOW_MODEL_MIN_TPS

    def run(self, engine, models, warmup_runs, force_all=False, save_fn=None):  # pragma: no cover — orchestrates real engine runs
        results = {}

        if not engine.ensure_running():
            Shared.err("Inference engine not reachable — skipping concurrency benchmark")
            Shared.err("Start with: ollama serve")
            return results

        crash_cache = Shared.load_crash_cache(ConcurrencyBenchmark.CONCURRENCY_CRASH_CACHE)

        for model in models:
            tag   = model["tag"]
            label = model["label"]
            short = model["short"]

            Shared.section(f"Concurrency ({engine.name}): {label}")

            if not engine.reachable_or_abort():
                break

            try:
                if not engine.model_pulled(tag):
                    Shared.warn(f"{tag} not pulled — skipping")
                    Shared.warn(f"Pull with: ollama pull {tag}")
                    continue

                skip_entry = Shared.check_crash_cache(tag, label, crash_cache,
                                                       ConcurrencyBenchmark.CONCURRENCY_CRASH_CACHE)
                if skip_entry is not None:
                    results[short] = skip_entry
                    continue

                results[short] = {}
                stopped_at = None

                for level in config.CONCURRENCY_LEVELS:
                    Shared.log(f"{label}: preparing {level}-way concurrency at "
                               f"{config.CONCURRENCY_CONTEXT} tokens/slot ...")

                    if not engine.prepare_concurrency(tag, level, config.CONCURRENCY_CONTEXT, warmup_runs):
                        Shared.warn(f"{label}: couldn't load at {level}-way concurrency — "
                                    "this is the model's real ceiling, stopping here")
                        results[short]["memory_at_failure"] = Shared.sample_memory_gb()
                        stopped_at = "load_failed"
                        break

                    memory = Shared.sample_memory_gb()  # right after load, before the batch
                    mem_bits = [f"{memory['system_ram_used_gb']:.1f}/{memory['system_ram_total_gb']:.1f} GB RAM"]
                    if memory["gpu_vram_used_gb"] is not None:
                        mem_bits.append(f"{memory['gpu_vram_used_gb']:.1f}/{memory['gpu_vram_total_gb']:.1f} GB VRAM")
                    Shared.log(f"{label}: loaded at {level}-way — {', '.join(mem_bits)} in use")

                    Shared.log(f"{label}: firing {level} concurrent request(s) ...")
                    prompts = [Shared.build_prompt_for_context(config.CONCURRENCY_CONTEXT)
                               for _ in range(level)]

                    batch_t0 = time.perf_counter()
                    try:
                        with ThreadPoolExecutor(max_workers=level) as pool:
                            futures = [
                                pool.submit(engine.generate, tag, p, config.RUN_TIMEOUT,
                                            config.CONCURRENCY_CONTEXT, level)
                                for p in prompts
                            ]
                            samples = [f.result() for f in futures]
                    except Exception as e:
                        if engine.is_connection_crash(e):
                            Shared.err(f"{label}: engine crashed during the {level}-way batch — "
                                       f"last server output:\n{engine.tail_log()}")
                            engine.wait_for_recovery()
                            results[short]["crashed_at"] = Shared.record_crash(
                                tag, crash_cache, ConcurrencyBenchmark.CONCURRENCY_CRASH_CACHE,
                                f"running {level}-way concurrency")
                            stopped_at = "crashed"
                        else:
                            Shared.err(f"{label}: {level}-way concurrency batch failed: {e}")
                            stopped_at = "failed"
                        break
                    batch_elapsed = time.perf_counter() - batch_t0

                    ttfts  = [s[0] for s in samples]
                    tokens = [s[1] for s in samples]
                    tpss   = [s[2] for s in samples]
                    total_tokens  = sum(tokens)
                    aggregate_tps = total_tokens / batch_elapsed if batch_elapsed > 0 else 0
                    mean_tps      = Shared.mean(tpss)

                    results[short][str(level)] = {
                        "ttft_mean_sec":     round(Shared.mean(ttfts), 3),
                        "ttft_stdev_sec":    round(Shared.stdev(ttfts), 3),
                        "tps_mean":          round(mean_tps, 2),
                        "tps_stdev":         round(Shared.stdev(tpss), 2),
                        "aggregate_tps":     round(aggregate_tps, 2),
                        "total_tokens":      total_tokens,
                        "batch_elapsed_sec": round(batch_elapsed, 3),
                        "memory":            memory,
                    }
                    Shared.ok(
                        f"{level}-way done: per-request TTFT={Shared.mean(ttfts):.2f}s "
                        f"TPS={mean_tps:.1f} — aggregate {aggregate_tps:.1f} tok/s"
                    )

                    if self.should_stop_escalating(level, mean_tps, force_all):
                        Shared.warn(f"{label}: per-request TPS ({mean_tps:.1f}) below "
                                    f"{config.SLOW_MODEL_MIN_TPS:.0f} tok/s at {level}-way "
                                    "— stopping here")
                        stopped_at = "slow"
                        break

                if stopped_at:
                    results[short]["stopped_at"] = stopped_at

                Shared.log(f"Unloading {label} ...")
                engine.unload(tag)
                engine.wait_until_unloaded(tag)
            finally:
                if save_fn:
                    save_fn(results)

        return results
