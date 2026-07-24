"""Shared implementation for the "tool" and "chat" concurrency tests — see docs/workloads.md#concurrency."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config
from shared import Shared


class ConcurrencyBenchmark:
    # Separate per test — see docs/workloads.md#concurrency's hard-stop bullet.
    TOOL_CRASH_CACHE = Path(".concurrency_tool_crash_cache.json")
    CHAT_CRASH_CACHE = Path(".concurrency_chat_crash_cache.json")

    @staticmethod
    def should_stop_escalating(level: int, mean_tps: float, force_all: bool,
                                soft_exit_floor: int | None) -> bool:
        """True if the sweep shouldn't climb past `level` — see docs/workloads.md#concurrency."""
        if force_all or soft_exit_floor is None:
            return False
        if level < soft_exit_floor:
            return False
        return mean_tps < config.SLOW_MODEL_MIN_TPS

    @staticmethod
    def slot_ctx_for(per_request_context: int) -> int:
        """Per-slot ctx budget: padded prompt plus generate()'s n_predict headroom,
        so a slot isn't sized to exactly the prompt with no room to generate."""
        return per_request_context + config.GENERATE_MAX_TOKENS

    @staticmethod
    def _fire_batch(engine, tag: str, level: int, per_request_context: int) -> list:
        """Fire `level` concurrent generate() requests — see docs/workloads.md#concurrency.
        Returns raw (ttft, tokens, tps) samples."""
        prompts = [Shared.build_prompt_for_context(per_request_context) for _ in range(level)]
        slot_ctx = ConcurrencyBenchmark.slot_ctx_for(per_request_context)
        with ThreadPoolExecutor(max_workers=level) as pool:
            futures = [
                pool.submit(engine.generate, tag, p, config.RUN_TIMEOUT, slot_ctx, level)
                for p in prompts
            ]
            return [f.result() for f in futures]

    @staticmethod
    def _fire_batch_with_crash_retries(engine, tag: str, level: int,
                                       per_request_context: int
                                       ) -> tuple[list, str, Exception | None, float]:
        for crash_i in range(Shared.CRASH_RETRY_MAX + 1):
            batch_t0 = time.perf_counter()
            try:
                samples = ConcurrencyBenchmark._fire_batch(
                    engine, tag, level, per_request_context,
                )
                return samples, "ok", None, time.perf_counter() - batch_t0
            except Exception as e:
                if not engine.is_connection_crash(e):
                    return [], "failed", e, 0
                recovered = engine.wait_for_recovery()
                if crash_i >= Shared.CRASH_RETRY_MAX or not recovered:
                    return [], "crashed", e, 0
                Shared.warn(
                    f"Engine crashed during {level}-way concurrency; retrying "
                    f"({crash_i + 1}/{Shared.CRASH_RETRY_MAX}) ..."
                )
        return [], "crashed", None, 0

    def run(self, engine, models, levels, per_request_context, warmup_runs,
            crash_cache_path: Path, section_label: str,
            soft_exit_floor: int | None = None, force_all=False,
            save_fn=None):  # pragma: no cover — orchestrates real engine runs
        results = {}

        if not engine.ensure_running():
            Shared.err(f"Inference engine not reachable — skipping {section_label} benchmark")
            return results

        crash_cache = Shared.load_crash_cache(crash_cache_path)

        for model in models:
            tag   = model["tag"]
            label = model["label"]
            short = model["short"]

            Shared.section(f"{section_label} ({engine.name}): {label}")

            if not engine.reachable_or_abort():
                break

            try:
                if not engine.model_pulled(tag):
                    Shared.warn(f"{tag} not pulled — skipping")
                    Shared.warn("Download it with: python setup_check.py")
                    continue

                skip_entry = Shared.check_crash_cache(tag, label, crash_cache, crash_cache_path)
                if skip_entry is not None:
                    results[short] = skip_entry
                    continue

                results[short] = {}
                stopped_at = None

                for level in levels:
                    Shared.log(f"{label}: preparing {level}-way concurrency at "
                               f"{per_request_context} tokens/slot ...")

                    if not engine.prepare_concurrency(
                        tag, level, self.slot_ctx_for(per_request_context), warmup_runs,
                        timeout=config.RUN_TIMEOUT,
                    ):
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

                    # Every level respawns llama-server — see docs/workloads.md#concurrency.
                    warmup_failed = False
                    for warmup_i in range(warmup_runs):
                        Shared.log(f"{label}: warming up {level}-way concurrency "
                                   f"(run {warmup_i+1}/{warmup_runs}) ...")
                        _, status, error, _ = self._fire_batch_with_crash_retries(
                            engine, tag, level, per_request_context,
                        )
                        if status != "ok":
                            if status == "crashed":
                                Shared.err(f"{label}: engine crashed repeatedly warming up {level}-way "
                                           f"concurrency — last server output:\n{engine.tail_log()}")
                                results[short]["crashed_at"] = Shared.record_crash(
                                    tag, crash_cache, crash_cache_path,
                                    f"warming up {level}-way concurrency")
                                stopped_at = "crashed"
                            else:
                                Shared.err(f"{label}: {level}-way concurrency warmup failed: {error}")
                                stopped_at = "failed"
                            warmup_failed = True
                            break
                    if warmup_failed:
                        break

                    Shared.log(f"{label}: firing {level} concurrent request(s) ...")
                    samples, status, error, batch_elapsed = self._fire_batch_with_crash_retries(
                        engine, tag, level, per_request_context,
                    )
                    if status != "ok":
                        if status == "crashed":
                            Shared.err(f"{label}: engine crashed repeatedly during the {level}-way batch — "
                                       f"last server output:\n{engine.tail_log()}")
                            results[short]["crashed_at"] = Shared.record_crash(
                                tag, crash_cache, crash_cache_path,
                                f"running {level}-way concurrency")
                            stopped_at = "crashed"
                        else:
                            Shared.err(f"{label}: {level}-way concurrency batch failed: {error}")
                            stopped_at = "failed"
                        break
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

                    if self.should_stop_escalating(level, mean_tps, force_all, soft_exit_floor):
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
