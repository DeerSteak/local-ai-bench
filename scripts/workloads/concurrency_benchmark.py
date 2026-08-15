"""Shared implementation for the "tool" and "chat" concurrency tests — see docs/workloads.md#concurrency."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.runtime import config
from scripts.runtime.engines.base import aggregate_generation_measurements, measurement_validation_errors
from scripts.runtime.shared import Shared
from scripts.runtime.failure_handling import unexpected_model_failure
from scripts.runtime.crash_cache import check_crash_cache, load_crash_cache, record_crash
from scripts.runtime.progress_events import emit_model_finished, emit_progress
from scripts.runtime.pause_control import wait_if_paused


def pending_concurrency_levels(levels, next_attempt):
    """Return uncompleted levels with their durable attempt allocation."""
    return [(level, attempt) for level in levels if (attempt := next_attempt(level)) is not None]


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
        Returns named measurement samples."""
        prompts = [Shared.build_prompt_for_context(per_request_context, variant=index)
                   for index in range(level)]
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

    @staticmethod
    def _fire_measured_batch(engine, tag: str, level: int, per_request_context: int,
                             label: str, progress_stage: str = "conc_chat") -> tuple[list, str, Exception | None, float]:
        def fire():
            return ConcurrencyBenchmark._fire_batch_with_crash_retries(
                engine, tag, level, per_request_context,
            )

        outcome = fire()
        samples, status, _, _ = outcome
        if status != "ok" or not any(sample.server_tps_implausible for sample in samples):
            return outcome
        Shared.warn(f"{label}: retrying the {level}-way batch after an implausible server TPS report")
        emit_progress("measurement", progress_stage, "retrying", label)
        outcome = fire()
        samples, status, _, _ = outcome
        if status == "ok" and any(sample.server_tps_implausible for sample in samples):
            Shared.warn(f"{label}: retry also reported implausible TPS; dropping affected requests")
            emit_progress("measurement", progress_stage, "invalid", label)
        elif status == "ok":
            emit_progress("measurement", progress_stage, "valid", label)
        return outcome

    def run(self, engine, models, levels, per_request_context, warmup_runs,
            crash_cache_path: Path, section_label: str,
            stage_name: str,
            soft_exit_floor: int | None = None, force_all=False,
            save_fn=None, journal=None):  # pragma: no cover — orchestrates real engine runs
        results = journal.export() if journal else {}

        if not engine.ensure_running():
            Shared.err(f"Inference engine not reachable — skipping {section_label} benchmark")
            return results

        crash_cache = load_crash_cache(crash_cache_path)

        for model in models:
            tag   = model["tag"]
            label = model["label"]
            short = model["short"]

            Shared.section(f"{section_label} ({engine.name}): {label}")

            if not engine.reachable_or_abort():
                break

            progress_stage = stage_name
            emit_progress("model", progress_stage, "running", label, model_id=tag)
            try:
                if not engine.model_pulled(tag):
                    Shared.warn(f"{tag} not pulled — skipping")
                    Shared.warn("Download it with: python setup_check.py")
                    if journal:
                        journal.record_model_state(model, "skipped", {
                            "skipped": True, "skip_reason": "not_installed",
                        })
                    continue

                skip_entry = check_crash_cache(
                    tag, label, crash_cache, crash_cache_path, engine_name=engine.name,
                )
                if skip_entry is not None:
                    results[short] = skip_entry
                    if journal:
                        journal.record_model_state(model, "skipped", skip_entry)
                    continue

                results[short] = {}
                stopped_at = None

                pending_levels = pending_concurrency_levels(
                    levels,
                    (lambda level: journal.next_context_attempt(model, level))
                    if journal else (lambda _level: 1),
                )
                for level, attempt_number in pending_levels:
                    wait_if_paused()
                    Shared.log(f"{label}: preparing {level}-way concurrency at "
                               f"{per_request_context} tokens/slot ...")

                    if journal:
                        journal.begin_model_load()
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
                                results[short]["crashed_at"] = record_crash(
                                    tag, crash_cache, crash_cache_path,
                                    f"warming up {level}-way concurrency", engine_name=engine.name)
                                stopped_at = "crashed"
                            else:
                                Shared.err(f"{label}: {level}-way concurrency warmup failed: {error}")
                                stopped_at = "failed"
                            warmup_failed = True
                            break
                    if warmup_failed:
                        break

                    if journal:
                        journal.begin_measured()
                    Shared.log(f"{label}: firing {level} concurrent request(s) ...")
                    samples, status, error, batch_elapsed = self._fire_measured_batch(
                        engine, tag, level, per_request_context, label, progress_stage,
                    )
                    if status != "ok":
                        if journal:
                            journal.record_case(
                                model, level, str(level), samples, status, level,
                                attempt_number=attempt_number,
                            )
                        if status == "crashed":
                            Shared.err(f"{label}: engine crashed repeatedly during the {level}-way batch — "
                                       f"last server output:\n{engine.tail_log()}")
                            results[short]["crashed_at"] = record_crash(
                                tag, crash_cache, crash_cache_path,
                                f"running {level}-way concurrency", engine_name=engine.name)
                            stopped_at = "crashed"
                        else:
                            Shared.err(f"{label}: {level}-way concurrency batch failed: {error}")
                            stopped_at = "failed"
                        break
                    valid_samples = [sample for sample in samples
                                     if not measurement_validation_errors(sample)]
                    aggregate = aggregate_generation_measurements(samples, level)
                    invalid_samples = aggregate["invalid_runs"]
                    if not valid_samples:
                        Shared.err(f"{label}: {level}-way batch had no valid measurements")
                        results[short][str(level)] = {
                            **aggregate,
                        }
                        if journal:
                            journal.record_case(
                                model, level, str(level), samples, "ok", level,
                                attempt_number=attempt_number,
                            )
                        stopped_at = "invalid"
                        break
                    ttfts = [sample.client_ttft_sec for sample in valid_samples]
                    tokens = [sample.generated_tokens for sample in valid_samples]
                    tpss = [sample.tokens_per_sec for sample in valid_samples]
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
                        **aggregate,
                    }
                    if journal:
                        journal.record_case(
                            model, level, str(level), samples, "ok", level,
                            result_fields={
                                "aggregate_tps": round(aggregate_tps, 2),
                                "total_tokens": total_tokens,
                                "batch_elapsed_sec": round(batch_elapsed, 3),
                                "memory": memory,
                            },
                            attempt_number=attempt_number,
                        )
                    Shared.ok(
                        f"{level}-way done: per-request TTFT={Shared.mean(ttfts):.2f}s "
                        f"decode-only TPS={mean_tps:.1f} — serving throughput "
                        f"(incl. TTFT) {aggregate_tps:.1f} tok/s"
                    )

                    if self.should_stop_escalating(level, mean_tps, force_all, soft_exit_floor):
                        Shared.warn(f"{label}: per-request TPS ({mean_tps:.1f}) below "
                                    f"{config.SLOW_MODEL_MIN_TPS:.0f} tok/s at {level}-way "
                                    "— stopping here")
                        stopped_at = "slow"
                        break

                if stopped_at:
                    results[short]["stopped_at"] = stopped_at
                    if journal:
                        state = ("complete" if stopped_at == "slow" else
                                 "invalid" if stopped_at == "invalid" else "failed")
                        markers = {key: value for key, value in results[short].items()
                                   if key in {"stopped_at", "crashed_at", "memory_at_failure"}}
                        journal.record_model_state(model, state, markers)

                Shared.log(f"Unloading {label} ...")
                engine.unload(tag)
                engine.wait_until_unloaded(tag)
            except Exception as exc:
                Shared.err(f"{label}: unexpected error running the {section_label} benchmark — {exc} — "
                           "skipping remaining work for this model")
                entry = unexpected_model_failure(label, exc)
                results.setdefault(short, {}).update(entry)
                if journal:
                    journal.record_model_state(model, "crashed", entry)
            finally:
                if save_fn:
                    save_fn(journal.export() if journal else results)
                emit_model_finished(progress_stage, label, results.get(short), model_id=tag)

        if journal:
            journal.finish()
            return journal.export()
        return results
