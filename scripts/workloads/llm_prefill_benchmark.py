"""llm_prefill_benchmark.py — single-shot LLM prefill/decode benchmark."""

from pathlib import Path

from scripts.runtime import config
from scripts.runtime.engines.base import aggregate_generation_measurements, measurement_validation_errors
from scripts.runtime.shared import Shared
from scripts.runtime.progress_events import emit_model_finished, emit_progress


class LLMPrefillBenchmark:
    LLM_CRASH_CACHE = Path(".llm_crash_cache.json")  # see docs/project-structure.md

    @staticmethod
    def prefill_server_ctx(ctx_len: int, model_max: int) -> int:
        """num_ctx to load the server at for a padded-to-ctx_len prompt — headroom
        for the n_predict generation on top, clamped to the model's real max."""
        return Shared.ctx_with_headroom(ctx_len, config.GENERATE_MAX_TOKENS, model_max)

    def run(self, engine, models, context_lengths, warmup_runs, force_all=False,
            save_fn=None, journal=None):  # pragma: no cover — orchestrates real engine runs
        results = journal.export() if journal else {}

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

            emit_progress("model", "llm", "running", label, model_id=tag)
            try:
                if not engine.model_pulled(tag):
                    Shared.warn(f"{tag} not pulled — skipping")
                    Shared.warn("Download it with: python setup_check.py")
                    continue

                skip_entry = Shared.check_crash_cache(tag, label, crash_cache, LLMPrefillBenchmark.LLM_CRASH_CACHE,
                                       engine_name=engine.name)
                if skip_entry is not None:
                    results[short] = skip_entry
                    if journal:
                        journal.record_model_state(model, "skipped", skip_entry)
                    continue

                model_max = engine.max_context_length(tag)
                results[short] = {}

                model_ctx_lengths = [c for c in context_lengths if c <= model_max]
                Shared.log(f"{label}: model supports {model_max} ctx")

                model_timed_out = False
                for ctx_len in model_ctx_lengths:
                    label_ctx = Shared.context_label(ctx_len)
                    attempt_number = journal.next_context_attempt(model, ctx_len) if journal else 1
                    if attempt_number is None:
                        Shared.log(f"Context {label_ctx} already complete — preserving it")
                        continue
                    server_ctx = LLMPrefillBenchmark.prefill_server_ctx(ctx_len, model_max)
                    if server_ctx <= ctx_len:
                        Shared.warn(f"{label}: no headroom left for generation at {label_ctx} "
                                    f"(model max {model_max}) — TPS at this depth may read as ~0")
                    if not engine.warmup(tag, label, server_ctx, warmup_runs,
                                         crash_cache, LLMPrefillBenchmark.LLM_CRASH_CACHE):
                        results[short]["crashed"] = label_ctx
                        results[short]["crashed_at"] = crash_cache.get(tag, {}).get(
                            "crashed_at", "during warmup",
                        )
                        if journal:
                            journal.record_case(
                                model, ctx_len, label_ctx, [], "crashed", config.N_RUNS,
                                {"crashed": label_ctx, "crashed_at": results[short]["crashed_at"]},
                                attempt_number=attempt_number,
                            )
                        engine.unload(tag)
                        break
                    Shared.log(f"Context {label_ctx} — {config.N_RUNS} runs ...")

                    def _prefill_once(run_i):
                        prompt = Shared.build_prompt_for_context(ctx_len)
                        measurement = Shared.retry_implausible_tps(
                            lambda: engine.generate(
                                tag, prompt, timeout=config.RUN_TIMEOUT, num_ctx=server_ctx,
                            ),
                            f"{tag} {label_ctx} run {run_i + 1}",
                            "llm",
                        )
                        ttft = measurement.client_ttft_sec
                        tps = measurement.tokens_per_sec
                        Shared.output(
                            f"    run {run_i+1}/{config.N_RUNS}: "
                            f"TTFT={ttft:.2f}s  "
                            f"TPS={tps:.1f}"
                        )
                        return measurement

                    samples, status, _, _metadata = Shared.run_measured_calls(
                        config.N_RUNS, _prefill_once, tag, crash_cache,
                        LLMPrefillBenchmark.LLM_CRASH_CACHE, f"running {label}", engine)
                    aggregate = aggregate_generation_measurements(samples, config.N_RUNS)
                    valid_samples = [sample for sample in samples
                                     if not measurement_validation_errors(sample)]
                    ttfts = [sample.client_ttft_sec for sample in valid_samples]
                    tps_list = [sample.tokens_per_sec for sample in valid_samples]

                    if ttfts:
                        results[short][label_ctx] = {
                            "ttft_mean_sec":  round(Shared.mean(ttfts),    3),
                            "ttft_stdev_sec": round(Shared.stdev(ttfts),   3),
                            "tps_mean":       round(Shared.mean(tps_list), 2),
                            "tps_stdev":      round(Shared.stdev(tps_list),2),
                            "ttft_runs":      [round(t, 3) for t in ttfts],
                            "tps_runs":       [round(t, 2) for t in tps_list],
                            **aggregate,
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
                        if journal:
                            journal.record_case(
                                model, ctx_len, label_ctx, samples, status, config.N_RUNS,
                                {"timed_out": label_ctx},
                                attempt_number=attempt_number,
                            )
                        break

                    if status == "crashed":
                        crashed_at = crash_cache.get(tag, {}).get("crashed_at", "an earlier run")
                        results[short]["crashed"] = label_ctx
                        results[short]["crashed_at"] = crashed_at
                        if journal:
                            journal.record_case(
                                model, ctx_len, label_ctx, samples, status, config.N_RUNS,
                                {"crashed": label_ctx, "crashed_at": crashed_at},
                                attempt_number=attempt_number,
                            )
                        break

                    is_first_ctx = ctx_len == model_ctx_lengths[0]
                    stopped_slow = Shared.slow_tps_early_exit(
                        results, short, label, label_ctx, is_first_ctx, tps_list, force_all,
                    )
                    if journal:
                        markers = ({"slow_tps": results[short]["slow_tps"]}
                                   if "slow_tps" in results[short] else {})
                        journal.record_case(
                            model, ctx_len, label_ctx, samples, status, config.N_RUNS, markers,
                            attempt_number=attempt_number,
                        )
                    if stopped_slow:
                        break

                if model_timed_out:
                    Shared.warn(f"{label}: timed out — moving to next model")
                Shared.log(f"Unloading {label} ...")
                engine.unload(tag)
                engine.wait_until_unloaded(tag)
            except Exception as exc:
                Shared.err(f"{label}: unexpected error running the LLM benchmark — {exc} — "
                           "skipping remaining work for this model")
                entry = Shared.unexpected_model_failure(label, exc)
                results.setdefault(short, {}).update(entry)
                if journal:
                    journal.record_model_state(model, "crashed", entry)
            finally:
                if save_fn:
                    save_fn(journal.export() if journal else results)
                emit_model_finished("llm", label, results.get(short), model_id=tag)

        if journal:
            journal.finish()
            return journal.export()
        return results
