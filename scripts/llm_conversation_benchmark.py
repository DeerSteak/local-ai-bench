"""Simulates a real multi-turn chat, grown from empty toward the model's real
context ceiling. See docs/workloads.md's conversation-test section."""

from pathlib import Path

import config
from shared import Shared


class LLMConversationBenchmark:
    CONV_CRASH_CACHE = Path(".conv_crash_cache.json")

    CONV_NUM_SECTIONS = 6
    CONV_RUNS = 1   # too expensive to repeat --runs times like the other benchmarks

    CONV_TARGET_CTX = 131072   # above the top checkpoint (96K) so growth never scrapes the ceiling

    CONV_CHECKPOINTS = [0, 2048, 4096, 8192, 16384, 32768, 49152, 65536, 81920, 98304]

    # See docs/workloads.md's conversation-test growth-step paragraph.
    CONV_STEP_MIN = 32
    CONV_STEP_MAX = 1024
    CONV_STEP_MAX_FAR = 4096
    CONV_STEP_DIVISOR = 4

    CONV_OPENING_PREDICT = 2048   # opening turn is a full structured answer, not a small growth step
    CONV_CTX_HEADROOM = 4096      # room beyond the top checkpoint so growth never scrapes num_ctx
    CONV_SAFETY_MARGIN = 64       # reserved so a non-final turn can't push the next one past num_ctx

    CONV_OPENING_PROMPT = (
        "Explain Plato's Allegory of the Cave in detail. Structure your answer into "
        f"{CONV_NUM_SECTIONS} numbered sections (Section 1 through Section {CONV_NUM_SECTIONS}): "
        "the setup and the prisoners, the escape and the ascent, the sun and the Form "
        "of the Good, the return to the cave, philosophical interpretation, and modern "
        "relevance. Write several detailed paragraphs for each section."
    )

    @staticmethod
    def compute_growth_step(cumulative_tokens: int, target: int, num_ctx: int,
                             is_last_checkpoint: bool) -> tuple[int | None, bool]:
        """num_predict for the next growth turn — see docs/workloads.md's
        conversation-test growth-step paragraph. Returns (step, out_of_room)."""
        remaining = target - cumulative_tokens
        step_max = (LLMConversationBenchmark.CONV_STEP_MAX_FAR
                    if remaining > 8192
                    else LLMConversationBenchmark.CONV_STEP_MAX)
        is_final_step = remaining <= step_max
        step = (remaining if is_final_step else
                max(LLMConversationBenchmark.CONV_STEP_MIN,
                    remaining // LLMConversationBenchmark.CONV_STEP_DIVISOR))
        step = max(LLMConversationBenchmark.CONV_STEP_MIN,
                   min(step_max, step))

        reserve = 0 if (is_last_checkpoint and is_final_step) \
            else LLMConversationBenchmark.CONV_SAFETY_MARGIN
        room = num_ctx - cumulative_tokens - reserve
        if room < LLMConversationBenchmark.CONV_STEP_MIN:
            return None, True
        return min(step, room), False

    @staticmethod
    def conv_ctx_plan(model_max: int) -> tuple[int, list[int], int]:
        """Resolve (target_ctx, checkpoints, num_ctx) from a model's real max
        context — every model gets the same plan, capped only by its own ceiling."""
        target_ctx = min(model_max, LLMConversationBenchmark.CONV_TARGET_CTX)
        checkpoints = [c for c in LLMConversationBenchmark.CONV_CHECKPOINTS if c <= target_ctx]
        num_ctx = Shared.ctx_with_headroom(target_ctx, LLMConversationBenchmark.CONV_CTX_HEADROOM, model_max)
        return target_ctx, checkpoints, num_ctx

    @staticmethod
    def _conv_followup_prompt(section_n: int) -> str:
        section = ((section_n - 1) % LLMConversationBenchmark.CONV_NUM_SECTIONS) + 1
        return (
            f"Give much more detail about Section {section}, including additional "
            "examples, counterarguments, and analysis."
        )

    def run(self, engine, models, warmup_runs, force_all=False, save_fn=None):  # pragma: no cover — orchestrates real engine runs
        results = {}

        if not engine.ensure_running():
            Shared.err("Inference engine not reachable — skipping LLM conversation benchmarks")
            return results

        crash_cache = Shared.load_crash_cache(LLMConversationBenchmark.CONV_CRASH_CACHE)

        for model in models:
            tag   = model["tag"]
            label = model["label"]
            short = model["short"]

            Shared.section(f"LLM Conversation ({engine.name}): {label}")

            if not engine.reachable_or_abort():
                break

            try:
                if not engine.model_pulled(tag):
                    Shared.warn(f"{tag} not pulled — skipping")
                    Shared.warn("Download it with: python setup_check.py")
                    continue

                skip_entry = Shared.check_crash_cache(tag, label, crash_cache, LLMConversationBenchmark.CONV_CRASH_CACHE)
                if skip_entry is not None:
                    results[short] = skip_entry
                    continue

                model_max = engine.max_context_length(tag)
                target_ctx, checkpoints, num_ctx = LLMConversationBenchmark.conv_ctx_plan(model_max)
                top_checkpoint = checkpoints[-1] if checkpoints else 0

                Shared.log(f"{label}: model supports {model_max} ctx — num_ctx={num_ctx}, "
                           f"sampling up to {top_checkpoint} ({len(checkpoints)} checkpoints)")

                if not engine.warmup(tag, label, num_ctx, warmup_runs,
                                     crash_cache, LLMConversationBenchmark.CONV_CRASH_CACHE):
                    engine.unload(tag)
                    continue

                results[short] = {}
                # label -> list of (ttft, tps, depth_tokens), one entry per run that reached it
                samples_by_label = {}
                timed_out_label = None
                slow_label       = None
                crashed          = False
                crashed_label    = None

                for run_i in range(LLMConversationBenchmark.CONV_RUNS):
                    Shared.log(f"{label}: run {run_i+1}/{LLMConversationBenchmark.CONV_RUNS} — starting a fresh conversation ...")

                    messages                = []
                    cumulative_tokens       = 0
                    pending_response_tokens = 0
                    section_n               = 1
                    first_turn_done         = False
                    run_timed_out           = False
                    run_failed              = False
                    run_crashed             = False

                    def _turn(prompt_text, num_predict):
                        nonlocal cumulative_tokens, pending_response_tokens
                        messages.append({"role": "user", "content": prompt_text})
                        ttft, eval_count, tps, prompt_eval_count, response_text = engine.chat(
                            tag, messages, timeout=config.RUN_TIMEOUT, num_ctx=num_ctx,
                            num_predict=num_predict,
                        )
                        messages.append({"role": "assistant", "content": response_text})
                        # prompt_eval_count is ground truth for what's in context; eval_count isn't —
                        # a reasoning model's thinking content can get silently dropped from history next turn.
                        cumulative_tokens = prompt_eval_count
                        # Not yet in cumulative_tokens until next turn's prompt_eval_count — see docs/workloads.md.
                        pending_response_tokens = eval_count
                        return ttft, tps

                    def _next_prompt():
                        nonlocal section_n, first_turn_done
                        if not first_turn_done:
                            first_turn_done = True
                            return LLMConversationBenchmark.CONV_OPENING_PROMPT
                        prompt_text = LLMConversationBenchmark._conv_followup_prompt(section_n)
                        section_n += 1
                        return prompt_text

                    try:
                        out_of_room = False
                        for idx, target in enumerate(checkpoints):
                            label_ctx = f"{target // 1024}K" if target > 0 else "0K"
                            if target == 0:
                                # Checkpoint 0 is just the opening turn — no growth to do first.
                                ttft, tps = _turn(_next_prompt(),
                                                   LLMConversationBenchmark.CONV_OPENING_PREDICT)
                            else:
                                is_last_checkpoint = idx == len(checkpoints) - 1
                                Shared.log(f"{label}: run {run_i+1}/{LLMConversationBenchmark.CONV_RUNS} — growing toward "
                                           f"{label_ctx} (currently ~{cumulative_tokens} tokens) ...")

                                # Stop growing if we are within 0.5% of the target context length
                                target_threshold = int(target * 0.995)
                                while cumulative_tokens < target_threshold:
                                    step, ran_out = LLMConversationBenchmark.compute_growth_step(
                                        cumulative_tokens + pending_response_tokens, target_threshold,
                                        num_ctx, is_last_checkpoint)
                                    if ran_out:
                                        out_of_room = True
                                        break

                                    ttft, tps = _turn(_next_prompt(), step)

                                if out_of_room:
                                    Shared.warn(f"{label}: run {run_i+1} ran out of context room "
                                                f"approaching {label_ctx} — stopping this run's growth here")
                                    break

                            # ttft/tps here are from the turn that just crossed `target`
                            # (or the opening turn for target == 0).
                            samples_by_label.setdefault(label_ctx, []).append(
                                (ttft, tps, cumulative_tokens))
                            Shared.output(
                                f"    run {run_i+1}/{LLMConversationBenchmark.CONV_RUNS}: "
                                f"{label_ctx}  TTFT={ttft:.2f}s  TPS={tps:.1f}  "
                                f"(depth~{cumulative_tokens})"
                            )

                            # See docs/workloads.md's within-conversation slow-model early exit.
                            if not force_all and tps < config.SLOW_MODEL_MIN_TPS:
                                Shared.warn(f"{label}: run {run_i+1} — {tps:.1f} tok/s at {label_ctx} is below "
                                            f"{config.SLOW_MODEL_MIN_TPS:.0f} tok/s cutoff — ending this run here")
                                slow_label = label_ctx
                                break

                    except Exception as e:
                        is_timeout = isinstance(e, TimeoutError) or "timed out" in str(e).lower()
                        if is_timeout:
                            Shared.err(f"{label}: run {run_i+1} timed out — stopping this run here")
                            partial_text = getattr(e, "partial_text", "")
                            if partial_text:
                                # Not scored (this test measures TTFT/TPS), just surfaced to tell a stall apart from a mid-stream cutoff.
                                Shared.warn(f"{label}: run {run_i+1} had streamed "
                                            f"{len(partial_text)} chars before the timeout: "
                                            f"{partial_text[:200]!r}")
                            run_timed_out = True
                            timed_out_label = timed_out_label or f"{cumulative_tokens // 1024}K"
                        elif engine.is_connection_crash(e):
                            # Mid-turn state makes retrying unsafe — stop the run, but wait for recovery before the next model.
                            Shared.err(f"{label}: run {run_i+1} — the engine's model runner appears to have crashed "
                                       f"— last server output:\n{engine.tail_log()}")
                            if not engine.wait_for_recovery():
                                Shared.warn("The engine did not become reachable again within 30s")
                            run_crashed = True
                            crashed_label = crashed_label or f"{cumulative_tokens // 1024}K"
                        else:
                            Shared.err(f"{label}: run {run_i+1} failed: {e}")
                            run_failed = True

                    if run_crashed:
                        crashed = True

                    if run_timed_out or run_failed or run_crashed:
                        Shared.warn(f"{label}: run {run_i+1} stopped early")

                for target in checkpoints:
                    label_ctx = f"{target // 1024}K"
                    samples = samples_by_label.get(label_ctx)
                    if not samples:
                        continue
                    ttfts  = [s[0] for s in samples]
                    tpss   = [s[1] for s in samples]
                    depths = [s[2] for s in samples]
                    results[short][label_ctx] = {
                        "ttft_mean_sec":  round(Shared.mean(ttfts), 3),
                        "ttft_stdev_sec": round(Shared.stdev(ttfts), 3),
                        "tps_mean":       round(Shared.mean(tpss), 2),
                        "tps_stdev":      round(Shared.stdev(tpss), 2),
                        "n_runs":         len(samples),
                        "ttft_runs":      [round(t, 3) for t in ttfts],
                        "tps_runs":       [round(t, 2) for t in tpss],
                        "depth_tokens":   round(Shared.mean(depths)),
                    }
                    Shared.ok(
                        f"{label_ctx} done ({len(samples)} run(s)): "
                        f"TTFT={results[short][label_ctx]['ttft_mean_sec']:.2f}s  "
                        f"TPS={results[short][label_ctx]['tps_mean']:.1f}"
                    )

                if timed_out_label:
                    results[short]["timed_out"] = timed_out_label
                if slow_label:
                    results[short]["slow_tps"] = slow_label
                if crashed:
                    results[short]["crashed"] = crashed_label or "0K"
                    results[short]["crashed_at"] = Shared.record_crash(
                        tag, crash_cache, LLMConversationBenchmark.CONV_CRASH_CACHE, f"running {label}")

                Shared.log(f"Unloading {label} ...")
                engine.unload(tag)
                engine.wait_until_unloaded(tag)
            finally:
                if save_fn:
                    save_fn(results)

        return results
