"""
llm_conversation_benchmark.py — simulates a real multi-turn chat (rather than
one padded single-shot prompt): the model explains Plato's Allegory of the
Cave in sections, then each turn asks for more detail on one. Every turn
sends the full history via /api/chat, so the slot cache carries prior turns
forward — TTFT/TPS at each depth reflect a new turn against an already-filled
context, not a cold fill. Expensive, so it runs one conversation per model
(--runs ignored), grown from empty toward the model's real context ceiling,
sampling at 0, 2K, 4K, 8K, 16K, 32K, 48K, 64K, 80K, and 96K. The
xsmall/small tiers stop at 48K. See docs/workloads.md.
"""

from pathlib import Path

import config
from shared import Shared


class LLMConversationBenchmark:
    # Deterministic-crash memo; delete to retry a skipped model.
    CONV_CRASH_CACHE = Path(".conv_crash_cache.json")

    CONV_NUM_SECTIONS = 6
    CONV_RUNS = 1   # too expensive to repeat --runs times like the other benchmarks

    # Higher than the top sampled checkpoint (96K) so the growth loop always has headroom, never scrapes the ceiling.
    CONV_TARGET_CTX = 131072

    # xsmall/small models can have a native context window (e.g. Qwen3.5's 256K) far
    # beyond what a constrained-memory machine can actually reserve KV-cache for — a small
    # model's value is being cheap to run everywhere, not exercising every token it's rated
    # for. Cap their context budget and top sampled checkpoint at half the large-tier ones,
    # same headroom ratio as CONV_TARGET_CTX leaves above the 96K checkpoint.
    CONV_SMALL_TIERS = {"xsmall", "small"}
    CONV_SMALL_TIER_TARGET_CTX = 65536
    CONV_SMALL_TIER_TOP_CHECKPOINT = 49152

    # 2K doubling to 64K, plus 48K/80K in the higher gaps and the 96K cap. Filtered per model to its real ceiling.
    CONV_CHECKPOINTS = [0, 2048, 4096, 8192, 16384, 32768, 49152, 65536, 81920, 98304]

    # Growth-turn num_predict bounds — crossing a gap in several steps (remaining // CONV_STEP_DIVISOR)
    # gives the smaller checkpoints real resolution instead of overshooting them.
    CONV_STEP_MIN = 32
    CONV_STEP_MAX = 1024
    CONV_STEP_MAX_FAR = 4096
    CONV_STEP_DIVISOR = 4

    CONV_OPENING_PREDICT = 2048   # opening turn is a full structured answer, not a small growth step

    # Room beyond the top checkpoint reached — growing right up against num_ctx forces a truncating context-shift.
    CONV_CTX_HEADROOM = 4096

    # Reserves room so a non-final turn's new message + generation can't push the *next* turn past num_ctx.
    CONV_SAFETY_MARGIN = 64

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
        """Compute num_predict for the next growth turn toward `target` in
        bounded steps, reserving CONV_SAFETY_MARGIN for the next turn (except
        the last step of the last checkpoint). Returns (step, out_of_room);
        step is None when out_of_room, meaning the caller should stop growing."""
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
    def conv_ctx_plan(tier: str | None, model_max: int) -> tuple[int, list[int], int]:
        """Resolve a model's conversation-test context budget from its tier and
        real max context: (target_ctx, checkpoints, num_ctx). xsmall/small-tier
        models get CONV_SMALL_TIER_TARGET_CTX/CONV_SMALL_TIER_TOP_CHECKPOINT
        instead of the full CONV_TARGET_CTX/CONV_CHECKPOINTS ceiling — see those
        constants' docstring. An unrecognized/missing tier (e.g. a custom
        --llm-models tag outside the catalog) falls back to the uncapped behavior."""
        is_small_tier = tier in LLMConversationBenchmark.CONV_SMALL_TIERS
        target_ctx_cap = (LLMConversationBenchmark.CONV_SMALL_TIER_TARGET_CTX if is_small_tier
                          else LLMConversationBenchmark.CONV_TARGET_CTX)
        target_ctx = min(model_max, target_ctx_cap)
        checkpoints = [c for c in LLMConversationBenchmark.CONV_CHECKPOINTS if c <= target_ctx]
        if is_small_tier:
            checkpoints = [c for c in checkpoints
                           if c <= LLMConversationBenchmark.CONV_SMALL_TIER_TOP_CHECKPOINT]
        num_ctx = min(target_ctx + LLMConversationBenchmark.CONV_CTX_HEADROOM, model_max)
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
                target_ctx, checkpoints, num_ctx = LLMConversationBenchmark.conv_ctx_plan(
                    model.get("tier"), model_max)
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

                    messages          = []
                    cumulative_tokens = 0
                    section_n         = 1
                    first_turn_done   = False
                    run_timed_out     = False
                    run_failed        = False
                    run_crashed       = False

                    def _turn(prompt_text, num_predict):
                        nonlocal cumulative_tokens
                        messages.append({"role": "user", "content": prompt_text})
                        ttft, eval_count, tps, prompt_eval_count, response_text = engine.chat(
                            tag, messages, timeout=config.RUN_TIMEOUT, num_ctx=num_ctx,
                            num_predict=num_predict,
                        )
                        messages.append({"role": "assistant", "content": response_text})
                        # prompt_eval_count is ground truth for what's in context; eval_count isn't —
                        # a reasoning model's thinking content can get silently dropped from history next turn.
                        cumulative_tokens = prompt_eval_count
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
                                        cumulative_tokens, target_threshold, num_ctx, is_last_checkpoint)
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

                            # A model below the cutoff at any depth isn't worth growing further —
                            # deeper turns only spend more of the expensive growth budget to
                            # confirm what we've already shown.
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
