"""
llm_conversation_benchmark.py — simulates a real multi-turn chat (rather than
one huge padded single-shot prompt): the model explains Plato's Allegory of
the Cave in numbered sections, then each subsequent turn asks for more detail
on one section. Every turn is sent with the full message history via
/api/chat, so llama.cpp's slot cache carries the prior turns forward —
TTFT/TPS at each context depth reflect processing the new turn against an
already-filled context, not a cold fill from empty.

This test is expensive, so it runs a single conversation per model — the
--runs flag (which repeats other tests) is deliberately ignored here.
The conversation is grown from a blank slate up to the model's real context
ceiling — 128K if the model supports it, otherwise whatever it actually
supports (looked up live via Shared.ollama_model_max_ctx, not hardcoded, so
it always matches what's actually pulled). Along the way it samples
TTFT/tokens-per-sec at 0, 2K, 4K, 8K, 16K, 32K, 64K, and 96K (whichever of
those the model's ceiling reaches). The model is still given the full 128K
of context room (so 96K is reached with headroom to spare, not scraped
against the ceiling) — we just stop sampling at 96K, since 96K-to-128K is
where models with an exact 128K native ceiling have no slack left and the
growth loop's final turns risk tipping over into truncation.
"""

from pathlib import Path

import config
from shared import Shared


class LLMConversationBenchmark:
    # Records models that crashed Ollama's runner repeatedly (deterministically,
    # not a transient blip) so future runs don't waste time rediscovering the
    # same crash. Delete this file to retry a skipped model.
    CONV_CRASH_CACHE = Path(".conv_crash_cache.json")

    CONV_NUM_SECTIONS = 6

    # Every conversation run gets a single try — this test is too expensive
    # (many turns growing all the way to the context ceiling) to repeat the
    # --runs number of times like the other benchmarks do.
    CONV_RUNS = 1

    # The context window handed to a model: 128K if it supports at least
    # that much, otherwise its own real ceiling (see Shared.ollama_model_max_ctx).
    # This is deliberately higher than the highest checkpoint we sample
    # (CONV_CHECKPOINTS below tops out at 96K) so the growth loop always has
    # real headroom left against num_ctx instead of scraping the ceiling.
    CONV_TARGET_CTX = 131072

    # Checkpoints to sample: 0, then 2K doubling up to 64K, plus the 96K cap.
    # Filtered per model down to whatever its real ceiling actually reaches.
    # Deliberately stops short of the 128K context window given to the model
    # (CONV_TARGET_CTX) so there's always headroom left to grow into.
    CONV_CHECKPOINTS = [0, 2048, 4096, 8192, 16384, 32768, 65536, 98304]

    # Bounds on any single growth turn's num_predict. Crossing a gap in a
    # handful of turns (remaining // CONV_STEP_DIVISOR, clamped to this
    # range) rather than one big jump is what gives the smaller checkpoints
    # (0->2K, 2K->4K, ...) real resolution instead of one turn blowing
    # straight past them — while the cap keeps any single generation call,
    # and the gap-to-128K's turn count, bounded.
    CONV_STEP_MIN = 32
    CONV_STEP_MAX = 1024
    CONV_STEP_DIVISOR = 4

    # Opening turn is a full structured answer, not a small growth step.
    CONV_OPENING_PREDICT = 2048

    # Extra num_ctx requested beyond the top checkpoint a model will actually
    # reach, when its real ceiling allows it — growing a turn's prompt right
    # up against num_ctx with nothing to spare forces Ollama to
    # truncate/context-shift, corrupting that measurement (see the room
    # clamp in run() for how this is used).
    CONV_CTX_HEADROOM = 4096

    # Reserved tokens so a non-final growth turn's own new user message (plus
    # its generation) can't push the *next* turn's prompt past num_ctx. Not
    # applied to the very last turn of a run, which has no next turn to
    # protect and can use the full remaining room up to num_ctx itself.
    CONV_SAFETY_MARGIN = 64

    CONV_OPENING_PROMPT = (
        "Explain Plato's Allegory of the Cave in detail. Structure your answer into "
        f"{CONV_NUM_SECTIONS} numbered sections (Section 1 through Section {CONV_NUM_SECTIONS}): "
        "the setup and the prisoners, the escape and the ascent, the sun and the Form "
        "of the Good, the return to the cave, philosophical interpretation, and modern "
        "relevance. Write several detailed paragraphs for each section."
    )

    @staticmethod
    def _conv_followup_prompt(section_n: int) -> str:
        section = ((section_n - 1) % LLMConversationBenchmark.CONV_NUM_SECTIONS) + 1
        return (
            f"Give much more detail about Section {section}, including additional "
            "examples, counterarguments, and analysis."
        )

    def run(self, models, warmup_runs, force_all=False, save_fn=None):
        results = {}

        if not Shared.ollama_available():
            Shared.err("Ollama server not reachable — skipping LLM conversation benchmarks")
            Shared.err("Start with: ollama serve")
            return results

        crash_cache = Shared.load_crash_cache(LLMConversationBenchmark.CONV_CRASH_CACHE)

        for model in models:
            tag   = model["tag"]
            label = model["label"]
            short = model["short"]

            Shared.section(f"LLM Conversation: {label}")

            if not Shared.ollama_reachable_or_abort():
                break

            try:
                if not Shared.model_pulled(tag):
                    Shared.warn(f"{tag} not pulled — skipping")
                    Shared.warn(f"Pull with: ollama pull {tag}")
                    continue

                skip_entry = Shared.check_crash_cache(tag, label, crash_cache, LLMConversationBenchmark.CONV_CRASH_CACHE)
                if skip_entry is not None:
                    results[short] = skip_entry
                    continue

                model_max = Shared.ollama_model_max_ctx(tag)
                target_ctx = min(model_max, LLMConversationBenchmark.CONV_TARGET_CTX)
                checkpoints = [c for c in LLMConversationBenchmark.CONV_CHECKPOINTS if c <= target_ctx]
                num_ctx = min(target_ctx + LLMConversationBenchmark.CONV_CTX_HEADROOM, model_max)
                top_checkpoint = checkpoints[-1] if checkpoints else 0

                Shared.log(f"{label}: model supports {model_max} ctx — num_ctx={num_ctx}, "
                           f"sampling up to {top_checkpoint} ({len(checkpoints)} checkpoints)")

                if not Shared.warmup_model(tag, label, num_ctx, warmup_runs,
                                           crash_cache, LLMConversationBenchmark.CONV_CRASH_CACHE):
                    Shared.unload_model(tag)
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
                        ttft, eval_count, tps, prompt_eval_count, response_text = Shared.ollama_chat(
                            tag, messages, timeout=config.RUN_TIMEOUT, num_ctx=num_ctx,
                            num_predict=num_predict,
                        )
                        messages.append({"role": "assistant", "content": response_text})
                        # prompt_eval_count is the *total* prompt length Ollama reports for this
                        # call (ground truth of what's actually in context going into it) — even
                        # when the slot cache means only the suffix was actually recomputed
                        # (that's why ttft stays flat as this grows). We deliberately don't add
                        # eval_count on top: for reasoning models a turn's generated tokens can
                        # include large amounts of thinking content that a template silently
                        # drops from history on the next turn, so eval_count doesn't reliably
                        # predict what will actually persist. The next turn's prompt_eval_count
                        # (i.e. this same assignment, one call later) is what tells us the truth.
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
                        # Checkpoint 0 is just the opening turn — there's no
                        # growth to do first, it's the start of the conversation.
                        ttft, tps = _turn(_next_prompt(),
                                           LLMConversationBenchmark.CONV_OPENING_PREDICT)
                        samples_by_label.setdefault("0K", []).append((ttft, tps, cumulative_tokens))
                        print(f"    run {run_i+1}/{LLMConversationBenchmark.CONV_RUNS}: 0K  TTFT={ttft:.2f}s  "
                              f"TPS={tps:.1f}  (depth~{cumulative_tokens})")

                        # A model already below the cutoff on the very first turn
                        # isn't worth growing further — every deeper turn only
                        # costs more of this test's expensive growth budget to
                        # confirm what the first turn already showed. End the
                        # conversation here rather than paying to grow it toward
                        # the context ceiling.
                        if not force_all and tps < config.SLOW_MODEL_MIN_TPS:
                            Shared.warn(f"{label}: run {run_i+1} — {tps:.1f} tok/s at 0K is below "
                                        f"{config.SLOW_MODEL_MIN_TPS:.0f} tok/s cutoff — ending this run here")
                            slow_label = "0K"
                        else:
                            out_of_room = False
                            for idx, target in enumerate(checkpoints[1:], start=1):
                                label_ctx = f"{target // 1024}K"
                                is_last_checkpoint = idx == len(checkpoints) - 1
                                Shared.log(f"{label}: run {run_i+1}/{LLMConversationBenchmark.CONV_RUNS} — growing toward "
                                           f"{label_ctx} (currently ~{cumulative_tokens} tokens) ...")

                                while cumulative_tokens < target:
                                    remaining = target - cumulative_tokens
                                    is_final_step = remaining <= LLMConversationBenchmark.CONV_STEP_MAX
                                    step = (remaining if is_final_step else
                                            max(LLMConversationBenchmark.CONV_STEP_MIN,
                                                remaining // LLMConversationBenchmark.CONV_STEP_DIVISOR))
                                    step = max(LLMConversationBenchmark.CONV_STEP_MIN,
                                               min(LLMConversationBenchmark.CONV_STEP_MAX, step))

                                    # No next turn follows the very last step of the very last
                                    # checkpoint in this run, so it doesn't need margin held back
                                    # for one — it can use every token of room left up to num_ctx.
                                    reserve = 0 if (is_last_checkpoint and is_final_step) \
                                        else LLMConversationBenchmark.CONV_SAFETY_MARGIN
                                    room = num_ctx - cumulative_tokens - reserve
                                    if room < LLMConversationBenchmark.CONV_STEP_MIN:
                                        out_of_room = True
                                        break
                                    step = min(step, room)

                                    ttft, tps = _turn(_next_prompt(), step)

                                if out_of_room:
                                    Shared.warn(f"{label}: run {run_i+1} ran out of context room "
                                                f"approaching {label_ctx} — stopping this run's growth here")
                                    break

                                # ttft/tps here are from the turn that just crossed `target`
                                # (the last iteration of the while loop above).
                                samples_by_label.setdefault(label_ctx, []).append(
                                    (ttft, tps, cumulative_tokens))
                                print(f"    run {run_i+1}/{LLMConversationBenchmark.CONV_RUNS}: {label_ctx}  TTFT={ttft:.2f}s  "
                                      f"TPS={tps:.1f}  (depth~{cumulative_tokens})")

                    except Exception as e:
                        is_timeout = isinstance(e, TimeoutError) or "timed out" in str(e).lower()
                        if is_timeout:
                            Shared.err(f"{label}: run {run_i+1} timed out — stopping this run here")
                            run_timed_out = True
                            timed_out_label = timed_out_label or f"{cumulative_tokens // 1024}K"
                        elif Shared.is_connection_crash(e):
                            # Ollama's model runner subprocess died mid-conversation
                            # (commonly OOM). This test only runs one conversation per
                            # model (CONV_RUNS), and mid-turn state (the message list
                            # already has this turn's user prompt appended) makes
                            # retrying the exact turn unsafe, so just stop the run here
                            # — but wait for the server to recover before moving on to
                            # the next model, instead of letting that model's own
                            # warmup discover Ollama is still down.
                            Shared.err(f"{label}: run {run_i+1} — Ollama's model runner appears to have crashed "
                                       f"— last server output:\n{Shared.tail_ollama_log()}")
                            if not Shared.wait_for_ollama_recovery():
                                Shared.warn("Ollama did not become reachable again within 30s")
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
                Shared.unload_model(tag)
                Shared.wait_until_unloaded(tag)
            finally:
                if save_fn:
                    save_fn(results)

        return results
