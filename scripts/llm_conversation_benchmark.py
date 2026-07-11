"""
llm_conversation_benchmark.py — simulates a real multi-turn chat (rather than
one huge padded single-shot prompt): the model explains Plato's Allegory of
the Cave in numbered sections, then each subsequent turn asks for more detail
on one section. Every turn is sent with the full message history via
/api/chat, so llama.cpp's slot cache carries the prior turns forward —
TTFT/TPS at each context depth reflect processing the new turn against an
already-filled context, not a cold fill from empty.
"""

import threading
import time

import config
from shared import Shared


class LLMConversationBenchmark:
    CONV_NUM_SECTIONS    = 6
    CONV_MIN_PREDICT     = 64    # measured-phase turn size floor, smallest depths
    CONV_MAX_PREDICT     = 1024  # measured-phase turn size cap, largest depths —
                                  # longer, steadier responses reduce turn-to-turn
                                  # variance. Governs only the N_RUNS measured runs
                                  # reported per checkpoint (via _conv_turn_budget
                                  # below) — growth turns are separate, see
                                  # CONV_GROWTH_PREDICT.
    CONV_GROWTH_PREDICT  = 2000  # turn size for every growth turn (opening answer
                                  # and all "give more detail" follow-ups used to
                                  # build up to each checkpoint) — large so growth
                                  # is a handful of substantial turns rather than
                                  # dozens of small ones, more like a real
                                  # conversation. Doesn't affect what's measured;
                                  # some overshoot past each checkpoint is expected
                                  # and fine.

    CONV_OPENING_PROMPT = (
        "Explain Plato's Allegory of the Cave in detail. Structure your answer into "
        f"{CONV_NUM_SECTIONS} numbered sections (Section 1 through Section {CONV_NUM_SECTIONS}): "
        "the setup and the prisoners, the escape and the ascent, the sun and the Form "
        "of the Good, the return to the cave, philosophical interpretation, and modern "
        "relevance. Write several detailed paragraphs for each section."
    )

    @staticmethod
    def _conv_turn_budget(ctx_len: int) -> int:
        """
        Per-turn generation length, scaled to the depth currently being targeted.

        Small checkpoints (e.g. 2K) use short turns so the crossing overshoot stays
        a small fraction of the target — otherwise a single 1024-token turn could
        blow straight past a 2K checkpoint. Large checkpoints (e.g. 64K) use long
        turns so we're not spending hundreds of tiny turns to get there.
        """
        return max(LLMConversationBenchmark.CONV_MIN_PREDICT,
                   min(LLMConversationBenchmark.CONV_MAX_PREDICT, ctx_len // 32))

    @staticmethod
    def _conv_followup_prompt(section_n: int) -> str:
        section = ((section_n - 1) % LLMConversationBenchmark.CONV_NUM_SECTIONS) + 1
        return (
            f"Give much more detail about Section {section}, including additional "
            "examples, counterarguments, and analysis."
        )

    def run(self, models, context_lengths, warmup_runs, force_all=False):
        results = {}

        if not Shared.ollama_available():
            Shared.err("Ollama server not reachable — skipping LLM conversation benchmarks")
            Shared.err("Start with: ollama serve")
            return results

        for model in models:
            tag   = model["tag"]
            label = model["label"]
            short = model["short"]

            Shared.section(f"LLM Conversation: {label}")

            if not Shared.model_pulled(tag):
                Shared.warn(f"{tag} not pulled — skipping")
                Shared.warn(f"Pull with: ollama pull {tag}")
                continue

            # Unlike the single-shot test, this session keeps growing past each
            # checkpoint within the *same* num_ctx for the whole conversation. If
            # num_ctx == the top checkpoint exactly, growth lands right on the
            # ceiling with zero headroom — Ollama has to truncate/context-shift and
            # fully reprocess, which looks like a cache miss (a 100x+ TTFT spike)
            # rather than the incremental cost we're trying to measure. Pad the
            # ceiling so the top checkpoint's crossing overshoot (one growth turn,
            # up to CONV_GROWTH_PREDICT) plus all its measured turns (up to
            # CONV_MAX_PREDICT each, plus 2 extra turns of buffer) still fit
            # comfortably inside num_ctx.
            headroom = (LLMConversationBenchmark.CONV_GROWTH_PREDICT
                        + (config.N_RUNS + 2) * LLMConversationBenchmark.CONV_MAX_PREDICT)
            session_ctx_ceiling = max(context_lengths) + headroom
            max_ctx = min(model.get("max_ctx", session_ctx_ceiling), session_ctx_ceiling)
            Shared.log(f"Warming up {label} at num_ctx={max_ctx} (timeout: {config.RUN_TIMEOUT}s per run) ...")
            warmup_ok = True
            for warmup_i in range(warmup_runs):
                result_box = [None]
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
                                 if c <= model.get("max_ctx", session_ctx_ceiling)]

            messages          = []
            cumulative_tokens = 0
            section_n         = 1
            first_turn_done   = False
            model_timed_out   = False
            model_failed      = False

            def _turn(prompt_text, num_predict):
                nonlocal cumulative_tokens
                messages.append({"role": "user", "content": prompt_text})
                ttft, eval_count, tps, prompt_eval_count, response_text = Shared.ollama_chat(
                    tag, messages, timeout=config.RUN_TIMEOUT, num_ctx=max_ctx,
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

            for ctx_len in model_ctx_lengths:
                label_ctx = f"{ctx_len // 1024}K"
                turn_budget = LLMConversationBenchmark._conv_turn_budget(ctx_len)

                # Grow the conversation (untimed) until we've crossed this depth,
                # using large CONV_GROWTH_PREDICT turns regardless of this
                # checkpoint's (much smaller) measured-phase turn_budget — see
                # CONV_GROWTH_PREDICT above for why. Once within CONV_GROWTH_PREDICT
                # of the ceiling, shrink num_predict to the remaining gap instead of
                # blindly using the full budget, so the crossing turn lands close to
                # the checkpoint rather than potentially overshooting it by up to
                # CONV_GROWTH_PREDICT tokens. num_predict is a cap, not a target —
                # a turn can still come in short (early stop) — so this can still
                # take more than one closing turn; the loop just keeps recomputing
                # the remaining gap each time.
                Shared.log(f"Conversation depth {label_ctx} — growing context "
                           f"(currently ~{cumulative_tokens} tokens) ...")
                try:
                    while cumulative_tokens < ctx_len:
                        remaining = ctx_len - cumulative_tokens
                        num_predict = (LLMConversationBenchmark.CONV_GROWTH_PREDICT
                                       if remaining > LLMConversationBenchmark.CONV_GROWTH_PREDICT
                                       else max(LLMConversationBenchmark.CONV_MIN_PREDICT, remaining))
                        _turn(_next_prompt(), num_predict)
                except Exception as e:
                    is_timeout = isinstance(e, TimeoutError) or "timed out" in str(e).lower()
                    if is_timeout:
                        Shared.err(f"Timed out growing context for {label} — skipping remaining depths")
                        model_timed_out = True
                    else:
                        Shared.err(f"Failed growing context for {label}: {e}")
                        model_failed = True
                    break

                Shared.log(f"Context {label_ctx} (~{cumulative_tokens} tokens actual) — "
                           f"{config.N_RUNS} runs ...")

                ttfts, tps_list = [], []
                ctx_timed_out = False

                for run_i in range(config.N_RUNS):
                    try:
                        ttft, tps = _turn(_next_prompt(), turn_budget)
                        ttfts.append(ttft)
                        tps_list.append(tps)
                        print(
                            f"    run {run_i+1}/{config.N_RUNS}: "
                            f"TTFT={ttft:.2f}s  "
                            f"TPS={tps:.1f}  "
                            f"(depth~{cumulative_tokens})"
                        )
                    except Exception as e:
                        is_timeout = isinstance(e, TimeoutError) or "timed out" in str(e).lower()
                        if is_timeout:
                            Shared.err(f"Run {run_i+1} timed out — skipping remaining runs and depths for {label}")
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
                        "depth_tokens":   cumulative_tokens,
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

            if model_timed_out or model_failed:
                Shared.warn(f"{label}: stopped early — moving to next model")
            Shared.log(f"Unloading {label} ...")
            Shared.unload_model(tag)
            Shared.wait_until_unloaded(tag)

        return results
