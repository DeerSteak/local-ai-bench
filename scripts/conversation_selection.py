"""Pure conversation preflight selection from single-shot results."""

import config


def conv_skip_entry(model: dict, llm_data: dict | None, first_ctx_label: str,
                    force_all: bool) -> dict | None:
    label = model["label"]
    if not llm_data:
        detail = "no LLM benchmark data (checkpoint skipped or model failed)"
        return {"label": label, "skipped": True,
                "skip_reason": "no_llm_data", "skip_detail": detail}
    if llm_data.get("skipped") or llm_data.get("crashed"):
        detail = llm_data.get("skip_detail") or (
            f"The engine's runner crashed repeatedly during the LLM test "
            f"(at {llm_data['crashed']} context)"
        )
        return {"label": label, "skipped": True,
                "skip_reason": llm_data.get("skip_reason", "known_crash"),
                "skip_detail": detail}
    if llm_data.get("timed_out") == first_ctx_label:
        detail = f"LLM test timed out at {llm_data['timed_out']} context"
        return {"label": label, "skipped": True,
                "skip_reason": "timed_out", "skip_detail": detail}
    slow_ctx = None if force_all else llm_data.get("slow_tps") or (
        first_ctx_label if isinstance(llm_data.get(first_ctx_label), dict)
        and llm_data[first_ctx_label].get("tps_mean") is not None
        and llm_data[first_ctx_label]["tps_mean"] < config.SLOW_MODEL_MIN_TPS
        else None
    )
    if slow_ctx is not None:
        ctx_data = llm_data.get(slow_ctx)
        detail = (f"{ctx_data['tps_mean']:.1f} tok/s at {slow_ctx} "
                  f"context (below {config.SLOW_MODEL_MIN_TPS:.0f} tok/s cutoff)"
                  if isinstance(ctx_data, dict) and ctx_data.get("tps_mean") is not None
                  else f"below {config.SLOW_MODEL_MIN_TPS:.0f} tok/s cutoff at {slow_ctx} context")
        return {"label": label, "skipped": True,
                "skip_reason": "slow_tps", "skip_detail": detail}
    return None
