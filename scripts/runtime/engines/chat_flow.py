"""Engine-neutral bounded chat finalization and measurement aggregation."""

import json
import time
from collections.abc import Callable

from scripts.runtime.engines.base import ChatMeasurement
from scripts.runtime.shared import EngineBudgetExceeded, EngineTimeout, split_token_budget


def graded_response(result: dict, tools: list | None) -> str:
    if tools is not None and result["tool_calls"]:
        return json.dumps(result["tool_calls"])
    return result["response_text"]


def validate_chat_budget(num_predict: int, token_budget: int | None) -> None:
    if token_budget is not None and num_predict != -1:
        raise ValueError("token_budget cannot be combined with finite num_predict")


def run_bounded_chat(request: Callable, messages: list, tools: list | None,
                     deadline: float, num_predict: int, check_loop: bool,
                     token_budget: int | None, finalize_fraction: float,
                     finalize_message: str, operation: str) -> tuple[dict, dict | None, bool]:
    validate_chat_budget(num_predict, token_budget)
    if token_budget is None:
        return request(messages, tools, deadline, num_predict, check_loop, False), None, False

    first_budget, second_budget = split_token_budget(token_budget, finalize_fraction)
    first = request(messages, tools, deadline, first_budget, check_loop, False)
    if first["finish_reason"] != "length":
        return first, None, False
    first_response = graded_response(first, tools)
    if second_budget == 0:
        raise EngineBudgetExceeded(
            f"{operation} exhausted its completion-token budget",
            partial_text=first_response, budget_nudged=False,
        )
    if time.perf_counter() >= deadline:
        raise EngineTimeout(
            f"{operation} exceeded its wall-clock deadline before finalization",
            partial_text=first_response,
        )
    followup = [dict(message) for message in messages]
    followup.extend([
        {"role": "assistant", "content": first_response},
        {"role": "user", "content": finalize_message},
    ])
    second = request(followup, tools, deadline, second_budget, check_loop, True)
    if second["finish_reason"] == "length":
        raise EngineBudgetExceeded(
            f"{operation} exhausted its completion-token budget",
            partial_text=graded_response(second, tools),
        )
    return first, second, True


def chat_measurement(first: dict, second: dict | None, budget_nudged: bool,
                     model_load_sec: float, sanitize_tps: Callable[[float, int, float, float], float]
                     | None = None, cpu_offload_gb: int = 0,
                     model_placement: dict | None = None) -> ChatMeasurement:
    graded = second or first
    parts = [first] if second is None else [first, second]
    tokens = sum(part["tokens"] for part in parts)
    decode_seconds = sum(part["decode_seconds"] for part in parts)
    wall_seconds = sum(part["wall_seconds"] for part in parts)
    raw_tps = tokens / decode_seconds if decode_seconds else 0
    tps = sanitize_tps(raw_tps, tokens, first["ttft"], wall_seconds) \
        if sanitize_tps is not None else raw_tps
    return ChatMeasurement(
        client_ttft_sec=first["ttft"], generated_tokens=tokens, tokens_per_sec=tps,
        client_wall_sec=wall_seconds, decode_sec=decode_seconds,
        server_prompt_sec=first["server_prompt_sec"],
        prompt_tokens=graded["prompt_eval_count"], response_text=graded["response_text"],
        finish_reason=graded["finish_reason"], tool_calls=graded["tool_calls"],
        budget_nudged=budget_nudged, model_load_sec=model_load_sec,
        server_tps_implausible=(
            tps != raw_tps or any(part.get("server_tps_implausible", False) for part in parts)
        ),
        cpu_offload_gb=cpu_offload_gb,
        gpu_layers=(model_placement or {}).get("gpu_layers"),
        total_layers=(model_placement or {}).get("total_layers"),
        cpu_model_buffer_gb=(model_placement or {}).get("cpu_model_buffer_gb"),
    )
