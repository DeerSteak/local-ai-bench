"""Continuous-generation soak workload with aligned resource telemetry."""

import math
import os
from pathlib import Path
import time
from typing import Mapping, Sequence

from scripts.runtime import config
from scripts.runtime.crash_cache import check_crash_cache, load_crash_cache
from scripts.runtime.engines.base import measurement_validation_errors
from scripts.runtime.failure_handling import unexpected_model_failure
from scripts.runtime.progress_events import emit_model_finished, emit_progress
from scripts.runtime.pause_control import PAUSE_CONTROL_ENV, read_pause_state, wait_if_paused
from scripts.runtime.shared import Shared
from scripts.workloads.sustained_analysis import analyze_sustained_series


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _samples(block: Mapping[str, object] | None) -> list[Mapping[str, object]]:
    if not isinstance(block, Mapping):
        return []
    result = []
    windows = block.get("windows")
    if not isinstance(windows, list):
        return result
    for window in windows:
        if not isinstance(window, Mapping) or not str(window.get("name", "")).startswith("measured"):
            continue
        values = window.get("samples")
        if isinstance(values, list):
            result.extend(value for value in values if isinstance(value, Mapping))
    return result


def _window_mean(samples: Sequence[Mapping[str, object]], key: str,
                 start: float, end: float, offset: float) -> float | None:
    values = [
        value for sample in samples
        if (timestamp := _number(sample.get("timestamp_sec"))) is not None
        and start <= timestamp - offset < end
        and (value := _number(sample.get(key))) is not None
    ]
    return sum(values) / len(values) if values else None


def aligned_sustained_windows(
        requests: Sequence[Mapping[str, object]], duration_sec: float, window_sec: float, *,
        measured_offset_sec: float = 0,
        memory: Mapping[str, object] | None = None,
        power: Mapping[str, object] | None = None,
        temperature: Mapping[str, object] | None = None) -> list[dict[str, object]]:
    if duration_sec <= 0 or window_sec <= 0:
        raise ValueError("sustained duration and window must be positive")
    memory_samples = _samples(memory)
    power_samples = _samples(power)
    temperature_samples = _samples(temperature)
    windows = []
    start = 0.0
    while start < duration_sec:
        end = min(start + window_sec, duration_sec)
        tokens = 0.0
        for request in requests:
            request_start = _number(request.get("start_sec"))
            request_end = _number(request.get("end_sec"))
            generated = _number(request.get("generated_tokens"))
            if (request_start is None or request_end is None or generated is None
                    or request_end <= request_start or generated < 0):
                continue
            overlap = max(0.0, min(end, request_end) - max(start, request_start))
            tokens += generated * overlap / (request_end - request_start)
        duration = end - start
        windows.append({
            "timestamp_sec": start,
            "duration_sec": duration,
            "tokens": tokens,
            "tokens_per_sec": tokens / duration,
            "host_ram_used_gb": _window_mean(
                memory_samples, "host_ram_used_gb", start, end, measured_offset_sec,
            ),
            "process_rss_gb": _window_mean(
                memory_samples, "process_rss_gb", start, end, measured_offset_sec,
            ),
            "accelerator_memory_used_gb": _window_mean(
                memory_samples, "accelerator_memory_used_gb", start, end, measured_offset_sec,
            ),
            "power_watts": _window_mean(
                power_samples, "watts", start, end, measured_offset_sec,
            ),
            "cpu_package_c": _window_mean(
                temperature_samples, "cpu_package_c", start, end, measured_offset_sec,
            ),
            "gpu_die_c": _window_mean(
                temperature_samples, "gpu_die_c", start, end, measured_offset_sec,
            ),
            "gpu_hotspot_c": _window_mean(
                temperature_samples, "gpu_hotspot_c", start, end, measured_offset_sec,
            ),
        })
        start = end
    return windows


def sustained_measurement_valid(request_count: int, valid_request_count: int) -> bool:
    return request_count > 0 and valid_request_count == request_count


class SustainedBenchmark:
    CRASH_CACHE = Path(".sustained_crash_cache.json")

    def run(self, engine, models, warmup_runs, duration_sec=config.SUSTAINED_DURATION_SEC,
            window_sec=config.SUSTAINED_WINDOW_SEC, ambient_temp_c=None, save_fn=None,
            telemetry_factory=None, journal=None, monotonic=time.monotonic):  # pragma: no cover
        results = journal.export() if journal else {}
        if not engine.ensure_running():
            Shared.err("Inference engine not reachable — skipping sustained benchmark")
            return results
        crash_cache = load_crash_cache(self.CRASH_CACHE)
        for model in models:
            tag, label, short = model["tag"], model["label"], model["short"]
            Shared.section(f"Sustained load ({engine.name}): {label}")
            emit_progress("model", "sustained", "running", label, model_id=tag)
            telemetry = journal.telemetry if journal else None
            owns_telemetry = False
            attempt_number = journal.next_attempt(model) if journal else 1
            if attempt_number is None:
                continue
            case_started = False
            telemetry_finished = False
            try:
                if not engine.model_pulled(tag):
                    Shared.warn(f"{tag} not pulled — skipping")
                    continue
                skip = check_crash_cache(
                    tag, label, crash_cache, self.CRASH_CACHE, engine_name=engine.name,
                )
                if skip is not None:
                    results[short] = skip
                    if journal:
                        journal.record_model_state(model, "skipped", skip)
                    continue
                model_max = engine.max_context_length(tag)
                server_context = Shared.ctx_with_headroom(
                    config.SUSTAINED_CONTEXT_TOKENS, config.GENERATE_MAX_TOKENS, model_max,
                )
                if server_context <= config.SUSTAINED_CONTEXT_TOKENS:
                    results[short] = {
                        "skipped": "context_unsupported",
                        "required_context_tokens": config.SUSTAINED_CONTEXT_TOKENS,
                        "model_max_context_tokens": model_max,
                    }
                    if journal:
                        journal.record_model_state(model, "skipped", results[short])
                    continue
                if telemetry is None and telemetry_factory:
                    telemetry = telemetry_factory().start()
                    owns_telemetry = True
                if telemetry:
                    telemetry.begin_model_load()
                if not engine.warmup(
                        tag, label, server_context, warmup_runs,
                        crash_cache, self.CRASH_CACHE):
                    results[short] = {"crashed": "during warmup"}
                    if journal:
                        journal.record_model_state(model, "failed", results[short])
                    continue
                if journal:
                    journal.begin_case(model, attempt_number)
                    case_started = True
                boundary = telemetry.begin_measured("measured:sustained") if telemetry else None
                started = monotonic()
                requests = []
                pause_invalidated = False
                prompt = Shared.build_prompt_for_context(config.SUSTAINED_CONTEXT_TOKENS)
                while monotonic() - started < duration_sec:
                    pause_path = os.environ.get(PAUSE_CONTROL_ENV)
                    if pause_path:
                        try:
                            pause_requested = read_pause_state(Path(pause_path)) == "paused"
                        except RuntimeError:
                            pause_requested = False
                        if pause_requested and telemetry:
                            telemetry.begin_pause()
                        if wait_if_paused():
                            pause_invalidated = True
                            if telemetry:
                                telemetry.begin_measured("measured:sustained")
                    request_start = monotonic() - started
                    measurement = engine.generate(
                        tag, prompt, timeout=config.RUN_TIMEOUT,
                        num_ctx=server_context,
                    )
                    request_end = monotonic() - started
                    requests.append({
                        "start_sec": request_start,
                        "end_sec": request_end,
                        "generated_tokens": measurement.generated_tokens,
                        "tokens_per_sec": measurement.tokens_per_sec,
                        "validation_errors": measurement_validation_errors(measurement),
                    })
                    if journal:
                        journal.record_request(
                            model, attempt_number, len(requests), measurement,
                            request_start, request_end,
                        )
                    elif save_fn:
                        save_fn(results)
                actual_duration = monotonic() - started
                memory = telemetry.finish_case() if telemetry else None
                telemetry_finished = telemetry is not None
                power = telemetry.last_power if telemetry else None
                temperature = telemetry.last_temperature if telemetry else None
                valid_requests = [request for request in requests if not request["validation_errors"]]
                windows = aligned_sustained_windows(
                    valid_requests, actual_duration, window_sec,
                    measured_offset_sec=boundary.timestamp_sec if boundary else 0,
                    memory=memory, power=power, temperature=temperature,
                )
                analysis = analyze_sustained_series(
                    windows,
                    measurement_valid=sustained_measurement_valid(
                        len(requests), len(valid_requests),
                    ),
                )
                if pause_invalidated:
                    analysis.update({
                        "performance": "indeterminate", "cause": "unavailable",
                        "retention_ratio": None, "throttle_onset_sec": None,
                    })
                results[short] = {
                    "context_tokens": config.SUSTAINED_CONTEXT_TOKENS,
                    "server_context_tokens": server_context,
                    "target_duration_sec": duration_sec,
                    "actual_duration_sec": actual_duration,
                    "window_sec": window_sec,
                    "ambient_temp_c": ambient_temp_c,
                    "pause_invalidated": pause_invalidated,
                    "request_count": len(requests),
                    "valid_request_count": len(valid_requests),
                    "requests": requests,
                    "series": windows,
                    "analysis": analysis,
                    "memory": memory,
                    "power": power,
                    "temperature": temperature,
                }
                if journal:
                    journal.complete_case(model, attempt_number, results[short])
                Shared.ok(
                    f"{label}: {analysis['performance']} — "
                    f"retention {analysis['retention_ratio'] if analysis['retention_ratio'] is not None else 'unknown'}"
                )
            except Exception as exc:
                Shared.err(f"{label}: sustained benchmark failed — {exc}")
                results[short] = unexpected_model_failure(label, exc)
                if telemetry and case_started and not telemetry_finished:
                    results[short]["memory"] = telemetry.finish_case()
                    results[short]["power"] = telemetry.last_power
                    results[short]["temperature"] = telemetry.last_temperature
                if journal:
                    if case_started:
                        journal.complete_case(model, attempt_number, results[short], "failed")
                    else:
                        journal.record_model_state(model, "failed", results[short])
            finally:
                if telemetry and owns_telemetry:
                    telemetry.stop()
                engine.unload(tag)
                engine.wait_until_unloaded(tag)
                if save_fn:
                    save_fn(results)
                emit_model_finished("sustained", label, results.get(short), model_id=tag)
        if journal:
            journal.finish()
            return journal.export()
        return results
