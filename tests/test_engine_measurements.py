from dataclasses import replace

import pytest

from scripts.runtime.engines.base import (
    EmbeddingMeasurement, GenerationMeasurement, aggregate_generation_measurements,
    embedding_validation_errors,
    is_valid_measurement, measurement_validation_errors,
)


def valid_measurement():
    return GenerationMeasurement(
        client_ttft_sec=0.2, generated_tokens=10, tokens_per_sec=20.0,
        client_wall_sec=0.7, decode_sec=0.5, server_prompt_sec=0.15,
    )


def test_valid_measurement_accepts_distinct_client_and_server_timings():
    assert is_valid_measurement(valid_measurement())


def test_measurement_validation_rejects_nonfinite_negative_and_boolean_values():
    measurement = replace(
        valid_measurement(), client_ttft_sec=float("nan"), decode_sec=-1,
        generated_tokens=True, tokens_per_sec=float("inf"),
    )
    assert set(measurement_validation_errors(measurement)) == {
        "client_ttft_sec", "decode_sec", "generated_tokens", "tokens_per_sec",
    }


def test_measurement_validation_rejects_ttft_after_request_completion():
    errors = measurement_validation_errors(replace(
        valid_measurement(), client_ttft_sec=0.8, client_wall_sec=0.7,
    ))
    assert errors == ["client_ttft_after_wall"]


def test_measurement_validation_rejects_implausible_server_timing_marker():
    measurement = replace(valid_measurement(), server_tps_implausible=True)
    assert measurement_validation_errors(measurement) == ["implausible_server_tps"]
    result = aggregate_generation_measurements([measurement], 1)
    assert result["valid_runs"] == 0
    assert result["invalid_runs"] == [
        {"run": 1, "errors": ["implausible_server_tps"]},
    ]


def test_embedding_validation_rejects_zero_wall_time_and_bad_payload():
    assert embedding_validation_errors(EmbeddingMeasurement(None, 0)) == [
        "client_wall_sec", "embeddings",
    ]


def test_aggregate_excludes_invalid_samples_but_keeps_completed_count_and_diagnostics():
    invalid = replace(valid_measurement(), client_ttft_sec=float("nan"))
    result = aggregate_generation_measurements([valid_measurement(), invalid], 3)
    assert (result["requested_runs"], result["completed_runs"], result["valid_runs"]) == (3, 2, 1)
    assert result["n_runs"] == 2
    assert result["invalid_runs"] == [{"run": 2, "errors": ["client_ttft_sec"]}]
    assert "client_ttft_median_sec" not in result


def test_aggregate_adds_median_and_cv_with_two_valid_samples():
    result = aggregate_generation_measurements([
        valid_measurement(), replace(valid_measurement(), client_ttft_sec=0.4,
                                     client_wall_sec=0.9),
    ], 2)
    assert result["client_ttft_median_sec"] == 0.3
    assert result["client_ttft_cv"] > 0
    assert result["server_prompt_mean_sec"] == 0.15


# ── prefill throughput ──

def test_prefill_tps_divides_prompt_tokens_by_the_server_reported_prompt_time():
    from scripts.runtime.engines.base import prefill_tokens_per_sec
    assert prefill_tokens_per_sec(2048, 0.5) == 4096.0


@pytest.mark.parametrize("prompt_tokens,server_prompt_sec", [
    # An engine that reports no prompt duration must never fall back to wall time.
    (2048, None),
    (None, 0.5),
    (0, 0.5),
    (-1, 0.5),
    (2048, 0),
    (2048, -0.1),
    (2048, float("nan")),
    (2048, float("inf")),
    # Booleans are ints in Python and would otherwise divide as 1/True.
    (True, 0.5),
    (2048, True),
])
def test_prefill_tps_is_none_when_either_input_is_missing_or_implausible(
        prompt_tokens, server_prompt_sec):
    from scripts.runtime.engines.base import prefill_tokens_per_sec
    assert prefill_tokens_per_sec(prompt_tokens, server_prompt_sec) is None


def test_aggregate_reports_prefill_throughput_per_run_and_averaged():
    measurements = [
        replace(valid_measurement(), prompt_tokens=2048, server_prompt_sec=0.5),
        replace(valid_measurement(), prompt_tokens=2048, server_prompt_sec=1.0),
    ]
    aggregate = aggregate_generation_measurements(measurements, 2)
    assert aggregate["prefill_tps_runs"] == [4096.0, 2048.0]
    assert aggregate["prefill_tps_mean"] == 3072.0
    assert aggregate["prefill_tps_stdev"] > 0
    assert aggregate["valid_samples"][0]["prefill_tps"] == 4096.0
    assert aggregate["valid_samples"][0]["prompt_tokens"] == 2048


def test_aggregate_omits_prefill_throughput_when_no_run_reported_prompt_timing():
    """An engine or build without the timing source leaves the key absent rather
    than reporting a zero that would plot as a real measurement."""
    measurements = [replace(valid_measurement(), prompt_tokens=2048, server_prompt_sec=None)]
    aggregate = aggregate_generation_measurements(measurements, 1)
    assert "prefill_tps_mean" not in aggregate
    assert aggregate["valid_samples"][0]["prefill_tps"] is None


def test_aggregate_averages_only_the_runs_that_reported_prompt_timing():
    measurements = [
        replace(valid_measurement(), prompt_tokens=2048, server_prompt_sec=0.5),
        replace(valid_measurement(), prompt_tokens=2048, server_prompt_sec=None),
    ]
    aggregate = aggregate_generation_measurements(measurements, 2)
    assert aggregate["prefill_tps_runs"] == [4096.0]
    assert aggregate["prefill_tps_mean"] == 4096.0
    assert aggregate["prefill_tps_stdev"] == 0
