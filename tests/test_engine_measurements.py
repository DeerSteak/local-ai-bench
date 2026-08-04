from dataclasses import replace

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
