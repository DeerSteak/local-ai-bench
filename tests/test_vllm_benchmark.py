from pathlib import Path

import pytest

from scripts.runtime import config
from scripts.workloads.vllm_benchmark import VllmBenchBenchmark


def test_error_log_excerpt_keeps_short_output_unchanged():
    assert VllmBenchBenchmark.error_log_excerpt("  root cause\ntraceback  ", 100) == (
        "root cause\ntraceback"
    )


def test_error_log_excerpt_keeps_both_ends_of_long_output():
    output = "ROOT" + ("x" * 20) + "TRACE"
    excerpt = VllmBenchBenchmark.error_log_excerpt(output, 10)
    assert excerpt.startswith("ROOTx")
    assert excerpt.endswith("TRACE")
    assert "19 characters omitted" in excerpt


def test_latency_command_pins_iteration_counts_instead_of_vllm_defaults():
    command = VllmBenchBenchmark.build_latency_command(
        "/venv/bin/vllm", "org/model", Path("/tmp/out.json"), 2048, 128,
    )
    assert command[:3] == ["/venv/bin/vllm", "bench", "latency"]
    assert command[command.index("--model") + 1] == "org/model"
    assert command[command.index("--input-len") + 1] == "2048"
    assert command[command.index("--output-len") + 1] == "128"
    assert command[command.index("--output-json") + 1] == "/tmp/out.json"
    # vllm's own defaults are 30 iterations and 10 warmups.
    assert command[command.index("--num-iters") + 1] == str(config.VLLMBENCH_ITERS)
    assert command[command.index("--num-iters-warmup") + 1] == str(config.VLLMBENCH_WARMUP_ITERS)


def test_throughput_command_uses_num_prompts_and_no_iteration_flags():
    command = VllmBenchBenchmark.build_throughput_command(
        "/venv/bin/vllm", "org/model", Path("/tmp/out.json"), 4096, 512,
    )
    assert command[:3] == ["/venv/bin/vllm", "bench", "throughput"]
    assert command[command.index("--num-prompts") + 1] == str(config.VLLMBENCH_NUM_PROMPTS)
    assert "--num-iters" not in command
    assert command[command.index("--input-len") + 1] == "4096"


def test_latency_result_parses_seconds_and_derives_an_output_rate():
    payload = {
        "avg_latency": 2.0,
        "latencies": [1.9, 2.0, 2.1],
        "percentiles": {"50": 2.0, "99": 2.1},
    }
    entry = VllmBenchBenchmark.parse_latency_result(payload, 2048, 128)
    assert entry is not None
    assert entry["avg_latency_sec"] == 2.0
    assert entry["latency_runs_sec"] == [1.9, 2.0, 2.1]
    assert entry["completed_iters"] == 3
    assert entry["percentiles_sec"] == {"50": 2.0, "99": 2.1}
    assert entry["output_tps"] == round(128 * config.VLLMBENCH_BATCH_SIZE / 2.0, 2)
    assert entry["input_len"] == 2048 and entry["output_len"] == 128


def test_latency_result_survives_a_payload_without_samples_or_percentiles():
    entry = VllmBenchBenchmark.parse_latency_result({"avg_latency": 1.5}, 512, 128)
    assert entry is not None
    assert entry["latency_runs_sec"] == []
    assert entry["completed_iters"] == 0
    assert entry["percentiles_sec"] == {}


@pytest.mark.parametrize("payload", [
    None, {}, "not-a-dict", {"avg_latency": 0}, {"avg_latency": -1},
    {"avg_latency": True}, {"avg_latency": "2.0"},
])
def test_latency_result_is_none_when_no_usable_average_was_reported(payload):
    assert VllmBenchBenchmark.parse_latency_result(payload, 512, 128) is None


def test_latency_result_drops_non_numeric_samples_rather_than_failing():
    payload = {"avg_latency": 2.0, "latencies": [1.9, None, "x", True, 2.1]}
    entry = VllmBenchBenchmark.parse_latency_result(payload, 512, 128)
    assert entry is not None
    assert entry["latency_runs_sec"] == [1.9, 2.1]


def test_throughput_result_derives_an_output_only_rate_from_request_count():
    payload = {
        "elapsed_time": 4.0, "num_requests": 32, "total_num_tokens": 8000,
        "requests_per_second": 8.0, "tokens_per_second": 2000.0,
    }
    entry = VllmBenchBenchmark.parse_throughput_result(payload, 2048, 128)
    assert entry is not None
    assert entry["elapsed_sec"] == 4.0
    assert entry["num_requests"] == 32
    assert entry["requests_per_sec"] == 8.0
    # total_num_tokens includes prompt tokens, so output rate cannot come from it.
    assert entry["output_tps"] == round(32 * 128 / 4.0, 2)
    assert entry["total_tps"] == 2000.0


def test_throughput_result_computes_rates_when_vllm_omits_them():
    payload = {"elapsed_time": 2.0, "num_requests": 10}
    entry = VllmBenchBenchmark.parse_throughput_result(payload, 512, 128)
    assert entry is not None
    assert entry["requests_per_sec"] == 5.0
    assert "total_tps" not in entry and "total_num_tokens" not in entry


@pytest.mark.parametrize("payload", [
    None, {}, "not-a-dict",
    {"elapsed_time": 0, "num_requests": 4},
    {"elapsed_time": -1, "num_requests": 4},
    {"elapsed_time": 4.0, "num_requests": 0},
    {"elapsed_time": 4.0, "num_requests": True},
    {"elapsed_time": 4.0},
    {"num_requests": 4},
])
def test_throughput_result_is_none_when_the_run_reported_nothing_usable(payload):
    assert VllmBenchBenchmark.parse_throughput_result(payload, 512, 128) is None


def test_sweep_sizes_pairs_every_input_with_every_output():
    assert VllmBenchBenchmark.sweep_sizes([512, 2048], [128, 512], None) == [
        (512, 128), (512, 512), (2048, 128), (2048, 512),
    ]


def test_sweep_sizes_drops_pairs_that_would_exceed_the_model_context():
    """vLLM rejects prompt + generation above --max-model-len outright, unlike
    llama-server which simply generates fewer tokens."""
    assert VllmBenchBenchmark.sweep_sizes([512, 4096], [128, 512], 4096) == [
        (512, 128), (512, 512),
    ]
    assert VllmBenchBenchmark.sweep_sizes([4096], [128], 4224) == [(4096, 128)]
    assert VllmBenchBenchmark.sweep_sizes([4096], [128], 4095) == []


def test_format_entry_labels_each_kind_with_its_own_headline_number():
    latency = VllmBenchBenchmark.parse_latency_result({"avg_latency": 2.0}, 2048, 128)
    assert latency is not None
    assert "in2048/out128" in VllmBenchBenchmark.format_entry("latency", latency)
    assert "s per batch" in VllmBenchBenchmark.format_entry("latency", latency)
    throughput = VllmBenchBenchmark.parse_throughput_result(
        {"elapsed_time": 4.0, "num_requests": 32}, 2048, 128,
    )
    assert throughput is not None
    assert "req/s" in VllmBenchBenchmark.format_entry("throughput", throughput)
