import pytest

from scripts.runtime.workload_runner import llm_benchmark_class
from scripts.workloads.llm_prefill_benchmark import LLMCachedPrefillBenchmark, LLMPrefillBenchmark


def test_uncached_prompt_plan_builds_unique_measured_prompts_without_priming():
    calls = []

    def build(ctx):
        calls.append(ctx)
        return f"prompt-{len(calls)}"

    prime, measured = LLMPrefillBenchmark.prompt_plan(2048, 3, False, build)
    assert prime is None
    assert measured == ["prompt-1", "prompt-2", "prompt-3"]


def test_cached_prompt_plan_primes_and_reuses_one_exact_prompt():
    calls = []

    def build(ctx):
        calls.append(ctx)
        return "cached-prompt"

    prime, measured = LLMPrefillBenchmark.prompt_plan(8192, 3, True, build)
    assert prime == "cached-prompt"
    assert measured == [prime, prime, prime]
    assert calls == [8192]


def test_runner_resolves_each_prefill_stage_to_its_methodology():
    assert llm_benchmark_class("llm") is LLMPrefillBenchmark
    assert llm_benchmark_class("llm_cached") is LLMCachedPrefillBenchmark
    with pytest.raises(ValueError, match="unsupported LLM stage"):
        llm_benchmark_class("conv")


@pytest.mark.parametrize("benchmark", [LLMPrefillBenchmark, LLMCachedPrefillBenchmark])
def test_prefill_terminal_metrics_show_ttft_prefill_and_generation_tps(benchmark):
    assert benchmark.format_terminal_metrics(0.456, 1234.56, 78.94) == (
        "TTFT=0.46s  Prefill=1234.6 tok/s  TPS=78.9"
    )


def test_prefill_terminal_metrics_mark_unavailable_prompt_speed():
    assert LLMPrefillBenchmark.format_terminal_metrics(0.5, None, 40.0) == (
        "TTFT=0.50s  Prefill=N/A  TPS=40.0"
    )
