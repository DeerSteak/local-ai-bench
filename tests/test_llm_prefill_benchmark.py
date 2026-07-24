import config
from llm_prefill_benchmark import LLMPrefillBenchmark


def test_prefill_server_ctx_adds_generation_headroom():
    assert (LLMPrefillBenchmark.prefill_server_ctx(32768, 131072)
            == 32768 + config.GENERATE_MAX_TOKENS)


def test_prefill_server_ctx_clamps_to_model_max():
    assert LLMPrefillBenchmark.prefill_server_ctx(65536, 65536 + 100) == 65536 + 100


def test_prefill_server_ctx_no_headroom_when_ctx_len_equals_model_max():
    assert LLMPrefillBenchmark.prefill_server_ctx(65536, 65536) == 65536
