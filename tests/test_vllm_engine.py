import json
import os
import subprocess
from pathlib import Path
from typing import cast

import pytest

from scripts.runtime import config
from scripts.runtime.engines.vllm import (
    VllmEngine, available_kv_cache_gib, load_attempt_deadline, load_timeout_error,
    next_cpu_offload_gb, offload_calibration_timeout, offload_retry_allowed,
    offload_stop_reason, offload_timeout_message, tensor_parallel_size,
)
import scripts.runtime.engines.vllm as vllm_module
from scripts.workloads.models import LLM_MODELS
from scripts.runtime.shared import EngineTimeout


@pytest.fixture
def engine(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SETUP_CONFIG_PATH", tmp_path / "absent.json")
    instance = VllmEngine()
    instance._cache_home = tmp_path / "cache"
    instance._launcher = None
    instance._executable = "/usr/bin/vllm"
    # Pretend a model is already loaded so inference calls skip the real spawn.
    instance._loaded_tag = "qwen3.5:9b-q4_K_M"
    instance._loaded_num_ctx = None
    instance._loaded_embedding = False
    instance._loaded_n_parallel = 1
    instance._proc = cast(subprocess.Popen, type("P", (), {"poll": staticmethod(lambda: None)})())
    return instance


class _Response:
    def __init__(self, chunks):
        self._lines = [f"data: {json.dumps(chunk)}".encode() for chunk in chunks] + [b"data: [DONE]"]

    def __enter__(self):
        return self._lines

    def __exit__(self, *exc):
        return False


def _patch_stream(monkeypatch, chunks):
    captured = {}

    def post(self, path, payload, timeout):
        captured["path"], captured["payload"] = path, payload
        return _Response(chunks)

    monkeypatch.setattr(VllmEngine, "_post", post)
    return captured


TEST_TAG = "qwen3.5:9b-q4_K_M"
TEST_REPO = next(m["vllm_repo"] for m in LLM_MODELS if m["tag"] == TEST_TAG)
TEST_REPO_DIR = "models--" + TEST_REPO.replace("/", "--")


def _text_chunk(text, finish=None):
    return {"choices": [{"text": text, "finish_reason": finish}]}


def _delta_chunk(content=None, finish=None, tool_calls=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {"choices": [{"delta": delta, "finish_reason": finish}]}


# ── command construction ──

def test_launcher_is_preferred_and_pins_the_port(engine):
    engine._launcher = "/usr/bin/vllm-launch"
    command = engine.server_command("org/m", 4096)
    assert command[0] == "/usr/bin/vllm-launch"
    assert command[command.index("-p") + 1] == str(config.VLLM_PORT)
    assert command[command.index("-m") + 1] == "org/m"


def test_bare_serve_is_used_without_a_launcher(engine):
    command = engine.server_command("org/m", 4096)
    assert command[:3] == ["/usr/bin/vllm", "serve", "org/m"]
    assert "--port" in command


def test_fp8_cache_is_selected_only_for_supported_accelerator_backends(engine):
    assert engine.configure_kv_cache("cuda") == "fp8"
    command = engine.server_command("org/m", 4096)
    assert command[command.index("--kv-cache-dtype") + 1] == "fp8"

    assert engine.configure_kv_cache("rocm") == "fp8"
    assert engine.configure_kv_cache("cpu") == "auto"
    assert "--kv-cache-dtype" not in engine.server_command("org/m", 4096)


def test_cuda_generation_avoids_uninstalled_flashinfer_backend(engine):
    engine.configure_kv_cache("cuda")
    command = engine.server_command("org/m", 4096)
    assert command[command.index("--attention-backend") + 1] == "FLASH_ATTN"


def test_embedding_does_not_use_quantized_kv_cache_or_decoder_attention_backend(engine):
    engine.configure_kv_cache("cuda")
    command = engine.server_command("org/m", None, embedding=True)
    assert "--kv-cache-dtype" not in command
    assert "--attention-backend" not in command


def test_external_server_cache_policy_remains_unmanaged(engine):
    engine._server_url = "http://external:8000"
    assert engine.configure_kv_cache("cuda") == "auto"


def test_max_model_len_is_per_sequence_and_not_scaled_by_parallelism(engine):
    """Unlike llama-server's -c, --max-model-len is per sequence."""
    command = engine.server_command("org/m", 4096, n_parallel=8)
    assert command[command.index("--max-model-len") + 1] == "4096"
    assert command[command.index("--max-num-seqs") + 1] == "8"


def test_context_is_omitted_when_unset(engine):
    assert "--max-model-len" not in engine.server_command("org/m", None)


def test_cpu_offload_is_only_added_when_calibrated(engine):
    assert "--cpu-offload-gb" not in engine.server_command("org/m", 4096)
    command = engine.server_command("org/m", 4096, cpu_offload_gb=8)
    assert command[command.index("--cpu-offload-gb") + 1] == "8"


@pytest.mark.parametrize(("available", "expected"), [
    (-3.53, 8),
    (-2.05, 6),
    (-0.01, 4),
])
def test_offload_calculation_uses_deficit_reserve_and_two_gib_steps(available, expected):
    log = (f"Available KV cache memory: {available} GiB\n"
           "ValueError: No available memory for the cache blocks")
    assert next_cpu_offload_gb(log) == expected


def test_offload_retry_adds_the_new_shortfall_to_the_current_value():
    log = ("Available KV cache memory: -0.25 GiB\n"
           "No available memory for the cache blocks")
    assert next_cpu_offload_gb(log, current_gb=8) == 10


def test_offload_calculation_handles_positive_but_insufficient_kv_memory():
    log = ("To serve at least one request with the model's max seq len (262144), "
           "(27.45 GiB KV cache is needed, which is larger than the available "
           "KV cache memory (23.43 GiB).")
    assert next_cpu_offload_gb(log) == 8


@pytest.mark.parametrize(("args", "expected"), [
    (["--tensor-parallel-size", "4"], 4),
    (["--tensor-parallel-size=2"], 2),
    (["-tp", "8"], 8),
    (["-tp=3"], 3),
    (["--other", "2"], 1),
    (["-tp", "bad"], 1),
])
def test_tensor_parallel_size_parses_launcher_forms(args, expected):
    assert tensor_parallel_size(args) == expected


def test_offload_retry_guard_caps_attempts_and_host_use(monkeypatch):
    monkeypatch.setattr(config, "VLLM_OFFLOAD_MAX_ATTEMPTS", 4)
    assert offload_retry_allowed(8, 10, 3)
    assert not offload_retry_allowed(12, 10, 3)
    assert not offload_retry_allowed(8, 10, 4)
    assert not offload_retry_allowed(None, 10, 0)


def test_unrecognized_failure_reason_takes_precedence_over_retry_limit(monkeypatch):
    monkeypatch.setattr(config, "VLLM_OFFLOAD_MAX_ATTEMPTS", 4)
    assert offload_stop_reason(None, 10, 4) == (
        "failure was not a recognized KV-cache memory shortage")


def test_offload_calculation_uses_last_profile_and_ignores_other_failures():
    log = ("Available KV cache memory: -9 GiB\n"
           "Available KV cache memory: -2.05 GiB\n"
           "No available memory for the cache blocks")
    assert available_kv_cache_gib(log) == -2.05
    assert next_cpu_offload_gb("Available KV cache memory: -2.05 GiB\nbad checkpoint") is None


def test_offload_timeout_identifies_attempt_and_value(monkeypatch):
    monkeypatch.setattr(config, "VLLM_OFFLOAD_MAX_ATTEMPTS", 4)
    message = offload_timeout_message("large:model", 10, 2)
    assert "attempt 3/5" in message
    assert "--cpu-offload-gb 10" in message


def test_load_attempt_deadline_preserves_the_caller_bound():
    assert load_attempt_deadline(100.0, 400.0, 900) == 400.0
    assert load_attempt_deadline(100.0, 1200.0, 900) == 1000.0
    assert load_attempt_deadline(100.0, None, 900) == 1000.0


def test_load_timeout_classification_distinguishes_caller_from_server():
    caller_error = load_timeout_error("large:model", 10, 2, 900, True)
    server_error = load_timeout_error("large:model", 10, 2, 900, False)
    assert isinstance(caller_error, EngineTimeout)
    assert isinstance(server_error, RuntimeError)
    assert not isinstance(server_error, EngineTimeout)


def test_offload_calibration_timeout_covers_every_attempt(monkeypatch):
    monkeypatch.setattr(config, "VLLM_OFFLOAD_MAX_ATTEMPTS", 4)
    assert offload_calibration_timeout(900, 300) == 4500
    assert offload_calibration_timeout(900, 5000) == 5000


def test_prepare_concurrency_uses_the_full_calibration_budget(engine, monkeypatch):
    captured = {}
    monkeypatch.setattr(config, "VLLM_OFFLOAD_MAX_ATTEMPTS", 4)
    monkeypatch.setattr(vllm_module.time, "perf_counter", lambda: 100.0)
    monkeypatch.setattr(
        engine, "_ensure_model",
        lambda *_args, **kwargs: captured.update(kwargs),
    )

    assert engine.prepare_concurrency(TEST_TAG, 4, 4096, timeout=300)
    assert captured["deadline"] == 100.0 + (engine.LOAD_TIMEOUT * 5)


def test_offload_cache_rejects_malformed_values(engine):
    engine._cache_home.mkdir()
    engine._offload_cache_path.write_text(json.dumps({
        "offload_gb": {"valid": 6, "zero": 0, "float": 4.0, "negative": -2},
    }))
    assert engine._load_offload_cache() == {"valid": 6}


def test_offload_cache_key_tracks_model_revision_and_visible_devices(engine, monkeypatch):
    revisions = iter((Path("snapshots/one"), Path("snapshots/two")))
    monkeypatch.setattr(engine, "_snapshot_dir", lambda _tag: next(revisions))
    first = engine._offload_key(TEST_TAG, TEST_REPO)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    second = engine._offload_key(TEST_TAG, TEST_REPO)
    assert first != second


def test_host_offload_limit_accounts_for_per_worker_tensor_parallel_allocation(
        engine, monkeypatch):
    engine._launcher_extra_args = ["--tensor-parallel-size", "4"]
    memory = type("Memory", (), {"available": 72 * 1024 ** 3})()
    monkeypatch.setattr(vllm_module.psutil, "virtual_memory", lambda: memory)
    assert engine._host_offload_limit_gb() == 16


def test_runtime_environment_exposes_vllm_venv_build_tools(engine, monkeypatch, tmp_path):
    venv = tmp_path / "vllm-env"
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents=True)
    monkeypatch.setattr(config, "VLLM_VENV", venv)
    monkeypatch.setattr(config, "SCRIPT_DIR", tmp_path)
    monkeypatch.setenv("PATH", "/usr/bin")

    environment = engine.runtime_environment()

    assert environment["PATH"].split(os.pathsep) == [str(bin_dir), "/usr/bin"]
    assert environment["HF_HOME"] == str(engine.cache_home())


def test_runtime_environment_enables_wsl2_pin_memory_for_managed_vllm(engine, monkeypatch):
    monkeypatch.setattr(vllm_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(vllm_module.platform, "release", lambda: "6.6.0-microsoft-standard-WSL2")
    monkeypatch.delenv("VLLM_WSL2_ENABLE_PIN_MEMORY", raising=False)

    assert engine.runtime_environment()["VLLM_WSL2_ENABLE_PIN_MEMORY"] == "1"


def test_runtime_environment_preserves_explicit_wsl2_pin_memory_override(engine, monkeypatch):
    monkeypatch.setattr(vllm_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(vllm_module.platform, "release", lambda: "6.6.0-microsoft-standard-WSL2")
    monkeypatch.setenv("VLLM_WSL2_ENABLE_PIN_MEMORY", "0")

    assert engine.runtime_environment()["VLLM_WSL2_ENABLE_PIN_MEMORY"] == "0"


def test_runtime_environment_leaves_external_vllm_environment_unmanaged(engine, monkeypatch):
    monkeypatch.setattr(vllm_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(vllm_module.platform, "release", lambda: "6.6.0-microsoft-standard-WSL2")
    monkeypatch.delenv("VLLM_WSL2_ENABLE_PIN_MEMORY", raising=False)
    engine._server_url = "http://external:8000"

    assert "VLLM_WSL2_ENABLE_PIN_MEMORY" not in engine.runtime_environment()


def test_embedding_mode_uses_the_pooling_runner(engine):
    """--task was replaced by --runner in current vLLM."""
    command = engine.server_command("org/m", None, embedding=True)
    assert command[command.index("--runner") + 1] == "pooling"
    assert "--task" not in command


def test_tool_parser_flags_are_only_passed_when_configured(engine):
    plain = engine.server_command("org/m", 1024)
    assert "--enable-auto-tool-choice" not in plain and "--tool-call-parser" not in plain

    with_tools = engine.server_command("org/m", 1024, tool_parser="hermes")
    assert "--enable-auto-tool-choice" in with_tools
    assert with_tools[with_tools.index("--tool-call-parser") + 1] == "hermes"


def test_no_command_ever_passes_a_device_flag(engine):
    """vLLM has no --device option; CPU needs a separately built wheel."""
    assert "--device" not in engine.server_command("org/m", 1024, embedding=True)


def test_no_runtime_raises_rather_than_building_a_broken_command(engine):
    engine._executable = None
    with pytest.raises(RuntimeError, match="no vLLM runtime"):
        engine.server_command("org/m", 1024)


# ── generate ──

def test_generate_counts_tokens_from_streamed_usage(engine, monkeypatch):
    captured = _patch_stream(monkeypatch, [
        _text_chunk("Hello"), _text_chunk(" world", finish="stop"),
        {"choices": [], "usage": {"completion_tokens": 7, "prompt_tokens": 31}},
    ])
    result = engine.generate("qwen3.5:9b-q4_K_M", "hi", timeout=30)
    assert result.generated_tokens == 7, "SSE fragments are never counted as tokens"
    assert result.prompt_tokens == 31
    assert result.response_text == "Hello world"
    assert result.finish_reason == "stop"
    assert captured["path"] == "/v1/completions"
    assert captured["payload"]["stream_options"] == {"include_usage": True}
    assert isinstance(captured["payload"]["cache_salt"], str)
    assert len(captured["payload"]["cache_salt"]) >= 32


def test_generate_uses_a_fresh_cache_salt_per_request(engine, monkeypatch):
    salts = []

    def post(self, path, payload, timeout):
        salts.append(payload["cache_salt"])
        return _Response([_text_chunk("x", finish="stop"),
                          {"choices": [], "usage": {"completion_tokens": 1}}])

    monkeypatch.setattr(VllmEngine, "_post", post)
    engine.generate(TEST_TAG, "same prompt", timeout=30)
    engine.generate(TEST_TAG, "same prompt", timeout=30)
    assert len(set(salts)) == 2


def test_generate_reports_no_server_prompt_time(engine, monkeypatch):
    _patch_stream(monkeypatch, [_text_chunk("x", finish="stop"),
                                {"choices": [], "usage": {"completion_tokens": 1}}])
    result = engine.generate("qwen3.5:9b-q4_K_M", "hi", timeout=30)
    assert result.server_prompt_sec is None, "vLLM reports no per-request prompt duration"


def test_generate_requests_the_repo_id_not_the_tag(engine, monkeypatch):
    captured = _patch_stream(monkeypatch, [_text_chunk("x", finish="stop"),
                                            {"choices": [], "usage": {"completion_tokens": 1}}])
    engine.generate("qwen3.5:9b-q4_K_M", "hi", timeout=30)
    assert captured["payload"]["model"] == TEST_REPO


def test_generate_measurement_is_internally_consistent(engine, monkeypatch):
    _patch_stream(monkeypatch, [_text_chunk("a"), _text_chunk("b", finish="stop"),
                                {"choices": [], "usage": {"completion_tokens": 4}}])
    from scripts.runtime.engines.base import measurement_validation_errors
    assert measurement_validation_errors(engine.generate("qwen3.5:9b-q4_K_M", "hi", 30)) == []


def test_generate_decode_sec_always_matches_the_measured_decode_window(engine, monkeypatch):
    """decode_sec must reflect the actual wall-clock decode window regardless of whether
    tps was sanitized — the two must never be derived from each other and drift apart."""
    _patch_stream(monkeypatch, [_text_chunk("a"), _text_chunk("b", finish="stop"),
                                {"choices": [], "usage": {"completion_tokens": 4}}])
    result = engine.generate("qwen3.5:9b-q4_K_M", "hi", timeout=30)
    assert result.decode_sec == pytest.approx(
        max(result.client_wall_sec - result.client_ttft_sec, 0))


def test_generate_times_out_with_a_gradeable_partial(engine, monkeypatch):
    _patch_stream(monkeypatch, [_text_chunk("partial answer")])
    with pytest.raises(EngineTimeout) as excinfo:
        engine.generate("qwen3.5:9b-q4_K_M", "hi", timeout=-1)
    assert "partial answer" in excinfo.value.partial_text


# ── chat ──

def test_chat_uses_usage_for_tokens_and_prompt_count(engine, monkeypatch):
    captured = _patch_stream(monkeypatch, [
        _delta_chunk("Yes"), _delta_chunk(" indeed", finish="stop"),
        {"choices": [], "usage": {"completion_tokens": 5, "prompt_tokens": 12}},
    ])
    result = engine.chat("qwen3.5:9b-q4_K_M", [{"role": "user", "content": "hi"}], timeout=30)
    assert (result.generated_tokens, result.prompt_tokens) == (5, 12)
    assert result.response_text == "Yes indeed"
    assert captured["path"] == "/v1/chat/completions"


def test_chat_measurement_routes_the_combined_tps_through_sanitize_tps(engine, monkeypatch):
    """The combined tokens_per_sec across (first, second) must go through the same
    implausible-tps guard as each individual request, not compute it inline unchecked."""
    from scripts.runtime.engines import openai_api
    calls = []
    real_sanitize = openai_api.sanitize_tps

    def spy(tps, tokens, ttft, total):
        calls.append((tps, tokens, ttft, total))
        return real_sanitize(tps, tokens, ttft, total)

    monkeypatch.setattr(openai_api, "sanitize_tps", spy)
    _patch_stream(monkeypatch, [_delta_chunk("Yes"), _delta_chunk(" indeed", finish="stop"),
                                {"choices": [], "usage": {"completion_tokens": 5, "prompt_tokens": 12}}])
    result = engine.chat("qwen3.5:9b-q4_K_M", [{"role": "user", "content": "hi"}], timeout=30)
    assert len(calls) == 2, "one call for the per-request tps, one for the combined tps"
    assert calls[-1][1] == result.generated_tokens
    expected_tps = result.generated_tokens / result.decode_sec if result.decode_sec else 0
    assert result.tokens_per_sec == pytest.approx(expected_tps)


def test_chat_omits_max_tokens_when_unbounded(engine, monkeypatch):
    captured = _patch_stream(monkeypatch, [_delta_chunk("x", finish="stop"),
                                            {"choices": [], "usage": {"completion_tokens": 1}}])
    engine.chat("qwen3.5:9b-q4_K_M", [{"role": "user", "content": "hi"}],
                timeout=30, num_predict=-1)
    assert "max_tokens" not in captured["payload"]


def test_chat_tools_refuses_a_model_with_no_configured_parser(engine, monkeypatch):
    """Without --tool-call-parser vLLM returns no tool_calls, which would score as a
    wrong answer rather than an unsupported configuration."""
    _patch_stream(monkeypatch, [_delta_chunk("x", finish="stop")])
    with pytest.raises(RuntimeError, match="tool-call parser"):
        engine.chat_tools("qwen3.5:9b-q4_K_M", [{"role": "user", "content": "hi"}],
                          tools=[{"type": "function"}], timeout=30)


def test_chat_tools_parses_streamed_tool_calls(engine, monkeypatch):
    monkeypatch.setattr(VllmEngine, "_tool_parser", classmethod(lambda cls, tag: "hermes"))
    engine._loaded_tool_parser = "hermes"  # already serving with the parser enabled
    _patch_stream(monkeypatch, [
        _delta_chunk(tool_calls=[{"index": 0, "function": {"name": "get_weather"}}]),
        _delta_chunk(tool_calls=[{"index": 0, "function": {"arguments": '{"city":'}}]),
        _delta_chunk(tool_calls=[{"index": 0, "function": {"arguments": '"Paris"}'}}],
                     finish="tool_calls"),
        {"choices": [], "usage": {"completion_tokens": 9}},
    ])
    result = engine.chat_tools("qwen3.5:9b-q4_K_M", [{"role": "user", "content": "weather?"}],
                                tools=[{"type": "function"}], timeout=30)
    assert result.tool_calls == [{"name": "get_weather", "arguments": {"city": "Paris"}}]


# ── model resolution ──

def test_model_pulled_reads_the_hf_cache(engine):
    tag = "qwen3.5:9b-q4_K_M"
    assert engine.model_pulled(tag) is False
    snapshot = (engine._cache_home / "hub" / TEST_REPO_DIR
                / "snapshots" / "abc")
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").touch()
    assert engine.model_pulled(tag) is True
    assert [m["tag"] for m in engine.list_installed_models()] == [tag]


def test_an_unknown_tag_has_no_repo(engine):
    assert engine._repo("not-a-model") is None


def test_registered_custom_model_resolves_and_lists_from_cache(engine, monkeypatch):
    record = {"engine": "vllm", "tag": "custom", "label": "Custom", "repo": "owner/model"}
    monkeypatch.setattr(vllm_module, "load_custom_models", lambda: [record])
    monkeypatch.setattr(
        vllm_module, "custom_model",
        lambda engine_name, tag: record if (engine_name, tag) == ("vllm", "custom") else None,
    )
    snapshot = engine._cache_home / "hub" / "models--owner--model" / "snapshots" / "commit"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")

    assert engine._repo("custom") == "owner/model"
    assert engine.model_pulled("custom")
    assert engine.list_installed_models() == [{"tag": "custom", "label": "Custom", "size": None}]
    assert not engine.supports_tool_calls("custom")


def test_external_vllm_server_cannot_receive_local_imports(engine):
    assert engine.supports_model_import()
    engine._server_url = "http://external:8000"
    assert not engine.supports_model_import()


def test_list_installed_models_skips_a_pulled_entry_missing_its_vllm_repo(engine, monkeypatch):
    """model_pulled() only checks the HF cache, so a catalog entry with no vllm_repo
    (a llamacpp-only model) must be skipped rather than raising KeyError."""
    fake_model = {"tag": "no-repo:1b", "vllm_repo": None}
    monkeypatch.setattr(VllmEngine, "model_pulled", lambda self, tag: True)
    monkeypatch.setattr("scripts.runtime.engines.vllm.LLM_MODELS", [fake_model])
    monkeypatch.setattr("scripts.runtime.engines.vllm.EMBED_MODELS", [])
    assert engine.list_installed_models() == []


def test_max_context_length_reads_the_snapshot_config(engine):
    tag = "qwen3.5:9b-q4_K_M"
    snapshot = (engine._cache_home / "hub" / TEST_REPO_DIR
                / "snapshots" / "abc")
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"max_position_embeddings": 32768}))
    assert engine.max_context_length(tag) == 32768


def test_max_context_length_falls_back_on_missing_or_broken_config(engine):
    tag = "qwen3.5:9b-q4_K_M"
    assert engine.max_context_length(tag, default=4096) == 4096
    snapshot = (engine._cache_home / "hub" / TEST_REPO_DIR
                / "snapshots" / "abc")
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{ not json")
    assert engine.max_context_length(tag, default=4096) == 4096


def test_max_context_length_reads_a_nested_text_config(engine):
    tag = "gemma3:27b-it-q4_K_M"
    repo = engine._repo(tag).replace("/", "--")
    snapshot = engine._cache_home / "hub" / f"models--{repo}" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 8192}}))
    assert engine.max_context_length(tag) == 8192


def test_changing_the_tool_parser_forces_a_respawn(engine, monkeypatch):
    """A server started without --tool-call-parser cannot serve a tool request."""
    spawned = []
    monkeypatch.setattr(VllmEngine, "_tool_parser", classmethod(lambda cls, tag: "hermes"))
    monkeypatch.setattr(VllmEngine, "stop", lambda self, **kw: None)
    monkeypatch.setattr(VllmEngine, "model_pulled", lambda self, tag: True)
    monkeypatch.setattr(VllmEngine, "available", lambda self: True)

    def popen(args, **kwargs):
        spawned.append(args)
        return type("P", (), {"poll": staticmethod(lambda: None), "returncode": 0})()

    monkeypatch.setattr("subprocess.Popen", popen)
    engine._ensure_model("qwen3.5:9b-q4_K_M", 1024, tool_parser="hermes")
    assert spawned, "the loaded server had no parser, so a respawn was required"
    assert "--tool-call-parser" in spawned[0]


# ── resume identity ──

def _cache_snapshot(engine, symlink, repo=None, files=("model.safetensors", "config.json")):
    repo = repo or TEST_REPO_DIR.removeprefix("models--")
    repo_dir = engine._cache_home / "hub" / f"models--{repo}"
    blobs, snapshot = repo_dir / "blobs", repo_dir / "snapshots" / "abc"
    blobs.mkdir(parents=True); snapshot.mkdir(parents=True)
    for index, name in enumerate(files):
        blob = blobs / f"b{index}"
        blob.write_text("weights")
        symlink(snapshot / name, blob)
    return snapshot


def test_resume_artifacts_resolve_through_the_cache_symlinks(engine, symlink_or_skip):
    _cache_snapshot(engine, symlink_or_skip, files=("model-00001.safetensors", "model-00002.safetensors", "config.json"))
    paths = engine.resume_artifact_paths("qwen3.5:9b-q4_K_M")
    assert len(paths) == 3
    assert all(path.parent.name == "blobs" for path in paths), "identity follows the blob, not the link"
    assert all(path.exists() for path in paths)


def test_resume_artifacts_raise_for_an_uncached_model(engine):
    with pytest.raises(ValueError, match="cannot identify local model artifact"):
        engine.resume_artifact_paths("qwen3.5:9b-q4_K_M")


def test_resume_runtime_prefers_the_launcher(engine):
    assert engine.resume_runtime_paths() == {"vllm": Path("/usr/bin/vllm").resolve()}
    engine._launcher = "/usr/bin/vllm-launch"
    assert engine.resume_runtime_paths() == {"vllm": Path("/usr/bin/vllm-launch").resolve()}


def test_resume_runtime_raises_without_any_vllm(engine):
    engine._launcher = engine._executable = None
    with pytest.raises(ValueError, match="cannot identify vLLM runtime"):
        engine.resume_runtime_paths()


def test_resume_identity_builds_for_a_cached_model(engine, symlink_or_skip):
    """The failure that stopped the first real run: the base class raised NotImplementedError."""
    from scripts.results.resume_policy import cached_file_identity
    _cache_snapshot(engine, symlink_or_skip)
    cache = {}
    for path in engine.resume_artifact_paths("qwen3.5:9b-q4_K_M"):
        assert cached_file_identity(path, cache)["sha256"]
    assert len(cache) == 2


# ── tool-call capability ──

def test_tool_support_follows_the_configured_parser(engine):
    from scripts.workloads.models import LLM_MODELS
    configured = [m["tag"] for m in LLM_MODELS if m.get("vllm_tool_parser")]
    unconfigured = [m["tag"] for m in LLM_MODELS if not m.get("vllm_tool_parser")]
    assert configured and unconfigured, "the catalog should exercise both cases"
    assert all(engine.supports_tool_calls(tag) for tag in configured)
    assert not any(engine.supports_tool_calls(tag) for tag in unconfigured)


def test_llamacpp_supports_tool_calls_for_everything():
    """Only vLLM needs a per-model parser; the default must not narrow llama.cpp."""
    from scripts.runtime.engines.llamacpp import LlamaCppEngine
    assert LlamaCppEngine().supports_tool_calls("anything") is True


def test_configured_parsers_are_passed_to_the_server(engine):
    command = engine.server_command("org/m", 1024, tool_parser=engine._tool_parser("granite4.1:8b-q4_K_M"))
    assert command[command.index("--tool-call-parser") + 1] == "granite4"


# ── context tolerance ──

def test_context_limit_adds_tolerance_for_approximate_padding(engine, monkeypatch):
    """Prompts are padded by characters, so a 512-token target can tokenize to 513+.
    vLLM rejects prompt+max_tokens over max_model_len outright."""
    monkeypatch.setattr(VllmEngine, "max_context_length", lambda self, tag, default=0: 32768)
    assert engine.context_limit("qwen3.5:9b-q4_K_M", 1024) == 1024 + config.VLLM_CTX_TOLERANCE


def test_context_limit_never_exceeds_the_models_real_maximum(engine, monkeypatch):
    """Asking for more than the model supports is rejected by vLLM just as firmly."""
    monkeypatch.setattr(VllmEngine, "max_context_length", lambda self, tag, default=0: 4096)
    assert engine.context_limit("qwen3.5:9b-q4_K_M", 4096) == 4096
    assert engine.context_limit("qwen3.5:9b-q4_K_M", 4000) == 4064


def test_context_limit_passes_through_none(engine):
    assert engine.context_limit("qwen3.5:9b-q4_K_M", None) is None


def test_context_limit_keeps_tolerance_when_the_snapshot_config_is_unreadable(engine, monkeypatch):
    """max_context_length falling back to its own default must not collapse the
    tolerance padding back to exactly num_ctx — that would defeat VLLM_CTX_TOLERANCE."""
    monkeypatch.setattr(VllmEngine, "max_context_length", lambda self, tag, default=131072: default)
    assert engine.context_limit("qwen3.5:9b-q4_K_M", 1024) == 1024 + config.VLLM_CTX_TOLERANCE


def test_the_tolerance_covers_the_reported_gemma_failure(engine, monkeypatch):
    """512-token checkpoint: server_ctx 1024, prompt measured 513, generation 512."""
    monkeypatch.setattr(VllmEngine, "max_context_length", lambda self, tag, default=0: 32768)
    limit = engine.context_limit("gemma3:1b-it-q4_K_M", 1024)
    assert 513 + config.GENERATE_MAX_TOKENS <= limit


# ── process teardown ──

def test_server_is_spawned_in_its_own_process_group(engine, monkeypatch):
    """vLLM forks an EngineCore child holding the weights; signalling only the API
    server orphans it, and the memory it holds is never released."""
    import os as _os
    captured = {}

    def popen(args, **kwargs):
        captured.update(kwargs)
        return type("P", (), {"poll": staticmethod(lambda: None), "returncode": 0, "pid": 4321})()

    monkeypatch.setattr("subprocess.Popen", popen)
    monkeypatch.setattr(VllmEngine, "stop", lambda self, **kw: None)
    monkeypatch.setattr(VllmEngine, "model_pulled", lambda self, tag: True)
    monkeypatch.setattr(VllmEngine, "available", lambda self: True)
    engine._ensure_model("qwen3.5:9b-q4_K_M", 1024)

    if _os.name == "nt":
        assert captured.get("creationflags")
    else:
        assert captured.get("start_new_session") is True


def test_stop_signals_the_group_then_escalates(engine, monkeypatch):
    signalled = []

    class Proc:
        pid = 4321
        def __init__(self): self.waits = 0
        def poll(self): return None
        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise __import__("subprocess").TimeoutExpired("vllm", timeout)
        def kill(self): signalled.append("kill")
        def send_signal(self, sig): signalled.append(("signal", sig))

    engine._proc = Proc()
    monkeypatch.setattr("os.getpgid", lambda pid: pid)
    monkeypatch.setattr("os.killpg", lambda pgid, sig: signalled.append(("killpg", pgid, sig)))
    engine._stop_process(timeout=0.01)

    import signal as _signal
    assert ("killpg", 4321, _signal.SIGTERM) in signalled
    assert ("killpg", 4321, _signal.SIGKILL) in signalled, "a hung group must be forced"


def test_launcher_stop_interrupts_before_escalating(engine, monkeypatch):
    signalled = []
    waited = []

    class Proc:
        pid = 4321
        def poll(self): return None
        def wait(self, timeout=None): return None
        def kill(self): pass
        def send_signal(self, sig): pass

    engine._launcher = "/usr/bin/vllm-launch"
    engine._proc = Proc()
    monkeypatch.setattr("os.getpgid", lambda pid: pid)
    monkeypatch.setattr("os.killpg", lambda pgid, sig: signalled.append(sig))
    monkeypatch.setattr(engine, "_wait_for_launcher_shutdown", lambda timeout: waited.append(timeout))
    engine._stop_process()

    import signal as _signal
    assert signalled == [_signal.SIGINT]
    assert waited == [engine.LAUNCHER_STOP_TIMEOUT]


def test_launcher_shutdown_waits_until_the_container_health_endpoint_disappears(
        engine, monkeypatch):
    health = iter([True, True, False])
    sleeps = []
    monkeypatch.setattr(engine, "available", lambda: next(health))
    monkeypatch.setattr("scripts.runtime.engines.vllm.time.sleep", sleeps.append)

    engine._wait_for_launcher_shutdown(timeout=30)

    assert sleeps == [1, 1]


def test_launcher_shutdown_refuses_to_continue_while_container_is_reachable(
        engine, monkeypatch):
    clock = iter([0.0, 1.0])
    monkeypatch.setattr(engine, "available", lambda: True)
    monkeypatch.setattr("scripts.runtime.engines.vllm.time.perf_counter", lambda: next(clock))

    with pytest.raises(RuntimeError, match="still reachable"):
        engine._wait_for_launcher_shutdown(timeout=1)


@pytest.mark.parametrize("operation", ["generate", "chat"])
def test_model_load_uses_calibration_timeout_not_request_timeout(
        engine, monkeypatch, operation):
    deadlines = []

    def ensure(*_args, **kwargs):
        deadlines.append(kwargs["deadline"])
        raise RuntimeError("captured")

    monkeypatch.setattr("scripts.runtime.engines.vllm.time.perf_counter", lambda: 100.0)
    monkeypatch.setattr(engine, "_ensure_model", ensure)

    with pytest.raises(RuntimeError, match="captured"):
        if operation == "generate":
            engine.generate(TEST_TAG, "hello", timeout=3)
        else:
            engine.chat(TEST_TAG, [{"role": "user", "content": "hello"}], timeout=3)

    expected = engine.LOAD_TIMEOUT * (config.VLLM_OFFLOAD_MAX_ATTEMPTS + 1)
    assert deadlines == [100.0 + expected]


def test_stop_falls_back_when_the_group_is_gone(engine, monkeypatch):
    """A process that already exited must not raise out of teardown."""
    sent = []

    class Proc:
        pid = 4321
        def poll(self): return None
        def wait(self, timeout=None): return None
        def kill(self): sent.append("kill")
        def send_signal(self, sig): sent.append(sig)

    engine._proc = Proc()
    monkeypatch.setattr("os.getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError()))
    engine._stop_process(timeout=0.01)
    assert sent, "falls back to signalling the process itself"
    assert engine._loaded_tag is None


# ── per-request prefill timing (scraped from /metrics) ──

METRICS_BODY = """\
# HELP vllm:request_prefill_time_seconds Request prefill time
# TYPE vllm:request_prefill_time_seconds histogram
vllm:request_prefill_time_seconds_bucket{le="0.1",model_name="m"} 0.0
vllm:request_prefill_time_seconds_bucket{le="+Inf",model_name="m"} 4.0
vllm:request_prefill_time_seconds_sum{model_name="m"} 12.5
vllm:request_prefill_time_seconds_count{model_name="m"} 4.0
vllm:time_to_first_token_seconds_sum{model_name="m"} 99.0
vllm:time_to_first_token_seconds_count{model_name="m"} 4.0
"""


def test_prefill_metric_parses_sum_and_count_ignoring_buckets_and_other_metrics():
    from scripts.runtime.engines.vllm import VllmEngine
    assert VllmEngine.parse_prefill_metric(METRICS_BODY) == (12.5, 4)


def test_prefill_metric_sums_across_label_sets():
    """Two loaded models each carry their own labelled series."""
    from scripts.runtime.engines.vllm import VllmEngine
    body = (
        'vllm:request_prefill_time_seconds_sum{model_name="a"} 1.5\n'
        'vllm:request_prefill_time_seconds_count{model_name="a"} 2.0\n'
        'vllm:request_prefill_time_seconds_sum{model_name="b"} 2.5\n'
        'vllm:request_prefill_time_seconds_count{model_name="b"} 3.0\n'
    )
    assert VllmEngine.parse_prefill_metric(body) == (4.0, 5)


def test_prefill_metric_parses_unlabelled_series():
    from scripts.runtime.engines.vllm import VllmEngine
    body = ("vllm:request_prefill_time_seconds_sum 1.25\n"
            "vllm:request_prefill_time_seconds_count 1.0\n")
    assert VllmEngine.parse_prefill_metric(body) == (1.25, 1)


@pytest.mark.parametrize("body", [
    "",
    "# only comments\n",
    # A build without the metric, or one exposing only part of the histogram.
    'vllm:time_to_first_token_seconds_sum{model_name="m"} 9.0\n',
    'vllm:request_prefill_time_seconds_sum{model_name="m"} 9.0\n',
    'vllm:request_prefill_time_seconds_count{model_name="m"} 9.0\n',
    'vllm:request_prefill_time_seconds_sum{model_name="m"} not-a-number\n',
])
def test_prefill_metric_returns_none_when_unavailable_or_malformed(body):
    from scripts.runtime.engines.vllm import VllmEngine
    assert VllmEngine.parse_prefill_metric(body) is None


def test_prefill_metric_does_not_match_a_longer_metric_name():
    """A name this one is a prefix of must not be mistaken for it."""
    from scripts.runtime.engines.vllm import VllmEngine
    body = ("vllm:request_prefill_time_seconds_extra_sum 5.0\n"
            "vllm:request_prefill_time_seconds_extra_count 1.0\n")
    assert VllmEngine.parse_prefill_metric(body) is None


def test_prefill_delta_attributes_a_single_request():
    from scripts.runtime.engines.vllm import VllmEngine
    assert VllmEngine.prefill_seconds_from_delta((12.5, 4), (13.25, 5)) == 0.75


@pytest.mark.parametrize("before,after", [
    (None, (1.0, 1)),
    ((1.0, 1), None),
    (None, None),
    # Nothing was recorded: an older build, or a request that never reached prefill.
    ((12.5, 4), (12.5, 4)),
    # More than one request landed in the window, so the sum is not ours alone.
    ((12.5, 4), (14.0, 6)),
    # A restarted server resets the histogram.
    ((12.5, 4), (0.5, 5)),
])
def test_prefill_delta_refuses_unattributable_readings(before, after):
    from scripts.runtime.engines.vllm import VllmEngine
    assert VllmEngine.prefill_seconds_from_delta(before, after) is None


# ── externally-managed server (server_url) ──

def test_is_installed_true_with_only_a_server_url(engine):
    """setup_check.py records `server_url` when it finds a reachable vLLM with no
    local binary/launcher — that alone must count as installed."""
    engine._launcher = None
    engine._executable = None
    engine._server_url = "http://gpu-box:8000"
    assert engine.is_installed() is True


def test_base_url_prefers_the_configured_server(engine):
    engine._server_url = "http://gpu-box:8000"
    assert engine.base_url == "http://gpu-box:8000"


def test_base_url_falls_back_to_the_local_port_when_unconfigured(engine):
    engine._server_url = None
    assert engine.base_url == config.VLLM_URL


def test_qualification_ignores_external_vllm_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_AI_BENCH_QUALIFICATION", "1")
    monkeypatch.setattr(config, "VLLM_VENV", tmp_path / "vllm-env")
    executable = config.VLLM_VENV / "bin" / "vllm"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(vllm_module, "find_vllm_launcher", lambda: "/usr/bin/vllm-launch")
    monkeypatch.setattr(vllm_module.shutil, "which", lambda _name: "/usr/bin/vllm")
    instance = VllmEngine()
    assert instance._launcher is None
    assert instance._server_url is None
    assert instance._executable == str(executable)


def test_ensure_running_succeeds_against_a_reachable_configured_server(engine, monkeypatch):
    """The setup-confirmed external server must not require a local launcher/executable
    or a populated model cache — both are irrelevant when we never spawn it ourselves."""
    engine._launcher = None
    engine._executable = None
    engine._server_url = "http://gpu-box:8000"
    monkeypatch.setattr(VllmEngine, "available", lambda self: True)
    assert engine.ensure_running() is True


def test_ensure_running_fails_when_the_configured_server_is_unreachable(engine, monkeypatch):
    engine._launcher = None
    engine._executable = None
    engine._server_url = "http://gpu-box:8000"
    monkeypatch.setattr(VllmEngine, "available", lambda self: False)
    assert engine.ensure_running() is False


def test_external_server_model_is_available_without_local_cache(engine, monkeypatch):
    engine._server_url = "http://gpu-box:8000"
    monkeypatch.setattr(VllmEngine, "_served_model_ids", lambda self: {TEST_REPO})

    assert engine.model_pulled(TEST_TAG) is True
    assert engine.list_installed_models() == [{"tag": TEST_TAG, "size": None}]


def test_external_server_accepts_catalog_tag_as_served_model_name(engine, monkeypatch):
    engine._server_url = "http://gpu-box:8000"
    monkeypatch.setattr(VllmEngine, "_served_model_ids", lambda self: {TEST_TAG})

    assert engine.model_pulled(TEST_TAG) is True


def test_external_server_does_not_expose_other_catalog_models(engine, monkeypatch):
    engine._server_url = "http://gpu-box:8000"
    monkeypatch.setattr(VllmEngine, "_served_model_ids", lambda self: {"org/other-model"})

    assert engine.model_pulled(TEST_TAG) is False
    assert engine.list_installed_models() == []


@pytest.mark.parametrize("payload", [None, {}, {"data": "bad"}, {"data": [{"id": 4}]}])
def test_external_server_model_discovery_rejects_unusable_responses(engine, monkeypatch, payload):
    engine._server_url = "http://gpu-box:8000"

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return payload

    monkeypatch.setattr("scripts.runtime.engines.vllm.requests.get", lambda *a, **k: Response())
    expected = set() if payload == {"data": [{"id": 4}]} else None
    assert engine._served_model_ids() == expected


def test_ensure_model_against_a_server_url_never_spawns_a_process(engine, monkeypatch):
    """A tag switch against an externally-managed server must not try to launch/replace
    the process we never started — just confirm reachability and update our bookkeeping."""
    engine._launcher = None
    engine._executable = None
    engine._server_url = "http://gpu-box:8000"
    engine._proc = None
    engine._loaded_tag = None
    monkeypatch.setattr(VllmEngine, "available", lambda self: True)
    monkeypatch.setattr(VllmEngine, "_served_model_ids", lambda self: {TEST_REPO})

    def fail_popen(*a, **k):
        raise AssertionError("must not spawn a process for an externally-managed server")
    monkeypatch.setattr("subprocess.Popen", fail_popen)

    engine._ensure_model("qwen3.5:9b-q4_K_M", 1024)
    assert engine._loaded_tag == "qwen3.5:9b-q4_K_M"
    assert engine._loaded_model_id == TEST_REPO
    assert engine._proc is None


def test_external_server_preserves_served_model_name_for_requests(engine, monkeypatch):
    engine._server_url = "http://gpu-box:8000"
    engine._proc = None
    engine._loaded_tag = None
    monkeypatch.setattr(VllmEngine, "available", lambda self: True)
    monkeypatch.setattr(VllmEngine, "_served_model_ids", lambda self: {TEST_TAG})

    engine._ensure_model(TEST_TAG, 1024)

    assert engine._loaded_model_id == TEST_TAG


def test_external_embedding_request_uses_the_server_advertised_model_id(engine, monkeypatch):
    engine._server_url = "http://gpu-box:8000"
    engine._proc = None
    engine._loaded_tag = None
    monkeypatch.setattr(VllmEngine, "available", lambda self: True)
    monkeypatch.setattr(VllmEngine, "_served_model_ids", lambda self: {TEST_TAG})
    captured = {}

    class Response:
        ok = True

        @staticmethod
        def json():
            return {"data": [{"index": 0, "embedding": [1.0, 2.0]}]}

    def post(_url, *, json, timeout):
        captured.update(json)
        return Response()

    monkeypatch.setattr("scripts.runtime.engines.vllm.requests.post", post)
    measurement = engine.embed(TEST_TAG, ["hello"])

    assert captured["model"] == TEST_TAG
    assert measurement.embeddings == [[1.0, 2.0]]


def test_ensure_model_against_a_server_url_raises_when_unreachable(engine, monkeypatch):
    engine._launcher = None
    engine._executable = None
    engine._server_url = "http://gpu-box:8000"
    engine._proc = None
    engine._loaded_tag = None
    monkeypatch.setattr(VllmEngine, "available", lambda self: False)
    with pytest.raises(RuntimeError, match="not reachable"):
        engine._ensure_model("qwen3.5:9b-q4_K_M", 1024)


def test_ensure_model_against_a_server_url_rejects_mismatched_model(engine, monkeypatch):
    engine._server_url = "http://gpu-box:8000"
    engine._proc = None
    engine._loaded_tag = None
    monkeypatch.setattr(VllmEngine, "available", lambda self: True)
    monkeypatch.setattr(VllmEngine, "_served_model_ids", lambda self: {"org/other-model"})

    with pytest.raises(RuntimeError, match=f"serves org/other-model, not {TEST_TAG}"):
        engine._ensure_model(TEST_TAG, 1024)
