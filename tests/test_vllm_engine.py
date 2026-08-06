import json
from pathlib import Path

import pytest

from scripts.runtime import config
from scripts.runtime.engines.vllm import VllmEngine
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
    instance._proc = type("P", (), {"poll": staticmethod(lambda: None)})()
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


def test_max_model_len_is_per_sequence_and_not_scaled_by_parallelism(engine):
    """Unlike llama-server's -c, --max-model-len is per sequence."""
    command = engine.server_command("org/m", 4096, n_parallel=8)
    assert command[command.index("--max-model-len") + 1] == "4096"
    assert command[command.index("--max-num-seqs") + 1] == "8"


def test_context_is_omitted_when_unset(engine):
    assert "--max-model-len" not in engine.server_command("org/m", None)


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


def test_generate_reports_no_server_prompt_time(engine, monkeypatch):
    _patch_stream(monkeypatch, [_text_chunk("x", finish="stop"),
                                {"choices": [], "usage": {"completion_tokens": 1}}])
    result = engine.generate("qwen3.5:9b-q4_K_M", "hi", timeout=30)
    assert result.server_prompt_sec is None, "vLLM reports no per-request prompt duration"


def test_generate_requests_the_repo_id_not_the_tag(engine, monkeypatch):
    captured = _patch_stream(monkeypatch, [_text_chunk("x", finish="stop"),
                                            {"choices": [], "usage": {"completion_tokens": 1}}])
    engine.generate("qwen3.5:9b-q4_K_M", "hi", timeout=30)
    assert captured["payload"]["model"] == "QuantTrio/Qwen3.5-9B-AWQ"


def test_generate_measurement_is_internally_consistent(engine, monkeypatch):
    _patch_stream(monkeypatch, [_text_chunk("a"), _text_chunk("b", finish="stop"),
                                {"choices": [], "usage": {"completion_tokens": 4}}])
    from scripts.runtime.engines.base import measurement_validation_errors
    assert measurement_validation_errors(engine.generate("qwen3.5:9b-q4_K_M", "hi", 30)) == []


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
    snapshot = (engine._cache_home / "hub" / "models--QuantTrio--Qwen3.5-9B-AWQ"
                / "snapshots" / "abc")
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").touch()
    assert engine.model_pulled(tag) is True
    assert [m["tag"] for m in engine.list_installed_models()] == [tag]


def test_an_unknown_tag_has_no_repo(engine):
    assert engine._repo("not-a-model") is None
    assert engine.model_pulled("not-a-model") is False


def test_max_context_length_reads_the_snapshot_config(engine):
    tag = "qwen3.5:9b-q4_K_M"
    snapshot = (engine._cache_home / "hub" / "models--QuantTrio--Qwen3.5-9B-AWQ"
                / "snapshots" / "abc")
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"max_position_embeddings": 32768}))
    assert engine.max_context_length(tag) == 32768


def test_max_context_length_falls_back_on_missing_or_broken_config(engine):
    tag = "qwen3.5:9b-q4_K_M"
    assert engine.max_context_length(tag, default=4096) == 4096
    snapshot = (engine._cache_home / "hub" / "models--QuantTrio--Qwen3.5-9B-AWQ"
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

def _cache_snapshot(engine, repo="QuantTrio--Qwen3.5-9B-AWQ", files=("model.safetensors", "config.json")):
    repo_dir = engine._cache_home / "hub" / f"models--{repo}"
    blobs, snapshot = repo_dir / "blobs", repo_dir / "snapshots" / "abc"
    blobs.mkdir(parents=True); snapshot.mkdir(parents=True)
    for index, name in enumerate(files):
        blob = blobs / f"b{index}"
        blob.write_text("weights")
        (snapshot / name).symlink_to(blob)
    return snapshot


def test_resume_artifacts_resolve_through_the_cache_symlinks(engine):
    _cache_snapshot(engine, files=("model-00001.safetensors", "model-00002.safetensors", "config.json"))
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


def test_resume_identity_builds_for_a_cached_model(engine):
    """The failure that stopped the first real run: the base class raised NotImplementedError."""
    from scripts.results.resume_policy import cached_file_identity
    _cache_snapshot(engine)
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
