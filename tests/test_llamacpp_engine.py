"""Tests for LlamaCppEngine — the llama.cpp (llama-server) implementation of
InferenceEngine that resolves each tag to GGUF file(s) downloaded ahead of
time by setup_check.py into config.MODELS_DIR.

Three groups:
  1. Local model-file resolution (_slug, _resolve_model_files, model_pulled,
     list_installed_models, max_context_length) — built against a fake
     catalog and tmp_path, monkeypatching config.MODELS_DIR and
     models.LLM_MODELS/EMBED_MODELS so no real download is needed.
  2. Streaming / timing parsing in generate()/chat() — mocked at the
     _urlopen seam.
  3. Maintenance seams (is_connection_crash, reachable_or_abort, unload,
     unload_all, wait_until_unloaded) — exercise LlamaCppEngine's
     single-process-per-model lifecycle.
"""

import json

import gguf
import pytest
import requests

import config
from engines.llamacpp import LlamaCppEngine
import engines.llamacpp as llamacpp_module
from shared import EngineLoopDetected, EngineTimeout


# ══════════════════════════════════════════════════════════════════════════
#  Group 1 — local model-file resolution
# ══════════════════════════════════════════════════════════════════════════


_FAKE_CATALOG = [
    {"tag": "phi4-mini", "hf_repo": "org/phi4-mini-gguf", "hf_file": "phi4-mini.Q4_K_M.gguf"},
    {"tag": "llama3.2:3b-instruct-q4_K_M", "hf_repo": "org/llama32-gguf",
     "hf_file": "llama32-3b.Q4_K_M.gguf"},
    {"tag": "split:model", "hf_repo": "org/split-gguf",
     "hf_file": ["split-00001-of-00002.gguf", "split-00002-of-00002.gguf"]},
]


@pytest.fixture
def fake_catalog(monkeypatch, tmp_path):
    """Point config.MODELS_DIR at tmp_path and swap in a small fixture
    catalog (LLM_MODELS + EMBED_MODELS combined) instead of the real one."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(llamacpp_module, "LLM_MODELS", _FAKE_CATALOG)
    monkeypatch.setattr(llamacpp_module, "EMBED_MODELS", [])
    return tmp_path


def _write_model_file(models_dir, tag, filename, content: bytes):
    slug = LlamaCppEngine._slug(tag)
    model_dir = models_dir / "llamacpp" / slug
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / filename).write_bytes(content)


def test_models_dir_namespaced_under_engine_name(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    assert LlamaCppEngine._models_dir() == tmp_path / "llamacpp"


def test_slug_replaces_colons_and_slashes():
    assert LlamaCppEngine._slug("llama3.2:3b-instruct-q4_K_M") == "llama3.2_3b-instruct-q4_K_M"
    assert LlamaCppEngine._slug("someorg/some-model") == "someorg_some-model"


def test_resolve_model_files_finds_single_file(fake_catalog):
    _write_model_file(fake_catalog, "phi4-mini", "phi4-mini.Q4_K_M.gguf", b"fake-gguf-bytes")
    paths = LlamaCppEngine._resolve_model_files("phi4-mini")
    assert paths is not None
    assert paths[0].read_bytes() == b"fake-gguf-bytes"


def test_resolve_model_files_missing_tag_returns_none(fake_catalog):
    assert LlamaCppEngine._resolve_model_files("not-in-catalog") is None


def test_resolve_model_files_finds_custom_dropped_in_model(fake_catalog):
    from benchmark import resolve_custom_models

    custom_dir = fake_catalog / "llamacpp" / "my-custom-model"
    custom_dir.mkdir(parents=True)
    model_path = custom_dir / "weights.gguf"
    model_path.write_bytes(b"custom")
    assert LlamaCppEngine._resolve_model_files("my-custom-model") == [model_path]
    engine = LlamaCppEngine()
    installed_tags = [model["tag"] for model in engine.list_installed_models()]
    selected = resolve_custom_models(["my-custom-model"], [], installed_tags)
    assert selected[0]["tag"] == "my-custom-model"
    assert engine.model_pulled(selected[0]["tag"]) is True


def test_resolve_model_files_requires_complete_custom_multipart_set(fake_catalog):
    custom_dir = fake_catalog / "llamacpp" / "custom-split"
    custom_dir.mkdir(parents=True)
    first = custom_dir / "weights-00001-of-00002.gguf"
    second = custom_dir / "weights-00002-of-00002.gguf"
    first.write_bytes(b"a")
    assert LlamaCppEngine._resolve_model_files("custom-split") is None
    second.write_bytes(b"b")
    assert LlamaCppEngine._resolve_model_files("custom-split") == [first, second]


def test_resolve_model_files_missing_file_returns_none(fake_catalog):
    # Catalog entry exists but the file was never downloaded.
    assert LlamaCppEngine._resolve_model_files("phi4-mini") is None


def test_resolve_model_files_requires_every_split_part(fake_catalog):
    _write_model_file(fake_catalog, "split:model", "split-00001-of-00002.gguf", b"a")
    assert LlamaCppEngine._resolve_model_files("split:model") is None  # part 2 missing
    _write_model_file(fake_catalog, "split:model", "split-00002-of-00002.gguf", b"b")
    paths = LlamaCppEngine._resolve_model_files("split:model")
    assert paths is not None
    assert [p.name for p in paths] == ["split-00001-of-00002.gguf", "split-00002-of-00002.gguf"]


def test_model_pulled_true_when_resolvable(fake_catalog):
    _write_model_file(fake_catalog, "phi4-mini", "phi4-mini.Q4_K_M.gguf", b"x")
    assert LlamaCppEngine().model_pulled("phi4-mini") is True


def test_model_pulled_false_when_not_resolvable(fake_catalog):
    assert LlamaCppEngine().model_pulled("phi4-mini") is False


def test_list_installed_models_lists_every_downloaded_catalog_tag(fake_catalog):
    _write_model_file(fake_catalog, "phi4-mini", "phi4-mini.Q4_K_M.gguf", b"aaa")
    _write_model_file(fake_catalog, "llama3.2:3b-instruct-q4_K_M", "llama32-3b.Q4_K_M.gguf", b"bbbbb")
    installed = {m["tag"]: m["size"] for m in LlamaCppEngine().list_installed_models()}
    assert installed == {"phi4-mini": 3, "llama3.2:3b-instruct-q4_K_M": 5}


def test_list_installed_models_includes_custom_dropped_in_model(fake_catalog):
    # A model dropped in manually, not in the catalog at all.
    custom_dir = fake_catalog / "llamacpp" / "my-custom-model"
    custom_dir.mkdir(parents=True)
    (custom_dir / "weights.gguf").write_bytes(b"cc")
    installed = {m["tag"]: m["size"] for m in LlamaCppEngine().list_installed_models()}
    assert installed == {"my-custom-model": 2}


def test_list_installed_models_omits_ambiguous_custom_directory(fake_catalog):
    custom_dir = fake_catalog / "llamacpp" / "ambiguous"
    custom_dir.mkdir(parents=True)
    (custom_dir / "one.gguf").write_bytes(b"a")
    (custom_dir / "two.gguf").write_bytes(b"b")
    assert LlamaCppEngine().list_installed_models() == []


def test_list_installed_models_empty_when_dir_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "does-not-exist")
    monkeypatch.setattr(llamacpp_module, "LLM_MODELS", [])
    monkeypatch.setattr(llamacpp_module, "EMBED_MODELS", [])
    assert LlamaCppEngine().list_installed_models() == []


def _write_gguf(path, context_length: int):
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_context_length(context_length)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_max_context_length_reads_architecture_prefixed_key(tmp_path, monkeypatch):
    gguf_path = tmp_path / "model.gguf"
    _write_gguf(gguf_path, 40960)
    monkeypatch.setattr(LlamaCppEngine, "_resolve_model_files", classmethod(lambda cls, tag: [gguf_path]))
    assert LlamaCppEngine().max_context_length("qwen3.5:4b") == 40960


def test_max_context_length_falls_back_to_default_when_not_pulled(fake_catalog):
    assert LlamaCppEngine().max_context_length("phi4-mini", default=8192) == 8192


def test_max_context_length_falls_back_to_default_on_unparseable_file(tmp_path, monkeypatch):
    bad_path = tmp_path / "not-really-gguf.bin"
    bad_path.write_bytes(b"not a gguf file")
    monkeypatch.setattr(LlamaCppEngine, "_resolve_model_files", classmethod(lambda cls, tag: [bad_path]))
    assert LlamaCppEngine().max_context_length("some-tag", default=2048) == 2048


# ══════════════════════════════════════════════════════════════════════════
#  Group 2 — streaming / timing parsing
# ══════════════════════════════════════════════════════════════════════════


class _FakeResponse:
    """Mimics urllib.request.urlopen's return value: a context manager that
    iterates raw SSE lines as bytes."""

    def __init__(self, chunks):
        lines = []
        for c in chunks:
            lines.append(f"data: {json.dumps(c)}\n".encode())
        lines.append(b"data: [DONE]\n")
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._lines)


def _patch_urlopen(monkeypatch, chunks):
    monkeypatch.setattr(LlamaCppEngine, "_urlopen",
                        staticmethod(lambda req, timeout: _FakeResponse(chunks)))


# ── _sanitize_tps ──

def test_sanitize_tps_passes_through_plausible_value():
    assert LlamaCppEngine._sanitize_tps(120.0, tokens=50, ttft=0.5, total=1.0) == 120.0


def test_sanitize_tps_passes_through_at_exact_ceiling():
    ceiling = config.MAX_PLAUSIBLE_TPS
    assert LlamaCppEngine._sanitize_tps(ceiling, tokens=50, ttft=0.5, total=1.0) == ceiling


def test_sanitize_tps_falls_back_to_wall_clock_when_implausible():
    # Reproduces the real bug: llama-server reports a tiny predicted_ms under
    # heavy slot contention, producing a tps ratio with no physical basis.
    huge = config.MAX_PLAUSIBLE_TPS + 1
    result = LlamaCppEngine._sanitize_tps(huge, tokens=5, ttft=1.0, total=3.0)
    assert result == pytest.approx(2.5)  # 5 tokens / (3.0 - 1.0)s decode time


def test_sanitize_tps_returns_zero_when_decode_elapsed_not_positive():
    huge = config.MAX_PLAUSIBLE_TPS + 1
    assert LlamaCppEngine._sanitize_tps(huge, tokens=5, ttft=2.0, total=2.0) == 0


def _patch_ensure_model(monkeypatch):
    monkeypatch.setattr(LlamaCppEngine, "_ensure_model", lambda self, *a, **kw: None)


def _clock(*values):
    iterator = iter(values)
    last = values[-1]

    def now():
        nonlocal last
        try:
            last = next(iterator)
        except StopIteration:
            pass
        return last

    return now


# ── generate ──

def test_generate_uses_server_reported_timings(monkeypatch):
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        *[{"content": "x", "tokens": [i]} for i in range(10)],
        {"content": "", "stop": True,
         "timings": {"predicted_n": 10, "predicted_ms": 2000, "prompt_ms": 500}},
    ])
    ttft, tokens, tps = LlamaCppEngine().generate("some-tag", "prompt", num_ctx=2048)
    assert ttft >= 0
    assert tokens == 10
    assert tps == pytest.approx(5.0)  # 10 tokens / 2 sec


def test_generate_falls_back_to_wall_clock_ttft_when_server_omits_it(monkeypatch):
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", _clock(100.0, 100.0, 101.5, 101.5, 102.0))
    _patch_urlopen(monkeypatch, [
        {"content": "Hi", "tokens": [1]},
        {"content": "", "stop": True, "timings": {"predicted_n": 1, "predicted_ms": 0}},
    ])
    ttft, tokens, tps = LlamaCppEngine().generate("some-tag", "prompt")
    assert ttft == pytest.approx(1.5)
    assert tps == pytest.approx(2.0)


def test_generate_sanitizes_implausible_server_reported_tps(monkeypatch):
    # Reproduces the exact real-world observed failure: under heavy
    # concurrent-slot contention, llama-server's timings.predicted_ms can be
    # implausibly tiny (predicted_n=1, predicted_ms=0.001 -> raw tps of
    # exactly 1000000.0) — this must fall back to a wall-clock estimate
    # instead of returning garbage.
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", _clock(100.0, 100.0, 100.5, 100.5, 110.5))
    _patch_urlopen(monkeypatch, [
        {"content": "Hi", "tokens": [1]},
        {"content": "", "stop": True, "timings": {"predicted_n": 1, "predicted_ms": 0.001}},
    ])
    ttft, tokens, tps = LlamaCppEngine().generate("some-tag", "prompt")
    assert ttft == pytest.approx(0.5)
    assert tps == pytest.approx(0.1)  # 1 token / (10.5 - 0.5)s wall-clock decode time


def test_generate_counts_native_token_ids_not_sse_fragments(monkeypatch):
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"content": "several decoded pieces", "tokens": [11, 12]},
        {"content": "x", "tokens": []},
        {"content": "", "stop": True, "timings": {"predicted_n": 1, "predicted_ms": 1000}},
    ])
    _, tokens, _ = LlamaCppEngine().generate("some-tag", "prompt")
    assert tokens == 2


def test_generate_logs_raw_server_values_when_sanitizing(monkeypatch):
    # The whole point of surfacing this warning is to make the raw
    # predicted_n/predicted_ms diagnosable, and to show whether our own
    # count agrees with the server's — assert the actual numbers appear,
    # not just that some warning fired.
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", _clock(100.0, 100.0, 100.5, 100.5, 110.5))
    _patch_urlopen(monkeypatch, [
        {"content": "Hi", "tokens": [1]},
        {"content": "", "stop": True, "timings": {"predicted_n": 1, "predicted_ms": 0.001}},
    ])
    warnings = []
    monkeypatch.setattr(llamacpp_module.Shared, "warn", staticmethod(lambda msg: warnings.append(msg)))
    LlamaCppEngine().generate("some-tag", "prompt")
    assert len(warnings) == 1
    assert "some-tag" in warnings[0]
    assert "server predicted_n=1" in warnings[0]
    assert "response tokens=1" in warnings[0]
    assert "predicted_ms=0.001" in warnings[0]


def test_generate_does_not_warn_when_tps_is_plausible(monkeypatch):
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"content": "Hel", "tokens": [1]},
        {"content": "lo", "tokens": [2]},
        {"content": "", "stop": True,
         "timings": {"predicted_n": 10, "predicted_ms": 2000, "prompt_ms": 500}},
    ])
    warnings = []
    monkeypatch.setattr(llamacpp_module.Shared, "warn", staticmethod(lambda msg: warnings.append(msg)))
    LlamaCppEngine().generate("some-tag", "prompt", num_ctx=2048)
    assert warnings == []


def test_generate_preserves_request_to_first_output_ttft(monkeypatch):
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", _clock(10.0, 10.0, 12.0, 12.0, 13.0))
    _patch_urlopen(monkeypatch, [
        {"content": "Hi", "tokens": [1]},
        {"content": "", "stop": True,
         "timings": {"predicted_n": 1, "predicted_ms": 500, "prompt_ms": 100}},
    ])
    ttft, _, _ = LlamaCppEngine().generate("tag", "prompt")
    assert ttft == pytest.approx(2.0)


def test_generate_enforces_total_deadline_and_keeps_partial_text(monkeypatch):
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", _clock(0.0, 0.0, 1.0, 1.0, 6.0))
    _patch_urlopen(monkeypatch, [
        {"content": "partial", "tokens": [1]},
        {"content": " too late", "tokens": [2]},
    ])
    with pytest.raises(EngineTimeout) as exc_info:
        LlamaCppEngine().generate("tag", "prompt", timeout=5)
    assert exc_info.value.partial_text == "partial too late"


def test_generate_enforces_deadline_during_sse_keepalive_lines(monkeypatch):
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", _clock(0.0, 0.0, 6.0))
    response = type("Response", (), {
        "__enter__": lambda self: self,
        "__exit__": lambda self, *args: False,
        "__iter__": lambda self: iter([b": ping\n"]),
    })()
    monkeypatch.setattr(LlamaCppEngine, "_urlopen", staticmethod(lambda req, timeout: response))
    with pytest.raises(EngineTimeout):
        LlamaCppEngine().generate("tag", "prompt", timeout=5)


# ── chat ──

def test_chat_returns_content_and_server_timings(monkeypatch):
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"content": "H"}}]},
        {"choices": [{"delta": {"content": "e"}}]},
        {"choices": [{"delta": {"content": "l"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "timings": {"predicted_n": 4, "predicted_ms": 1000, "prompt_ms": 200, "prompt_n": 50}},
        {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 4, "total_tokens": 54}},
    ])
    ttft, tokens, tps, prompt_eval_count, text = LlamaCppEngine().chat(
        "tag", [{"role": "user", "content": "hi"}])
    assert text == "Hello"
    assert prompt_eval_count == 50
    assert ttft == pytest.approx(0.2)
    assert tokens == 4
    assert tps == pytest.approx(4.0)


def test_chat_sanitizes_implausible_server_reported_tps(monkeypatch):
    # Same real bug as generate()'s equivalent test, via the chat() code path.
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(
        llamacpp_module.time, "perf_counter",
        _clock(100.0, 100.0, 100.5, 100.5, 105.0, 110.5),
    )
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"content": "Hi"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "timings": {"predicted_n": 1, "predicted_ms": 0.001}},
    ])
    ttft, tokens, tps, _, _ = LlamaCppEngine().chat("tag", [{"role": "user", "content": "hi"}])
    assert ttft == pytest.approx(0.5)
    assert tps == pytest.approx(0.1)  # 1 token / (10.5 - 0.5)s wall-clock decode time


def test_chat_prefers_usage_prompt_tokens_over_timings_prompt_n(monkeypatch):
    # usage.prompt_tokens (true total) must win over timings.prompt_n
    # (cache-miss-only count) — see chat()'s docstring.
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"content": "Hi"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "timings": {"predicted_n": 4, "predicted_ms": 1000, "prompt_n": 12}},
        {"choices": [], "usage": {"prompt_tokens": 2048, "completion_tokens": 4, "total_tokens": 2052}},
    ])
    _, _, _, prompt_eval_count, _ = LlamaCppEngine().chat(
        "tag", [{"role": "user", "content": "hi"}])
    assert prompt_eval_count == 2048


def test_chat_falls_back_to_reasoning_text_when_content_empty(monkeypatch):
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"reasoning_content": "Let me consider... "}}]},
        {"choices": [{"delta": {"reasoning_content": "the answer is 42."}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "timings": {"predicted_n": 8, "predicted_ms": 1000, "prompt_n": 10}},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}},
    ])
    _, tokens, _, _, text = LlamaCppEngine().chat("tag", [{"role": "user", "content": "hi"}])
    assert text == "Let me consider... the answer is 42."
    assert tokens == 8


def test_chat_prefers_content_over_reasoning_when_both_present(monkeypatch):
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"content": "answer", "reasoning_content": "reasoning"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "timings": {"predicted_n": 1, "predicted_ms": 1000, "prompt_n": 10}},
    ])
    _, _, _, _, text = LlamaCppEngine().chat("tag", [{"role": "user", "content": "hi"}])
    assert text == "answer"


def test_chat_check_loop_raises_early_on_repeated_hedging(monkeypatch):
    _patch_ensure_model(monkeypatch)
    import itertools

    counter = itertools.count(0, 1.0)
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", lambda: next(counter))
    monkeypatch.setattr(llamacpp_module.config, "LOOP_CHECK_INTERVAL", 0)
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"content": "wait, "}}]},
        {"choices": [{"delta": {"content": "wait, "}}]},
        {"choices": [{"delta": {"content": "wait, "}}]},
        {"choices": [{"delta": {"content": "wait, "}}]},
        {"choices": [{"delta": {"content": "wait, still stuck"}}]},
        {"choices": [{"delta": {"content": "this chunk should never be reached"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "timings": {"predicted_n": 6, "predicted_ms": 1000, "prompt_n": 10}},
    ])
    with pytest.raises(EngineLoopDetected) as exc_info:
        LlamaCppEngine().chat("tag", [{"role": "user", "content": "hi"}], check_loop=True)
    assert "loop" in str(exc_info.value).lower()
    assert "wait, wait, wait, wait, wait, still stuck" == exc_info.value.partial_text
    assert "never be reached" not in exc_info.value.partial_text


def test_chat_raises_engine_timeout_type_for_run_measured_calls_compat(monkeypatch):
    # run_measured_calls (shared.py) does isinstance(e, EngineLoopDetected) /
    # isinstance(e, TimeoutError) checks by type — LlamaCppEngine must raise
    # the *same* shared.py types, not engine-specific subclasses, or that
    # dispatch silently breaks for this engine.
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", _clock(0.0, 0.0, 1.0, 1.0, 100.0))
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"content": "hi"}}]},
        {"choices": [{"delta": {"content": "still going"}}]},
    ])
    with pytest.raises(EngineTimeout):
        LlamaCppEngine().chat("tag", [{"role": "user", "content": "hi"}], timeout=5)


# ── chat_tools ──

def test_chat_tools_accumulates_fragmented_arguments(monkeypatch):
    # arguments streams as partial JSON text across chunks and must be
    # reassembled by index before parsing; name arrives once up front.
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "get_weather", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"location": "Par'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": 'is", "unit": "celsius"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
         "timings": {"predicted_n": 3, "predicted_ms": 1000, "prompt_n": 20}},
    ])
    _, _, _, prompt_eval_count, text, tool_calls = LlamaCppEngine().chat_tools(
        "tag", [{"role": "user", "content": "weather?"}], tools=[{"type": "function"}])
    assert tool_calls == [{"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}]
    assert prompt_eval_count == 20
    assert text == ""  # no content chunks, only tool calls


def test_chat_tools_zero_tool_calls_returns_empty_list(monkeypatch):
    # A model that answers in prose instead of calling anything yields an
    # empty tool_calls list plus the response text.
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"content": "I can't help with that."}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "timings": {"predicted_n": 5, "predicted_ms": 1000, "prompt_n": 15}},
    ])
    _, _, _, _, text, tool_calls = LlamaCppEngine().chat_tools(
        "tag", [{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    assert tool_calls == []
    assert text == "I can't help with that."


def test_chat_tools_malformed_arguments_falls_back_to_empty_dict(monkeypatch):
    # A truncated/invalid arguments string must not crash the parse — it falls
    # back to {} while still reporting the tool name, and is marked incomplete
    # even though this stream completed normally (didn't time out) — a
    # completed-but-unparseable call is not a genuine empty-argument call.
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "set_timer", "arguments": '{"minutes":'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
         "timings": {"predicted_n": 1, "predicted_ms": 1000, "prompt_n": 10}},
    ])
    _, _, _, _, _, tool_calls = LlamaCppEngine().chat_tools(
        "tag", [{"role": "user", "content": "timer"}], tools=[{"type": "function"}])
    assert tool_calls == [{"name": "set_timer", "arguments": {}, "incomplete": True}]


def test_chat_tools_multiple_calls_ordered_by_index(monkeypatch):
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 1, "id": "c2", "function": {"name": "second", "arguments": "{}"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "first", "arguments": "{}"}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
         "timings": {"predicted_n": 2, "predicted_ms": 1000, "prompt_n": 10}},
    ])
    _, _, _, _, _, tool_calls = LlamaCppEngine().chat_tools(
        "tag", [{"role": "user", "content": "two"}], tools=[{"type": "function"}])
    assert [c["name"] for c in tool_calls] == ["first", "second"]


def test_chat_tools_timeout_serializes_completed_fragmented_call(monkeypatch):
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", _clock(0.0, 0.0, 1.0, 1.0, 2.0, 10.0))
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "get_weather", "arguments": '{"city":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '"Paris"}'}}]}}]},
        {"choices": [{"delta": {"content": "late"}}]},
    ])
    with pytest.raises(EngineTimeout) as exc_info:
        LlamaCppEngine().chat_tools("tag", [], [], timeout=5)
    assert json.loads(exc_info.value.partial_text) == [
        {"name": "get_weather", "arguments": {"city": "Paris"}},
    ]


def test_chat_tools_timeout_marks_incomplete_argument_evidence(monkeypatch):
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", _clock(0.0, 0.0, 1.0, 10.0))
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "set_timer", "arguments": '{"minutes":'}}]}}]},
    ])
    with pytest.raises(EngineTimeout) as exc_info:
        LlamaCppEngine().chat_tools("tag", [], [], timeout=5)
    assert json.loads(exc_info.value.partial_text) == [
        {"name": "set_timer", "arguments": {}, "incomplete": True},
    ]


def test_chat_tools_timeout_with_text_only_keeps_text(monkeypatch):
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", _clock(0.0, 0.0, 1.0, 10.0))
    _patch_urlopen(monkeypatch, [{"choices": [{"delta": {"content": "not calling"}}]}])
    with pytest.raises(EngineTimeout) as exc_info:
        LlamaCppEngine().chat_tools("tag", [], [], timeout=5)
    assert exc_info.value.partial_text == "not calling"


def test_chat_tools_falls_back_to_reasoning_text_when_content_empty(monkeypatch):
    # Mirrors chat()'s reasoning fallback (test_chat_falls_back_to_reasoning_
    # text_when_content_empty) — a reasoning model that declines to call
    # anything can stream its whole turn via reasoning_content.
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"reasoning_content": "Let me consider... "}}]},
        {"choices": [{"delta": {"reasoning_content": "no tool fits."}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "timings": {"predicted_n": 8, "predicted_ms": 1000, "prompt_n": 10}},
    ])
    _, tokens, _, _, text, tool_calls = LlamaCppEngine().chat_tools(
        "tag", [{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    assert text == "Let me consider... no tool fits."
    assert tool_calls == []
    assert tokens == 8


def test_chat_tools_check_loop_raises_during_reasoning_phase(monkeypatch):
    # Before the fix, check_loop only inspected `content`, so a model stuck
    # looping in reasoning_content (with content still empty) would never
    # trip loop detection and would burn the full accuracy timeout instead.
    _patch_ensure_model(monkeypatch)
    import itertools

    counter = itertools.count(0, 1.0)
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", lambda: next(counter))
    monkeypatch.setattr(llamacpp_module.config, "LOOP_CHECK_INTERVAL", 0)
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"reasoning_content": "wait, "}}]},
        {"choices": [{"delta": {"reasoning_content": "wait, "}}]},
        {"choices": [{"delta": {"reasoning_content": "wait, "}}]},
        {"choices": [{"delta": {"reasoning_content": "wait, "}}]},
        {"choices": [{"delta": {"reasoning_content": "wait, still stuck"}}]},
        {"choices": [{"delta": {"reasoning_content": "this chunk should never be reached"}}]},
    ])
    with pytest.raises(EngineLoopDetected) as exc_info:
        LlamaCppEngine().chat_tools(
            "tag", [{"role": "user", "content": "hi"}], tools=[{"type": "function"}], check_loop=True)
    assert "wait, wait, wait, wait, wait, still stuck" == exc_info.value.partial_text
    assert "never be reached" not in exc_info.value.partial_text


def test_chat_tools_timeout_with_reasoning_only_keeps_reasoning_text(monkeypatch):
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", _clock(0.0, 0.0, 1.0, 10.0))
    _patch_urlopen(monkeypatch, [{"choices": [{"delta": {"reasoning_content": "still thinking"}}]}])
    with pytest.raises(EngineTimeout) as exc_info:
        LlamaCppEngine().chat_tools("tag", [], [], timeout=5)
    assert exc_info.value.partial_text == "still thinking"


# ── embed ──

def test_embed_returns_embeddings_in_index_order(monkeypatch):
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(
        "engines.llamacpp.requests.post",
        lambda url, json=None, timeout=None: type("R", (), {
            "ok": True,
            "json": lambda self: {"data": [
                {"index": 1, "embedding": [0.2]},
                {"index": 0, "embedding": [0.1]},
            ]},
        })(),
    )
    embeddings, elapsed = LlamaCppEngine().embed("nomic-embed-text", ["a", "b"])
    assert embeddings == [[0.1], [0.2]]
    assert elapsed >= 0


def test_embed_raises_on_rejected_request(monkeypatch):
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(
        "engines.llamacpp.requests.post",
        lambda url, json=None, timeout=None: type("R", (), {
            "ok": False, "status_code": 500, "json": lambda self: {"error": "oom"},
            "text": "oom",
        })(),
    )
    with pytest.raises(RuntimeError, match="oom"):
        LlamaCppEngine().embed("nomic-embed-text", ["a"])


# ══════════════════════════════════════════════════════════════════════════
#  Group 3 — maintenance seams
# ══════════════════════════════════════════════════════════════════════════


def test_is_connection_crash_true_for_connection_error():
    assert LlamaCppEngine().is_connection_crash(requests.exceptions.ConnectionError("down")) is True


def test_is_connection_crash_false_for_unrelated_error():
    assert LlamaCppEngine().is_connection_crash(ValueError("bad json")) is False


@pytest.mark.parametrize(("listing", "expected"), [
    ("CUDA0: NVIDIA RTX", "cuda"),
    ("HIP0: AMD Radeon", "rocm"),
    ("Metal: Apple M3", "metal"),
    ("Available devices:\n  MTL0: Apple M4 (18186 MiB, 18185 MiB free)", "metal"),
    ("SYCL0: Intel Arc", "xpu"),
    ("Vulkan0: AMD Radeon", "vulkan"),
    ("Available devices:\n", "cpu"),
])
def test_backend_from_device_listing(listing, expected):
    assert LlamaCppEngine._backend_from_device_listing(listing) == expected


def test_runtime_backend_uses_binary_device_listing_and_cpu_override(monkeypatch):
    completed = type("Completed", (), {
        "stdout": "Available devices:\n  Vulkan0: AMD Radeon",
        "stderr": "",
        "returncode": 0,
    })()
    monkeypatch.setattr(LlamaCppEngine, "_binary_path", staticmethod(lambda: "llama-server"))
    monkeypatch.setattr(llamacpp_module.subprocess, "run", lambda *args, **kwargs: completed)
    engine = LlamaCppEngine()
    assert engine.runtime_backend("rocm") == "vulkan"
    assert engine.runtime_backend("rocm", cpu_only=True) == "cpu"


def test_ensure_model_fast_path_does_not_probe_health(monkeypatch):
    engine = LlamaCppEngine()
    engine._loaded_tag = "tag"
    engine._loaded_num_ctx = 2048
    engine._loaded_embedding = False
    engine._loaded_n_parallel = 4
    engine._proc = type("Proc", (), {"poll": lambda self: None})()
    monkeypatch.setattr(engine, "available", lambda: pytest.fail("health probe should not run"))
    engine._ensure_model("tag", 2048, n_parallel=4)


def test_ensure_model_deadline_stops_in_progress_process(monkeypatch, tmp_path):
    class Proc:
        returncode = None

        def __init__(self):
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

        def kill(self):
            self.terminated = True

    proc = Proc()
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"x")
    monkeypatch.setattr(LlamaCppEngine, "_resolve_model_files", classmethod(lambda cls, tag: [model_path]))
    monkeypatch.setattr(LlamaCppEngine, "_binary_path", staticmethod(lambda: "llama-server"))
    monkeypatch.setattr(llamacpp_module.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(llamacpp_module.Shared, "_managed_procs", [])
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", _clock(0.0, 0.0, 0.0, 2.0))
    engine = LlamaCppEngine()
    monkeypatch.setattr(engine, "available", lambda: False)

    with pytest.raises(EngineTimeout):
        engine._ensure_model("tag", 2048, deadline=1.0)

    assert proc.terminated is True
    assert engine._proc is None


@pytest.mark.parametrize(("n_parallel", "num_ctx", "expected_ctx_arg"), [
    (1, 2048, "2048"),
    (4, 2048, "8192"),
])
def test_ensure_model_always_pins_parallel_flag(monkeypatch, tmp_path, n_parallel, num_ctx, expected_ctx_arg):
    """--parallel must be passed even at 1 — omitting it lets llama-server
    fall back to its own auto-slot resolution instead of the single slot
    this engine records via _loaded_n_parallel."""
    captured_args = {}

    class Proc:
        returncode = None

        def poll(self):
            return None

    def fake_popen(args, **kwargs):
        captured_args["args"] = args
        return Proc()

    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"x")
    monkeypatch.setattr(LlamaCppEngine, "_resolve_model_files", classmethod(lambda cls, tag: [model_path]))
    monkeypatch.setattr(LlamaCppEngine, "_binary_path", staticmethod(lambda: "llama-server"))
    monkeypatch.setattr(llamacpp_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(llamacpp_module.Shared, "_managed_procs", [])
    engine = LlamaCppEngine()
    monkeypatch.setattr(engine, "available", lambda: True)

    engine._ensure_model("tag", num_ctx, n_parallel=n_parallel)

    args = captured_args["args"]
    assert "--parallel" in args
    assert args[args.index("--parallel") + 1] == str(n_parallel)
    assert args[args.index("-c") + 1] == expected_ctx_arg
    assert engine._loaded_n_parallel == n_parallel


def test_reachable_or_abort_always_true_regardless_of_available(monkeypatch):
    # There's no always-on daemon to check between models — llama-server
    # spawns fresh per model, so reachable_or_abort() never blocks a loop
    # over models on it (see its docstring).
    monkeypatch.setattr(LlamaCppEngine, "available", lambda self: False)
    assert LlamaCppEngine().reachable_or_abort() is True
    monkeypatch.setattr(LlamaCppEngine, "available", lambda self: True)
    assert LlamaCppEngine().reachable_or_abort() is True


def test_wait_for_recovery_always_true_regardless_of_available(monkeypatch):
    # No passive self-heal to poll for (see wait_for_recovery's docstring) —
    # recovery happens synchronously on the next generate/chat/embed call.
    monkeypatch.setattr(LlamaCppEngine, "available", lambda self: False)
    assert LlamaCppEngine().wait_for_recovery() is True


def test_unload_stops_process_when_tag_matches(monkeypatch):
    engine = LlamaCppEngine()
    engine._loaded_tag = "phi4-mini"
    stopped = []
    monkeypatch.setattr(LlamaCppEngine, "_stop_process", lambda self: stopped.append(True))
    engine.unload("phi4-mini")
    assert stopped == [True]


def test_unload_noop_when_tag_does_not_match(monkeypatch):
    engine = LlamaCppEngine()
    engine._loaded_tag = "other-model"
    stopped = []
    monkeypatch.setattr(LlamaCppEngine, "_stop_process", lambda self: stopped.append(True))
    engine.unload("phi4-mini")
    assert stopped == []


def test_unload_all_unloads_the_loaded_model(monkeypatch):
    engine = LlamaCppEngine()
    engine._loaded_tag = "phi4-mini"
    unloaded = []
    monkeypatch.setattr(LlamaCppEngine, "unload", lambda self, tag: unloaded.append(tag))
    engine.unload_all()
    assert unloaded == ["phi4-mini"]


def test_unload_all_noop_when_nothing_loaded(monkeypatch):
    engine = LlamaCppEngine()
    unloaded = []
    monkeypatch.setattr(LlamaCppEngine, "unload", lambda self, tag: unloaded.append(tag))
    engine.unload_all()
    assert unloaded == []


def test_wait_until_unloaded_true_once_tag_no_longer_loaded():
    engine = LlamaCppEngine()
    engine._loaded_tag = None
    assert engine.wait_until_unloaded("phi4-mini") is True


def test_wait_until_unloaded_false_while_still_loaded():
    engine = LlamaCppEngine()
    engine._loaded_tag = "phi4-mini"
    assert engine.wait_until_unloaded("phi4-mini") is False
