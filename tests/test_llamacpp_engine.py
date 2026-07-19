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


def _patch_ensure_model(monkeypatch):
    monkeypatch.setattr(LlamaCppEngine, "_ensure_model", lambda self, *a, **kw: None)


# ── generate ──

def test_generate_uses_server_reported_timings(monkeypatch):
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"content": "Hel"},
        {"content": "lo"},
        {"content": "", "stop": True,
         "timings": {"predicted_n": 10, "predicted_ms": 2000, "prompt_ms": 500}},
    ])
    ttft, tokens, tps = LlamaCppEngine().generate("some-tag", "prompt", num_ctx=2048)
    assert ttft == pytest.approx(0.5)
    assert tokens == 10
    assert tps == pytest.approx(5.0)  # 10 tokens / 2 sec


def test_generate_falls_back_to_wall_clock_ttft_when_server_omits_it(monkeypatch):
    _patch_ensure_model(monkeypatch)
    fake_time = iter([100.0, 101.5, 102.0])
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", lambda: next(fake_time))
    _patch_urlopen(monkeypatch, [
        {"content": "Hi"},
        {"content": "", "stop": True, "timings": {"predicted_n": 1, "predicted_ms": 0}},
    ])
    ttft, tokens, tps = LlamaCppEngine().generate("some-tag", "prompt")
    assert ttft == pytest.approx(1.5)
    assert tps == 0


# ── chat ──

def test_chat_returns_content_and_server_timings(monkeypatch):
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "timings": {"predicted_n": 4, "predicted_ms": 1000, "prompt_ms": 200, "prompt_n": 50}},
    ])
    ttft, tokens, tps, prompt_eval_count, text = LlamaCppEngine().chat(
        "tag", [{"role": "user", "content": "hi"}])
    assert text == "Hello"
    assert prompt_eval_count == 50
    assert ttft == pytest.approx(0.2)
    assert tps == pytest.approx(4.0)


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
    ])
    _, _, _, _, text = LlamaCppEngine().chat("tag", [{"role": "user", "content": "hi"}])
    assert text == "Let me consider... the answer is 42."


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
    counter = iter([0.0, 0.0, 100.0])
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", lambda: next(counter))
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"content": "hi"}}]},
        {"choices": [{"delta": {"content": "still going"}}]},
    ])
    with pytest.raises(EngineTimeout):
        LlamaCppEngine().chat("tag", [{"role": "user", "content": "hi"}], timeout=5)


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
