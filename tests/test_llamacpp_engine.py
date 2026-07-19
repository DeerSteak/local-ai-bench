"""Tests for LlamaCppEngine — the llama.cpp (llama-server) implementation of
InferenceEngine that reuses models already pulled via `ollama pull`.

Three groups:
  1. Ollama blob-store resolution (_split_tag, _resolve_blob_path,
     model_pulled, list_installed_models, max_context_length) — built against
     a fake Ollama store under tmp_path, monkeypatching _ollama_models_dir so
     no real Ollama installation is needed.
  2. Streaming / timing parsing in generate()/chat() — mocked at the
     _urlopen seam, same pattern as test_ollama_engine.py.
  3. Maintenance seams (is_connection_crash, reachable_or_abort, unload,
     unload_all, wait_until_unloaded) — mirror the OllamaEngine cases but
     against LlamaCppEngine's single-process-per-model model.
"""

import hashlib
import json

import gguf
import pytest
import requests

from engines.llamacpp import LlamaCppEngine
import engines.llamacpp as llamacpp_module
from shared import OllamaLoopDetected, OllamaTimeout


# ══════════════════════════════════════════════════════════════════════════
#  Group 1 — Ollama blob-store resolution
# ══════════════════════════════════════════════════════════════════════════


def _build_ollama_store(tmp_path, name: str, version: str, blob_bytes: bytes):
    """Write a minimal but real Ollama manifest + blob under tmp_path, in the
    exact layout `ollama pull` produces, and return that tmp_path."""
    digest = hashlib.sha256(blob_bytes).hexdigest()
    blobs_dir = tmp_path / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)
    (blobs_dir / f"sha256-{digest}").write_bytes(blob_bytes)

    manifest_dir = tmp_path / "manifests" / "registry.ollama.ai" / "library" / name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "layers": [
            {"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{digest}"},
            {"mediaType": "application/vnd.ollama.image.params", "digest": "sha256:deadbeef"},
        ]
    }
    (manifest_dir / version).write_text(json.dumps(manifest))
    return tmp_path


@pytest.fixture
def fake_store(tmp_path, monkeypatch):
    store = _build_ollama_store(tmp_path, "phi4-mini", "latest", b"fake-gguf-bytes")
    monkeypatch.setattr(LlamaCppEngine, "_ollama_models_dir", staticmethod(lambda: store))
    return store


def test_split_tag_with_explicit_version():
    assert LlamaCppEngine._split_tag("llama3.2:3b-instruct-q4_K_M") == ("llama3.2", "3b-instruct-q4_K_M")


def test_split_tag_implies_latest():
    assert LlamaCppEngine._split_tag("phi4-mini") == ("phi4-mini", "latest")


def test_resolve_blob_path_finds_model_layer(fake_store):
    blob = LlamaCppEngine._resolve_blob_path("phi4-mini")
    assert blob is not None
    assert blob.read_bytes() == b"fake-gguf-bytes"


def test_resolve_blob_path_missing_manifest_returns_none(fake_store):
    assert LlamaCppEngine._resolve_blob_path("not-pulled:latest") is None


def test_resolve_blob_path_missing_blob_file_returns_none(tmp_path, monkeypatch):
    # Manifest exists but references a digest with no matching blob on disk —
    # a partial/corrupt pull.
    manifest_dir = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "broken"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "latest").write_text(json.dumps({
        "layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:" + "0" * 64}]
    }))
    monkeypatch.setattr(LlamaCppEngine, "_ollama_models_dir", staticmethod(lambda: tmp_path))
    assert LlamaCppEngine._resolve_blob_path("broken:latest") is None


def test_model_pulled_true_when_resolvable(fake_store):
    assert LlamaCppEngine().model_pulled("phi4-mini") is True


def test_model_pulled_false_when_not_resolvable(fake_store):
    assert LlamaCppEngine().model_pulled("nope:latest") is False


def test_list_installed_models_lists_every_resolvable_tag(tmp_path, monkeypatch):
    store = _build_ollama_store(tmp_path, "phi4-mini", "latest", b"aaa")
    _build_ollama_store(store, "llama3.2", "3b-instruct-q4_K_M", b"bbbbb")
    monkeypatch.setattr(LlamaCppEngine, "_ollama_models_dir", staticmethod(lambda: store))

    installed = {m["tag"]: m["size"] for m in LlamaCppEngine().list_installed_models()}
    assert installed == {"phi4-mini:latest": 3, "llama3.2:3b-instruct-q4_K_M": 5}


def test_list_installed_models_empty_when_store_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(LlamaCppEngine, "_ollama_models_dir", staticmethod(lambda: tmp_path / "does-not-exist"))
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
    monkeypatch.setattr(LlamaCppEngine, "_resolve_blob_path", classmethod(lambda cls, tag: gguf_path))
    assert LlamaCppEngine().max_context_length("qwen3.5:4b") == 40960


def test_max_context_length_falls_back_to_default_when_not_pulled(fake_store):
    assert LlamaCppEngine().max_context_length("not-pulled:latest", default=8192) == 8192


def test_max_context_length_falls_back_to_default_on_unparseable_file(tmp_path, monkeypatch):
    bad_path = tmp_path / "not-really-gguf.bin"
    bad_path.write_bytes(b"not a gguf file")
    monkeypatch.setattr(LlamaCppEngine, "_resolve_blob_path", classmethod(lambda cls, tag: bad_path))
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
    with pytest.raises(OllamaLoopDetected) as exc_info:
        LlamaCppEngine().chat("tag", [{"role": "user", "content": "hi"}], check_loop=True)
    assert "loop" in str(exc_info.value).lower()
    assert "wait, wait, wait, wait, wait, still stuck" == exc_info.value.partial_text
    assert "never be reached" not in exc_info.value.partial_text


def test_chat_raises_ollama_timeout_type_for_run_measured_calls_compat(monkeypatch):
    # run_measured_calls (shared.py) does isinstance(e, OllamaLoopDetected) /
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
    with pytest.raises(OllamaTimeout):
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
    # Unlike OllamaEngine, there's no always-on daemon to check between
    # models — llama-server spawns fresh per model, so reachable_or_abort()
    # never blocks a loop over models on it (see its docstring).
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
    engine._loaded_tag = "phi4-mini:latest"
    stopped = []
    monkeypatch.setattr(LlamaCppEngine, "_stop_process", lambda self: stopped.append(True))
    engine.unload("phi4-mini")  # implicit :latest should still match
    assert stopped == [True]


def test_unload_noop_when_tag_does_not_match(monkeypatch):
    engine = LlamaCppEngine()
    engine._loaded_tag = "other-model:latest"
    stopped = []
    monkeypatch.setattr(LlamaCppEngine, "_stop_process", lambda self: stopped.append(True))
    engine.unload("phi4-mini")
    assert stopped == []


def test_unload_all_unloads_the_loaded_model(monkeypatch):
    engine = LlamaCppEngine()
    engine._loaded_tag = "phi4-mini:latest"
    unloaded = []
    monkeypatch.setattr(LlamaCppEngine, "unload", lambda self, tag: unloaded.append(tag))
    engine.unload_all()
    assert unloaded == ["phi4-mini:latest"]


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
    engine._loaded_tag = "phi4-mini:latest"
    assert engine.wait_until_unloaded("phi4-mini") is False
