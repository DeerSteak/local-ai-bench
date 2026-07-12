"""Tests for the NDJSON stream-parsing logic in Shared.ollama_generate /
Shared.ollama_chat, and the HTTP-error reformatting in Shared._ollama_urlopen.
These are mocked at the Shared._ollama_urlopen seam (or urllib.request.urlopen
directly for the error-handling test) rather than hitting a real server."""

import json
import urllib.error

import pytest

import shared as shared_module
from shared import Shared


class _FakeResponse:
    """Mimics the object returned by urllib.request.urlopen: a context
    manager that iterates raw NDJSON lines as bytes."""

    def __init__(self, chunks):
        self._lines = [json.dumps(c).encode() + b"\n" for c in chunks]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._lines)


def _patch_urlopen(monkeypatch, chunks):
    monkeypatch.setattr(Shared, "_ollama_urlopen", staticmethod(lambda req, timeout: _FakeResponse(chunks)))


# ── ollama_generate ──

def test_ollama_generate_uses_server_reported_timings(monkeypatch):
    _patch_urlopen(monkeypatch, [
        {"response": "Hel"},
        {"response": "lo"},
        {"done": True, "eval_count": 10, "eval_duration": 2_000_000_000,
         "prompt_eval_duration": 500_000_000},
    ])
    ttft, tokens, tps = Shared.ollama_generate("some-tag", "prompt", num_ctx=2048)
    assert ttft == pytest.approx(0.5)
    assert tokens == 10
    assert tps == pytest.approx(5.0)  # 10 tokens / 2 sec


def test_ollama_generate_falls_back_to_wall_clock_ttft_when_server_omits_it(monkeypatch):
    fake_time = iter([100.0, 101.5, 102.0])  # t_start, ttft calc, total calc
    monkeypatch.setattr(shared_module.time, "perf_counter", lambda: next(fake_time))
    _patch_urlopen(monkeypatch, [
        {"response": "Hi"},
        {"done": True, "eval_count": 1, "eval_duration": 0, "prompt_eval_duration": 0},
    ])
    ttft, tokens, tps = Shared.ollama_generate("some-tag", "prompt")
    assert ttft == pytest.approx(1.5)
    assert tps == 0  # no eval_duration reported, tps stays at its initial 0


def test_ollama_generate_skips_blank_and_unparsable_lines(monkeypatch):
    class _MixedResponse(_FakeResponse):
        def __init__(self):
            super().__init__([{"response": "ok"}, {"done": True, "eval_count": 1, "eval_duration": 1_000_000_000}])
            self._lines = [b"", b"   ", b"not json at all"] + self._lines

    monkeypatch.setattr(Shared, "_ollama_urlopen", staticmethod(lambda req, timeout: _MixedResponse()))
    ttft, tokens, tps = Shared.ollama_generate("some-tag", "prompt")
    assert tokens == 1
    assert tps == pytest.approx(1.0)


# ── ollama_chat ──

def test_ollama_chat_returns_content_and_server_timings(monkeypatch):
    _patch_urlopen(monkeypatch, [
        {"message": {"content": "Hel"}},
        {"message": {"content": "lo"}},
        {"done": True, "eval_count": 4, "eval_duration": 1_000_000_000,
         "prompt_eval_duration": 200_000_000, "prompt_eval_count": 50},
    ])
    ttft, tokens, tps, prompt_eval_count, text = Shared.ollama_chat("tag", [{"role": "user", "content": "hi"}])
    assert text == "Hello"
    assert prompt_eval_count == 50
    assert ttft == pytest.approx(0.2)
    assert tps == pytest.approx(4.0)


def test_ollama_chat_falls_back_to_thinking_text_when_content_empty(monkeypatch):
    """Reasoning models can stream their whole turn through message.thinking
    with content left empty — the reply text must fall back to that so the
    next turn's history isn't an empty assistant message."""
    _patch_urlopen(monkeypatch, [
        {"message": {"thinking": "Let me consider... "}},
        {"message": {"thinking": "the answer is 42."}},
        {"done": True, "eval_count": 8, "eval_duration": 1_000_000_000, "prompt_eval_count": 10},
    ])
    ttft, tokens, tps, prompt_eval_count, text = Shared.ollama_chat("tag", [{"role": "user", "content": "hi"}])
    assert text == "Let me consider... the answer is 42."


def test_ollama_chat_prefers_content_over_thinking_when_both_present(monkeypatch):
    _patch_urlopen(monkeypatch, [
        {"message": {"content": "answer", "thinking": "reasoning"}},
        {"done": True, "eval_count": 1, "eval_duration": 1_000_000_000, "prompt_eval_count": 10},
    ])
    _, _, _, _, text = Shared.ollama_chat("tag", [{"role": "user", "content": "hi"}])
    assert text == "answer"


# ── _ollama_urlopen error handling ──

def _http_error(code, body: bytes):
    err = urllib.error.HTTPError(url="http://x", code=code, msg="err", hdrs=None, fp=None)
    monkeypatch_read = lambda: body
    err.read = monkeypatch_read
    return err


def test_ollama_urlopen_surfaces_json_error_body(monkeypatch):
    err = _http_error(500, json.dumps({"error": "model requires more system memory"}).encode())

    def fake_urlopen(req, timeout):
        raise err

    monkeypatch.setattr(shared_module.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="model requires more system memory"):
        Shared._ollama_urlopen(req=None, timeout=5)


def test_ollama_urlopen_falls_back_to_raw_body_when_not_json(monkeypatch):
    err = _http_error(500, b"Internal Server Error (not json)")

    def fake_urlopen(req, timeout):
        raise err

    monkeypatch.setattr(shared_module.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="Internal Server Error"):
        Shared._ollama_urlopen(req=None, timeout=5)


def test_ollama_urlopen_passes_through_on_success(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(shared_module.urllib.request, "urlopen", lambda req, timeout: sentinel)
    assert Shared._ollama_urlopen(req=None, timeout=5) is sentinel
