"""Tests for OllamaEngine — the Ollama implementation of InferenceEngine.

Relocated here from test_shared_ollama_streaming.py and
test_shared_ollama_maintenance.py when the Ollama HTTP/process client moved
off Shared into engines/ollama.py: same cases and reasoning, retargeted onto
OllamaEngine instances/methods and the engines.ollama.* seams
(engines.ollama.requests.*, engines.ollama.urllib.request.urlopen, ...)
instead of Shared / "shared.requests.*".

Two groups:
  1. NDJSON stream-parsing in generate()/chat() and HTTP-error reformatting in
     OllamaEngine._ollama_urlopen — mocked at the _ollama_urlopen seam (or
     urllib.request.urlopen directly for the error-handling test).
  2. Maintenance seams (reachable_or_abort, model_pulled, max_context_length,
     unload/unload_all, wait_until_unloaded) — mocked at the HTTP layer.
"""

import json
import urllib.error

import pytest
import requests

import config
from engines.ollama import OllamaEngine
import engines.ollama as ollama_module
from shared import OllamaLoopDetected


# ══════════════════════════════════════════════════════════════════════════
#  Group 1 — streaming / HTTP-error parsing
# ══════════════════════════════════════════════════════════════════════════


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
    monkeypatch.setattr(OllamaEngine, "_ollama_urlopen",
                        staticmethod(lambda req, timeout: _FakeResponse(chunks)))


# ── generate ──

def test_generate_uses_server_reported_timings(monkeypatch):
    _patch_urlopen(monkeypatch, [
        {"response": "Hel"},
        {"response": "lo"},
        {"done": True, "eval_count": 10, "eval_duration": 2_000_000_000,
         "prompt_eval_duration": 500_000_000},
    ])
    ttft, tokens, tps = OllamaEngine().generate("some-tag", "prompt", num_ctx=2048)
    assert ttft == pytest.approx(0.5)
    assert tokens == 10
    assert tps == pytest.approx(5.0)  # 10 tokens / 2 sec


def test_generate_falls_back_to_wall_clock_ttft_when_server_omits_it(monkeypatch):
    fake_time = iter([100.0, 101.5, 102.0])  # t_start, ttft calc, total calc
    monkeypatch.setattr(ollama_module.time, "perf_counter", lambda: next(fake_time))
    _patch_urlopen(monkeypatch, [
        {"response": "Hi"},
        {"done": True, "eval_count": 1, "eval_duration": 0, "prompt_eval_duration": 0},
    ])
    ttft, tokens, tps = OllamaEngine().generate("some-tag", "prompt")
    assert ttft == pytest.approx(1.5)
    assert tps == 0  # no eval_duration reported, tps stays at its initial 0


def test_generate_skips_blank_and_unparsable_lines(monkeypatch):
    class _MixedResponse(_FakeResponse):
        def __init__(self):
            super().__init__([{"response": "ok"}, {"done": True, "eval_count": 1, "eval_duration": 1_000_000_000}])
            self._lines = [b"", b"   ", b"not json at all"] + self._lines

    monkeypatch.setattr(OllamaEngine, "_ollama_urlopen", staticmethod(lambda req, timeout: _MixedResponse()))
    ttft, tokens, tps = OllamaEngine().generate("some-tag", "prompt")
    assert tokens == 1
    assert tps == pytest.approx(1.0)


# ── chat ──

def test_chat_returns_content_and_server_timings(monkeypatch):
    _patch_urlopen(monkeypatch, [
        {"message": {"content": "Hel"}},
        {"message": {"content": "lo"}},
        {"done": True, "eval_count": 4, "eval_duration": 1_000_000_000,
         "prompt_eval_duration": 200_000_000, "prompt_eval_count": 50},
    ])
    ttft, tokens, tps, prompt_eval_count, text = OllamaEngine().chat("tag", [{"role": "user", "content": "hi"}])
    assert text == "Hello"
    assert prompt_eval_count == 50
    assert ttft == pytest.approx(0.2)
    assert tps == pytest.approx(4.0)


def test_chat_falls_back_to_thinking_text_when_content_empty(monkeypatch):
    """Reasoning models can stream their whole turn through message.thinking
    with content left empty — the reply text must fall back to that so the
    next turn's history isn't an empty assistant message."""
    _patch_urlopen(monkeypatch, [
        {"message": {"thinking": "Let me consider... "}},
        {"message": {"thinking": "the answer is 42."}},
        {"done": True, "eval_count": 8, "eval_duration": 1_000_000_000, "prompt_eval_count": 10},
    ])
    ttft, tokens, tps, prompt_eval_count, text = OllamaEngine().chat("tag", [{"role": "user", "content": "hi"}])
    assert text == "Let me consider... the answer is 42."


def test_chat_prefers_content_over_thinking_when_both_present(monkeypatch):
    _patch_urlopen(monkeypatch, [
        {"message": {"content": "answer", "thinking": "reasoning"}},
        {"done": True, "eval_count": 1, "eval_duration": 1_000_000_000, "prompt_eval_count": 10},
    ])
    _, _, _, _, text = OllamaEngine().chat("tag", [{"role": "user", "content": "hi"}])
    assert text == "answer"


def test_chat_check_loop_raises_early_on_repeated_hedging(monkeypatch):
    """With check_loop=True, a stuck response should be cut off as soon as a
    loop is detectable — well before the remaining chunks (and the eventual
    `done`) ever stream in — rather than only after the full timeout."""
    import itertools

    counter = itertools.count(0, 1.0)
    monkeypatch.setattr(ollama_module.time, "perf_counter", lambda: next(counter))
    monkeypatch.setattr(ollama_module.config, "LOOP_CHECK_INTERVAL", 0)
    _patch_urlopen(monkeypatch, [
        {"message": {"content": "wait, "}},
        {"message": {"content": "wait, "}},
        {"message": {"content": "wait, still stuck"}},
        {"message": {"content": "this chunk should never be reached"}},
        {"done": True, "eval_count": 4, "eval_duration": 1_000_000_000, "prompt_eval_count": 10},
    ])
    with pytest.raises(OllamaLoopDetected) as exc_info:
        OllamaEngine().chat("tag", [{"role": "user", "content": "hi"}], check_loop=True)
    assert "loop" in str(exc_info.value).lower()
    assert "wait, wait, wait, still stuck" == exc_info.value.partial_text
    assert "never be reached" not in exc_info.value.partial_text


def test_chat_without_check_loop_ignores_repeated_hedging(monkeypatch):
    """check_loop defaults to False, so callers that don't opt in (e.g. the
    conversation benchmark) see no behavior change — a repetitive-but-still-
    streaming response runs to completion as before."""
    _patch_urlopen(monkeypatch, [
        {"message": {"content": "wait, "}},
        {"message": {"content": "wait, "}},
        {"message": {"content": "wait, still stuck"}},
        {"done": True, "eval_count": 3, "eval_duration": 1_000_000_000, "prompt_eval_count": 10},
    ])
    _, _, _, _, text = OllamaEngine().chat("tag", [{"role": "user", "content": "hi"}])
    assert text == "wait, wait, wait, still stuck"


# ── _ollama_urlopen error handling ──

def _http_error(code, body: bytes):
    err = urllib.error.HTTPError(url="http://x", code=code, msg="err", hdrs=None, fp=None)
    err.read = lambda: body
    return err


def test_urlopen_surfaces_json_error_body(monkeypatch):
    err = _http_error(500, json.dumps({"error": "model requires more system memory"}).encode())

    def fake_urlopen(req, timeout):
        raise err

    monkeypatch.setattr(ollama_module.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="model requires more system memory"):
        OllamaEngine._ollama_urlopen(req=None, timeout=5)


def test_urlopen_falls_back_to_raw_body_when_not_json(monkeypatch):
    err = _http_error(500, b"Internal Server Error (not json)")

    def fake_urlopen(req, timeout):
        raise err

    monkeypatch.setattr(ollama_module.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="Internal Server Error"):
        OllamaEngine._ollama_urlopen(req=None, timeout=5)


def test_urlopen_passes_through_on_success(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(ollama_module.urllib.request, "urlopen", lambda req, timeout: sentinel)
    assert OllamaEngine._ollama_urlopen(req=None, timeout=5) is sentinel


# ══════════════════════════════════════════════════════════════════════════
#  Group 2 — maintenance seams
# ══════════════════════════════════════════════════════════════════════════


class _JsonResp:
    def __init__(self, payload, ok=True, status_code=200):
        self._payload = payload
        self.ok = ok
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


# ── reachable_or_abort ──

def test_reachable_or_abort_true_when_available(monkeypatch):
    monkeypatch.setattr(OllamaEngine, "available", lambda self: True)
    assert OllamaEngine().reachable_or_abort() is True


def test_reachable_or_abort_false_when_unavailable(monkeypatch):
    monkeypatch.setattr(OllamaEngine, "available", lambda self: False)
    assert OllamaEngine().reachable_or_abort() is False


# ── model_pulled ──

def test_model_pulled_exact_name_match(monkeypatch):
    monkeypatch.setattr(
        "engines.ollama.requests.get",
        lambda url, timeout=None: _JsonResp({"models": [{"name": "llama3.2:3b-instruct-q4_K_M"}]}),
    )
    assert OllamaEngine().model_pulled("llama3.2:3b-instruct-q4_K_M") is True


def test_model_pulled_partial_match(monkeypatch):
    # Ollama sometimes reports tags with an implicit ":latest" suffix.
    monkeypatch.setattr(
        "engines.ollama.requests.get",
        lambda url, timeout=None: _JsonResp({"models": [{"name": "phi4-mini:latest"}]}),
    )
    assert OllamaEngine().model_pulled("phi4-mini") is True


def test_model_pulled_no_match(monkeypatch):
    monkeypatch.setattr(
        "engines.ollama.requests.get",
        lambda url, timeout=None: _JsonResp({"models": [{"name": "other-model"}]}),
    )
    assert OllamaEngine().model_pulled("phi4-mini") is False


def test_model_pulled_request_failure_returns_false(monkeypatch):
    def fake_get(url, timeout=None):
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr("engines.ollama.requests.get", fake_get)
    assert OllamaEngine().model_pulled("phi4-mini") is False


# ── max_context_length ──

def test_max_context_length_finds_architecture_prefixed_key(monkeypatch):
    monkeypatch.setattr(
        "engines.ollama.requests.post",
        lambda url, json=None, timeout=None: _JsonResp(
            {"model_info": {"general.architecture": "qwen35", "qwen35.context_length": 40960}}
        ),
    )
    assert OllamaEngine().max_context_length("qwen3.5:4b") == 40960


def test_max_context_length_ignores_non_matching_keys(monkeypatch):
    monkeypatch.setattr(
        "engines.ollama.requests.post",
        lambda url, json=None, timeout=None: _JsonResp(
            {"model_info": {"qwen35.context_length_something_else": 999, "gptoss.context_length": 131072}}
        ),
    )
    assert OllamaEngine().max_context_length("gpt-oss:20b") == 131072


def test_max_context_length_falls_back_to_default_when_missing(monkeypatch):
    monkeypatch.setattr(
        "engines.ollama.requests.post",
        lambda url, json=None, timeout=None: _JsonResp({"model_info": {}}),
    )
    assert OllamaEngine().max_context_length("some-tag", default=8192) == 8192


def test_max_context_length_falls_back_to_default_on_request_failure(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        raise requests.exceptions.Timeout("slow")

    monkeypatch.setattr("engines.ollama.requests.post", fake_post)
    assert OllamaEngine().max_context_length("some-tag", default=4096) == 4096


def test_max_context_length_falls_back_to_default_on_http_error(monkeypatch):
    monkeypatch.setattr(
        "engines.ollama.requests.post",
        lambda url, json=None, timeout=None: _JsonResp({}, ok=False, status_code=404),
    )
    assert OllamaEngine().max_context_length("some-tag", default=2048) == 2048


# ── unload / unload_all ──

def test_unload_posts_keep_alive_zero(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append((url, json))
        return _JsonResp({})

    monkeypatch.setattr("engines.ollama.requests.post", fake_post)
    OllamaEngine().unload("phi4-mini")

    assert len(calls) == 1
    url, payload = calls[0]
    assert url == f"{config.OLLAMA_URL}/api/generate"
    assert payload == {"model": "phi4-mini", "keep_alive": 0}


def test_unload_swallows_request_errors(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr("engines.ollama.requests.post", fake_post)
    OllamaEngine().unload("phi4-mini")  # should not raise


def test_unload_all_unloads_every_loaded_model(monkeypatch):
    monkeypatch.setattr(
        "engines.ollama.requests.get",
        lambda url, timeout=None: _JsonResp({"models": [{"name": "a"}, {"name": "b"}]}),
    )
    unloaded = []
    monkeypatch.setattr(OllamaEngine, "unload", lambda self, tag: unloaded.append(tag))

    OllamaEngine().unload_all()

    assert unloaded == ["a", "b"]


def test_unload_all_noop_when_nothing_loaded(monkeypatch):
    monkeypatch.setattr("engines.ollama.requests.get", lambda url, timeout=None: _JsonResp({"models": []}))
    unloaded = []
    monkeypatch.setattr(OllamaEngine, "unload", lambda self, tag: unloaded.append(tag))

    OllamaEngine().unload_all()

    assert unloaded == []


# ── wait_until_unloaded ──

def test_wait_until_unloaded_returns_true_once_model_disappears(monkeypatch):
    responses = [
        {"models": [{"name": "phi4-mini"}]},
        {"models": []},
    ]

    def fake_get(url, timeout=None):
        return _JsonResp(responses.pop(0))

    sleeps = []
    monkeypatch.setattr("engines.ollama.requests.get", fake_get)
    monkeypatch.setattr("engines.ollama.time.sleep", lambda s: sleeps.append(s))

    assert OllamaEngine().wait_until_unloaded("phi4-mini", timeout=30) is True
    assert sleeps == [1]


def test_wait_until_unloaded_times_out_if_model_never_disappears(monkeypatch):
    monkeypatch.setattr(
        "engines.ollama.requests.get",
        lambda url, timeout=None: _JsonResp({"models": [{"name": "phi4-mini"}]}),
    )
    # Simulate the timeout deadline passing after a couple of polls.
    fake_clock = iter([0, 1, 2, 31])
    monkeypatch.setattr("engines.ollama.time.time", lambda: next(fake_clock))
    monkeypatch.setattr("engines.ollama.time.sleep", lambda s: None)

    assert OllamaEngine().wait_until_unloaded("phi4-mini", timeout=30) is False
