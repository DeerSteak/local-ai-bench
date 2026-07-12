import requests

import config
from shared import Shared


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


# ── ollama_reachable_or_abort ──

def test_ollama_reachable_or_abort_true_when_available(monkeypatch):
    monkeypatch.setattr(Shared, "ollama_available", staticmethod(lambda: True))
    assert Shared.ollama_reachable_or_abort() is True


def test_ollama_reachable_or_abort_false_when_unavailable(monkeypatch):
    monkeypatch.setattr(Shared, "ollama_available", staticmethod(lambda: False))
    assert Shared.ollama_reachable_or_abort() is False


# ── model_pulled ──

def test_model_pulled_exact_name_match(monkeypatch):
    monkeypatch.setattr(
        "shared.requests.get",
        lambda url, timeout=None: _JsonResp({"models": [{"name": "llama3.2:3b-instruct-q4_K_M"}]}),
    )
    assert Shared.model_pulled("llama3.2:3b-instruct-q4_K_M") is True


def test_model_pulled_partial_match(monkeypatch):
    # Ollama sometimes reports tags with an implicit ":latest" suffix.
    monkeypatch.setattr(
        "shared.requests.get",
        lambda url, timeout=None: _JsonResp({"models": [{"name": "phi4-mini:latest"}]}),
    )
    assert Shared.model_pulled("phi4-mini") is True


def test_model_pulled_no_match(monkeypatch):
    monkeypatch.setattr(
        "shared.requests.get",
        lambda url, timeout=None: _JsonResp({"models": [{"name": "other-model"}]}),
    )
    assert Shared.model_pulled("phi4-mini") is False


def test_model_pulled_request_failure_returns_false(monkeypatch):
    def fake_get(url, timeout=None):
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr("shared.requests.get", fake_get)
    assert Shared.model_pulled("phi4-mini") is False


# ── ollama_model_max_ctx ──

def test_ollama_model_max_ctx_finds_architecture_prefixed_key(monkeypatch):
    monkeypatch.setattr(
        "shared.requests.post",
        lambda url, json=None, timeout=None: _JsonResp(
            {"model_info": {"general.architecture": "qwen35", "qwen35.context_length": 40960}}
        ),
    )
    assert Shared.ollama_model_max_ctx("qwen3.5:4b") == 40960


def test_ollama_model_max_ctx_ignores_non_matching_keys(monkeypatch):
    monkeypatch.setattr(
        "shared.requests.post",
        lambda url, json=None, timeout=None: _JsonResp(
            {"model_info": {"qwen35.context_length_something_else": 999, "gptoss.context_length": 131072}}
        ),
    )
    assert Shared.ollama_model_max_ctx("gpt-oss:20b") == 131072


def test_ollama_model_max_ctx_falls_back_to_default_when_missing(monkeypatch):
    monkeypatch.setattr(
        "shared.requests.post",
        lambda url, json=None, timeout=None: _JsonResp({"model_info": {}}),
    )
    assert Shared.ollama_model_max_ctx("some-tag", default=8192) == 8192


def test_ollama_model_max_ctx_falls_back_to_default_on_request_failure(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        raise requests.exceptions.Timeout("slow")

    monkeypatch.setattr("shared.requests.post", fake_post)
    assert Shared.ollama_model_max_ctx("some-tag", default=4096) == 4096


def test_ollama_model_max_ctx_falls_back_to_default_on_http_error(monkeypatch):
    monkeypatch.setattr(
        "shared.requests.post",
        lambda url, json=None, timeout=None: _JsonResp({}, ok=False, status_code=404),
    )
    assert Shared.ollama_model_max_ctx("some-tag", default=2048) == 2048


# ── unload_model / unload_all_models ──

def test_unload_model_posts_keep_alive_zero(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append((url, json))
        return _JsonResp({})

    monkeypatch.setattr("shared.requests.post", fake_post)
    Shared.unload_model("phi4-mini")

    assert len(calls) == 1
    url, payload = calls[0]
    assert url == f"{config.OLLAMA_URL}/api/generate"
    assert payload == {"model": "phi4-mini", "keep_alive": 0}


def test_unload_model_swallows_request_errors(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr("shared.requests.post", fake_post)
    Shared.unload_model("phi4-mini")  # should not raise


def test_unload_all_models_unloads_every_loaded_model(monkeypatch):
    monkeypatch.setattr(
        "shared.requests.get",
        lambda url, timeout=None: _JsonResp({"models": [{"name": "a"}, {"name": "b"}]}),
    )
    unloaded = []
    monkeypatch.setattr(Shared, "unload_model", staticmethod(lambda tag: unloaded.append(tag)))

    Shared.unload_all_models()

    assert unloaded == ["a", "b"]


def test_unload_all_models_noop_when_nothing_loaded(monkeypatch):
    monkeypatch.setattr("shared.requests.get", lambda url, timeout=None: _JsonResp({"models": []}))
    unloaded = []
    monkeypatch.setattr(Shared, "unload_model", staticmethod(lambda tag: unloaded.append(tag)))

    Shared.unload_all_models()

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
    monkeypatch.setattr("shared.requests.get", fake_get)
    monkeypatch.setattr("shared.time.sleep", lambda s: sleeps.append(s))

    assert Shared.wait_until_unloaded("phi4-mini", timeout=30) is True
    assert sleeps == [1]


def test_wait_until_unloaded_times_out_if_model_never_disappears(monkeypatch):
    monkeypatch.setattr(
        "shared.requests.get",
        lambda url, timeout=None: _JsonResp({"models": [{"name": "phi4-mini"}]}),
    )
    # Simulate the timeout deadline passing after a couple of polls.
    fake_clock = iter([0, 1, 2, 31])
    monkeypatch.setattr("shared.time.time", lambda: next(fake_clock))
    monkeypatch.setattr("shared.time.sleep", lambda s: None)

    assert Shared.wait_until_unloaded("phi4-mini", timeout=30) is False
