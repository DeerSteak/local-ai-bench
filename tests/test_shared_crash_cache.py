import urllib.error
import http.client

import pytest
import requests

from engines.llamacpp import LlamaCppEngine
from shared import Shared


def test_load_crash_cache_missing_file_returns_empty(tmp_path):
    assert Shared.load_crash_cache(tmp_path / "does_not_exist.json") == {}


def test_load_crash_cache_invalid_json_returns_empty(tmp_path):
    path = tmp_path / "crash.json"
    path.write_text("not json")
    assert Shared.load_crash_cache(path) == {}


def test_save_and_load_crash_cache_roundtrip(tmp_path):
    path = tmp_path / "crash.json"
    cache = {"llama3.2:3b": {"crashed_at": "2026-01-01T00:00:00"}}
    Shared.save_crash_cache(path, cache)
    assert Shared.load_crash_cache(path) == cache


def test_save_crash_cache_swallows_write_failures(tmp_path):
    # Directory as the target path makes write_text() raise — save_crash_cache
    # should warn and not propagate the exception.
    unwritable = tmp_path / "not_a_file"
    unwritable.mkdir()
    Shared.save_crash_cache(unwritable, {"tag": {"crashed_at": "now"}})  # should not raise


def test_check_crash_cache_returns_none_when_not_present(tmp_path):
    path = tmp_path / "crash.json"
    assert Shared.check_crash_cache("some-tag", "Some Model", {}, path) is None


def test_check_crash_cache_returns_skip_entry_when_present(tmp_path):
    path = tmp_path / "crash.json"
    cache = {"some-tag": {"crashed_at": "2026-01-01T00:00:00"}}
    entry = Shared.check_crash_cache("some-tag", "Some Model", cache, path)
    assert entry["skipped"] is True
    assert entry["skip_reason"] == "known_crash"
    assert entry["label"] == "Some Model"


def test_record_crash_persists_to_cache(tmp_path):
    path = tmp_path / "crash.json"
    cache = {}
    crashed_at = Shared.record_crash("some-tag", cache, path, "running Some Model")
    assert cache["some-tag"]["crashed_at"] == crashed_at
    assert Shared.load_crash_cache(path)["some-tag"]["crashed_at"] == crashed_at


@pytest.mark.parametrize("exc", [
    requests.exceptions.ConnectionError("boom"),
    urllib.error.URLError("boom"),
    http.client.IncompleteRead(b""),
    ConnectionResetError("boom"),
    ConnectionAbortedError("boom"),
    BrokenPipeError("boom"),
])
def test_is_connection_crash_true_for_connection_errors(exc):
    # is_connection_crash lives on the engine now, not Shared — same cases, retargeted.
    assert LlamaCppEngine().is_connection_crash(exc) is True


def test_is_connection_crash_true_for_actively_refused_message():
    assert LlamaCppEngine().is_connection_crash(RuntimeError("connection actively refused")) is True


@pytest.mark.parametrize("exc", [
    ValueError("bad value"),
    TimeoutError("timed out"),
    RuntimeError("llama-server returned HTTP 500: something else"),
])
def test_is_connection_crash_false_for_unrelated_errors(exc):
    assert LlamaCppEngine().is_connection_crash(exc) is False
