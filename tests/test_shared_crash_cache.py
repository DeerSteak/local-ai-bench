import urllib.error
import http.client

import pytest
import requests

from scripts.runtime import config
from scripts.runtime.engines.llamacpp import LlamaCppEngine
from scripts.runtime.shared import Shared


def test_load_crash_cache_missing_file_returns_empty(tmp_path):
    assert Shared.load_crash_cache(tmp_path / "does_not_exist.json") == {}


def test_unexpected_model_failure_carries_the_label_and_exception_detail():
    entry = Shared.unexpected_model_failure("Some Model", TypeError("missing 'x'"))
    assert entry["label"] == "Some Model"
    assert entry["unexpected_error"] is True
    assert entry["error"] == "TypeError: missing 'x'"
    assert entry["crashed_at"]


def test_unexpected_model_failure_omits_crashed_by_default():
    """llm/llm_conversation read `crashed` as a context-label string (dashboard/src/utils/
    llm.ts) — leaving it unset (rather than a bool) avoids corrupting any real checkpoint
    data already merged into the same results entry."""
    entry = Shared.unexpected_model_failure("Some Model", RuntimeError("boom"))
    assert "crashed" not in entry


def test_unexpected_model_failure_sets_crashed_bool_when_requested():
    """The accuracy path's `crashed` field is a plain boolean (dashboard/src/utils/
    accuracy.ts) — opt in explicitly rather than defaulting every caller to it."""
    entry = Shared.unexpected_model_failure("Some Model", RuntimeError("boom"), crashed=True)
    assert entry["crashed"] is True


def test_unexpected_model_failure_never_corrupts_prior_checkpoint_data():
    """Merging this entry into a results dict that already has real per-checkpoint data
    must not introduce a boolean `crashed` that would make every checkpoint look skipped."""
    results = {"m": {"2K": {"ttft_mean_sec": 0.5}, "8K": {"ttft_mean_sec": 1.2}}}
    results["m"].update(Shared.unexpected_model_failure("m", RuntimeError("boom")))
    assert results["m"]["2K"] == {"ttft_mean_sec": 0.5}
    assert results["m"]["8K"] == {"ttft_mean_sec": 1.2}
    assert "crashed" not in results["m"]


def test_unexpected_model_failure_never_raises_even_on_a_weird_exception():
    """Building the crash entry must not itself raise — that would defeat the whole
    point of the top-level guard this feeds into."""
    class Weird(Exception):
        def __str__(self):
            raise RuntimeError("boom")
    entry = Shared.unexpected_model_failure("m", Weird())
    assert "could not be formatted" in entry["error"]


def test_load_crash_cache_invalid_json_returns_empty(tmp_path):
    path = tmp_path / "crash.json"
    path.write_text("not json")
    assert Shared.load_crash_cache(path) == {}


def test_save_and_load_crash_cache_roundtrip(tmp_path):
    path = tmp_path / "crash.json"
    cache = {"llamacpp": {"llama3.2:3b": {"crashed_at": "2026-01-01T00:00:00"}}}
    Shared.save_crash_cache(path, cache)
    assert Shared.load_crash_cache(path) == cache


def test_crash_cache_paths_discovers_every_cache_type_only(tmp_path):
    llm = tmp_path / ".llm_crash_cache.json"
    future = tmp_path / ".future_workload_crash_cache.json"
    unrelated = tmp_path / ".llm_cache.json"
    directory = tmp_path / ".directory_crash_cache.json"
    for path in (llm, future, unrelated):
        path.write_text("{}", encoding="utf-8")
    directory.mkdir()

    assert Shared.crash_cache_paths(tmp_path) == [future, llm]


def test_clear_crash_caches_removes_all_types_and_preserves_unrelated_files(tmp_path):
    caches = [tmp_path / ".llm_crash_cache.json", tmp_path / ".tool_crash_cache.json"]
    for path in caches:
        path.write_text("{}", encoding="utf-8")
    unrelated = tmp_path / ".benchmark_frontend_state.json"
    unrelated.write_text("{}", encoding="utf-8")

    removed, failures = Shared.clear_crash_caches(tmp_path)

    assert removed == caches
    assert failures == {}
    assert unrelated.is_file()


def test_clear_crash_caches_unlinks_symlink_without_touching_target(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / ".llm_crash_cache.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")

    removed, failures = Shared.clear_crash_caches(tmp_path)

    assert removed == [link]
    assert failures == {}
    assert target.is_file()


def test_clear_crash_caches_continues_after_one_file_fails(tmp_path, monkeypatch):
    blocked = tmp_path / ".llm_crash_cache.json"
    removable = tmp_path / ".tool_crash_cache.json"
    for path in (blocked, removable):
        path.write_text("{}", encoding="utf-8")
    unlink = type(blocked).unlink

    def selective_unlink(path):
        if path == blocked:
            raise OSError("locked")
        unlink(path)

    monkeypatch.setattr(type(blocked), "unlink", selective_unlink)

    removed, failures = Shared.clear_crash_caches(tmp_path)

    assert removed == [removable]
    assert failures == {blocked: "locked"}
    assert blocked.is_file() and not removable.exists()


def test_save_crash_cache_swallows_write_failures(tmp_path):
    # Directory as the target path makes write_text() raise — save_crash_cache
    # should warn and not propagate the exception.
    unwritable = tmp_path / "not_a_file"
    unwritable.mkdir()
    Shared.save_crash_cache(unwritable, {"tag": {"crashed_at": "now"}})  # should not raise


def test_check_crash_cache_returns_none_when_not_present(tmp_path):
    path = tmp_path / "crash.json"
    assert Shared.check_crash_cache("some-tag", "Some Model", {}, path, engine_name="llamacpp") is None


def test_check_crash_cache_returns_skip_entry_when_present(tmp_path):
    path = tmp_path / "crash.json"
    cache = {"llamacpp": {"some-tag": {"crashed_at": "2026-01-01T00:00:00"}}}
    entry = Shared.check_crash_cache("some-tag", "Some Model", cache, path, engine_name="llamacpp")
    assert entry is not None
    assert entry["skipped"] is True
    assert entry["skip_reason"] == "known_crash"
    assert entry["label"] == "Some Model"


def test_check_crash_cache_can_be_bypassed_for_current_run(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RETRY_CRASHED_MODELS", True)
    cache = {"llamacpp": {"some-tag": {"crashed_at": "yesterday"}}}
    assert Shared.check_crash_cache(
        "some-tag", "Some Model", cache, tmp_path / "cache", engine_name="llamacpp") is None


def test_record_crash_persists_to_cache(tmp_path):
    path = tmp_path / "crash.json"
    cache = {}
    crashed_at = Shared.record_crash("some-tag", cache, path, "running Some Model", engine_name="llamacpp")
    assert cache["llamacpp"]["some-tag"]["crashed_at"] == crashed_at
    assert Shared.load_crash_cache(path)["llamacpp"]["some-tag"]["crashed_at"] == crashed_at


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
