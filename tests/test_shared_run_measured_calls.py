import requests

from shared import OllamaLoopDetected, OllamaTimeout, Shared
import config


def test_run_measured_calls_all_succeed(tmp_path):
    cache_path = tmp_path / "crash.json"
    calls = []

    def call(run_i):
        calls.append(run_i)
        return run_i * 2

    samples, status, partial_text = Shared.run_measured_calls(3, call, "tag", {}, cache_path, "testing")
    assert samples == [0, 2, 4]
    assert status == "ok"
    assert partial_text == ""
    assert calls == [0, 1, 2]


def test_run_measured_calls_timeout_stops_immediately(tmp_path):
    cache_path = tmp_path / "crash.json"

    def call(run_i):
        if run_i == 1:
            raise TimeoutError("timed out")
        return run_i

    samples, status, partial_text = Shared.run_measured_calls(3, call, "tag", {}, cache_path, "testing")
    assert samples == [0]
    assert status == "timed_out"
    assert partial_text == ""


def test_run_measured_calls_timeout_captures_partial_text(tmp_path):
    """A timeout that cut off a response already in progress should surface
    that text, not just a bare 'timed_out' status — the caller needs to tell
    a genuine blank apart from a cut-off (possibly wrong-format) answer."""
    cache_path = tmp_path / "crash.json"

    def call(run_i):
        raise OllamaTimeout("timed out", partial_text="The answer is B")

    samples, status, partial_text = Shared.run_measured_calls(3, call, "tag", {}, cache_path, "testing")
    assert samples == []
    assert status == "timed_out"
    assert partial_text == "The answer is B"


def test_run_measured_calls_loop_detected_is_a_distinct_status(tmp_path):
    """OllamaLoopDetected (raised by ollama_chat's check_loop) must surface as
    its own "loop_detected" status, not get folded into "timed_out" — the two
    are independent buckets for the caller (see run_accuracy_benchmark)."""
    cache_path = tmp_path / "crash.json"

    def call(run_i):
        raise OllamaLoopDetected("detected a generation loop after 8s", partial_text="wait, wait, wait,")

    samples, status, partial_text = Shared.run_measured_calls(3, call, "tag", {}, cache_path, "testing")
    assert samples == []
    assert status == "loop_detected"
    assert partial_text == "wait, wait, wait,"


def test_run_measured_calls_ordinary_failure_skips_and_continues(tmp_path):
    cache_path = tmp_path / "crash.json"

    def call(run_i):
        if run_i == 1:
            raise ValueError("some ordinary failure")
        return run_i

    samples, status, _ = Shared.run_measured_calls(3, call, "tag", {}, cache_path, "testing")
    # run_i=1 fails but still counts as attempted (advances), so only 2 samples collected
    assert samples == [0, 2]
    assert status == "ok"


def test_run_measured_calls_crash_retries_then_gives_up(tmp_path, monkeypatch):
    cache_path = tmp_path / "crash.json"
    crash_cache = {}

    # Recovery always succeeds so the retry loop doesn't stall on real sleeps.
    monkeypatch.setattr(Shared, "wait_for_ollama_recovery", lambda timeout=30: True)

    def call(run_i):
        raise requests.exceptions.ConnectionError("actively refused")

    samples, status, _ = Shared.run_measured_calls(3, call, "tag", crash_cache, cache_path, "testing")
    assert samples == []
    assert status == "crashed"
    assert "tag" in crash_cache
    assert Shared.load_crash_cache(cache_path)["tag"]["crashed_at"] == crash_cache["tag"]["crashed_at"]


def test_run_measured_calls_crash_recovers_and_retries_same_run(tmp_path, monkeypatch):
    cache_path = tmp_path / "crash.json"
    monkeypatch.setattr(Shared, "wait_for_ollama_recovery", lambda timeout=30: True)

    attempts = {"n": 0}

    def call(run_i):
        if run_i == 0 and attempts["n"] == 0:
            attempts["n"] += 1
            raise requests.exceptions.ConnectionError("actively refused")
        return run_i

    samples, status, _ = Shared.run_measured_calls(2, call, "tag", {}, cache_path, "testing")
    assert samples == [0, 1]
    assert status == "ok"


def test_run_measured_calls_crash_gives_up_if_recovery_fails(tmp_path, monkeypatch):
    cache_path = tmp_path / "crash.json"
    crash_cache = {}
    monkeypatch.setattr(Shared, "wait_for_ollama_recovery", lambda timeout=30: False)

    def call(run_i):
        raise requests.exceptions.ConnectionError("actively refused")

    samples, status, _ = Shared.run_measured_calls(3, call, "tag", crash_cache, cache_path, "testing")
    assert samples == []
    assert status == "crashed"
    assert "tag" in crash_cache


def test_slow_tps_early_exit_triggers_below_cutoff():
    results = {"short": {}}
    stopped = Shared.slow_tps_early_exit(
        results, "short", "Model", "2K", True, [config.SLOW_MODEL_MIN_TPS - 1.0], False,
    )
    assert stopped is True
    assert results["short"]["slow_tps"] == "2K"


def test_slow_tps_early_exit_force_all_ignores_cutoff():
    results = {"short": {}}
    stopped = Shared.slow_tps_early_exit(
        results, "short", "Model", "2K", True, [config.SLOW_MODEL_MIN_TPS - 1.0], True,
    )
    assert stopped is False
    assert "slow_tps" not in results["short"]


def test_slow_tps_early_exit_not_first_ctx_never_triggers():
    results = {"short": {}}
    stopped = Shared.slow_tps_early_exit(
        results, "short", "Model", "8K", False, [config.SLOW_MODEL_MIN_TPS - 1.0], False,
    )
    assert stopped is False
    assert "slow_tps" not in results["short"]


def test_slow_tps_early_exit_above_cutoff_does_not_trigger():
    results = {"short": {}}
    stopped = Shared.slow_tps_early_exit(
        results, "short", "Model", "2K", True, [config.SLOW_MODEL_MIN_TPS + 10.0], False,
    )
    assert stopped is False
    assert "slow_tps" not in results["short"]
