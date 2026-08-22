from types import SimpleNamespace

from scripts.app.benchmark import eta_match_config, runtime_shaping_config
from scripts.results.result_history import ETA_MATCH_KEYS


def test_eta_match_config_covers_the_runtime_shape_registry(monkeypatch):
    from scripts.runtime import config
    monkeypatch.setattr(config, "N_RUNS", 2)
    args = SimpleNamespace(
        warmup=1, cpu_only=True, force_all=False, max_prompt_tokens=2048, sample=5,
    )
    matched = eta_match_config(args)
    assert tuple(matched) == ETA_MATCH_KEYS
    assert matched["runs"] == 2
    assert matched["warmup_runs"] == 1
    assert matched["cpu_only"] is True
    assert matched["sample_size"] == 5
    assert matched["mtp_enabled"] is False
    shaping = runtime_shaping_config(args)
    assert all(shaping[key] == value for key, value in matched.items()
               if key != "mtp_enabled")


def test_mtp_eta_does_not_reuse_a_non_mtp_duration():
    args = SimpleNamespace(
        warmup=1, cpu_only=False, force_all=False, max_prompt_tokens=None, sample=None,
    )
    assert eta_match_config(args, mtp_enabled=False)["mtp_enabled"] is False
    assert eta_match_config(args, mtp_enabled=True)["mtp_enabled"] is True


def test_sustained_eta_adds_only_its_runtime_shaping_settings():
    args = SimpleNamespace(
        warmup=1, cpu_only=False, force_all=False, max_prompt_tokens=None, sample=None,
        tests=["sustained"], sustained_duration=900,
    )
    matched = eta_match_config(args)
    assert matched["sustained_duration_sec"] == 900
    assert matched["sustained_window_sec"] > 0
    assert matched["sustained_context_tokens"] > 0
