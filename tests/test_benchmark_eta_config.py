from types import SimpleNamespace

from scripts.app.benchmark import eta_match_config
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
