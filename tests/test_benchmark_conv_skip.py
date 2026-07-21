import config
from benchmark import conv_skip_entry
from shared import Shared

MODEL = {"label": "Llama 3.2 3B Q4_K_M", "short": "llama3.2-3b-q4"}
FIRST_CTX = Shared.context_label(config.CONTEXT_LENGTHS[0])


def test_no_llm_data_skips():
    entry = conv_skip_entry(MODEL, None, FIRST_CTX, force_all=False)
    assert entry["skip_reason"] == "no_llm_data"
    assert entry["skipped"] is True
    assert entry["label"] == MODEL["label"]


def test_empty_llm_data_dict_skips_as_no_data():
    entry = conv_skip_entry(MODEL, {}, FIRST_CTX, force_all=False)
    assert entry["skip_reason"] == "no_llm_data"


def test_skipped_llm_data_propagates_reason_and_detail():
    llm_data = {"skipped": True, "skip_reason": "known_crash", "skip_detail": "custom detail"}
    entry = conv_skip_entry(MODEL, llm_data, FIRST_CTX, force_all=False)
    assert entry["skip_reason"] == "known_crash"
    assert entry["skip_detail"] == "custom detail"


def test_crashed_llm_data_without_skip_detail_builds_generic_message():
    llm_data = {"crashed": "8K"}
    entry = conv_skip_entry(MODEL, llm_data, FIRST_CTX, force_all=False)
    assert entry["skip_reason"] == "known_crash"
    assert "8K" in entry["skip_detail"]


def test_timeout_at_first_context_skips():
    llm_data = {"timed_out": FIRST_CTX}
    entry = conv_skip_entry(MODEL, llm_data, FIRST_CTX, force_all=False)
    assert entry["skip_reason"] == "timed_out"


def test_timeout_at_deeper_context_does_not_disqualify():
    # Timed out at 8K, not the configured first context — it passed prefill there,
    # so it should fall through to the tok/s check rather than being skipped
    # outright.
    llm_data = {"timed_out": "8K", FIRST_CTX: {"tps_mean": 50.0}}
    entry = conv_skip_entry(MODEL, llm_data, FIRST_CTX, force_all=False)
    assert entry is None


def test_slow_tps_flag_skips():
    llm_data = {"slow_tps": FIRST_CTX}
    entry = conv_skip_entry(MODEL, llm_data, FIRST_CTX, force_all=False)
    assert entry["skip_reason"] == "slow_tps"
    assert FIRST_CTX in entry["skip_detail"]


def test_slow_tps_derived_from_first_context_tps_mean():
    llm_data = {FIRST_CTX: {"tps_mean": config.SLOW_MODEL_MIN_TPS - 1.0}}
    entry = conv_skip_entry(MODEL, llm_data, FIRST_CTX, force_all=False)
    assert entry["skip_reason"] == "slow_tps"
    assert f"{config.SLOW_MODEL_MIN_TPS - 1.0:.1f} tok/s" in entry["skip_detail"]


def test_fast_enough_model_is_not_skipped():
    llm_data = {FIRST_CTX: {"tps_mean": config.SLOW_MODEL_MIN_TPS + 10.0}}
    entry = conv_skip_entry(MODEL, llm_data, FIRST_CTX, force_all=False)
    assert entry is None


def test_force_all_ignores_explicit_slow_tps_flag():
    llm_data = {"slow_tps": FIRST_CTX}
    entry = conv_skip_entry(MODEL, llm_data, FIRST_CTX, force_all=True)
    assert entry is None


def test_force_all_ignores_derived_slow_tps():
    llm_data = {FIRST_CTX: {"tps_mean": config.SLOW_MODEL_MIN_TPS - 1.0}}
    entry = conv_skip_entry(MODEL, llm_data, FIRST_CTX, force_all=True)
    assert entry is None


def test_force_all_does_not_override_crash_or_timeout():
    # force_all only overrides the slow-tps cutoff, not real failures.
    assert conv_skip_entry(MODEL, {"crashed": "2K"}, FIRST_CTX, force_all=True) is not None
    assert conv_skip_entry(MODEL, {"timed_out": FIRST_CTX}, FIRST_CTX, force_all=True) is not None
    assert conv_skip_entry(MODEL, None, FIRST_CTX, force_all=True) is not None
