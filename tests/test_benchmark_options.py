from scripts.runtime import config

from scripts.app.benchmark_options import (
    GUI_OPTION_FLAGS, PUBLIC_OPTION_SCHEMA, TEST_CHOICES, TG_TOKEN_CHOICES,
    TIER_CHOICES, gui_option_defaults, option_value_errors,
)


def test_schema_defines_every_gui_default_and_cli_choice_set():
    assert gui_option_defaults() == {
        "warmup": config.WARMUP_RUNS, "runs": config.N_RUNS, "timeout": 300,
        "acc_timeout": config.ACC_TIMEOUT, "acc_token_budget": config.ACC_TOKEN_BUDGET,
        "cpu_only": False, "gpu_split_mode": "layer", "force_all": False,
        "llamacpp_no_repack": False,
        "retry_crashed_models": False, "offline": False, "memory_telemetry": False,
        "out": "", "comfyui": "",
    }
    assert set(GUI_OPTION_FLAGS.values()) <= set(PUBLIC_OPTION_SCHEMA)
    assert PUBLIC_OPTION_SCHEMA["--tests"].choices == TEST_CHOICES
    assert PUBLIC_OPTION_SCHEMA["--tg-tokens"].choices == TG_TOKEN_CHOICES
    assert PUBLIC_OPTION_SCHEMA["--maxtier"].choices == TIER_CHOICES
    assert PUBLIC_OPTION_SCHEMA["--quick"].default is False
    assert PUBLIC_OPTION_SCHEMA["--quick"].ui_status == "missing"
    assert PUBLIC_OPTION_SCHEMA["--dry-run"].default is False


def test_schema_validates_numeric_types_and_bounds():
    assert option_value_errors({"--warmup": 0, "--runs": 10, "--timeout": None}) == []
    assert option_value_errors({"--warmup": -1}) == ["--warmup must be at least 0."]
    assert option_value_errors({"--runs": 11}) == ["--runs must be at most 10."]
    assert option_value_errors({"--timeout": True}) == ["--timeout must be a whole number."]
    assert option_value_errors({"--gpu-split-mode": "row"}) == [
        "--gpu-split-mode must be one of: single, layer, tensor.",
    ]


def test_schema_has_complete_frontend_policy_for_each_option():
    assert all(spec.value_type for spec in PUBLIC_OPTION_SCHEMA.values())
    assert all(spec.classification in {
        "guided", "advanced", "contextual", "developer-only", "unsafe", "unsupported",
    } for spec in PUBLIC_OPTION_SCHEMA.values())
    assert all(spec.ui_status in {"exposed", "equivalent", "excluded", "missing"}
               for spec in PUBLIC_OPTION_SCHEMA.values())
