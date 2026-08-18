import json
import subprocess
from pathlib import Path
from typing import cast

import pytest

from scripts.runtime import config
from scripts.app.benchmark_frontend import (
    GUI_OPTION_DEFAULTS,
    MenuEntry,
    TEST_DEFINITIONS,
    build_benchmark_command,
    build_frontend_state,
    load_frontend_state,
    validate_gui_options,
)
from scripts.app.benchmark_gui import (
    authorize_macos_power_telemetry,
    BENCHMARK_PRESETS, CUSTOM_PRESET, BenchmarkLaunchError, BenchmarkLaunchReady,
    PsutilLike, apply_hardware_model_defaults,
    build_discovery_report, build_plan_preview, custom_option_defaults, default_control_values,
    effective_gui_options, estimate_remaining_seconds, format_run_outcome,
    gpu_split_mode_labels, gpu_split_mode_value, history_row_height,
    launch_controlled_process, open_path_command, parse_progress_line,
    normalize_gui_option_values, prepare_benchmark_launch, restored_tg_tokens,
    process_completion_state, resolve_engine_selection,
    resolve_engine_names,
    parse_gpu_process_memory, parse_gpu_usage, plan_preview_sections,
    query_gpu_process_memory, query_gpu_usage,
    query_vram_usage, show_vram_usage,
    progress_event_engine, progress_model_identity,
    progress_summary_rows,
    reconcile_imported_model_state,
    resolve_preset,
    preset_control_values, process_resource_usage, preset_after_control_change,
    restored_preset_name, should_finalize_process_exit,
    resource_usage_rows, system_memory_usage,
    windows_host_utc_offset_minutes,
    update_progress_metrics, workload_preflight_errors,
)
from scripts.app.result_actions import (
    completed_result_paths, dashboard_launcher_command, record_result_path,
    result_paths_for_log, run_log_path, selected_result_paths, write_run_logs,
)
from scripts.app.recovery_actions import (
    fork_executor_command, fork_review_report, format_recovery_inspection,
    recovery_executor_command, recovery_progress_entries, retry_executor_command,
)
from scripts.results.run_plan import RunPlan
from scripts.runtime.shared import Shared


def test_effective_gui_options_uses_defaults_without_saved_gui_settings():
    assert effective_gui_options(None) == GUI_OPTION_DEFAULTS
    assert effective_gui_options({"tests": ["llm"]}) == GUI_OPTION_DEFAULTS
    assert effective_gui_options(None) is not GUI_OPTION_DEFAULTS


def test_dual_gpu_modes_have_user_facing_labels_and_round_trip():
    modes = ("single", "layer", "tensor")
    labels = gpu_split_mode_labels(modes)
    assert labels == (
        "Single GPU", "Layer split (recommended)", "Tensor parallel (experimental)",
    )
    assert tuple(gpu_split_mode_value(label) for label in labels) == modes
    assert tuple(gpu_split_mode_value(mode) for mode in modes) == modes
    with pytest.raises(ValueError, match="Unknown GPU mode"):
        gpu_split_mode_value("invalid")


def test_process_exit_waits_for_reader_or_quiet_drain_period():
    assert not should_finalize_process_exit(-2, False, 10.0, 10.0, 10.2, grace=0.25)
    assert should_finalize_process_exit(-2, False, 10.0, 10.0, 10.25, grace=0.25)
    assert should_finalize_process_exit(-2, True, 10.0, 10.2, 10.2, grace=0.25)
    assert not should_finalize_process_exit(None, True, None, 10.0, 20.0, grace=0.25)


def test_process_exit_quiet_period_restarts_after_late_output():
    assert not should_finalize_process_exit(-2, False, 10.0, 10.2, 10.4, grace=0.25)
    assert should_finalize_process_exit(-2, False, 10.0, 10.2, 10.45, grace=0.25)


def test_import_refresh_preserves_valid_selections_selects_import_and_prunes_stale_state():
    rebuilt = [
        MenuEntry("kept", "Kept", "llm", "LLM", False),
        MenuEntry("imported", "Imported", "llm", "LLM", False),
        MenuEntry("new-other", "Other", "llm", "LLM", True),
    ]

    selected, dropped, added, defaults = reconcile_imported_model_state(
        {"kept", "removed"}, {"kept", "removed"}, {"kept": True, "removed": True},
        rebuilt, "imported",
    )

    assert selected == {"kept", "imported"}
    assert dropped == {"removed"}
    assert added == {"imported", "new-other"}
    assert defaults == {"kept": True, "imported": False, "new-other": True}


def test_import_refresh_does_not_select_tag_absent_from_rebuilt_entries():
    rebuilt = [MenuEntry("kept", "Kept", "llm", "LLM", False)]

    selected, dropped, added, defaults = reconcile_imported_model_state(
        {"kept"}, set(), {"kept": False}, rebuilt, "not-installed",
    )

    assert selected == set()
    assert dropped == set() and added == set()
    assert defaults == {"kept": False}


def test_selected_result_paths_supports_multiple_and_enforces_action_limits(tmp_path):
    mapping = {"one": tmp_path / "one.json", "two": tmp_path / "two.json"}
    assert selected_result_paths(("one", "two"), mapping) == [
        (tmp_path / "one.json").resolve(), (tmp_path / "two.json").resolve(),
    ]
    with pytest.raises(ValueError, match="exactly 1"):
        selected_result_paths(("one", "two"), mapping, exact=1)
    with pytest.raises(ValueError, match="no more than 1"):
        selected_result_paths(("one", "two"), mapping, maximum=1)


def test_run_log_sidecar_uses_result_suffix_and_completion_paths(tmp_path):
    first = tmp_path / "results_workstation_20260810_120000_llamacpp.json"
    second = tmp_path / "results_workstation_20260810_120000_vllm.json"
    log = f"noise\nResults saved to: {first}\nResults saved to: {second}\n"

    assert run_log_path(first) == tmp_path / "log_workstation_20260810_120000_llamacpp.txt"
    assert completed_result_paths(log) == [first.resolve(), second.resolve()]
    assert write_run_logs(log, [first, second]) == [run_log_path(first), run_log_path(second)]
    assert run_log_path(first).read_text(encoding="utf-8") == log


def test_completed_result_paths_ignores_partial_save_messages():
    assert completed_result_paths("Partial results saved to /tmp/results_one.json (llm)\n") == []


def test_completed_result_paths_rejects_redacted_home_placeholder():
    log = r"Results saved to: <home>\projects\local-ai-bench\results\results_pc.json"
    assert completed_result_paths(log) == []


def test_trusted_result_paths_override_display_log_and_preserve_multiple(tmp_path):
    trusted = [tmp_path / "results_pc_llamacpp.json", tmp_path / "results_pc_vllm.json"]
    redacted = r"Results saved to: <home>\projects\results_pc_llamacpp.json"
    paths = result_paths_for_log(redacted, trusted)
    assert paths == [path.resolve() for path in trusted]
    written = write_run_logs(redacted, paths)
    assert written == [run_log_path(path) for path in trusted]
    assert all(path.read_text(encoding="utf-8") == redacted for path in written)


def test_structured_result_paths_accumulate_once_for_multi_engine_run(tmp_path):
    first = tmp_path / "results_pc_llamacpp.json"
    second = tmp_path / "results_pc_vllm.json"
    paths = record_result_path([], str(first))
    paths = record_result_path(paths, str(second))
    assert record_result_path(paths, str(first)) == [first.resolve(), second.resolve()]


def test_dashboard_launcher_command_passes_each_result_as_a_separate_argument(tmp_path):
    paths = [tmp_path / "first result.json", tmp_path / "second.json"]
    assert dashboard_launcher_command(paths, "Darwin", tmp_path) == [
        "bash", str((tmp_path / "launch_dashboard.sh").resolve()),
        "--result", str(paths[0].resolve()), "--result", str(paths[1].resolve()),
    ]
    assert dashboard_launcher_command(paths[:1], "Windows", tmp_path)[:3] == [
        "cmd", "/c", str((tmp_path / "launch_dashboard.bat").resolve()),
    ]


def test_gui_options_round_trip_in_frontend_state(tmp_path):
    path = tmp_path / "state.json"
    options = dict(GUI_OPTION_DEFAULTS, runs=5, cpu_only=True, out="results/custom.json")
    state = build_frontend_state(
        "llamacpp", ["llm"], [MenuEntry("model", "Model", "llm", "LLM", True)],
        gui_options=options,
    )
    from scripts.app.benchmark_frontend import save_frontend_state
    assert save_frontend_state(state, path)
    loaded = load_frontend_state(path)
    assert loaded is not None and loaded["gui_options"] == options


def test_validate_gui_options_rejects_bounds_types_and_missing_keys():
    assert validate_gui_options(dict(GUI_OPTION_DEFAULTS, runs=0))
    assert validate_gui_options(dict(GUI_OPTION_DEFAULTS, timeout="300"))
    assert validate_gui_options({})
    assert validate_gui_options(GUI_OPTION_DEFAULTS) == []


def test_normalize_gui_option_values_converts_controls_and_trims_paths():
    raw = {
        **GUI_OPTION_DEFAULTS, "runs": "5", "timeout": "bad",
        "gpu_split_mode": "Tensor parallel (experimental)", "out": " result.json ",
        "comfyui": " /ComfyUI ",
    }
    options = normalize_gui_option_values(raw)
    assert options["runs"] == 5
    assert options["timeout"] == "bad"
    assert options["gpu_split_mode"] == "tensor"
    assert options["out"] == "result.json"
    assert options["comfyui"] == "/ComfyUI"


def test_normalize_gui_option_values_accepts_canonical_gpu_mode():
    options = normalize_gui_option_values(dict(GUI_OPTION_DEFAULTS, gpu_split_mode="layer"))
    assert options["gpu_split_mode"] == "layer"


def test_restored_tg_tokens_distinguishes_empty_selection_from_legacy_default():
    assert restored_tg_tokens({"tg_tokens": []}) == set()
    assert restored_tg_tokens({"tg_tokens": [128, 1024]}) == {128, 1024}
    assert restored_tg_tokens({"tg_tokens": None}) == set(config.LLAMABENCH_TG)
    assert restored_tg_tokens(None) == set(config.LLAMABENCH_TG)


def test_prepare_benchmark_launch_returns_validation_errors_without_launch_data(tmp_path):
    preparation = prepare_benchmark_launch(
        engine="llamacpp", tests=[], entries=[], max_prompt_tokens=None, tg_tokens=[],
        gui_options=dict(GUI_OPTION_DEFAULTS), selected_preset="Custom",
        detected_tools={}, found_comfyui=None, detected_comfyui=tmp_path,
    )
    assert isinstance(preparation, BenchmarkLaunchError)
    assert "Select at least one benchmark test." in preparation.errors


def test_prepare_benchmark_launch_builds_review_state_and_command(tmp_path):
    entries = [MenuEntry("model", "Model", "llm", "LLM", True)]
    options = dict(GUI_OPTION_DEFAULTS, runs=4, out="result.json")
    preparation = prepare_benchmark_launch(
        engine="llamacpp", tests=["llm"], entries=entries,
        max_prompt_tokens=8192, tg_tokens=[1024], gui_options=options,
        selected_preset="Quick run", detected_tools={"llama-server": "/bin/server"},
        found_comfyui=None, detected_comfyui=tmp_path,
    )
    assert isinstance(preparation, BenchmarkLaunchReady)
    assert preparation.preview is not None and "Measured runs: 4" in preparation.preview
    assert preparation.state is not None
    assert preparation.state["selected_preset"] == "Quick run"
    assert preparation.state["tg_tokens"] == [1024]
    assert preparation.command is not None
    assert "--tg-tokens" not in preparation.command
    assert preparation.command[preparation.command.index("--max-prompt-tokens") + 1] == "8192"
    assert preparation.command[preparation.command.index("--out") + 1] == "result.json"


def test_prepare_benchmark_launch_requires_tg_selection_for_llamabench(tmp_path):
    entries = [MenuEntry("model", "Model", "llm", "LLM", True)]
    preparation = prepare_benchmark_launch(
        engine="llamacpp", tests=["llamabench"], entries=entries,
        max_prompt_tokens=None, tg_tokens=[], gui_options=dict(GUI_OPTION_DEFAULTS),
        selected_preset="Custom", detected_tools={"llama-bench": "/bin/bench"},
        found_comfyui=None, detected_comfyui=tmp_path,
    )
    assert isinstance(preparation, BenchmarkLaunchError)
    assert "Select at least one llama-bench generation size." in preparation.errors


def test_prepare_benchmark_launch_requires_a_selected_model(tmp_path):
    entries = [MenuEntry("model", "Model", "llm", "LLM", False)]
    preparation = prepare_benchmark_launch(
        engine="llamacpp", tests=["llm"], entries=entries,
        max_prompt_tokens=None, tg_tokens=[], gui_options=dict(GUI_OPTION_DEFAULTS),
        selected_preset="Custom", detected_tools={"llama-server": "/bin/server"},
        found_comfyui=None, detected_comfyui=tmp_path,
    )
    assert isinstance(preparation, BenchmarkLaunchError)
    assert any("model" in error.lower() for error in preparation.errors)


def test_resolve_engine_names_restores_default_without_model_checks():
    assert resolve_engine_names([], ["llamacpp", "vllm"]) == ["llamacpp"]
    assert resolve_engine_names(["vllm"], ["llamacpp", "vllm"]) == ["vllm"]


def test_resolve_engine_selection_restores_default_and_gates_models():
    models = [
        MenuEntry("shared", "Shared", "llm", "LLM", True),
        MenuEntry("vllm-only", "vLLM", "llm", "LLM", True),
    ]
    owners = {"shared": {"llamacpp", "vllm"}, "vllm-only": {"vllm"}}
    defaulted = resolve_engine_selection([], ["llamacpp", "vllm"], models, owners)
    assert defaulted.engines == ["llamacpp"]
    assert defaulted.value == "llamacpp"
    assert defaulted.model_availability == {"shared": True, "vllm-only": False}

    combined = resolve_engine_selection(
        ["vllm", "llamacpp"], ["llamacpp", "vllm"], models, owners,
    )
    assert combined.engines == ["llamacpp", "vllm"]
    assert combined.model_availability == {"shared": True, "vllm-only": True}
    assert "2 passes" in combined.note


@pytest.mark.parametrize(("kind", "exit_code", "status"), [
    ("recovery", 0, "Recovery completed successfully. Results are ready to review."),
    ("retry", 7, "Selected retry stopped with exit code 7. Preserved evidence remains available."),
    ("fork", 2, "Forked run stopped with exit code 2. Preserved evidence remains available."),
])
def test_process_completion_state_formats_history_outcomes(kind, exit_code, status):
    completion = process_completion_state(kind, exit_code)
    assert completion.status == status
    assert completion.write_run_log is (exit_code == 0)


def test_process_completion_state_uses_standard_benchmark_outcome():
    completion = process_completion_state("benchmark", 0)
    assert completion.status == "Benchmark completed successfully. Results are ready to review."
    assert completion.write_run_log is True


def test_build_command_includes_every_gui_execution_setting():
    options = dict(
        GUI_OPTION_DEFAULTS, warmup=2, runs=4, timeout=600, acc_timeout=90,
        acc_token_budget=2048, cpu_only=True, force_all=True,
        out="chosen.json", comfyui="/chosen/ComfyUI",
    )
    command = build_benchmark_command(
        # An image test is included so --comfyui is emitted at all; it is omitted without one.
        "llamacpp", Path("/detected/ComfyUI"), ["llm", "img"],
        [MenuEntry("model", "Model", "llm", "LLM", True),
         MenuEntry("sdxl", "SDXL", "image", "Images", True)],
        python_executable="python", benchmark_path=Path("benchmark.py"), gui_options=options,
    )
    assert command[command.index("--comfyui") + 1] == "/chosen/ComfyUI"
    for flag, value in (
        ("--warmup", "2"), ("--runs", "4"), ("--timeout", "600"),
        ("--acc-timeout", "90"), ("--acc-token-budget", "2048"),
        ("--out", "chosen.json"),
    ):
        assert command[command.index(flag) + 1] == value
    assert "--cpu-only" in command
    assert "--force-all" in command


def test_open_path_command_uses_each_desktop_platform_launcher():
    path = Path("/tmp/results")
    assert open_path_command(path, "Darwin") == ["open", "/tmp/results"]
    assert open_path_command(path, "Linux") == ["xdg-open", "/tmp/results"]
    assert open_path_command(path, "Windows") == ["explorer", "/tmp/results"]


def test_launch_controlled_process_supplies_progress_environment(monkeypatch, tmp_path):
    control_path = tmp_path / "pause.json"
    control_path.write_text("{}", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        "scripts.app.benchmark_gui.windows_host_utc_offset_minutes", lambda: -300,
    )

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return object()

    process, path = launch_controlled_process(
        ["python", "benchmark.py"], creationflags=7,
        pause_path_factory=lambda: control_path,
        popen=cast("type[subprocess.Popen]", fake_popen),
    )

    assert process is not None and path == control_path
    assert calls[0][1]["env"]["PYTHONUNBUFFERED"] == "1"
    assert calls[0][1]["env"]["PYTHONIOENCODING"] == "utf-8"
    assert calls[0][1]["env"]["NO_COLOR"] == "1"
    assert calls[0][1]["env"]["LOCAL_AI_BENCH_PROGRESS"] == "1"
    assert calls[0][1]["env"]["LOCAL_AI_BENCH_PAUSE_CONTROL"] == str(control_path)
    assert calls[0][1]["env"]["LOCAL_AI_BENCH_RUN_LOG_UTC_OFFSET_MINUTES"] == "-300"
    assert calls[0][1]["creationflags"] == 7
    assert calls[0][1]["encoding"] == "utf-8"
    assert calls[0][1]["errors"] == "replace"


def test_macos_power_authorization_uses_graphical_askpass(tmp_path):
    helper = tmp_path / "askpass"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return type("Result", (), {"returncode": 0})()

    assert authorize_macos_power_telemetry(
        True, system=lambda: "Darwin", run=run, environ={"SAFE": "yes"},
        askpass_path=helper,
    ) is None
    assert calls[0][0] == ["/usr/bin/sudo", "-A", "-v"]
    assert calls[0][1]["env"] == {"SAFE": "yes", "SUDO_ASKPASS": str(helper)}
    assert calls[0][1]["stdin"] is subprocess.DEVNULL
    assert calls[0][1]["timeout"] == 120


def test_power_authorization_skips_other_paths_and_blocks_denial(tmp_path):
    run = lambda *_args, **_kwargs: pytest.fail("authorization must not run")
    assert authorize_macos_power_telemetry(False, system=lambda: "Darwin", run=run) is None
    assert authorize_macos_power_telemetry(True, system=lambda: "Linux", run=run) is None

    helper = tmp_path / "askpass"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    denied = lambda *_args, **_kwargs: type("Result", (), {"returncode": 1})()
    error = authorize_macos_power_telemetry(
        True, system=lambda: "Darwin", run=denied, askpass_path=helper,
    )
    assert error is not None and "canceled or denied" in error


def test_power_authorization_rejects_missing_or_hung_prompt(tmp_path):
    missing_error = authorize_macos_power_telemetry(
        True, system=lambda: "Darwin", askpass_path=tmp_path / "missing",
    )
    assert missing_error is not None and "missing or not executable" in missing_error
    helper = tmp_path / "askpass"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("sudo", 120)

    timeout_error = authorize_macos_power_telemetry(
        True, system=lambda: "Darwin", run=timeout, askpass_path=helper,
    )
    assert timeout_error is not None and "could not be requested" in timeout_error


def test_windows_host_offset_comes_from_powershell_only_under_wsl():
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return type("Result", (), {"stdout": "-300\n", "returncode": 0})()

    assert windows_host_utc_offset_minutes(
        system=lambda: "Linux", release=lambda: "microsoft-standard-WSL2", run=run,
    ) == -300
    assert calls[0][0][0] == "powershell.exe"
    assert windows_host_utc_offset_minutes(
        system=lambda: "Linux", release=lambda: "6.8.0-generic",
        run=lambda *_args, **_kwargs: pytest.fail("native Linux must not call PowerShell"),
    ) is None


@pytest.mark.parametrize("output,returncode", [("not-a-number", 0), ("900", 0), ("60", 1)])
def test_windows_host_offset_rejects_invalid_powershell_results(output, returncode):
    run = lambda *_args, **_kwargs: type(
        "Result", (), {"stdout": output, "returncode": returncode},
    )()
    assert windows_host_utc_offset_minutes(
        system=lambda: "Linux", release=lambda: "microsoft-standard-WSL2", run=run,
    ) is None


def test_launch_controlled_process_removes_control_file_when_launch_fails(tmp_path):
    control_path = tmp_path / "pause.json"
    control_path.write_text("{}", encoding="utf-8")

    def fail_popen(*_args, **_kwargs):
        raise OSError("executable missing")

    with pytest.raises(OSError, match="executable missing"):
        launch_controlled_process(
            ["missing"], pause_path_factory=lambda: control_path,
            popen=cast("type[subprocess.Popen]", fail_popen),
        )
    assert not control_path.exists()


def test_recovery_command_and_inspection_are_explicit_and_readable(tmp_path):
    result = tmp_path / "result.json"
    command = recovery_executor_command(result, python_executable="python")
    assert command == [
        "python", "-m", "scripts.results.recovery_executor", str(result.resolve()),
    ]
    detail = format_recovery_inspection({
        "action": "fork", "plan_id": "plan-1", "interrupted_attempts": 2,
        "stage_states": {"llm": "interrupted"},
        "case_counts": {"complete": 17, "timed_out": 1},
        "stage_case_counts": {"llm": {"complete": 17, "timed_out": 1}},
        "retryable_cases": [{
            "case_id": "case_1", "stage": "llm", "label": "model · 8K",
            "state": "timed_out", "model": "model",
        }],
        "reasons": ["runtime identity changed"],
    })
    assert "Decision: FORK" in detail
    assert "llm: interrupted" in detail
    assert "complete: 17" in detail
    assert "llm: complete=17, timed_out=1" in detail
    assert "llm: model · 8K (timed_out)" in detail
    assert "runtime identity changed" in detail
    fork = fork_executor_command(result, tmp_path / "fork.json", python_executable="python")
    assert fork == [
        "python", "-m", "scripts.results.fork_executor",
        str(result.resolve()), str((tmp_path / "fork.json").resolve()),
    ]
    assert retry_executor_command(result, ["case_a"], python_executable="python") == [
        "python", "-m", "scripts.results.retry_executor",
        str(result.resolve()), "case_a",
    ]


def test_fork_review_does_not_require_an_event_journal(tmp_path):
    plan = RunPlan.create(
        application_version="4.1", engine_name="llamacpp", tests=["emb"],
        stage_order=["emb"], models={
            "llm": [], "concurrency": [],
            "embeddings": [{"tag": "embed", "short": "embed"}], "images": [],
        }, effective_config={"warmup_runs": 0, "cpu_only": False, "force_all": False},
    )
    result = tmp_path / "legacy.json"
    result.write_text(json.dumps({
        "run": {"plan": plan.to_dict(), "stages": {"emb": {"status": "failed"}}},
    }), encoding="utf-8")
    report = fork_review_report(result)
    assert report["action"] == "fork" and report["can_resume"] is False
    assert report["stage_states"] == {"emb": "failed"}
    assert report["case_counts"] == {}


def test_recovery_progress_entries_deduplicate_models_and_use_catalog_labels():
    tag = "gemma3:1b-it-q4_K_M"
    plan = RunPlan.create(
        application_version="4.1", engine_name="llamacpp",
        tests=["llm", "conc_chat"], stage_order=["llm", "conc_chat"],
        models={
            "llm": [{"tag": tag, "short": "gemma3-1b"}],
            "concurrency": [{"tag": tag, "short": "gemma3-1b"}],
            "embeddings": [], "images": [],
        },
        effective_config={"cpu_only": False, "force_all": False, "warmup_runs": 0},
    )
    entries = recovery_progress_entries(plan)
    assert [(entry.kind, entry.value, entry.label) for entry in entries] == [
        ("llm", tag, "Gemma 3 1B"),
    ]
    assert recovery_progress_entries(plan, {"other"}) == []


def test_recovery_progress_entries_include_embedding_and_image_models():
    plan = RunPlan.create(
        application_version="6.0-pre7", engine_name="llamacpp",
        tests=["emb", "img"], stage_order=["emb", "img"], models={
            "llm": [], "concurrency": [],
            "embeddings": [{"tag": "nomic-embed-text", "short": "nomic-embed-text"}],
            "images": [{"short": "sdxl"}],
        }, effective_config={"cpu_only": False, "force_all": False, "warmup_runs": 0},
    )
    assert [(entry.kind, entry.value, entry.label)
            for entry in recovery_progress_entries(plan)] == [
        ("embedding", "nomic-embed-text", "Nomic Embed Text"),
        ("image", "sdxl", "SDXL"),
    ]


def test_discovery_report_summarizes_readiness_without_mutation():
    inventory = {
        "llm": [{"tag": "one"}], "custom": [{"tag": "custom"}],
        "embedding": [{"tag": "embed"}], "image": [],
    }
    report = build_discovery_report(
        platform_name="Darwin", architecture="arm64", ram_gb=64.0, backend="metal",
        tools={"llama-server": "/bin/server", "llama-bench": None},
        comfyui_dir=None, inventory=inventory, free_storage_gb=120.5,
    )
    assert report["system"] == "Darwin arm64 · 64.0 GB RAM · metal"
    assert report["models"] == "1 LLM, 1 custom LLM, 1 embedding, 0 image"
    assert report["runtime"] == "llama-server: found, llama-bench: missing"
    assert report["storage"].endswith("120.5 GB free")
    assert report["issues"] == []


def test_discovery_report_identifies_blockers_and_image_runtime_gap():
    empty = {"llm": [], "custom": [], "embedding": [], "image": []}
    report = build_discovery_report(
        platform_name="Linux", architecture="x86_64", ram_gb=32.0, backend="cpu",
        tools={"llama-server": None}, comfyui_dir=None, inventory=empty,
    )
    assert report["issues"] == [
        "llama-server was not found; LLM-backed tests cannot start.",
        "No benchmark models were found; run Setup to add models.",
    ]

    empty["image"] = [{"short": "sdxl"}]
    report = build_discovery_report(
        platform_name="Linux", architecture="x86_64", ram_gb=32.0, backend="cpu",
        tools={"llama-server": "/bin/server"}, comfyui_dir=None, inventory=empty,
    )
    assert report["issues"] == ["Image models are installed, but ComfyUI was not found."]


def test_progress_line_parser_accepts_only_supported_structured_events():
    assert parse_progress_line(
        '::local-ai-bench-progress::{"kind":"stage","stage":"llm","status":"running"}\n'
    ) == {"kind": "stage", "stage": "llm", "status": "running"}
    assert parse_progress_line(
        '::local-ai-bench-progress::{"kind":"model","stage":"llm","status":"complete","model":"Qwen: 4B"}\n'
    ) == {"kind": "model", "stage": "llm", "status": "complete", "model": "Qwen: 4B"}
    assert parse_progress_line(
        '::local-ai-bench-progress::{"kind":"model","stage":"conv","status":"skipped","model":"Slow"}\n'
    ) == {"kind": "model", "stage": "conv", "status": "skipped", "model": "Slow"}
    assert parse_progress_line("ordinary benchmark output") is None
    assert parse_progress_line("::local-ai-bench-progress::{bad json") is None
    retrying_event = parse_progress_line(
        '::local-ai-bench-progress::{"kind":"measurement","stage":"llm",'
        '"status":"retrying","model":"Qwen 2K run 1"}\n'
    )
    assert retrying_event is not None and retrying_event["status"] == "retrying"
    result_event = parse_progress_line(
        '::local-ai-bench-progress::{"kind":"result","stage":"run",'
        '"status":"complete","path":"C:\\\\results\\\\run.json"}'
    )
    assert result_event is not None and result_event["path"] == "C:\\results\\run.json"


def test_progress_model_identity_prefers_stable_id_over_custom_display_label():
    event = {"model": "nemotron:30b (custom)", "model_id": "nemotron:30b"}
    assert progress_model_identity(event) == "nemotron:30b"


@pytest.mark.parametrize(("line_height", "expected"), ((12, 38), (18, 46), (27, 64)))
def test_history_row_height_tracks_scaled_font_height(line_height, expected):
    assert history_row_height(line_height) == expected


def test_progress_metrics_count_terminal_models_and_measurement_quality_once():
    metrics = {
        "total_models": 2, "finished_models": set(), "usable_models": set(),
        "retries": 0, "valid": 0, "invalid": 0,
    }
    metrics = update_progress_metrics(
        metrics, {"kind": "measurement", "stage": "llm", "status": "retrying", "model": "A"},
    )
    metrics = update_progress_metrics(
        metrics, {"kind": "measurement", "stage": "llm", "status": "invalid", "model": "A"},
    )
    terminal = {
        "kind": "model", "stage": "llm", "status": "complete", "model": "A", "usable": True,
    }
    metrics = update_progress_metrics(metrics, terminal)
    metrics = update_progress_metrics(metrics, terminal)
    metrics = update_progress_metrics(
        metrics, {"kind": "model", "stage": "conv", "status": "skipped", "model": "B"},
    )
    assert (metrics["retries"], metrics["invalid"], len(metrics["finished_models"])) == (1, 1, 2)
    assert metrics["usable_models"] == {("llm", "A")}
    assert progress_summary_rows(metrics) == {
        "Finished models": "2 / 2",
        "Usable coverage": "1 / 2",
        "Invalid measurements": "1",
        "Retries": "1",
    }


@pytest.mark.parametrize(("elapsed", "completed", "total", "expected"), [
    (60, 1, 4, 180), (60, 4, 4, 0), (60, 0, 4, None), (-1, 1, 4, None),
])
def test_remaining_time_estimate(elapsed, completed, total, expected):
    assert estimate_remaining_seconds(elapsed, completed, total) == expected


def test_process_resource_usage_includes_child_processes():
    class Memory:
        def __init__(self, rss):
            self.rss = rss

    class Process:
        def __init__(self, cpu, rss, children=(), pid=0):
            self.cpu = cpu
            self.rss = rss
            self._children = children
            self.pid = pid

        def children(self, recursive):
            assert recursive
            return list(self._children)

        def cpu_percent(self, interval):
            assert interval is None
            return self.cpu

        def memory_info(self):
            return Memory(self.rss)

    child = Process(20, 1024 ** 3)
    parent = Process(30, 2 * 1024 ** 3, [child])

    class Psutil:
        Error = RuntimeError

        @staticmethod
        def Process(pid):
            assert pid == 42
            return parent

    assert process_resource_usage(42, cast(PsutilLike, Psutil)) == (50, 3.0)


def test_process_resource_usage_skips_inaccessible_children():
    class ResourceError(Exception):
        pass

    class Process:
        pid = 0

        def __init__(self, cpu, rss, *, inaccessible=False, children=()):
            self.cpu = cpu
            self.rss = rss
            self.inaccessible = inaccessible
            self._children = children

        def children(self, recursive):
            assert recursive
            return list(self._children)

        def cpu_percent(self, interval):
            assert interval is None
            return self.cpu

        def memory_info(self):
            if self.inaccessible:
                raise ResourceError("privileged child")
            return type("Memory", (), {"rss": self.rss})()

    accessible = Process(20, 1024 ** 3)
    privileged = Process(90, 8 * 1024 ** 3, inaccessible=True)
    parent = Process(30, 2 * 1024 ** 3, children=[accessible, privileged])

    class Psutil:
        Error = ResourceError

        @staticmethod
        def Process(_pid):
            return parent

    assert process_resource_usage(42, cast(PsutilLike, Psutil)) == (50, 3.0)


def test_process_resource_usage_retains_parent_when_child_enumeration_races():
    class ResourceError(Exception):
        pass

    class Parent:
        pid = 42

        def children(self, recursive):
            assert recursive
            raise ResourceError("child exited")

        def cpu_percent(self, interval):
            assert interval is None
            return 30

        def memory_info(self):
            return type("Memory", (), {"rss": 2 * 1024 ** 3})()

    class Psutil:
        Error = ResourceError

        @staticmethod
        def Process(_pid):
            return Parent()

    assert process_resource_usage(42, cast(PsutilLike, Psutil)) == (30, 2.0)


def test_system_memory_usage_reports_used_and_total_gibibytes():
    memory = type("Memory", (), {"used": 32 * 1024 ** 3, "total": 128 * 1024 ** 3})()
    psutil_module = cast(PsutilLike,
        type("Psutil", (), {"virtual_memory": staticmethod(lambda: memory)}))
    assert system_memory_usage(psutil_module) == (32.0, 128.0)


@pytest.mark.parametrize(
    ("platform_name", "output", "expected"),
    [
        ("Darwin", '"PerformanceStatistics" = {"Device Utilization %"=67}', 67.0),
        ("Linux", "12\n84\n", 84.0),
        ("Linux", '{"card0": {"GPU use (%)": "53"}}', 53.0),
        ("Linux", "not available", None),
    ],
)
def test_parse_gpu_usage_handles_apple_nvidia_and_amd(platform_name, output, expected):
    assert parse_gpu_usage(platform_name, output) == expected


def test_query_gpu_usage_uses_non_privileged_apple_statistics():
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return type("Result", (), {
            "returncode": 0,
            "stdout": '"Device Utilization %"=72',
        })()

    assert query_gpu_usage("Darwin", run_fn=run, which_fn=lambda _: "/usr/sbin/ioreg") == 72
    assert calls[0][0] == ["/usr/sbin/ioreg", "-r", "-d", "1", "-c", "AGXAccelerator"]
    assert calls[0][1]["timeout"] == 2


def test_gpu_process_memory_filters_to_benchmark_process_tree():
    assert parse_gpu_process_memory("42, 1024\n43, 2048 MiB\n99, 4096\n", {42, 43}) == 3.0
    assert parse_gpu_process_memory("N/A\n", {42}) is None


def test_query_gpu_process_memory_uses_nvidia_process_accounting():
    child = type("Child", (), {"pid": 43})()
    parent = type("Parent", (), {"pid": 42, "children": lambda self, recursive: [child]})()
    psutil_module = type("Psutil", (), {
        "Error": RuntimeError,
        "Process": staticmethod(lambda pid: parent),
    })
    result = type("Result", (), {"returncode": 0, "stdout": "42, 512\n43, 1536\n"})()

    assert query_gpu_process_memory(
        42, run_fn=lambda *args, **kwargs: result,
        which_fn=lambda name: "/usr/bin/nvidia-smi", psutil_module=cast(PsutilLike, psutil_module),
    ) == 2.0


def test_resource_usage_rows_format_table_values_and_fallbacks():
    assert resource_usage_rows(
        (50, 3.25), (40, 128), 35, 77, 20, (12, 16), include_vram=True,
    ) == {
        "CPU": "50%",
        "Process RAM": "3.2 GB",
        "System RAM": "40.0 / 128.0 GB (Δ +5.0 GB)",
        "GPU": "77% utilization · 20.0 GB process memory",
        "VRAM": "12.0 / 16.0 GB used",
    }


def test_vram_line_only_applies_to_discrete_memory_devices(monkeypatch):
    assert show_vram_usage([
        {"name": "NVIDIA GeForce RTX 4090", "vendor": "nvidia", "vram_gb": 24},
    ])
    assert show_vram_usage([
        {"name": "AMD Radeon Pro W7900", "vendor": "amd", "vram_gb": 48},
    ])
    assert not show_vram_usage([
        {"name": "AMD Radeon Graphics", "vendor": "amd", "vram_gb": 2},
        {"name": "NVIDIA GB10", "vendor": "nvidia", "vram_gb": None},
    ])

    monkeypatch.setattr(
        Shared, "sample_memory_gb", staticmethod(lambda: {
            "gpu_vram_used_gb": 10.25, "gpu_vram_total_gb": 24.0,
        }),
    )
    assert query_vram_usage() == (10.25, 24.0)
    assert resource_usage_rows(None, None, 0, None, None) == {
        "CPU": "Unavailable", "Process RAM": "Unavailable",
        "System RAM": "Unavailable", "GPU": "Unavailable",
    }
    assert resource_usage_rows(
        None, None, 0, None, None, include_vram=True,
    )["VRAM"] == "Unavailable"


def test_workload_preflight_reports_specific_runtime_resolutions():
    errors = workload_preflight_errors(
        ["llm", "llamabench", "llamabenchconc", "img"],
        {"llama-server": None, "llama-bench": None, "llama-batched-bench": None}, False,
    )
    assert len(errors) == 4
    assert all("Run Setup" in error for error in errors)
    assert workload_preflight_errors(
        ["llm", "img"], {"llama-server": "/bin/server"}, True,
    ) == []


def test_run_outcome_explains_preserved_data_cleanup_and_next_step():
    assert format_run_outcome(0) == "Benchmark completed successfully. Results are ready to review."
    failure = format_run_outcome(2)
    assert "exit code 2" in failure
    assert "Checkpointed measurements" in failure
    assert "remain usable" in failure
    assert "Automatic cleanup" in failure
    assert "Review the final Run Log" in failure


def test_custom_option_defaults_reset_paths_without_mutating_global_defaults():
    defaults = custom_option_defaults(Path("/chosen/ComfyUI"))
    assert defaults["comfyui"] == "/chosen/ComfyUI"
    assert defaults["runs"] == GUI_OPTION_DEFAULTS["runs"]
    assert GUI_OPTION_DEFAULTS["comfyui"] == ""


def test_default_control_values_describe_the_actual_default_run():
    tests = [MenuEntry("llm", "LLM", "test", "", True), MenuEntry("img", "Images", "test", "", False)]
    models = [MenuEntry("small", "Small", "llm", "", True)]
    values = default_control_values(tests, models, "llamacpp", Path("/ComfyUI"))
    assert values["tests"] == {"llm": True, "img": False}
    assert values["models"] == {"small": True}
    assert values["engine"] == "llamacpp"
    assert values["max_prompt_tokens"] == "No cap"
    assert values["tg_tokens"] == set(config.LLAMABENCH_TG)
    assert values["options"] == custom_option_defaults(Path("/ComfyUI"))


def test_preset_selection_restores_named_custom_and_legacy_states():
    assert restored_preset_name(None) == "Consumer guidance"
    assert restored_preset_name({"selected_preset": "Quick run"}) == "Quick run"
    assert restored_preset_name({"selected_preset": CUSTOM_PRESET}) == CUSTOM_PRESET
    assert restored_preset_name({}) == CUSTOM_PRESET
    assert restored_preset_name({"selected_preset": "Removed preset"}) == CUSTOM_PRESET


def test_control_changes_switch_named_presets_to_custom_except_during_application():
    assert preset_after_control_change("Quick run", False) == CUSTOM_PRESET
    assert preset_after_control_change("Quick run", True) == "Quick run"
    assert preset_after_control_change(CUSTOM_PRESET, False) == CUSTOM_PRESET


def test_commercial_presets_cover_named_use_cases_and_filter_unavailable_tests():
    assert set(BENCHMARK_PRESETS) == {
        "Consumer guidance", "Vendor validation", "Neutral comparison", "Platform optimized",
        "Offline / private", "Quick run", "Full run",
        "Role: Orchestrator", "Role: Agent / tool caller", "Role: Coding assistant",
        "Role: Chat assistant", "Role: RAG / retrieval",
    }
    quick = resolve_preset("Quick run", {"llm"})
    assert quick == {"tests": ["llm"], "runs": 1, "max_prompt_tokens": 8192, "force_all": False}
    full = resolve_preset("Full run", {"llm", "img"})
    assert full["tests"] == ["llm", "img"]
    assert full["force_all"]


def test_role_presets_select_role_relevant_tests_and_depth():
    every_test = {name for name, *_ in TEST_DEFINITIONS}
    orchestrator = resolve_preset("Role: Orchestrator", every_test)
    assert orchestrator == {
        "tests": ["llm", "conv", "reasoning", "tool", "conc_chat"], "runs": config.N_RUNS,
        "max_prompt_tokens": None, "force_all": False,
    }
    agent = resolve_preset("Role: Agent / tool caller", every_test)
    assert agent["tests"] == ["llm", "conv", "tool", "code", "conc_tool"]
    assert agent["max_prompt_tokens"] == 32768
    coding = resolve_preset("Role: Coding assistant", every_test)
    assert coding["tests"] == ["llm", "conv", "code", "reasoning"]
    assert coding["max_prompt_tokens"] == 32768
    chat = resolve_preset("Role: Chat assistant", every_test)
    assert chat["tests"] == ["llm", "conv", "mcq", "reasoning", "conc_chat"]
    assert chat["max_prompt_tokens"] == 8192
    rag = resolve_preset("Role: RAG / retrieval", every_test)
    assert rag["tests"] == ["llm", "conv", "emb", "mcq"]
    assert rag["max_prompt_tokens"] == 32768
    assert resolve_preset("Role: RAG / retrieval", {"llm", "emb"})["tests"] == ["llm", "emb"]
    assert restored_preset_name({"selected_preset": "Role: Orchestrator"}) == "Role: Orchestrator"


def test_named_preset_replaces_the_complete_control_configuration():
    defaults = {
        "tests": {"llm": True, "emb": True, "img": True},
        "models": {"small": True}, "engine": "llamacpp",
        "max_prompt_tokens": "No cap", "tg_tokens": {128, 256},
        "options": dict(GUI_OPTION_DEFAULTS),
    }
    values = preset_control_values("Quick run", {"llm", "emb", "img"}, defaults)
    # A preset must not touch the engine selection — see apply_control_values.
    assert "engine" not in values
    assert values["tests"] == {"llm": True, "emb": True, "img": False}
    assert values["max_prompt_tokens"] == "8192"
    assert values["options"]["runs"] == 1
    assert values["options"]["force_all"] is False
    assert values["options"]["out"] == ""
    values["options"]["out"] = "changed.json"
    assert defaults["options"]["out"] == ""


def test_hardware_defaults_uncheck_models_that_exceed_usable_ram():
    entries = [
        MenuEntry("small", "Small", "llm", "LLM", True),
        MenuEntry("huge", "Huge", "llm", "LLM", True),
    ]
    inventory = {
        "llm": [{"tag": "small", "size": 4e9}, {"tag": "huge", "size": 30e9}],
        "custom": [], "embedding": [], "image": [],
    }
    apply_hardware_model_defaults(entries, inventory, ram_gb=32)
    assert [entry.checked for entry in entries] == [True, False]
    entries[1].checked = True
    apply_hardware_model_defaults(entries, inventory, ram_gb=64)
    assert entries[1].checked


def test_plan_preview_shows_resolved_measurement_values_and_destinations():
    options = {**GUI_OPTION_DEFAULTS, "runs": 5, "cpu_only": True, "out": "chosen.json"}
    preview = build_plan_preview(
        engine="llamacpp", tests=["llm"],
        entries=[MenuEntry("m", "Model", "llm", "LLM", True)], options=options,
        max_prompt_tokens=32768, tg_tokens=[128, 512], comfyui_dir=Path("/ComfyUI"),
    )
    for expected in (
        "Engine: llamacpp", "Tests: llm", "Models: Model", "Measured runs: 5",
        "Prompt cap: 32768", "CPU only: Yes", "Results: chosen.json", "ComfyUI: /ComfyUI",
        "Broad cases: 1 model-workload passes", "Model loads: at least 1",
        "Duration range: minutes to hours", "Processes: llama-server", "Network use: none expected",
    ):
        assert expected in preview


def test_plan_preview_sections_group_the_review_for_scanning():
    sections = dict(plan_preview_sections(
        "Engine: llamacpp\nTests: llm\nWarmups: 2\nBroad cases: 3 passes\n"
        "Results: automatic\nNetwork use: none expected"
    ))
    assert sections == {
        "Selection": ["Engine: llamacpp", "Tests: llm"],
        "Measurement settings": ["Warmups: 2"],
        "Scope and duration": ["Broad cases: 3 passes"],
        "Output and environment": ["Results: automatic", "Network use: none expected"],
    }


# ── progress-row engine attribution ──

def test_progress_event_engine_uses_the_name_the_event_carries():
    from scripts.app.benchmark_gui import progress_event_engine
    event = {"kind": "model", "stage": "llm", "engine": "vllm"}
    assert progress_event_engine(event, ["llamacpp", "vllm"]) == "vllm"


def test_progress_event_engine_assumes_the_only_engine_when_unnamed():
    from scripts.app.benchmark_gui import progress_event_engine
    assert progress_event_engine({"kind": "model", "stage": "llm"}, ["llamacpp"]) == "llamacpp"


def test_progress_event_engine_refuses_to_guess_during_a_multi_engine_run():
    """An unnamed event previously landed on the first engine's rows, crediting the
    vLLM pass's progress to llamacpp while vLLM's rows stayed queued."""
    from scripts.app.benchmark_gui import progress_event_engine
    assert progress_event_engine({"kind": "model", "stage": "llm"}, ["llamacpp", "vllm"]) is None


def test_progress_event_engine_rejects_an_engine_that_is_not_in_this_run():
    from scripts.app.benchmark_gui import progress_event_engine
    event = {"kind": "stage", "stage": "llm", "engine": "mlx"}
    assert progress_event_engine(event, ["llamacpp", "vllm"]) is None
    assert progress_event_engine({"engine": "", "kind": "stage"}, ["llamacpp"]) == "llamacpp"
