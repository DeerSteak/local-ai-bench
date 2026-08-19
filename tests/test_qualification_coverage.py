import json

import pytest

from scripts.release.qualification_coverage import (
    RESULT_SECTIONS, qualification_command, qualification_workloads,
    run_qualification_coverage, workload_coverage_errors,
)


def complete_result(workloads):
    result: dict = {"run": {"status": "complete"}}
    for workload in workloads:
        marker = next(iter({
            "sustained": {"series"}, "llamabench": {"completed_cases"},
            "llamabenchconc": {"entries"}, "vllmbench": {"entries"},
            "mcq": {"answered"}, "math": {"answered"}, "reasoning": {"answered"},
            "code": {"answered"}, "tool": {"answered"},
            "conc_tool": {"valid_runs"}, "conc_chat": {"valid_runs"},
        }.get(workload, {"valid_runs"})))
        value = [{"measured": True}] if marker in {"series", "entries"} else 1
        result[RESULT_SECTIONS[workload]] = {"smallest-model": {marker: value}}
    return result


@pytest.mark.parametrize("engine", ("llamacpp", "vllm"))
def test_every_required_workload_must_produce_evidence(engine):
    workloads = qualification_workloads(engine)
    result = complete_result(workloads)
    assert workload_coverage_errors(result, workloads) == []
    result[RESULT_SECTIONS[workloads[-1]]] = {}
    assert workload_coverage_errors(result, workloads) == [
        f"{workloads[-1]} produced no measured result evidence",
    ]


def test_nonempty_skipped_or_error_record_is_not_measurement_evidence():
    result = {"run": {"status": "complete"}, "llm": {
        "smallest-model": {"skipped": True, "error": "model unavailable"},
    }}
    assert workload_coverage_errors(result, ["llm"]) == [
        "llm produced no measured result evidence",
    ]


def test_real_native_and_server_concurrency_shapes_count_as_evidence():
    result = {
        "run": {"status": "complete"},
        "concurrency_tool": {"model": {"1": {"valid_runs": 1}}},
        "concurrency_chat": {"model": {"1": {"valid_runs": 1}}},
        "llamabench": {"model": {
            "prompt_entries": [{"avg_ts": 100}], "completed_cases": 4,
        }},
    }
    assert workload_coverage_errors(
        result, ["conc_tool", "conc_chat", "llamabench"],
    ) == []


def test_completed_valid_result_is_reused_without_repeating_workloads(monkeypatch, tmp_path):
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(complete_result(qualification_workloads("vllm"))))
    monkeypatch.setattr(
        "scripts.release.qualification_coverage.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("completed coverage must not be repeated"),
    )
    run_qualification_coverage("vllm", "tiny", result_path)


def test_incomplete_run_never_qualifies_even_with_populated_sections():
    workloads = qualification_workloads("vllm")
    result = complete_result(workloads)
    result["run"]["status"] = "failed"
    assert workload_coverage_errors(result, workloads)[0] == \
        "qualification result run is not complete"


def test_llamacpp_coverage_uses_smallest_models_and_all_native_workloads(tmp_path):
    command = qualification_command(
        "llamacpp", "gemma3:1b-it-q4_K_M", tmp_path / "result.json",
        tmp_path / "ComfyUI",
    )
    assert command[command.index("--embedding-models") + 1] == "nomic-embed-text"
    assert command[command.index("--image-models") + 1] == "sd15"
    tests = command[command.index("--tests") + 1:command.index("--llm-models")]
    assert tests == qualification_workloads("llamacpp")
    assert {"llamabench", "llamabenchconc", "img", "sustained"} <= set(tests)


def test_vllm_coverage_uses_vllmbench_without_engine_independent_images(tmp_path):
    command = qualification_command("vllm", "tiny", tmp_path / "result.json")
    tests = command[command.index("--tests") + 1:command.index("--llm-models")]
    assert "vllmbench" in tests
    assert "img" not in tests
    assert "--ack-experimental-engine" in command
