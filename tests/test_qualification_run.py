from pathlib import Path

import pytest

from scripts.release.qualification_coverage import qualification_arguments
from scripts.release.qualification_run import (
    benchmark_wrapper, default_result_path, qualification_failure_summary,
)
from scripts.release.qualification_targets import (
    TARGET_ENGINES, TARGETS, qualification_target, qualification_target_errors, target_engine,
)


def test_target_engine_is_explicit_and_rejects_unknown_targets():
    assert target_engine("dgx-spark-vllm-cuda") == "vllm"
    assert target_engine("macos-m5-pro-llamacpp-metal") == "llamacpp"
    assert target_engine("radeon-linux-vllm-rocm") == "vllm"
    with pytest.raises(ValueError, match="unknown qualification target"):
        target_engine("guessed-platform")


def test_target_engines_are_derived_from_the_platform_registry():
    assert TARGET_ENGINES == {target["id"]: target["runtime"] for target in TARGETS}


def test_qualification_target_returns_the_shared_target_record():
    assert qualification_target("radeon-wsl2-llamacpp-rocm")["backend"] == "rocm"


def test_qualification_target_rejects_cpu_fallback_and_wrong_platform():
    target = qualification_target("radeon-wsl2-llamacpp-rocm")
    profile = {
        "os": "Linux 6.18.33.2-microsoft-standard-WSL2",
        "wsl": True,
        "arch": "x86_64",
        "backend": "cpu",
        "hostname": "AMD Ryzen 7 5800XT\nAMD Radeon RX 9060 XT 31 GB",
    }
    assert qualification_target_errors(target, profile) == [
        "requires backend rocm; detected cpu",
    ]
    profile["wsl"] = False
    assert qualification_target_errors(target, profile) == [
        "requires platform wsl2; detected linux",
        "requires backend rocm; detected cpu",
    ]


def test_qualification_target_accepts_architecture_alias_and_accelerator_substring():
    target = qualification_target("dgx-spark-vllm-cuda")
    profile = {
        "os": "Linux 6.17",
        "arch": "arm64",
        "backend": "cuda",
        "hostname": "NVIDIA GB10 119 GB",
    }
    assert qualification_target_errors(target, profile) == []


def test_intel_linux_target_accepts_b65_pci_codename():
    target = qualification_target("intel-arc-linux-llamacpp-sycl")
    profile = {
        "os": "Linux 6.17",
        "arch": "x86_64",
        "backend": "xpu",
        "hostname": (
            "Intel Core Ultra\nIntel Corporation Battlemage G31 "
            "[Intel Graphics] [8086:e222] 64 GB"
        ),
    }
    assert qualification_target_errors(target, profile) == []


def test_vulkan_target_uses_accelerator_identity_instead_of_compute_backend_probe():
    target = qualification_target("radeon-windows-llamacpp-vulkan")
    profile = {
        "os": "Windows 11",
        "arch": "AMD64",
        "backend": "rocm",
        "hostname": "AMD Ryzen 7 5800XT\nAMD Radeon RX 9060 XT 16 GB",
    }
    assert qualification_target_errors(target, profile) == []


def test_qualification_arguments_are_the_normal_smallest_model_benchmark():
    arguments = qualification_arguments("llamacpp", "tiny", Path("result.json"))
    assert arguments[:4] == ["--ui", "none", "--engine", "llamacpp"]
    assert arguments[arguments.index("--llm-models") + 1] == "tiny"
    assert arguments[arguments.index("--embedding-models") + 1] == "nomic-embed-text"
    assert arguments[arguments.index("--image-models") + 1] == "sd15"
    assert arguments[arguments.index("--out") + 1] == "result.json"


def test_unix_qualification_uses_the_shipped_benchmark_wrapper(tmp_path):
    assert benchmark_wrapper(tmp_path) == [str(tmp_path / "run_bench.sh")]


def test_default_qualification_result_is_grouped_under_ignored_target_directory(tmp_path):
    assert default_result_path(tmp_path, "dgx-spark-vllm-cuda") == (
        tmp_path / "qualification-evidence" / "dgx-spark-vllm-cuda"
        / "results_qualification_dgx-spark-vllm-cuda.json"
    )


def test_comfyui_failure_distinguishes_completed_llamacpp_evidence(monkeypatch):
    result = {
        "run": {"status": "failed", "stages": {"img": {"status": "failed"}}},
    }
    monkeypatch.setattr(
        "scripts.release.qualification_run.workload_coverage_errors",
        lambda _result, workloads: [] if "img" not in workloads else ["img failed"],
    )
    assert qualification_failure_summary(result, "llamacpp") == (
        "llama.cpp workloads passed; ComfyUI image generation did not pass"
    )


def test_comfyui_failure_does_not_hide_an_incomplete_llamacpp_workload(monkeypatch):
    result = {
        "run": {"status": "failed", "stages": {
            "llamabench": {"status": "complete"}, "img": {"status": "failed"},
        }},
    }
    monkeypatch.setattr(
        "scripts.release.qualification_run.workload_coverage_errors",
        lambda _result, _workloads: ["llamabench produced no evidence"],
    )
    assert qualification_failure_summary(result, "llamacpp") == "benchmark failed during img"


def test_failure_without_a_failed_stage_reports_incomplete_exit():
    assert qualification_failure_summary({"run": {"stages": {}}}, "vllm") == (
        "benchmark exited before qualification completed"
    )
