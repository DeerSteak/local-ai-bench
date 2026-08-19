from pathlib import Path

import pytest

from scripts.release.qualification_coverage import qualification_arguments
from scripts.release.qualification_run import benchmark_wrapper
from scripts.release.qualification_targets import TARGET_ENGINES, TARGETS, target_engine


def test_target_engine_is_explicit_and_rejects_unknown_targets():
    assert target_engine("dgx-spark-vllm-cuda") == "vllm"
    assert target_engine("macos-m5-pro-llamacpp-metal") == "llamacpp"
    with pytest.raises(ValueError, match="unknown qualification target"):
        target_engine("guessed-platform")


def test_target_engines_are_derived_from_the_platform_registry():
    assert TARGET_ENGINES == {target["id"]: target["runtime"] for target in TARGETS}


def test_qualification_arguments_are_the_normal_smallest_model_benchmark():
    arguments = qualification_arguments("llamacpp", "tiny", Path("result.json"))
    assert arguments[:4] == ["--ui", "none", "--engine", "llamacpp"]
    assert arguments[arguments.index("--llm-models") + 1] == "tiny"
    assert arguments[arguments.index("--embedding-models") + 1] == "nomic-embed-text"
    assert arguments[arguments.index("--image-models") + 1] == "sd15"
    assert arguments[arguments.index("--out") + 1] == "result.json"


def test_unix_qualification_uses_the_shipped_benchmark_wrapper(tmp_path):
    assert benchmark_wrapper(tmp_path) == [str(tmp_path / "run_bench.sh")]
