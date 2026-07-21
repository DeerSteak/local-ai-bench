from pathlib import Path

import config
from benchmark import sidecar_path


def test_swaps_results_prefix_for_given_prefix():
    out_path = str(config.RESULTS_DIR / "results_Mac_Studio_20260711_090000.json")
    p = sidecar_path(out_path, "answers_mcq_")
    assert p == config.RESULTS_DIR / "answers_mcq_Mac_Studio_20260711_090000.json"


def test_falls_back_to_prepending_prefix_when_stem_lacks_results_prefix():
    out_path = str(config.RESULTS_DIR / "custom_name.json")
    p = sidecar_path(out_path, "answers_math_")
    assert p == config.RESULTS_DIR / "answers_math_custom_name.json"


def test_different_prefixes_share_the_same_hostname_timestamp_suffix():
    out_path = str(config.RESULTS_DIR / "results_Host_20260101_000000.json")
    mcq_path = sidecar_path(out_path, "answers_mcq_")
    math_path = sidecar_path(out_path, "answers_math_")
    code_path = sidecar_path(out_path, "answers_code_")
    tool_path = sidecar_path(out_path, "answers_tool_")
    images_path = sidecar_path(out_path, "images_")
    suffix = "Host_20260101_000000"
    assert mcq_path.name == f"answers_mcq_{suffix}.json"
    assert math_path.name == f"answers_math_{suffix}.json"
    assert code_path.name == f"answers_code_{suffix}.json"
    assert tool_path.name == f"answers_tool_{suffix}.json"
    assert images_path.name == f"images_{suffix}.json"
