import pytest

from scripts.stage_registry import (
    ACCURACY_TESTS, CONCURRENCY_TESTS, LLM_TESTS, STAGE_BY_KEY, STAGE_ORDER, STAGE_SPECS,
    engine_incompatible_tests, stage_spec,
)


def test_stage_registry_has_unique_ordered_keys():
    assert STAGE_ORDER == tuple(spec.key for spec in STAGE_SPECS)
    assert len(STAGE_ORDER) == len(set(STAGE_ORDER)) == len(STAGE_BY_KEY)


def test_stage_groups_are_derived_from_categories():
    assert ACCURACY_TESTS == ["mcq", "math", "reasoning", "code", "tool"]
    assert CONCURRENCY_TESTS == ["conc_tool", "conc_chat"]
    assert set(LLM_TESTS) == {
        "llm", "conv", "llamabench", "llamabenchconc", "vllmbench",
        *ACCURACY_TESTS,
    }


def test_every_stage_has_result_and_model_ownership():
    assert all(spec.section and spec.model_family and spec.ui_family for spec in STAGE_SPECS)
    assert stage_spec("conv").section == "llm_conversation"
    assert stage_spec("llamabench").menu_label == "llama-bench (throughput + concurrency)"
    with pytest.raises(ValueError, match="unknown benchmark stage"):
        stage_spec("unknown")


def test_native_stages_reject_only_the_wrong_engine():
    tests = ["llm", "llamabench", "llamabenchconc", "vllmbench"]
    assert engine_incompatible_tests(tests, "llamacpp") == ["vllmbench"]
    assert engine_incompatible_tests(tests, "vllm") == ["llamabench", "llamabenchconc"]
