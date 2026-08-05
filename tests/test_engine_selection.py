from scripts.setup.engine_selection import (
    LLAMACPP,
    VLLM,
    build_engine_entries,
    engine_summary_line,
    engines_needing_install,
    find_entry,
    selected_engine_names,
    toggle_engine,
)
from scripts.setup.vllm_install import VllmSupport


SUPPORTED = VllmSupport("supported", "cuda_wheel", "CUDA wheels available")
EXPERIMENTAL = VllmSupport("experimental", "metal_plugin", "Apple Silicon plugin")
UNSUPPORTED = VllmSupport("unsupported", None, "no upstream Windows support")


def test_llamacpp_starts_selected_and_vllm_does_not():
    entries = build_engine_entries(vllm_support=SUPPORTED)
    assert find_entry(entries, LLAMACPP)["checked"] is True
    assert find_entry(entries, VLLM)["checked"] is False
    assert selected_engine_names(entries) == [LLAMACPP]


def test_vllm_is_selectable_when_supported():
    entries = build_engine_entries(vllm_support=SUPPORTED)
    assert toggle_engine(entries, VLLM) is True
    assert selected_engine_names(entries) == [LLAMACPP, VLLM]


def test_vllm_is_disabled_and_unselectable_when_unsupported():
    entries = build_engine_entries(vllm_support=UNSUPPORTED)
    vllm = find_entry(entries, VLLM)
    assert vllm["enabled"] is False
    assert vllm["checked"] is False
    assert toggle_engine(entries, VLLM) is False
    assert vllm["checked"] is False
    assert selected_engine_names(entries) == [LLAMACPP]


def test_missing_support_record_also_disables_vllm():
    entries = build_engine_entries(vllm_support=None)
    assert find_entry(entries, VLLM)["enabled"] is False


def test_experimental_support_is_selectable_but_flagged():
    entries = build_engine_entries(vllm_support=EXPERIMENTAL)
    vllm = find_entry(entries, VLLM)
    assert vllm["enabled"] is True and vllm["experimental"] is True
    assert toggle_engine(entries, VLLM) is True


def test_the_last_selected_engine_cannot_be_cleared():
    entries = build_engine_entries(vllm_support=SUPPORTED)
    assert toggle_engine(entries, LLAMACPP) is False
    assert selected_engine_names(entries) == [LLAMACPP]


def test_llamacpp_can_be_cleared_once_vllm_is_selected():
    entries = build_engine_entries(vllm_support=SUPPORTED)
    toggle_engine(entries, VLLM)
    assert toggle_engine(entries, LLAMACPP) is True
    assert selected_engine_names(entries) == [VLLM]
    assert toggle_engine(entries, VLLM) is False  # now vLLM is the last one


def test_toggling_an_unknown_engine_is_a_no_op():
    entries = build_engine_entries(vllm_support=SUPPORTED)
    assert toggle_engine(entries, "mlx") is False
    assert selected_engine_names(entries) == [LLAMACPP]


def test_already_installed_engines_are_not_reinstalled():
    entries = build_engine_entries(vllm_support=SUPPORTED, vllm_found=True, llamacpp_found=True)
    toggle_engine(entries, VLLM)
    assert engines_needing_install(entries) == []
    assert find_entry(entries, VLLM)["note"] == "already installed"


def test_only_missing_selected_engines_need_installing():
    entries = build_engine_entries(vllm_support=SUPPORTED, vllm_found=False, llamacpp_found=True)
    assert engines_needing_install(entries) == []  # vLLM not selected yet
    toggle_engine(entries, VLLM)
    assert engines_needing_install(entries) == [VLLM]


def test_a_disabled_engine_never_counts_as_selected_or_installable():
    entries = build_engine_entries(vllm_support=UNSUPPORTED)
    find_entry(entries, VLLM)["checked"] = True  # forced state, e.g. a stale saved plan
    assert selected_engine_names(entries) == [LLAMACPP]
    assert VLLM not in engines_needing_install(entries)


def test_summary_lines_show_state_and_reason():
    entries = build_engine_entries(vllm_support=UNSUPPORTED, llamacpp_found=True)
    assert engine_summary_line(entries[0]) == "[x] llama.cpp — already installed"
    assert engine_summary_line(entries[1]) == "[ ] vLLM (unavailable) — no upstream Windows support"
    flagged = build_engine_entries(vllm_support=EXPERIMENTAL)
    assert "(experimental)" in engine_summary_line(flagged[1])
