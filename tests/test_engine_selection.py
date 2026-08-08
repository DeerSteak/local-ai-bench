from scripts.setup.engine_selection import (
    LLAMACPP,
    VLLM,
    build_engine_entries,
    engine_summary_line,
    engines_needing_install,
    needs_python_headers,
    find_entry as _find_entry,
    selected_engine_names,
    toggle_engine,
)
from scripts.setup.vllm_install import VllmSupport


def find_entry(entries: list[dict], name: str) -> dict:
    """Test-only strict wrapper: every test call expects the entry to exist."""
    entry = _find_entry(entries, name)
    assert entry is not None, f"no engine entry named {name!r}"
    return entry


SUPPORTED = VllmSupport("supported", "cuda_wheel", "CUDA wheels available")
EXPERIMENTAL = VllmSupport("experimental", "nightly_cu130", "DGX Spark nightly wheels")
UNSUPPORTED = VllmSupport("unsupported", None, "no upstream Windows support")


def test_an_uninstalled_vllm_starts_unselected():
    entries = build_engine_entries(vllm_support=SUPPORTED)
    assert find_entry(entries, LLAMACPP)["checked"] is True
    assert find_entry(entries, VLLM)["checked"] is False
    assert selected_engine_names(entries) == [LLAMACPP]


def test_an_installed_engine_starts_selected():
    """Otherwise setup silently stops maintaining an engine that is already present:
    no headers, no build tools, no weights, while the row says "already installed"."""
    entries = build_engine_entries(vllm_support=SUPPORTED, vllm_found=True, llamacpp_found=True)
    assert find_entry(entries, VLLM)["checked"] is True
    assert selected_engine_names(entries) == [LLAMACPP, VLLM]


def test_an_installed_but_unsupported_vllm_is_still_selected():
    """A present vLLM needs maintaining regardless of whether we could install one."""
    entries = build_engine_entries(vllm_support=UNSUPPORTED, vllm_found=True)
    vllm = find_entry(entries, VLLM)
    assert (vllm["enabled"], vllm["checked"]) == (True, True)


def test_an_installed_engine_can_still_be_deselected():
    entries = build_engine_entries(vllm_support=SUPPORTED, vllm_found=True, llamacpp_found=True)
    assert toggle_engine(entries, VLLM) is True
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


# ── an already-present vLLM overrides the install gate ──

def test_a_running_server_makes_vllm_selectable_on_an_unsupported_platform():
    entries = build_engine_entries(vllm_support=UNSUPPORTED, vllm_found=True,
                                   llamacpp_found=True,
                                   vllm_note="preconfigured server at http://localhost:8000")
    vllm = find_entry(entries, VLLM)
    assert vllm["enabled"] is True, "nothing is being installed, so the gate does not apply"
    assert toggle_engine(entries, VLLM) is True
    assert engines_needing_install(entries) == []
    assert vllm["note"] == "preconfigured server at http://localhost:8000"


def test_a_present_vllm_is_not_labelled_experimental():
    entries = build_engine_entries(vllm_support=EXPERIMENTAL, vllm_found=True)
    assert find_entry(entries, VLLM)["experimental"] is False
    assert "(experimental)" not in engine_summary_line(entries[1])


def test_an_absent_vllm_on_an_experimental_platform_stays_flagged():
    entries = build_engine_entries(vllm_support=EXPERIMENTAL, vllm_found=False)
    assert find_entry(entries, VLLM)["experimental"] is True


def test_python_headers_are_needed_whenever_vllm_is_selected():
    """The bug this replaces: gating on *installation* skipped an already-installed vLLM,
    which still cannot start without the headers."""
    installed = build_engine_entries(vllm_support=SUPPORTED, vllm_found=True, llamacpp_found=True)
    assert engines_needing_install(installed) == [], "nothing to install"
    assert needs_python_headers(installed, "/usr/include/python3.12/Python.h") is True


def test_python_headers_are_not_needed_when_vllm_is_unselected_or_present():
    entries = build_engine_entries(vllm_support=SUPPORTED)
    assert needs_python_headers(entries, "/usr/include/python3.12/Python.h") is False
    toggle_engine(entries, VLLM)
    assert needs_python_headers(entries, None) is False


def test_a_disabled_vllm_never_triggers_a_header_install():
    entries = build_engine_entries(vllm_support=UNSUPPORTED)
    find_entry(entries, VLLM)["checked"] = True
    assert needs_python_headers(entries, "/usr/include/python3.12/Python.h") is False
