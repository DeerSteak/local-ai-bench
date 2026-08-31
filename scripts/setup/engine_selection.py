"""Pure engine-picker rules shared by both setup interfaces — see docs/setup.md."""

from scripts.setup.setup_console import section, warn
from scripts.runtime.engine_identity import LLAMACPP, LLAMACPP_VULKAN, VLLM


def build_engine_entries(*, vllm_support=None, vllm_found: bool = False,
                          llamacpp_found: bool = False,
                          llamacpp_vulkan_supported: bool = False,
                          llamacpp_vulkan_found: bool = False,
                          llamacpp_vulkan_note: str | None = None,
                          vllm_note: str | None = None) -> list[dict]:
    """Initial picker state. An already-installed engine starts checked, so setup keeps
    maintaining it; only an engine that would need installing starts unchecked."""
    vllm_enabled = bool(vllm_found) or bool(
        vllm_support and vllm_support.status != "unsupported")
    return [
        {
            "name": LLAMACPP,
            "label": "llama.cpp",
            "checked": True,
            "enabled": True,
            "installed": llamacpp_found,
            "note": "already installed" if llamacpp_found else "will be installed",
        },
        {
            "name": VLLM,
            "label": "vLLM",
            "checked": bool(vllm_found),
            "enabled": vllm_enabled,
            "installed": vllm_found,
            "note": (
                vllm_note or "already installed" if vllm_found
                else vllm_support.reason if vllm_support
                else "unavailable"
            ),
            "experimental": not vllm_found and bool(
                vllm_support and vllm_support.status == "experimental"),
        },
        {
            "name": LLAMACPP_VULKAN,
            "label": "llama.cpp Vulkan",
            "checked": bool(llamacpp_vulkan_found),
            "enabled": llamacpp_vulkan_supported or llamacpp_vulkan_found,
            "installed": llamacpp_vulkan_found,
            "note": (
                "already installed" if llamacpp_vulkan_found
                else llamacpp_vulkan_note or "unavailable on this platform"
            ),
        },
    ]


def model_engine_names(engines: list[str]) -> list[str]:
    """Deduplicate runtime choices that consume the same model family."""
    names = []
    for engine in engines:
        from scripts.runtime.engine_identity import engine_family
        model_engine = engine_family(engine)
        if model_engine not in names:
            names.append(model_engine)
    return names


def llamacpp_vulkan_setup_state(platform_name: str, machine: str, *,
                                 runtime_present: bool, backend: str | None,
                                 toolset_ready: bool) -> dict:
    windows_x64 = platform_name == "Windows" and machine.lower() in {"amd64", "x86_64"}
    supported = platform_name == "Linux" or windows_x64
    found = runtime_present and backend == "vulkan" and toolset_ready
    note = (
        "will be built from source" if platform_name == "Linux"
        else "will use the official Windows Vulkan package" if windows_x64
        else "requires Linux or 64-bit Windows"
    )
    problem = None
    if runtime_present and backend != "vulkan":
        problem = "wrong_backend"
    elif runtime_present and not toolset_ready:
        problem = "incomplete_toolset"
    return {"supported": supported, "found": found, "note": note, "problem": problem}


def find_entry(entries: list[dict], name: str) -> dict | None:
    return next((entry for entry in entries if entry["name"] == name), None)


def toggle_engine(entries: list[dict], name: str) -> bool:
    """Toggle one engine. Refuses to enable a disabled engine or clear the last one."""
    entry = find_entry(entries, name)
    if entry is None or not entry["enabled"]:
        return False
    if entry["checked"] and len(selected_engine_names(entries)) == 1:
        return False
    entry["checked"] = not entry["checked"]
    return True


def selected_engine_names(entries: list[dict]) -> list[str]:
    return [entry["name"] for entry in entries if entry["checked"] and entry["enabled"]]


def engines_needing_install(entries: list[dict]) -> list[str]:
    """Selected engines that aren't already present."""
    return [
        entry["name"] for entry in entries
        if entry["checked"] and entry["enabled"] and not entry["installed"]
    ]


def qualification_engines_needing_install(entries: list[dict], engine: str | None,
                                           *, vllm_bench_found: bool,
                                           vllm_runtime_ready: bool = True,
                                           llamacpp_runtime_ready: bool = True) -> list[str]:
    pending = engines_needing_install(entries)
    if engine == LLAMACPP and not llamacpp_runtime_ready and LLAMACPP not in pending:
        pending.append(LLAMACPP)
    if engine == VLLM and (not vllm_bench_found or not vllm_runtime_ready) \
            and VLLM not in pending:
        pending.append(VLLM)
    return pending


def qualification_setup_failed(engine: str | None, issues: list[str]) -> bool:
    return engine is not None and bool(issues)


def apply_engine_preset(entries: list[dict], engine: str) -> list[dict]:
    target = find_entry(entries, engine)
    if target is None or not target["enabled"]:
        reason = target["note"] if target else "unknown engine"
        raise ValueError(f"{engine} is unavailable: {reason}")
    for entry in entries:
        entry["checked"] = entry is target
    return entries


def needs_python_headers(entries: list[dict], missing_header: str | None) -> bool:
    """True when vLLM is selected and its Python headers are absent. Selection, not
    installation: an already-installed vLLM needs them just as much at run time."""
    return bool(missing_header) and VLLM in selected_engine_names(entries)


def engine_summary_line(entry: dict) -> str:
    """One picker row, e.g. '[x] llama.cpp — already installed'."""
    box = "x" if entry["checked"] else " "
    label = entry["label"]
    if entry.get("experimental") and entry["enabled"]:
        label += " (experimental)"
    if not entry["enabled"]:
        label += " (unavailable)"
    return f"[{box}] {label} — {entry['note']}"


def select_engines(entries: list[dict], *, input_fn=input) -> list[dict]:
    section("Engines")
    print("  Models selected later are downloaded for every checked engine.\n")
    while True:
        for index, entry in enumerate(entries, start=1):
            print(f"   {index}. {engine_summary_line(entry)}")
        choice = input_fn("\n  Number to toggle, Enter to continue, q to cancel: ").strip().lower()
        if not choice:
            return entries
        if choice == "q":
            print("\n  Setup cancelled — nothing was installed.\n")
            raise SystemExit(0)
        if choice.isdigit() and 1 <= int(choice) <= len(entries):
            target = entries[int(choice) - 1]
            if not toggle_engine(entries, target["name"]):
                reason = target["note"] if not target["enabled"] else "At least one engine must stay selected"
                warn(reason)
        else:
            warn("Enter a listed number, Enter, or q")
        print()
