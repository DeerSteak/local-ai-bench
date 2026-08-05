"""Pure engine-picker rules shared by both setup interfaces — see docs/setup.md's
"Choosing engines". Model selection is deliberately not per-engine: whatever models
you pick are downloaded for every selected engine.
"""

LLAMACPP = "llamacpp"
VLLM = "vllm"


def build_engine_entries(*, vllm_support=None, vllm_found: bool = False,
                          llamacpp_found: bool = False) -> list[dict]:
    """Initial picker state: llama.cpp on, vLLM off, and vLLM disabled where it can't run."""
    vllm_enabled = bool(vllm_support and vllm_support.status != "unsupported")
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
            "checked": False,
            "enabled": vllm_enabled,
            "installed": vllm_found,
            "note": (
                "already installed" if vllm_found
                else vllm_support.reason if vllm_support
                else "unavailable"
            ),
            "experimental": bool(vllm_support and vllm_support.status == "experimental"),
        },
    ]


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
    """Selected engines that aren't already present — mirrors llama.cpp's
    system-first policy, so nothing is reinstalled over a working copy."""
    return [
        entry["name"] for entry in entries
        if entry["checked"] and entry["enabled"] and not entry["installed"]
    ]


def engine_summary_line(entry: dict) -> str:
    """One picker row, e.g. '[x] llama.cpp — already installed'."""
    box = "x" if entry["checked"] else " "
    label = entry["label"]
    if entry.get("experimental") and entry["enabled"]:
        label += " (experimental)"
    if not entry["enabled"]:
        label += " (unavailable)"
    return f"[{box}] {label} — {entry['note']}"
