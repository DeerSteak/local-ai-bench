"""Model-selection policy and terminal interaction shared by setup interfaces."""

import os
from pathlib import Path

from scripts.runtime import config, hardware
from scripts.setup.engine_selection import LLAMACPP, VLLM
from scripts.setup.model_inventory import (
    engine_fit_report, engine_fit_warnings, find_non_catalog_model_dirs,
    find_non_catalog_vllm_repos, fits_any_engine, format_engine_sizes,
)
from scripts.setup.setup_console import BOLD, CYAN, RESET, YELLOW, warn
from scripts.workloads.models import (
    EMBED_MODELS, IMAGE_MODELS, LLM_MODELS_LARGE, LLM_MODELS_MEDIUM,
    LLM_MODELS_SMALL, LLM_MODELS_XSMALL,
)


def toggle_all_models(entries: list[dict]) -> None:
    """Toggle every install entry without changing destructive cleanup choices."""
    model_entries = [entry for entry in entries if entry["kind"] != "cleanup"]
    checked = not all(entry["checked"] for entry in model_entries)
    for entry in model_entries:
        entry["checked"] = checked


def selected_cleanup_names(entries: list[dict], kind: str = "cleanup") -> list[str]:
    """Return names from explicitly selected cleanup entries of one kind."""
    return [
        name
        for entry in entries
        if entry["kind"] == kind and entry["checked"]
        for name in entry["item"]["directory_names"]
    ]


def save_hf_token(path: Path, token: str) -> None:
    """Save a Hugging Face token with private permissions where supported."""
    value = token.strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError("Hugging Face token must be a non-empty single line")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as token_file:
            token_file.write(value + "\n")
    finally:
        if os.name != "nt":
            path.chmod(0o600)


def additional_disk_space_needed(free_gb: float, download_gb: float) -> float:
    """Return the download shortfall in GB, or zero when it fits."""
    return max(0.0, download_gb - free_gb)


def qualification_model_selection(engine: str) -> tuple[list[dict], list[dict], list[dict]]:
    if engine not in {LLAMACPP, VLLM}:
        raise ValueError(f"unknown qualification engine: {engine}")
    images = [IMAGE_MODELS[0]] if engine == LLAMACPP else []
    return [LLM_MODELS_XSMALL[0]], images, [EMBED_MODELS[0]]


def select_models(memory_ceiling_gb=None, engines=(LLAMACPP,), *,
                  vllm_cache_home: Path, cancel):
    """Flat numbered model picker — see docs/setup.md's "What the setup scripts do".
    Returns (selected_llm, selected_images, selected_embed, cleanup_names)."""
    TIER_KEYS = {"xs": "xsmall", "s": "small", "m": "medium", "l": "large"}
    non_catalog_dirs = find_non_catalog_model_dirs(config.MODELS_DIR / "llamacpp")
    folder_word = "folder" if len(non_catalog_dirs) == 1 else "folders"
    cleanup_items = [{
        "label": f"Delete {len(non_catalog_dirs)} installed non-catalog model {folder_word}",
        "directory_names": [path.name for path in non_catalog_dirs],
    }] if non_catalog_dirs else []
    # One entry per cached repo, not one aggregate: the vLLM cache is shared with
    # anything else on the machine, so each deletion is chosen individually.
    vllm_cleanup_items = [{
        "label": f"Delete {entry['repo']}  (~{entry['size'] / 1e9:.1f} GB)",
        "directory_names": [entry["directory_name"]],
    } for entry in find_non_catalog_vllm_repos(vllm_cache_home)]
    groups = [
        ("LLM — Extra-small tier (<6B params)", LLM_MODELS_XSMALL, "llm",   "xs"),
        ("LLM — Small tier (≤20B params)",   LLM_MODELS_SMALL,  "llm",   "s"),
        ("LLM — Medium tier (26–35B params)", LLM_MODELS_MEDIUM, "llm",   "m"),
        ("LLM — Large tier (70B+ params)",   LLM_MODELS_LARGE,  "llm",   "l"),
        ("Embeddings models",                 EMBED_MODELS,      "embed", "emb"),
        ("Image generation models",           IMAGE_MODELS,      "image", "img"),
        ("Optional cleanup — downloaded llama.cpp models not in the catalog",
         cleanup_items, "cleanup", "clean"),
        ("Optional cleanup — cached vLLM weights not in the catalog",
         vllm_cleanup_items, "vllm_cleanup", "vclean"),
    ]
    group_keys = {group_key for _, items, _, group_key in groups if items}

    def hardware_fit_report(model):
        return engine_fit_report(model, engines, memory_ceiling_gb)

    entries = []
    for _, items, kind, group_key in groups:
        # LLM groups are already one-per-tier; image models carry their own
        # "tier" field (see models.py) so xs/s/m/l can reach them too.
        tier = TIER_KEYS.get(group_key) if kind == "llm" else None
        for m in items:
            entry_tier = tier if kind == "llm" else m.get("tier")
            if kind in ("cleanup", "vllm_cleanup"):
                fits = None
                checked = False
            elif kind == "llm":
                report = hardware_fit_report(m)
                fits = fits_any_engine(report)
                checked = fits is not False
            elif kind == "image":
                fits = hardware.image_model_fits(m["checkpoint"], m["short"], memory_ceiling_gb)
                checked = fits is not False
            else:
                fits = True
                checked = True
            entries.append({"item": m, "kind": kind, "group": group_key,
                            "tier": entry_tier, "checked": checked,
                            "fits": fits,
                            "report": hardware_fit_report(m) if kind in ("llm", "embed") else {}})

    def size_label(e, m, kind):
        if kind in ("cleanup", "vllm_cleanup"):
            return "  (unchecked by default)"
        if kind == "embed":
            return f"  ({format_engine_sizes(e['report'])})"
        if kind == "llm":
            label = f"  ({format_engine_sizes(e['report'])})"
            for warning in engine_fit_warnings(e["report"], memory_ceiling_gb):
                label += f"  {YELLOW}⚠ {warning}{RESET}"
            return label
        gb = hardware.CHECKPOINT_SIZES_GB.get(m["checkpoint"])
        label = f"  (~{gb:.1f} GB)" if gb else ""
        if kind == "image" and e["fits"] is False:
            needed = hardware.image_model_memory_requirement_gb(m["checkpoint"], m["short"])
            label += f"  {YELLOW}⚠ needs ~{needed:.1f} GB, ~{memory_ceiling_gb:.1f} GB available{RESET}"
        return label

    def render():
        header_note = ("all selected by default" if memory_ceiling_gb is None
                        else "selected by default, except models that likely won't fit in memory")
        print(f"  {BOLD}Choose which models to install ({header_note}){RESET}")
        n = 1
        for header, items, kind, group_key in groups:
            if not items:
                continue
            print(f"  {CYAN}{header} [{group_key}]{RESET}")
            for m in items:
                e = entries[n - 1]
                box = "[x]" if e["checked"] else "[ ]"
                print(f"    {box} {n:>2}  {m['label']}{size_label(e, m, kind)}")
                if kind in ("cleanup", "vllm_cleanup"):
                    for name in m["directory_names"]:
                        print(f"             {name!r}")
                n += 1
            print()

    render()
    print("  Type numbers to toggle (e.g. '2 4 7-9'), a size tier (xs/s/m/l — LLM")
    print("  and image checkpoints together) or 'emb'/'img' to toggle a whole")
    cleanup_hint = ", 'clean' to toggle non-catalog cleanup" if cleanup_items else ""
    print(f"  section{cleanup_hint}, 'a' to select/deselect")
    print("  all models (cleanup is excluded), 'q' to cancel,")
    while True:
        try:
            raw = input("  or press Enter to install everything checked above: ").strip().lower()
        except EOFError:
            print()
            break
        if raw == "":
            break
        if raw in ("q", "quit", "cancel"):
            cancel()
        if raw in ("a", "all"):
            toggle_all_models(entries)
            print()
            render()
            continue
        if raw in TIER_KEYS:
            matching = [e for e in entries if e["tier"] == TIER_KEYS[raw]]
            all_checked = all(e["checked"] for e in matching)
            for e in matching:
                e["checked"] = not all_checked
            print()
            render()
            continue
        if raw in group_keys:
            matching = [e for e in entries if e["group"] == raw]
            all_checked = all(e["checked"] for e in matching)
            for e in matching:
                e["checked"] = not all_checked
            print()
            render()
            continue

        nums = set()
        valid = True
        for tok in raw.replace(",", " ").split():
            if "-" in tok:
                a, b = tok.split("-", 1)
                if a.isdigit() and b.isdigit():
                    nums.update(range(int(a), int(b) + 1))
                else:
                    valid = False
                    break
            elif tok.isdigit():
                nums.add(int(tok))
            else:
                valid = False
                break
        if not valid or not nums or any(x < 1 or x > len(entries) for x in nums):
            warn("Couldn't parse that — use numbers/ranges like '2 4 7-9', 'a', or Enter to continue")
            continue

        for x in nums:
            entries[x - 1]["checked"] = not entries[x - 1]["checked"]
        print()
        render()

    selected_llm    = [e["item"] for e in entries if e["checked"] and e["kind"] == "llm"]
    selected_images = [e["item"] for e in entries if e["checked"] and e["kind"] == "image"]
    selected_embed  = [e["item"] for e in entries if e["checked"] and e["kind"] == "embed"]
    cleanup_names = selected_cleanup_names(entries)
    vllm_cleanup_names = selected_cleanup_names(entries, "vllm_cleanup")
    return selected_llm, selected_images, selected_embed, cleanup_names, vllm_cleanup_names
