#!/usr/bin/env python3
"""Interactive launcher that translates selections into benchmark CLI flags."""

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts.results.result_store import atomic_write_json
from scripts.results.run_plan import RunPlan
from scripts.app.benchmark_options import (
    GUI_OPTION_FLAGS, PUBLIC_OPTION_SCHEMA, TG_TOKEN_CHOICES, gui_option_defaults,
    option_value_errors,
)

from scripts.runtime import config
from scripts.runtime.comfyui_installation import find_comfyui_installation
from scripts.app.benchmark import CONCURRENCY_TESTS, LLM_TESTS, engine_incompatible_tests
from scripts.runtime.engines import engine_names, get_engine
from scripts.setup.model_inventory import build_model_inventory
from scripts.workloads.models import EMBED_MODELS, IMAGE_MODELS, LLM_MODELS
from scripts.runtime.shared import Shared
from scripts.setup.setup_config import configured_comfyui_dir, load_setup_config
from scripts.stage_registry import STAGE_SPECS


TEST_DEFINITIONS = [
    (spec.key, "llama-bench (throughput + concurrency)" if spec.key == "llamabench"
     else spec.label, spec.ui_family, spec.default_enabled)
    for spec in STAGE_SPECS if spec.menu_visible
]
# One frontend toggle can cover several CLI tests. The CLI keeps them separate so
# `--tests llamabenchconc` alone still works; only the menus combine them.
TEST_ENTRY_TESTS = {"llamabench": ("llamabench", "llamabenchconc")}
# Every CLI test name a menu toggle can produce, including ones folded into another
# toggle, so the progress window can title their rows.
TEST_STAGE_LABELS = {
    spec.key: spec.label for spec in STAGE_SPECS
}
TEST_SHORTCUT_GROUPS = {
    "l": {"llm", "conv", "llamabench", "vllmbench"},
    "x": {"mcq", "math", "reasoning", "code", "tool"},
    "c": {"conc_tool", "conc_chat"},
    "e": {"emb"},
    "i": {"img"},
}
TIER_KEYS = {"xs": "xsmall", "s": "small", "m": "medium", "l": "large"}
LLM_BACKED_TESTS = set(LLM_TESTS + CONCURRENCY_TESTS + ["vllmbench"])
MAX_PROMPT_TOKEN_TESTS = {"llm", "conv", "llamabench", "llamabenchconc", "vllmbench"}
MAX_PROMPT_TOKEN_OPTIONS: list[int] = sorted(set(config.CONTEXT_LENGTHS) | set(config.LLAMABENCH_PP))
TG_TOKEN_TESTS = {"llamabench", "llamabenchconc"}
TG_TOKEN_OPTIONS: list[int] = list(TG_TOKEN_CHOICES)
FRONTEND_OPTION_INVENTORY = {
    flag: (spec.ui_status, spec.ui_location) for flag, spec in PUBLIC_OPTION_SCHEMA.items()
}
FRONTEND_OPTION_CLASSIFICATION = {
    flag: spec.classification for flag, spec in PUBLIC_OPTION_SCHEMA.items()
}
FRONTEND_CONTROL_BINDINGS = {
    **{flag: f"execution_setting:{key}" for key, flag in GUI_OPTION_FLAGS.items()},
    "--tests": "test_selector", "--engine": "engine_selector",
    "--llm-models": "llm_model_selector",
    "--embedding-models": "embedding_model_selector",
    "--image-models": "image_model_selector",
    "--max-prompt-tokens": "prompt_cap_selector",
    "--tg-tokens": "generation_size_selector",
}
FRONTEND_STATE_PATH = config.SCRIPT_DIR / ".benchmark_frontend_state.json"
FRONTEND_STATE_VERSION = 2
GUI_OPTION_DEFAULTS = gui_option_defaults()
FRONTEND_MODEL_FAMILIES = {
    "llm": {"llm", "custom"},
    "embedding": {"embedding"},
    "image": {"image"},
}


class FrontendCancelled(Exception):
    pass


def expand_selected_tests(values) -> list[str]:
    """Menu selections to CLI test names, preserving order and de-duplicating."""
    expanded = []
    for value in values:
        for name in TEST_ENTRY_TESTS.get(value, (value,)):
            if name not in expanded:
                expanded.append(name)
    return expanded


def collapse_tests_to_entries(tests) -> set[str]:
    """CLI test names back to the menu values that cover them, so a saved selection
    or preset written before two tests were combined still restores its toggle."""
    selected = set()
    for value, covered in TEST_ENTRY_TESTS.items():
        if any(name in tests for name in covered):
            selected.add(value)
    covered_names = {name for names in TEST_ENTRY_TESTS.values() for name in names}
    selected.update(name for name in tests if name not in covered_names)
    return selected


def frontend_option_gaps(inventory=None, bindings=None) -> list[str]:
    inventory = FRONTEND_OPTION_INVENTORY if inventory is None else inventory
    bindings = FRONTEND_CONTROL_BINDINGS if bindings is None else bindings
    missing = {flag for flag, (status, _) in inventory.items() if status == "missing"}
    missing.update(
        flag for flag, (status, _) in inventory.items()
        if status == "exposed" and flag not in bindings
    )
    return sorted(missing)


@dataclass
class MenuEntry:
    value: str
    label: str
    kind: str
    section: str
    checked: bool
    available: bool = True
    tier: str | None = None


def load_frontend_state(path: Path = FRONTEND_STATE_PATH) -> dict | None:
    try:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required_keys = {"version", "engine", "tests", "models", "max_prompt_tokens", "tg_tokens"}
    allowed_keys = required_keys | {"gui_options", "selected_preset"}
    if not isinstance(state, dict) or not required_keys.issubset(state) or not set(state).issubset(allowed_keys):
        return None
    if state["version"] != FRONTEND_STATE_VERSION or not isinstance(state["engine"], str):
        return None
    tests = state["tests"]
    models = state["models"]
    if (not isinstance(tests, list) or not tests
            or not all(isinstance(test, str) for test in tests)
            or len(tests) != len(set(tests))):
        return None
    if not isinstance(models, dict) or set(models) != set(FRONTEND_MODEL_FAMILIES):
        return None
    for values in models.values():
        if (not isinstance(values, list)
                or not all(isinstance(value, str) for value in values)
                or len(values) != len(set(values))):
            return None
    max_prompt_tokens = state["max_prompt_tokens"]
    if max_prompt_tokens is not None and not (isinstance(max_prompt_tokens, int) and max_prompt_tokens > 0):
        return None
    tg_tokens = state["tg_tokens"]
    if tg_tokens is not None and (
            not isinstance(tg_tokens, list) or not tg_tokens
            or not all(isinstance(v, int) and v > 0 for v in tg_tokens)
            or len(tg_tokens) != len(set(tg_tokens))):
        return None
    if "gui_options" in state:
        options = state["gui_options"]
        if isinstance(options, dict):
            missing = set(GUI_OPTION_DEFAULTS) - set(options)
            if missing <= {
                "offline", "gpu_split_mode", "retry_crashed_models", "llamacpp_no_repack",
            }:
                for key in missing:
                    options[key] = GUI_OPTION_DEFAULTS[key]
        if validate_gui_options(options):
            return None
    if "selected_preset" in state and (
            not isinstance(state["selected_preset"], str)
            or not state["selected_preset"].strip()):
        return None
    return state


def validate_gui_options(options: object) -> list[str]:
    if not isinstance(options, dict) or set(options) != set(GUI_OPTION_DEFAULTS):
        return ["GUI settings are incomplete."]
    errors = option_value_errors({GUI_OPTION_FLAGS[key]: value for key, value in options.items()})
    if any(not isinstance(options[key], bool) for key in (
            "cpu_only", "force_all", "retry_crashed_models", "offline",
            "llamacpp_no_repack")):
        errors.append("Execution mode settings must be true or false.")
    if not isinstance(options["out"], str) or not isinstance(options["comfyui"], str):
        errors.append("Output and ComfyUI paths must be text.")
    return errors


def save_frontend_state(state: dict, path: Path = FRONTEND_STATE_PATH) -> bool:
    try:
        atomic_write_json(Path(path), state)
        return True
    except (OSError, TypeError, ValueError):
        return False


def build_frontend_state(engine_name: str, tests: list[str],
                         entries: list[MenuEntry],
                         max_prompt_tokens: int | None = None,
                         tg_tokens: list[int] | None = None,
                         gui_options: dict | None = None,
                         selected_preset: str | None = None) -> dict:
    selected = [entry for entry in entries if entry.checked]
    state = {
        "version": FRONTEND_STATE_VERSION,
        "engine": engine_name,
        "tests": list(tests),
        "models": {
            family: [
                entry.value for entry in selected if entry.kind in kinds
            ]
            for family, kinds in FRONTEND_MODEL_FAMILIES.items()
        },
        "max_prompt_tokens": max_prompt_tokens,
        "tg_tokens": list(tg_tokens) if tg_tokens is not None else None,
    }
    if gui_options is not None:
        state["gui_options"] = dict(gui_options)
    if selected_preset is not None:
        state["selected_preset"] = selected_preset
    return state


def frontend_state_from_run_plan(plan: RunPlan, gui_options: dict | None = None) -> dict:
    effective = plan.effective_config
    if effective.get("sample_size") is not None:
        raise ValueError("Sampled developer runs cannot be represented in the benchmark GUI.")
    options = dict(GUI_OPTION_DEFAULTS if gui_options is None else gui_options)
    option_mapping = {
        "runs": "runs", "warmup_runs": "warmup", "run_timeout_seconds": "timeout",
        "accuracy_timeout_seconds": "acc_timeout",
        "accuracy_token_budget": "acc_token_budget", "cpu_only": "cpu_only",
        "gpu_split_mode": "gpu_split_mode", "force_all": "force_all",
        "llamacpp_no_repack": "llamacpp_no_repack",
        "retry_crashed_models": "retry_crashed_models", "offline": "offline",
    }
    for plan_key, option_key in option_mapping.items():
        if plan_key in effective:
            options[option_key] = effective[plan_key]
    models = plan.models
    llm_models = []
    for family in ("llm", "concurrency"):
        for model in models.get(family, []):
            tag = model.get("tag")
            if not tag:
                raise ValueError(f"Run plan contains an unidentified {family} model.")
            if tag not in llm_models:
                llm_models.append(tag)
    embedding_models = [model.get("tag") for model in models.get("embeddings", [])]
    image_models = [model.get("short") or model.get("tag") for model in models.get("images", [])]
    if any(model is None for model in embedding_models + image_models):
        raise ValueError("Run plan contains an unidentified embedding or image model.")
    state = {
        "version": FRONTEND_STATE_VERSION,
        "engine": plan.engine_name,
        "tests": list(plan.tests),
        "models": {
            "llm": llm_models,
            "embedding": embedding_models,
            "image": image_models,
        },
        "max_prompt_tokens": effective.get("max_prompt_tokens"),
        "tg_tokens": effective.get("llamabench_tg"),
        "gui_options": options,
    }
    if validate_gui_options(options):
        raise ValueError("Run plan contains execution settings unsupported by the benchmark GUI.")
    return state


def frontend_state_availability_errors(state: dict, engines: list[str],
                                       test_entries: list[MenuEntry],
                                       model_entries: list[MenuEntry]) -> list[str]:
    errors = []
    # Portable presets carry no engine — an absent key keeps the live selection.
    if "engine" in state:
        selected_engines = parse_engine_selection(state["engine"])
        missing_engines = [name for name in selected_engines if name not in engines]
        if not selected_engines or missing_engines:
            errors.append("Engine is not installed: "
                          + (", ".join(missing_engines) or state["engine"]))
    available_tests = {entry.value for entry in test_entries if entry.available}
    missing_tests = sorted(set(state["tests"]) - available_tests)
    if missing_tests:
        errors.append(f"Tests are unavailable: {', '.join(missing_tests)}")
    available_models = {entry.value for entry in model_entries if entry.available}
    selected_models = set().union(*map(set, state["models"].values()))
    missing_models = sorted(selected_models - available_models)
    if missing_models:
        errors.append(f"Models are not installed: {', '.join(missing_models)}")
    return errors


def apply_saved_test_selection(entries: list[MenuEntry], state: dict | None) -> bool:
    if state is None:
        return False
    saved = collapse_tests_to_entries(state["tests"])
    if not any(entry.available and entry.value in saved for entry in entries):
        return False
    for entry in entries:
        entry.checked = entry.available and entry.value in saved
    return True


def apply_saved_model_selection(entries: list[MenuEntry], state: dict | None) -> None:
    if state is None:
        return
    for family, kinds in FRONTEND_MODEL_FAMILIES.items():
        family_entries = [entry for entry in entries if entry.kind in kinds]
        saved = set(state["models"][family])
        if not any(entry.value in saved for entry in family_entries):
            continue
        for entry in family_entries:
            entry.checked = entry.value in saved


def read_choice(prompt: str, input_fn, output_fn) -> str:
    output_fn(prompt)
    try:
        return input_fn().strip()
    except (EOFError, KeyboardInterrupt) as exc:
        raise FrontendCancelled from exc


def parse_toggle_numbers(raw: str, count: int) -> set[int]:
    numbers = set()
    for token in raw.replace(",", " ").split():
        if "-" in token:
            start, end = token.split("-", 1)
            if not start.isdigit() or not end.isdigit() or int(start) > int(end):
                raise ValueError
            numbers.update(range(int(start), int(end) + 1))
        elif token.isdigit():
            numbers.add(int(token))
        else:
            raise ValueError
    if not numbers or any(number < 1 or number > count for number in numbers):
        raise ValueError
    return numbers


def toggle_group(entries: list[MenuEntry], predicate) -> bool:
    matching = [entry for entry in entries if predicate(entry)]
    if not matching:
        return False
    new_state = not all(entry.checked for entry in matching)
    for entry in matching:
        entry.checked = new_state
    return True


def apply_test_shortcut(entries: list[MenuEntry], shortcut: str) -> bool:
    if shortcut == "a":
        available = [entry for entry in entries if entry.available]
        if not available:
            return False
        for entry in available:
            entry.checked = True
        return True
    values = TEST_SHORTCUT_GROUPS.get(shortcut)
    if values is None:
        return False
    return toggle_group(
        entries, lambda entry: entry.available and entry.value in values,
    )


def choose_engine(available: list[str], input_fn, output_fn, clear_fn=lambda: None,
                  preferred: str | None = None) -> str:
    if len(available) == 1:
        output_fn(f"Engine: {available[0]}")
        return available[0]

    selected = available.index(preferred) if preferred in available else 0
    feedback = None
    redraw = False
    while True:
        if redraw:
            clear_fn()
        redraw = True
        output_fn("Choose one inference engine (`--engine all` remains CLI-only):")
        for index, name in enumerate(available, 1):
            box = "[x]" if index - 1 == selected else "[ ]"
            output_fn(f"  {box} {index:>2}  {name}")
        if feedback:
            output_fn(feedback)
        raw = read_choice("Enter a number, or press Enter to accept:", input_fn, output_fn).lower()
        if raw in ("q", "quit", "cancel"):
            raise FrontendCancelled
        if raw == "":
            return available[selected]
        if raw.isdigit() and 1 <= int(raw) <= len(available):
            selected = int(raw) - 1
            return available[selected]
        feedback = "Couldn't parse that engine selection."


def build_test_entries(inventory: dict[str, list[dict]]) -> list[MenuEntry]:
    availability = {
        "llm": bool(inventory["llm"] or inventory["custom"]),
        "embedding": bool(inventory["embedding"]),
        "image": bool(inventory["image"]),
    }
    return [
        MenuEntry(
            value=name,
            label=label,
            kind=family,
            section="Tests",
            checked=default_checked and availability[family],
            available=availability[family],
        )
        for name, label, family, default_checked in TEST_DEFINITIONS
    ]


def render_test_menu(entries: list[MenuEntry], output_fn,
                     selection_note: str | None = None) -> None:
    output_fn("Choose benchmark tests:")
    if selection_note:
        output_fn(selection_note)
    for index, entry in enumerate(entries, 1):
        box = "[x]" if entry.checked else "[ ]"
        unavailable = "  (no installed model available)" if not entry.available else ""
        output_fn(f"  {box} {index:>2}  {entry.label}{unavailable}")
    output_fn("Shortcuts: a all | l LLM | x accuracy | c concurrency | e embeddings | i images")
    output_fn("Numbers and ranges toggle individual tests; group shortcuts toggle their available tests.")


def choose_tests(entries: list[MenuEntry], input_fn, output_fn,
                 clear_fn=lambda: None, selection_note: str | None = None) -> list[str]:
    feedback = None
    redraw = False
    while True:
        if redraw:
            clear_fn()
        redraw = True
        render_test_menu(entries, output_fn, selection_note)
        if feedback:
            output_fn(feedback)
        feedback = None
        raw = read_choice(
            "Toggle tests with numbers/ranges, press Enter to continue, or q to cancel:",
            input_fn, output_fn,
        ).lower()
        if raw in ("q", "quit", "cancel"):
            raise FrontendCancelled
        if raw == "":
            selected = [entry.value for entry in entries if entry.checked]
            if selected:
                return selected
            feedback = "Select at least one available test."
            continue
        if raw in {"a", *TEST_SHORTCUT_GROUPS}:
            if not apply_test_shortcut(entries, raw):
                feedback = "No tests in that group are available for the installed models."
            continue
        try:
            numbers = parse_toggle_numbers(raw, len(entries))
        except ValueError:
            feedback = "Couldn't parse that selection; use numbers/ranges or a shortcut from the legend."
            continue
        unavailable = [number for number in numbers if not entries[number - 1].available]
        if unavailable:
            feedback = "A test with no applicable installed model cannot be selected."
            continue
        for number in numbers:
            entries[number - 1].checked = not entries[number - 1].checked


def choose_max_prompt_tokens(input_fn, output_fn, clear_fn=lambda: None,
                             options: list[int] | None = None,
                             preferred: int | None = None) -> int | None:
    options = options if options is not None else MAX_PROMPT_TOKEN_OPTIONS
    feedback = None
    redraw = False
    while True:
        if redraw:
            clear_fn()
        redraw = True
        output_fn(
            "Cap the max prompt-processing size tested (applies to Single-shot LLM, "
            "conversation, llama-bench throughput, and llama-bench concurrency):"
        )
        output_fn(f"   0  No cap (test every configured depth){' (restored)' if preferred is None else ''}")
        for index, value in enumerate(options, 1):
            marker = " (restored)" if value == preferred else ""
            output_fn(f"  {index:>2}  {value}{marker}")
        if feedback:
            output_fn(feedback)
        feedback = None
        raw = read_choice(
            "Enter a number, or press Enter to accept:", input_fn, output_fn,
        ).strip().lower()
        if raw in ("q", "quit", "cancel"):
            raise FrontendCancelled
        if raw == "":
            return preferred
        if raw == "0":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        feedback = "Couldn't parse that selection; enter a listed number, or press Enter to accept."


def choose_tg_tokens(input_fn, output_fn, clear_fn=lambda: None,
                     options: list[int] | None = None,
                     default_checked: set[int] | None = None) -> list[int]:
    options = options if options is not None else TG_TOKEN_OPTIONS
    default_checked = default_checked if default_checked is not None else set(config.LLAMABENCH_TG)
    checked = {value: value in default_checked for value in options}
    feedback = None
    redraw = False
    while True:
        if redraw:
            clear_fn()
        redraw = True
        output_fn("Choose generation (tg) sizes to sweep for llama-bench / llama-bench concurrency:")
        for index, value in enumerate(options, 1):
            box = "[x]" if checked[value] else "[ ]"
            output_fn(f"  {box} {index:>2}  {value}")
        if feedback:
            output_fn(feedback)
        feedback = None
        raw = read_choice(
            "Toggle numbers with spaces/ranges, or press Enter to continue:", input_fn, output_fn,
        ).lower()
        if raw in ("q", "quit", "cancel"):
            raise FrontendCancelled
        if raw == "":
            selected = [value for value in options if checked[value]]
            if selected:
                return selected
            feedback = "Select at least one generation size."
            continue
        try:
            numbers = parse_toggle_numbers(raw, len(options))
        except ValueError:
            feedback = "Couldn't parse that selection; use numbers/ranges such as `1 3`."
            continue
        for number in numbers:
            value = options[number - 1]
            checked[value] = not checked[value]


def parse_engine_selection(value: str | None) -> list[str]:
    """Engine names from a --engine value. One name, or several comma-separated;
    "all" is expanded later, against the registry."""
    return [name.strip() for name in str(value or "").split(",") if name.strip()]


def format_engine_selection(names) -> str:
    return ",".join(names)


def merge_model_inventories(inventories: dict[str, dict]) -> tuple[dict, dict[str, set[str]]]:
    """Union of every engine's inventory, plus which engines hold each model. Lets a
    frontend list all models once and enable only those the selected engine can run."""
    merged: dict[str, list[dict]] = {}
    owners: dict[str, set[str]] = {}
    for engine_name, inventory in inventories.items():
        for section, models in inventory.items():
            seen = {entry.get("tag") or entry.get("short") for entry in merged.setdefault(section, [])}
            for model in models:
                key = model.get("tag") or model.get("short")
                owners.setdefault(key, set()).add(engine_name)
                if key not in seen:
                    merged[section].append(model)
                    seen.add(key)
    for model in merged.get("custom", []):
        key = model.get("tag")
        model["engines"] = sorted(owners.get(key, ())) if isinstance(key, str) else []
    return merged, owners


def models_runnable_by(entries, engine_name: str, owners: dict[str, set[str]]) -> dict[str, bool]:
    """Which menu entries the given engine can actually run. Image models are engine-
    independent (ComfyUI), so they stay available whatever engine is selected."""
    return {
        entry.value: entry.kind == "image" or engine_name in owners.get(entry.value, set())
        for entry in entries
    }


def build_model_entries(inventory: dict[str, list[dict]], tests: list[str]) -> list[MenuEntry]:
    entries = []
    if any(test in LLM_BACKED_TESTS for test in tests):
        for model in inventory["llm"]:
            tier = model["tier"]
            entries.append(MenuEntry(
                model["tag"], model["label"], "llm", f"LLM — {tier}",
                checked=tier != "large", tier=tier,
            ))
        for model in inventory["custom"]:
            engines = ", ".join(model.get("engines", ()))
            section = f"Custom LLM — {engines}" if engines else "Custom LLM"
            entries.append(MenuEntry(
                model["tag"], model["label"], "custom", section,
                checked=False,
            ))
    if "emb" in tests:
        for model in inventory["embedding"]:
            entries.append(MenuEntry(
                model["tag"], model["label"], "embedding", "Embeddings",
                checked=True,
            ))
    if "img" in tests:
        for model in inventory["image"]:
            tier = model["tier"]
            entries.append(MenuEntry(
                model["short"], model["label"], "image", f"Images — {tier}",
                checked=tier != "large", tier=tier,
            ))
    return entries


def missing_catalog_hint(inventory: dict[str, list[dict]], system: str) -> str | None:
    missing = {
        "LLM": len(LLM_MODELS) - len(inventory["llm"]),
        "embedding": len(EMBED_MODELS) - len(inventory["embedding"]),
        "image": len(IMAGE_MODELS) - len(inventory["image"]),
    }
    parts = [f"{count} {label}" for label, count in missing.items() if count]
    if not parts:
        return None
    setup_command = "setup.bat" if system == "Windows" else "bash setup.sh"
    return (
        f"Not shown because they are not installed: {', '.join(parts)} models. "
        f"Run `{setup_command}` to add catalog models."
    )


def render_model_menu(entries: list[MenuEntry], hint: str | None, output_fn,
                      selection_note: str | None = None) -> None:
    output_fn("Choose installed models:")
    if selection_note:
        output_fn(selection_note)
    previous_section = None
    for index, entry in enumerate(entries, 1):
        if entry.section != previous_section:
            output_fn(f"  {entry.section}:")
            previous_section = entry.section
        box = "[x]" if entry.checked else "[ ]"
        output_fn(f"    {box} {index:>2}  {entry.label}")
    output_fn(
        "Tier keys xs/s/m/l toggle catalog LLM and image models together; "
        "custom and embedding models use their own toggles."
    )
    if hint:
        output_fn(hint)


def model_selection_error(entries: list[MenuEntry], tests: list[str]) -> str | None:
    selected = {entry.kind for entry in entries if entry.checked}
    if any(test in LLM_BACKED_TESTS for test in tests) and not ({"llm", "custom"} & selected):
        return "Select at least one LLM model for the selected LLM-backed tests."
    if "emb" in tests and "embedding" not in selected:
        return "Select at least one embedding model."
    if "img" in tests and "image" not in selected:
        return "Select at least one image model."
    return None


def choose_models(entries: list[MenuEntry], tests: list[str], hint: str | None,
                  input_fn, output_fn, clear_fn=lambda: None,
                  selection_note: str | None = None) -> list[MenuEntry]:
    feedback = None
    redraw = False
    while True:
        if redraw:
            clear_fn()
        redraw = True
        render_model_menu(entries, hint, output_fn, selection_note)
        if feedback:
            output_fn(feedback)
        feedback = None
        raw = read_choice(
            "Toggle numbers/ranges, xs/s/m/l, custom, or emb; press Enter to continue:",
            input_fn, output_fn,
        ).lower()
        if raw in ("q", "quit", "cancel"):
            raise FrontendCancelled
        if raw == "":
            error = model_selection_error(entries, tests)
            if error:
                feedback = error
                continue
            return entries
        if raw in TIER_KEYS:
            tier = TIER_KEYS[raw]
            if not toggle_group(
                entries, lambda entry: entry.kind in ("llm", "image") and entry.tier == tier,
            ):
                feedback = f"No installed catalog LLM/image models are available in tier {tier}."
            continue
        if raw in ("custom", "emb"):
            kind = "custom" if raw == "custom" else "embedding"
            if not toggle_group(entries, lambda entry: entry.kind == kind):
                feedback = f"No installed {kind} models are available in this selection."
            continue
        try:
            numbers = parse_toggle_numbers(raw, len(entries))
        except ValueError:
            feedback = "Couldn't parse that selection; use numbers/ranges or a documented group key."
            continue
        for number in numbers:
            entries[number - 1].checked = not entries[number - 1].checked


def build_benchmark_command(engine_name: str, comfyui_dir: Path, tests: list[str],
                            entries: list[MenuEntry], python_executable: str = sys.executable,
                            benchmark_path: Path | None = None,
                            max_prompt_tokens: int | None = None,
                            tg_tokens: list[int] | None = None,
                            gui_options: dict | None = None) -> list[str]:
    benchmark_target = [str(benchmark_path)] if benchmark_path else ["-m", "scripts.app.benchmark"]
    command = [python_executable, *benchmark_target, "--engine", engine_name]
    # Sending a path for a ComfyUI that was never installed fails validation, and without
    # image tests there is nothing for it to point at anyway.
    if "img" in tests:
        command.extend(["--comfyui", str(comfyui_dir)])
    command.extend(["--tests", *tests])
    if max_prompt_tokens is not None:
        command.extend(["--max-prompt-tokens", str(max_prompt_tokens)])
    if tg_tokens is not None:
        command.extend(["--tg-tokens", *[str(v) for v in tg_tokens]])
    if gui_options is not None:
        command.extend(["--warmup", str(gui_options["warmup"])])
        command.extend(["--runs", str(gui_options["runs"])])
        command.extend(["--timeout", str(gui_options["timeout"])])
        command.extend(["--acc-timeout", str(gui_options["acc_timeout"])])
        command.extend(["--acc-token-budget", str(gui_options["acc_token_budget"])])
        command.extend(["--gpu-split-mode", gui_options["gpu_split_mode"]])
        if gui_options["cpu_only"]:
            command.append("--cpu-only")
        if gui_options["llamacpp_no_repack"]:
            command.append("--llamacpp-no-repack")
        if gui_options["force_all"]:
            command.append("--force-all")
        if gui_options["retry_crashed_models"]:
            command.append("--retry-crashed-models")
        if gui_options["offline"]:
            command.append("--offline")
        if gui_options["out"]:
            command.extend(["--out", gui_options["out"]])
        if gui_options["comfyui"] and "--comfyui" in command:
            command[command.index("--comfyui") + 1] = gui_options["comfyui"]
    selected = [entry for entry in entries if entry.checked]
    if any(test in LLM_BACKED_TESTS for test in tests):
        command.extend([
            "--llm-models",
            *[entry.value for entry in selected if entry.kind in ("llm", "custom")],
        ])
    if "emb" in tests:
        command.extend([
            "--embedding-models",
            *[entry.value for entry in selected if entry.kind == "embedding"],
        ])
    if "img" in tests:
        command.extend([
            "--image-models",
            *[entry.value for entry in selected if entry.kind == "image"],
        ])
    return command


def render_summary(engine_name: str, comfyui_dir: Path, tests: list[str],
                   entries: list[MenuEntry], output_fn,
                   max_prompt_tokens: int | None = None,
                   tg_tokens: list[int] | None = None) -> None:
    output_fn("Benchmark selection:")
    output_fn(f"  Engine: {engine_name}")
    output_fn(f"  ComfyUI: {comfyui_dir}")
    output_fn(f"  Tests: {', '.join(tests)}")
    if max_prompt_tokens is not None:
        output_fn(f"  Max prompt-processing size: {max_prompt_tokens} tokens")
    if tg_tokens is not None:
        output_fn(f"  Generation (tg) sizes: {', '.join(str(v) for v in tg_tokens)}")
    for label, kinds in (
        ("LLM models", {"llm", "custom"}),
        ("Embedding models", {"embedding"}),
        ("Image models", {"image"}),
    ):
        names = [entry.label for entry in entries if entry.checked and entry.kind in kinds]
        if names:
            output_fn(f"  {label}: {', '.join(names)}")


def run_frontend(input_fn=input, output_fn=Shared.plain_output, process_runner=None,
                 engine_names_fn=engine_names, engine_factory=get_engine,
                 inventory_builder=build_model_inventory, system: str | None = None,
                 python_executable: str = sys.executable,
                 benchmark_path: Path | None = None,
                 clear_fn=Shared.clear_terminal,
                 state_path: Path | None = None) -> int:
    process_runner = process_runner or subprocess.run
    system = system or platform.system()
    state_path = state_path or FRONTEND_STATE_PATH
    saved_state = load_frontend_state(state_path)
    selection_note = None
    if saved_state:
        selection_note = (
            f"Restored saved selections from `{Path(state_path).name}`; "
            "delete this file to reset them."
        )
    try:
        clear_fn()
        output_fn("Local AI Bench interactive launcher")
        available_engines = engine_names_fn()
        selected_engine = choose_engine(
            available_engines, input_fn, output_fn, clear_fn,
            preferred=saved_state["engine"] if saved_state else None,
        )
        setup_config = load_setup_config(config.SETUP_CONFIG_PATH)
        comfyui_dir = find_comfyui_installation(
            saved_path=configured_comfyui_dir(setup_config),
            managed_dir=config.COMFYUI_DIR,
        ) or config.COMFYUI_DIR
        inventory = inventory_builder(engine_factory(selected_engine), config.COMFYUI_MODELS_DIR)
        test_entries = build_test_entries(inventory)
        apply_saved_test_selection(test_entries, saved_state)
        if not any(entry.available for entry in test_entries):
            output_fn("No installed benchmark models were found. Run setup to add catalog models.")
            return 1

        if len(available_engines) > 1:
            clear_fn()
        tests = expand_selected_tests(choose_tests(
            test_entries, input_fn, output_fn, clear_fn,
            selection_note=selection_note,
        ))
        model_entries = build_model_entries(inventory, tests)
        apply_saved_model_selection(model_entries, saved_state)
        hint = missing_catalog_hint(inventory, system)
        clear_fn()
        choose_models(
            model_entries, tests, hint, input_fn, output_fn, clear_fn,
            selection_note=selection_note,
        )
        max_prompt_tokens = None
        if MAX_PROMPT_TOKEN_TESTS & set(tests):
            clear_fn()
            max_prompt_tokens = choose_max_prompt_tokens(
                input_fn, output_fn, clear_fn,
                preferred=saved_state["max_prompt_tokens"] if saved_state else None,
            )
        tg_tokens = None
        if TG_TOKEN_TESTS & set(tests):
            clear_fn()
            saved_tg_tokens = saved_state["tg_tokens"] if saved_state else None
            tg_tokens = choose_tg_tokens(
                input_fn, output_fn, clear_fn,
                default_checked=set(saved_tg_tokens) if saved_tg_tokens else None,
            )
        render_summary(
            selected_engine, comfyui_dir, tests, model_entries, output_fn,
            max_prompt_tokens=max_prompt_tokens, tg_tokens=tg_tokens,
        )
        confirmation = read_choice("Start this benchmark? [Y/n]", input_fn, output_fn).lower()
        if confirmation not in ("", "y", "yes"):
            raise FrontendCancelled
        state = build_frontend_state(
            selected_engine, tests, model_entries,
            max_prompt_tokens=max_prompt_tokens, tg_tokens=tg_tokens,
            gui_options=saved_state.get("gui_options") if saved_state else None,
        )
        if not save_frontend_state(state, state_path):
            output_fn("Could not save this launcher selection; continuing without persistence.")
        command = build_benchmark_command(
            selected_engine, comfyui_dir, tests, model_entries,
            python_executable=python_executable, benchmark_path=benchmark_path,
            max_prompt_tokens=max_prompt_tokens, tg_tokens=tg_tokens,
        )
        output_fn("Launching benchmark.py with the confirmed selection.")
        result = process_runner(command)
        return result if isinstance(result, int) else result.returncode
    except FrontendCancelled:
        output_fn("Benchmark selection cancelled.")
        return 0


def main():  # pragma: no cover — real terminal/subprocess entrypoint
    raise SystemExit(run_frontend())


if __name__ == "__main__":
    main()
