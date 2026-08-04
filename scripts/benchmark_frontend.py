#!/usr/bin/env python3
"""Interactive launcher that translates selections into benchmark CLI flags."""

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from result_store import atomic_write_json

import config
from comfyui_installation import find_comfyui_installation
from benchmark import CONCURRENCY_TESTS, LLM_TESTS
from engines import engine_names, get_engine
from model_inventory import build_model_inventory
from models import EMBED_MODELS, IMAGE_MODELS, LLM_MODELS
from shared import Shared
from setup_config import configured_comfyui_dir, load_setup_config


TEST_DEFINITIONS = [
    ("llm", "Single-shot LLM", "llm", True),
    ("conv", "Conversation", "llm", True),
    ("llamabench", "llama-bench throughput", "llm", False),
    ("emb", "Embeddings", "embedding", True),
    ("mcq", "MCQ accuracy", "llm", False),
    ("math", "Math accuracy", "llm", False),
    ("reasoning", "Reasoning accuracy", "llm", False),
    ("code", "Code accuracy", "llm", False),
    ("tool", "Tool accuracy", "llm", False),
    ("conc_tool", "Tool concurrency", "llm", False),
    ("conc_chat", "Chat concurrency", "llm", False),
    ("llamabenchconc", "llama-bench concurrency", "llm", False),
    ("img", "Image generation", "image", True),
]
TEST_SHORTCUT_GROUPS = {
    "l": {"llm", "conv", "llamabench"},
    "x": {"mcq", "math", "reasoning", "code", "tool"},
    "c": {"conc_tool", "conc_chat", "llamabenchconc"},
    "e": {"emb"},
    "i": {"img"},
}
TIER_KEYS = {"xs": "xsmall", "s": "small", "m": "medium", "l": "large"}
LLM_BACKED_TESTS = set(LLM_TESTS + CONCURRENCY_TESTS)
MAX_PROMPT_TOKEN_TESTS = {"llm", "conv", "llamabench", "llamabenchconc"}
MAX_PROMPT_TOKEN_OPTIONS = sorted(set(config.CONTEXT_LENGTHS) | set(config.LLAMABENCH_PP))
TG_TOKEN_TESTS = {"llamabench", "llamabenchconc"}
TG_TOKEN_OPTIONS = [128, 512, 1024]
FRONTEND_OPTION_INVENTORY = {
    "--tests": ("exposed", "Test selection screen"),
    "--engine": ("exposed", "Engine selection screen"),
    "--llm-models": ("exposed", "LLM model selection screen"),
    "--embedding-models": ("exposed", "Embedding model selection screen"),
    "--image-models": ("exposed", "Image model selection screen"),
    "--max-prompt-tokens": ("exposed", "Prompt-processing cap screen"),
    "--tg-tokens": ("exposed", "Generation-size screen"),
    "--maxtier": ("equivalent", "Tier shortcuts and explicit model selection are more precise"),
    "--models": ("equivalent", "Backward-compatible alias for --llm-models"),
    "--list-models": ("equivalent", "Installed models are shown in the selection screens"),
    "--sample": ("excluded", "Developer-only non-comparable accuracy sampling"),
    "--warmup": ("exposed", "Graphical execution settings"),
    "--runs": ("exposed", "Graphical execution settings"),
    "--timeout": ("exposed", "Graphical execution settings"),
    "--acc-timeout": ("exposed", "Graphical execution settings"),
    "--acc-token-budget": ("exposed", "Graphical execution settings"),
    "--cpu-only": ("exposed", "Graphical execution settings"),
    "--force-all": ("exposed", "Graphical execution settings"),
    "--out": ("exposed", "Graphical path settings"),
    "--comfyui": ("exposed", "Graphical path settings"),
}
FRONTEND_STATE_PATH = config.SCRIPT_DIR / ".benchmark_frontend_state.json"
FRONTEND_STATE_VERSION = 2
GUI_OPTION_DEFAULTS = {
    "warmup": config.WARMUP_RUNS,
    "runs": config.N_RUNS,
    "timeout": 300,
    "acc_timeout": config.ACC_TIMEOUT,
    "acc_token_budget": config.ACC_TOKEN_BUDGET,
    "cpu_only": False,
    "force_all": False,
    "out": "",
    "comfyui": "",
}
FRONTEND_MODEL_FAMILIES = {
    "llm": {"llm", "custom"},
    "embedding": {"embedding"},
    "image": {"image"},
}


class FrontendCancelled(Exception):
    pass


def frontend_option_gaps() -> list[str]:
    return sorted(flag for flag, (status, _) in FRONTEND_OPTION_INVENTORY.items()
                  if status == "missing")


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
    allowed_keys = required_keys | {"gui_options"}
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
    if "gui_options" in state and validate_gui_options(state["gui_options"]):
        return None
    return state


def validate_gui_options(options: object) -> list[str]:
    if not isinstance(options, dict) or set(options) != set(GUI_OPTION_DEFAULTS):
        return ["GUI settings are incomplete."]
    errors = []
    for key in ("warmup", "runs", "timeout", "acc_timeout", "acc_token_budget"):
        value = options[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < (0 if key == "warmup" else 1):
            errors.append(f"{key} must be a valid whole number.")
    if isinstance(options.get("runs"), int) and not 1 <= options["runs"] <= 10:
        errors.append("runs must be between 1 and 10.")
    if not isinstance(options["cpu_only"], bool) or not isinstance(options["force_all"], bool):
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
                         gui_options: dict | None = None) -> dict:
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
    return state


def apply_saved_test_selection(entries: list[MenuEntry], state: dict | None) -> bool:
    if state is None:
        return False
    saved = set(state["tests"])
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
            entries.append(MenuEntry(
                model["tag"], model["label"], "custom", "Custom LLM",
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
    benchmark_path = benchmark_path or config.SCRIPT_DIR / "scripts" / "benchmark.py"
    command = [
        python_executable, str(benchmark_path),
        "--engine", engine_name,
        "--comfyui", str(comfyui_dir),
        "--tests", *tests,
    ]
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
        if gui_options["cpu_only"]:
            command.append("--cpu-only")
        if gui_options["force_all"]:
            command.append("--force-all")
        if gui_options["out"]:
            command.extend(["--out", gui_options["out"]])
        if gui_options["comfyui"]:
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
        tests = choose_tests(
            test_entries, input_fn, output_fn, clear_fn,
            selection_note=selection_note,
        )
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
