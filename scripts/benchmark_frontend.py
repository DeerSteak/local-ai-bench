#!/usr/bin/env python3
"""Interactive launcher that translates selections into benchmark CLI flags."""

import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import config
from benchmark import CONCURRENCY_TESTS, LLM_TESTS
from engines import engine_names, get_engine
from model_inventory import build_model_inventory
from models import EMBED_MODELS, IMAGE_MODELS, LLM_MODELS
from shared import Shared


TEST_DEFINITIONS = [
    ("llm", "Single-shot LLM", "llm", True),
    ("conv", "Conversation", "llm", True),
    ("emb", "Embeddings", "embedding", True),
    ("img", "Image generation", "image", True),
    ("mcq", "MCQ accuracy", "llm", False),
    ("math", "Math accuracy", "llm", False),
    ("reasoning", "Reasoning accuracy", "llm", False),
    ("code", "Code accuracy", "llm", False),
    ("tool", "Tool accuracy", "llm", False),
    ("conc_tool", "Tool concurrency", "llm", False),
    ("conc_chat", "Chat concurrency", "llm", False),
]
TIER_KEYS = {"xs": "xsmall", "s": "small", "m": "medium", "l": "large"}
LLM_BACKED_TESTS = set(LLM_TESTS + CONCURRENCY_TESTS)
FRONTEND_STATE_PATH = config.SCRIPT_DIR / ".benchmark_frontend_state.json"
FRONTEND_STATE_VERSION = 1
FRONTEND_MODEL_FAMILIES = {
    "llm": {"llm", "custom"},
    "embedding": {"embedding"},
    "image": {"image"},
}


class FrontendCancelled(Exception):
    pass


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
        state = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(state, dict) or set(state) != {"version", "engine", "tests", "models"}:
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
    return state


def save_frontend_state(state: dict, path: Path = FRONTEND_STATE_PATH) -> bool:
    path = Path(path)
    temporary_path = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                mode="w", dir=path.parent, prefix=f".{path.name}.",
                suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            json.dump(state, stream, indent=2, allow_nan=False)
        os.replace(temporary_path, path)
        return True
    except (OSError, TypeError, ValueError):
        return False
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def build_frontend_state(engine_name: str, tests: list[str],
                         entries: list[MenuEntry]) -> dict:
    selected = [entry for entry in entries if entry.checked]
    return {
        "version": FRONTEND_STATE_VERSION,
        "engine": engine_name,
        "tests": list(tests),
        "models": {
            family: [
                entry.value for entry in selected if entry.kind in kinds
            ]
            for family, kinds in FRONTEND_MODEL_FAMILIES.items()
        },
    }


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
        try:
            numbers = parse_toggle_numbers(raw, len(entries))
        except ValueError:
            feedback = "Couldn't parse that selection; use numbers/ranges such as `2 4 7-9`."
            continue
        unavailable = [number for number in numbers if not entries[number - 1].available]
        if unavailable:
            feedback = "A test with no applicable installed model cannot be selected."
            continue
        for number in numbers:
            entries[number - 1].checked = not entries[number - 1].checked


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
                            benchmark_path: Path | None = None) -> list[str]:
    benchmark_path = benchmark_path or config.SCRIPT_DIR / "scripts" / "benchmark.py"
    command = [
        python_executable, str(benchmark_path),
        "--engine", engine_name,
        "--comfyui", str(comfyui_dir),
        "--tests", *tests,
    ]
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
                   entries: list[MenuEntry], output_fn) -> None:
    output_fn("Benchmark selection:")
    output_fn(f"  Engine: {engine_name}")
    output_fn(f"  ComfyUI: {comfyui_dir}")
    output_fn(f"  Tests: {', '.join(tests)}")
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
        comfyui_dir = config.COMFYUI_DIR
        inventory = inventory_builder(engine_factory(selected_engine), comfyui_dir)
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
        render_summary(selected_engine, comfyui_dir, tests, model_entries, output_fn)
        confirmation = read_choice("Start this benchmark? [Y/n]", input_fn, output_fn).lower()
        if confirmation not in ("", "y", "yes"):
            raise FrontendCancelled
        state = build_frontend_state(selected_engine, tests, model_entries)
        if not save_frontend_state(state, state_path):
            output_fn("Could not save this launcher selection; continuing without persistence.")
        command = build_benchmark_command(
            selected_engine, comfyui_dir, tests, model_entries,
            python_executable=python_executable, benchmark_path=benchmark_path,
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
