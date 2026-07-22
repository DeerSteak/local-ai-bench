#!/usr/bin/env python3
"""Interactive launcher that translates selections into benchmark CLI flags."""

import platform
import subprocess
import sys
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
    ("code", "Code accuracy", "llm", False),
    ("tool", "Tool accuracy", "llm", False),
    ("conc_tool", "Tool concurrency", "llm", False),
    ("conc_chat", "Chat concurrency", "llm", False),
]
TIER_KEYS = {"xs": "xsmall", "s": "small", "m": "medium", "l": "large"}
LLM_BACKED_TESTS = set(LLM_TESTS + CONCURRENCY_TESTS)


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


def choose_engine(available: list[str], input_fn, output_fn) -> str:
    if len(available) == 1:
        output_fn(f"Engine: {available[0]}")
        return available[0]

    selected = 0
    while True:
        output_fn("Choose one inference engine (`--engine all` remains CLI-only):")
        for index, name in enumerate(available, 1):
            box = "[x]" if index - 1 == selected else "[ ]"
            output_fn(f"  {box} {index:>2}  {name}")
        raw = read_choice("Enter a number, or press Enter to accept:", input_fn, output_fn).lower()
        if raw in ("q", "quit", "cancel"):
            raise FrontendCancelled
        if raw == "":
            return available[selected]
        if raw.isdigit() and 1 <= int(raw) <= len(available):
            selected = int(raw) - 1
            return available[selected]
        output_fn("Couldn't parse that engine selection.")


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


def render_test_menu(entries: list[MenuEntry], output_fn) -> None:
    output_fn("Choose benchmark tests:")
    for index, entry in enumerate(entries, 1):
        box = "[x]" if entry.checked else "[ ]"
        unavailable = "  (no installed model available)" if not entry.available else ""
        output_fn(f"  {box} {index:>2}  {entry.label}{unavailable}")


def choose_tests(entries: list[MenuEntry], input_fn, output_fn) -> list[str]:
    while True:
        render_test_menu(entries, output_fn)
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
            output_fn("Select at least one available test.")
            continue
        try:
            numbers = parse_toggle_numbers(raw, len(entries))
        except ValueError:
            output_fn("Couldn't parse that selection; use numbers/ranges such as `2 4 7-9`.")
            continue
        unavailable = [number for number in numbers if not entries[number - 1].available]
        if unavailable:
            output_fn("A test with no applicable installed model cannot be selected.")
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


def render_model_menu(entries: list[MenuEntry], hint: str | None, output_fn) -> None:
    output_fn("Choose installed models:")
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
                  input_fn, output_fn) -> list[MenuEntry]:
    while True:
        render_model_menu(entries, hint, output_fn)
        raw = read_choice(
            "Toggle numbers/ranges, xs/s/m/l, custom, or emb; press Enter to continue:",
            input_fn, output_fn,
        ).lower()
        if raw in ("q", "quit", "cancel"):
            raise FrontendCancelled
        if raw == "":
            error = model_selection_error(entries, tests)
            if error:
                output_fn(error)
                continue
            return entries
        if raw in TIER_KEYS:
            tier = TIER_KEYS[raw]
            if not toggle_group(
                entries, lambda entry: entry.kind in ("llm", "image") and entry.tier == tier,
            ):
                output_fn(f"No installed catalog LLM/image models are available in tier {tier}.")
            continue
        if raw in ("custom", "emb"):
            kind = "custom" if raw == "custom" else "embedding"
            if not toggle_group(entries, lambda entry: entry.kind == kind):
                output_fn(f"No installed {kind} models are available in this selection.")
            continue
        try:
            numbers = parse_toggle_numbers(raw, len(entries))
        except ValueError:
            output_fn("Couldn't parse that selection; use numbers/ranges or a documented group key.")
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


def run_frontend(input_fn=input, output_fn=Shared.output, process_runner=None,
                 engine_names_fn=engine_names, engine_factory=get_engine,
                 inventory_builder=build_model_inventory, system: str | None = None,
                 python_executable: str = sys.executable,
                 benchmark_path: Path | None = None) -> int:
    process_runner = process_runner or subprocess.run
    system = system or platform.system()
    try:
        output_fn("Local AI Bench interactive launcher")
        selected_engine = choose_engine(engine_names_fn(), input_fn, output_fn)
        comfyui_dir = config.COMFYUI_DIR
        inventory = inventory_builder(engine_factory(selected_engine), comfyui_dir)
        test_entries = build_test_entries(inventory)
        if not any(entry.available for entry in test_entries):
            output_fn("No installed benchmark models were found. Run setup to add catalog models.")
            return 1

        tests = choose_tests(test_entries, input_fn, output_fn)
        model_entries = build_model_entries(inventory, tests)
        hint = missing_catalog_hint(inventory, system)
        choose_models(model_entries, tests, hint, input_fn, output_fn)
        render_summary(selected_engine, comfyui_dir, tests, model_entries, output_fn)
        confirmation = read_choice("Start this benchmark? [Y/n]", input_fn, output_fn).lower()
        if confirmation not in ("", "y", "yes"):
            raise FrontendCancelled
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
