from datetime import datetime
from pathlib import Path

import config
import shared
from shared import Shared


def fixed_now():
    return datetime(2026, 7, 22, 9, 8, 7)


def test_neutral_output_has_compact_timestamp(monkeypatch, capsys):
    monkeypatch.setattr(shared, "_console_now", fixed_now)
    Shared.output("progress")
    assert capsys.readouterr().out == "[09:08:07] progress\n"


def test_plain_output_has_no_timestamp(monkeypatch, capsys):
    monkeypatch.setattr(shared, "_console_now", fixed_now)
    Shared.plain_output("menu")
    assert capsys.readouterr().out == "menu\n"


def test_clear_terminal_emits_ansi_clear_without_timestamp(monkeypatch, capsys):
    monkeypatch.setattr(shared.platform, "system", lambda: "Linux")
    Shared.clear_terminal()
    assert capsys.readouterr().out == "\033[2J\033[H"


def test_clear_terminal_uses_native_windows_command(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(shared.platform, "system", lambda: "Windows")
    monkeypatch.setattr(shared.os, "system", lambda command: calls.append(command) or 0)
    Shared.clear_terminal()
    assert calls == ["cls"]
    assert capsys.readouterr().out == ""


def test_status_helpers_share_timestamp_and_keep_color_after_it(monkeypatch, capsys):
    monkeypatch.setattr(shared, "_console_now", fixed_now)
    Shared.log("working")
    Shared.ok("done")
    Shared.warn("careful")
    Shared.err("failed")
    lines = capsys.readouterr().out.splitlines()
    assert all(line.startswith("[09:08:07] ") for line in lines)
    assert lines[0].startswith(f"[09:08:07]   {config.CYAN}")
    assert lines[1].startswith(f"[09:08:07]   {config.GREEN}")
    assert lines[2].startswith(f"[09:08:07]   {config.YELLOW}")
    assert lines[3].startswith(f"[09:08:07]   {config.RED}")


def test_section_uses_one_timestamp_for_its_logical_block(monkeypatch, capsys):
    monkeypatch.setattr(shared, "_console_now", fixed_now)
    Shared.section("Models")
    output = capsys.readouterr().out
    assert output.startswith(f"\n[09:08:07]\n{config.BOLD}{'─' * 50}\n")
    assert output.count("[09:08:07]") == 1
    assert "Models" in output


def test_neutral_output_can_put_content_below_timestamp(monkeypatch, capsys):
    monkeypatch.setattr(shared, "_console_now", fixed_now)
    Shared.output("heading", timestamp_newline=True)
    assert capsys.readouterr().out == "[09:08:07]\nheading\n"


def test_neutral_output_supports_multiline_block_and_custom_end(monkeypatch, capsys):
    monkeypatch.setattr(shared, "_console_now", fixed_now)
    Shared.output("first\nsecond", end="")
    assert capsys.readouterr().out == "[09:08:07] first\nsecond"


def test_runtime_modules_do_not_bypass_shared_console_output():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    runtime_files = (
        "benchmark.py",
        "benchmark_frontend.py",
        "llm_prefill_benchmark.py",
        "llm_conversation_benchmark.py",
        "embedding_benchmark.py",
        "image_benchmark.py",
        "reasoning_benchmark.py",
    )
    for filename in runtime_files:
        assert "print(" not in (scripts_dir / filename).read_text()
