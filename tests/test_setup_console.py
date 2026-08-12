import builtins

from scripts.setup import setup_console


def test_status_lines_use_expected_symbols(capsys):
    setup_console.ok("ready")
    setup_console.warn("careful")
    setup_console.fail("broken")
    setup_console.info("working")

    output = capsys.readouterr().out
    assert "✓" in output and "ready" in output
    assert "!" in output and "careful" in output
    assert "✗" in output and "broken" in output
    assert "→" in output and "working" in output


def test_section_prints_heading_and_rule(capsys):
    setup_console.section("Models")

    output = capsys.readouterr().out
    assert "Models" in output
    assert "─" * 50 in output


def test_link_uses_label_or_url():
    assert "Docs" in setup_console.link("https://example.com", "Docs")
    assert "https://example.com" in setup_console.link("https://example.com")


def test_confirm_accepts_yes_and_no(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda _prompt: " YES ")
    assert setup_console.confirm("Continue?") is True

    monkeypatch.setattr(builtins, "input", lambda _prompt: "n")
    assert setup_console.confirm("Continue?") is False


def test_confirm_uses_default_for_empty_input(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda _prompt: "")
    assert setup_console.confirm("Continue?", default=True) is True
    assert setup_console.confirm("Continue?", default=False) is False


def test_confirm_uses_default_at_eof(monkeypatch, capsys):
    def raise_eof(_prompt):
        raise EOFError

    monkeypatch.setattr(builtins, "input", raise_eof)
    assert setup_console.confirm("Continue?", default=False) is False
    assert capsys.readouterr().out == "\n"
