import pytest

from log_redaction import redact_log_text
from shared import Shared


@pytest.mark.parametrize("secret", [
    "hf_abcdefghijklmnopqrstuvwxyz", "Bearer abcdefghijklmnopqrstuvwxyz",
    "https://example.test/?token=sensitive-value", "https://example.test/?access_token=secret&x=1",
    "api_key=abcdefghijklmnopqrstuvwxyz", "password=abcdefghijklmnopqrstuvwxyz",
    "--api-key abcdefghijklmnopqrstuvwxyz", "ghp_abcdefghijklmnopqrstuvwxyz1234",
    "sk-abcdefghijklmnopqrstuvwxyz1234",
])
def test_log_redaction_removes_secret_but_retains_context(secret):
    output = redact_log_text(f"request failed: {secret}", home="/private/home")
    assert "sensitive-value" not in output
    assert "abcdefghijklmnopqrstuvwxyz" not in output
    assert "request failed:" in output
    assert "<secret>" in output


@pytest.mark.parametrize("path", [
    "/Users/alice/private/model.gguf", "/home/alice/private/model.gguf",
    "C:\\Users\\Alice\\private\\model.gguf",
])
def test_log_redaction_replaces_private_home_identity_and_keeps_relative_context(path):
    output = redact_log_text(f"could not read {path}", home="/not/the/test/home")
    assert "alice" not in output.lower()
    assert "<home>" in output
    assert "private" in output and "model.gguf" in output


def test_shared_output_redacts_before_printing(capsys):
    Shared.warn("failed /Users/alice/model.gguf using hf_abcdefghijklmnop")
    output = capsys.readouterr().out
    assert "/Users/alice" not in output
    assert "hf_abcdefghijklmnop" not in output
    assert "<home>/model.gguf" in output
    assert "<secret>" in output
