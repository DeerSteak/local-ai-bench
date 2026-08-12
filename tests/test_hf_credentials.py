from scripts.setup import hf_credentials
from scripts.setup.hf_credentials import HfTokenProvider


def test_load_prefers_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", " environment ")
    (tmp_path / "hf.txt").write_text("file", encoding="utf-8")

    assert HfTokenProvider(tmp_path, False).load() == "environment"


def test_load_uses_saved_token(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / "hf.txt").write_text(" saved \n", encoding="utf-8")

    assert HfTokenProvider(tmp_path, False).load() == "saved"


def test_prompted_token_is_cached(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    replies = iter(["prompted"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(replies))
    monkeypatch.setattr(hf_credentials, "confirm", lambda *_args, **_kwargs: False)
    provider = HfTokenProvider(tmp_path, True)

    assert provider.load() == "prompted"
    assert provider.load() == "prompted"


def test_set_allows_gui_to_supply_empty_token(tmp_path):
    provider = HfTokenProvider(tmp_path, False)
    provider.set(None)

    assert provider.load() == ""
