"""Shared pytest configuration for package-based script imports."""

import pytest


@pytest.fixture(autouse=True)
def isolate_saved_llamacpp_tool_config(monkeypatch):
    monkeypatch.setattr(
        "scripts.runtime.llamacpp_tools.load_setup_config", lambda _path: {},
    )
