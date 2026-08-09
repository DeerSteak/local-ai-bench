"""Shared pytest configuration for package-based script imports."""

import pytest


@pytest.fixture
def symlink_or_skip():
    """Create a symlink, skipping where the platform forbids it — Windows needs Developer
    Mode or admin. These tests cover POSIX symlink semantics, which junctions do not match."""
    def _make(link, target, *, directory=False):
        try:
            link.symlink_to(target, target_is_directory=directory)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are unavailable on this platform")
        return link
    return _make


@pytest.fixture(autouse=True)
def isolate_saved_llamacpp_tool_config(monkeypatch):
    monkeypatch.setattr(
        "scripts.runtime.llamacpp_tools.load_setup_config", lambda _path: {},
    )
