import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from conftest import tk_display_unavailable_reason


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_non_macos_preserves_tk_display_detection(platform):
    loader = Mock(side_effect=AssertionError("must not load macOS frameworks"))
    assert tk_display_unavailable_reason(platform, loader) is None
    loader.assert_not_called()


@pytest.mark.parametrize("session", [None, 1234])
def test_macos_display_check_releases_available_session_and_detects_denial(session):
    copy_session = Mock(return_value=session)
    release = Mock()
    loader = Mock(side_effect=[
        SimpleNamespace(CGSessionCopyCurrentDictionary=copy_session),
        SimpleNamespace(CFRelease=release),
    ])
    reason = tk_display_unavailable_reason("darwin", loader)
    assert copy_session.restype is ctypes.c_void_p
    assert release.argtypes == [ctypes.c_void_p]
    if session is None:
        assert reason == "macOS WindowServer session unavailable (headless or sandboxed)"
        release.assert_not_called()
    else:
        assert reason is None
        release.assert_called_once_with(session)


def test_macos_framework_failure_is_not_silently_skipped():
    with pytest.raises(OSError, match="missing framework"):
        tk_display_unavailable_reason("darwin", Mock(side_effect=OSError("missing framework")))


@pytest.mark.parametrize("reason", [None, "WindowServer unavailable"])
def test_guard_covers_indirect_tk_initialization_without_skipping_other_code(monkeypatch, reason):
    from conftest import guard_tk_initialization

    initialized = []

    class Tk:
        def __init__(self):
            initialized.append(True)

    tk = SimpleNamespace(Tk=Tk)
    guard_tk_initialization(tk, reason, monkeypatch)
    if reason is None:
        tk.Tk()
        assert initialized == [True]
    else:
        with pytest.raises(pytest.skip.Exception, match=reason):
            tk.Tk()
        assert initialized == []
