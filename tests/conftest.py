"""Shared pytest configuration for package-based script imports."""

import ctypes
import sys

import pytest


def tk_display_unavailable_reason(platform, load_library=None):
    if platform != "darwin":
        return None
    if load_library is None:
        load_library = ctypes.CDLL
    graphics = load_library("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
    foundation = load_library("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    graphics.CGSessionCopyCurrentDictionary.restype = ctypes.c_void_p
    foundation.CFRelease.argtypes = [ctypes.c_void_p]
    session = graphics.CGSessionCopyCurrentDictionary()
    if not session:
        return "macOS WindowServer session unavailable (headless or sandboxed)"
    foundation.CFRelease(session)
    return None


def guard_tk_initialization(tk, reason, patcher):
    if reason is None:
        return

    def unavailable(*_args, **_kwargs):
        pytest.skip(reason)

    patcher.setattr(tk.Tk, "__init__", unavailable)


@pytest.fixture(scope="session", autouse=True)
def tk_display_guard():
    # macOS Tk can abort in native code before Python can catch TclError.
    reason = tk_display_unavailable_reason(sys.platform)
    if reason is None:
        yield
        return
    try:
        import tkinter
    except ImportError:
        yield
        return
    with pytest.MonkeyPatch.context() as patcher:
        guard_tk_initialization(tkinter, reason, patcher)
        yield


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
