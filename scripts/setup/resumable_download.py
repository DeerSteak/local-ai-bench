"""Small resumable HTTPS downloader for setup-owned runtime artifacts."""

import os
import re
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+|\*)$")


def download_file(url, destination, *, expected_size=None, opener=urllib.request.urlopen,
                  timeout=60, chunk_size=1024 * 1024, cancel_check=lambda: False):
    """Resume into a sibling part file and atomically publish only a complete response."""
    if urlparse(url).scheme != "https":
        raise ValueError("setup downloads require HTTPS")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    if expected_size is not None and offset > expected_size:
        partial.unlink()
        offset = 0
    headers = {"User-Agent": "local-ai-bench-setup/4.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with opener(request, timeout=timeout) as response:
        final_url = response.geturl() if hasattr(response, "geturl") else url
        if urlparse(final_url).scheme != "https":
            raise ValueError("setup download redirected outside HTTPS")
        status = getattr(response, "status", response.getcode())
        mode = "wb"
        if offset and status == 206:
            _validate_content_range(response.getheader("Content-Range"), offset, expected_size)
            mode = "ab"
        elif status != 200:
            raise ValueError(f"unexpected download response status: {status}")
        with partial.open(mode) as output:
            while True:
                if cancel_check():
                    raise InterruptedError("download cancelled")
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    size = partial.stat().st_size
    if expected_size is not None and size != expected_size:
        raise ValueError(f"download size mismatch: expected {expected_size}, received {size}")
    os.replace(partial, destination)
    _sync_directory(destination.parent)
    return destination


def _validate_content_range(value, offset, expected_size):
    match = CONTENT_RANGE.match(value or "")
    if not match or int(match.group(1)) != offset:
        raise ValueError("download server returned an inconsistent range")
    end = int(match.group(2))
    if end < offset or (expected_size is not None and end != expected_size - 1):
        raise ValueError("download server returned an inconsistent range")
    total = match.group(3)
    if expected_size is not None and total != "*" and int(total) != expected_size:
        raise ValueError("download server returned an unexpected total size")


def _sync_directory(path):
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
