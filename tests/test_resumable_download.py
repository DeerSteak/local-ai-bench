import io

import pytest

from scripts.setup.resumable_download import download_file


class Response(io.BytesIO):
    def __init__(self, data, *, status=200, headers=None, url="https://downloads.example/file"):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def getcode(self):
        return self.status

    def getheader(self, name):
        return self.headers.get(name)

    def geturl(self):
        return self.url


def opener_for(response, captured):
    def open_request(request, timeout):
        captured.append((dict(request.headers), timeout))
        return response
    return open_request


def test_new_download_is_atomically_published(tmp_path):
    destination = tmp_path / "runtime.zip"
    captured = []
    result = download_file(
        "https://downloads.example/runtime.zip", destination, expected_size=7,
        opener=opener_for(Response(b"content"), captured),
    )
    assert result == destination
    assert destination.read_bytes() == b"content"
    assert not destination.with_name("runtime.zip.part").exists()
    assert "Range" not in captured[0][0]


def test_partial_download_resumes_with_validated_range(tmp_path):
    destination = tmp_path / "runtime.zip"
    partial = tmp_path / "runtime.zip.part"
    partial.write_bytes(b"first-")
    captured = []
    download_file(
        "https://downloads.example/runtime.zip", destination, expected_size=10,
        opener=opener_for(Response(
            b"rest", status=206, headers={"Content-Range": "bytes 6-9/10"},
        ), captured),
    )
    assert destination.read_bytes() == b"first-rest"
    assert captured[0][0]["Range"] == "bytes=6-"


def test_server_ignoring_range_restarts_instead_of_duplicating(tmp_path):
    destination = tmp_path / "runtime.zip"
    partial = tmp_path / "runtime.zip.part"
    partial.write_bytes(b"ol")
    download_file(
        "https://downloads.example/runtime.zip", destination, expected_size=3,
        opener=opener_for(Response(b"new"), []),
    )
    assert destination.read_bytes() == b"new"


def test_inconsistent_range_or_size_preserves_partial_for_safe_retry(tmp_path):
    destination = tmp_path / "runtime.zip"
    partial = tmp_path / "runtime.zip.part"
    partial.write_bytes(b"abc")
    with pytest.raises(ValueError, match="inconsistent range"):
        download_file(
            "https://downloads.example/runtime.zip", destination, expected_size=6,
            opener=opener_for(Response(
                b"def", status=206, headers={"Content-Range": "bytes 4-6/7"},
            ), []),
        )
    assert partial.read_bytes() == b"abc"
    with pytest.raises(ValueError, match="size mismatch"):
        download_file(
            "https://downloads.example/runtime.zip", destination, expected_size=8,
            opener=opener_for(Response(b"short"), []),
        )
    assert partial.read_bytes() == b"short"
    assert not destination.exists()


def test_range_end_must_match_requested_remainder(tmp_path):
    destination = tmp_path / "runtime.zip"
    destination.with_name("runtime.zip.part").write_bytes(b"abc")
    with pytest.raises(ValueError, match="inconsistent range"):
        download_file(
            "https://downloads.example/runtime.zip", destination, expected_size=6,
            opener=opener_for(Response(
                b"def", status=206, headers={"Content-Range": "bytes 3-9/6"},
            ), []),
        )


def test_failed_stream_keeps_partial_and_existing_destination(tmp_path):
    class FailingResponse(Response):
        def read(self, size: int | None = -1):
            if self.tell() >= 3:
                raise OSError("connection lost")
            return super().read(3)

    destination = tmp_path / "runtime.zip"
    destination.write_bytes(b"known-good")
    with pytest.raises(OSError, match="connection lost"):
        download_file(
            "https://downloads.example/runtime.zip", destination,
            opener=opener_for(FailingResponse(b"partial-data"), []), chunk_size=3,
        )
    assert destination.read_bytes() == b"known-good"
    assert destination.with_name("runtime.zip.part").read_bytes() == b"par"


def test_cancelled_download_keeps_partial_for_resume(tmp_path):
    checks = iter((False, True))
    destination = tmp_path / "runtime.zip"
    with pytest.raises(InterruptedError, match="cancelled"):
        download_file(
            "https://downloads.example/runtime.zip", destination,
            opener=opener_for(Response(b"abcdef"), []), chunk_size=3,
            cancel_check=lambda: next(checks),
        )
    assert destination.with_name("runtime.zip.part").read_bytes() == b"abc"
    assert not destination.exists()


@pytest.mark.parametrize("url", ["http://downloads.example/file", "file:///private/file"])
def test_non_https_downloads_are_rejected(url, tmp_path):
    with pytest.raises(ValueError, match="HTTPS"):
        download_file(url, tmp_path / "file", opener=lambda *_args, **_kwargs: None)


def test_https_downgrade_redirect_is_rejected(tmp_path):
    response = Response(b"content", url="http://downloads.example/file")
    with pytest.raises(ValueError, match="redirected"):
        download_file(
            "https://downloads.example/file", tmp_path / "file",
            opener=opener_for(response, []),
        )
