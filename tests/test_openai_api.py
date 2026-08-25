import io
import urllib.error
from email.message import Message

import pytest

from scripts.runtime.engines import openai_api


def test_streamed_usage_preserves_prior_counts_until_the_usage_chunk_arrives():
    assert openai_api.streamed_usage({"choices": []}, 3, 7) == (3, 7)
    assert openai_api.streamed_usage(
        {"usage": {"completion_tokens": 5, "prompt_tokens": 11}}, 3, 7,
    ) == (5, 11)


def test_stream_timing_handles_missing_first_token_and_zero_decode_window():
    assert openai_api.stream_timing(2.5, None, 4) == (2.5, 0, 0, 0)
    assert openai_api.stream_timing(3.0, 1.0, 4) == (1.0, 2.0, 2.0, 2.0)


def _http_error(code: int, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://server.test", code, "failed", Message(), io.BytesIO(body),
    )


@pytest.mark.parametrize(("body", "detail"), [
    (b'{"error":"model unavailable"}', "model unavailable"),
    (b"plain failure", "plain failure"),
    (b'["unexpected", "shape"]', '["unexpected", "shape"]'),
])
def test_urlopen_with_detail_surfaces_json_and_raw_errors(monkeypatch, body, detail):
    monkeypatch.setattr(
        openai_api.urllib.request, "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(_http_error(503, body)),
    )
    with pytest.raises(RuntimeError) as error:
        openai_api.urlopen_with_detail(object(), 2, "vLLM")
    assert str(error.value) == f"vLLM returned HTTP 503: {detail}"


def test_urlopen_with_detail_truncates_large_error_body(monkeypatch):
    body = b"x" * 700
    monkeypatch.setattr(
        openai_api.urllib.request, "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(_http_error(500, body)),
    )
    with pytest.raises(RuntimeError) as error:
        openai_api.urlopen_with_detail(object(), 2, "server")
    assert str(error.value).endswith("x" * 500)
    assert len(str(error.value).split(": ", 1)[1]) == 500


def test_iter_sse_preserves_keepalives_done_and_malformed_lines():
    response = [
        b": keepalive\n", b"event: ping\n", b"data: {bad json}\n",
        b'data: {"choices":[]}\n', b"data: [DONE]\n",
    ]
    assert list(openai_api.iter_sse(response)) == [
        {}, {}, {}, {"choices": []}, {},
    ]


def test_tool_calls_from_fragments_marks_truncated_arguments_incomplete():
    calls = openai_api.tool_calls_from_fragments({
        2: {"name": "later", "arguments": '{"x":'},
        0: {"name": "first", "arguments": ""},
    })
    assert calls == [
        {"name": "first", "arguments": {}},
        {"name": "later", "arguments": {}, "incomplete": True},
    ]


def test_accumulate_tool_fragments_merges_out_of_order_and_partial_functions():
    fragments = {}
    openai_api.accumulate_tool_fragments(fragments, [
        {"index": 3, "function": {"arguments": '{"city"'}},
        {"index": 1},
    ])
    openai_api.accumulate_tool_fragments(fragments, [
        {"index": 3, "function": {"name": "weather", "arguments": ':"Paris"}'}},
        {"index": 1, "function": {"arguments": "{}"}},
    ])
    assert fragments == {
        1: {"name": "", "arguments": "{}"},
        3: {"name": "weather", "arguments": '{"city":"Paris"}'},
    }
    assert openai_api.tool_calls_from_fragments(fragments) == [
        {"name": "", "arguments": {}},
        {"name": "weather", "arguments": {"city": "Paris"}},
    ]
