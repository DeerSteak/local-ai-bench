"""OpenAI-compatible streaming helpers shared by engines — see docs/engines.md."""

import json
import urllib.error
import urllib.request

from scripts.runtime import config


def urlopen_with_detail(req, timeout, server_label: str):
    """urlopen that surfaces the server's JSON error body instead of "HTTP Error 500"."""
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            parsed = json.loads(body)
            detail = parsed.get("error", body) if isinstance(parsed, dict) else body
        except json.JSONDecodeError:
            detail = body
        raise RuntimeError(f"{server_label} returned HTTP {e.code}: {str(detail)[:500]}") from None


def iter_sse(resp):
    """Yield parsed JSON from an SSE body. Empty dicts for comments/[DONE]/malformed
    lines, so callers can still enforce a deadline on keepalive-only traffic."""
    for raw_line in resp:
        line = raw_line.decode(errors="replace") if isinstance(raw_line, bytes) else raw_line
        line = line.strip()
        if not line.startswith("data:"):
            yield {}
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            yield {}
            continue
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            yield {}


def sanitize_tps(tps: float, tokens: int, ttft: float, total: float) -> float:
    """Replace an implausible self-reported tps with a wall-clock estimate."""
    if tps <= config.MAX_PLAUSIBLE_TPS:
        return tps
    decode_elapsed = total - ttft
    return tokens / decode_elapsed if decode_elapsed > 0 else 0


def tool_calls_from_fragments(tool_fragments: dict[int, dict]) -> list[dict]:
    """Assemble streamed tool-call fragments into [{"name", "arguments"}]."""
    calls = []
    for index in sorted(tool_fragments):
        fragment = tool_fragments[index]
        call = {"name": fragment["name"], "arguments": {}}
        try:
            call["arguments"] = json.loads(fragment["arguments"]) if fragment["arguments"] else {}
        except json.JSONDecodeError:
            call["incomplete"] = True
        calls.append(call)
    return calls


def accumulate_tool_fragments(tool_fragments: dict[int, dict], tool_calls: list) -> None:
    """Merge one chunk's streamed tool-call deltas into `tool_fragments`."""
    for call in tool_calls:
        index = call.get("index", 0)
        fragment = tool_fragments.setdefault(index, {"name": "", "arguments": ""})
        function = call.get("function") or {}
        if function.get("name"):
            fragment["name"] = function["name"]
        if function.get("arguments"):
            fragment["arguments"] += function["arguments"]
