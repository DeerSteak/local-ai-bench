"""User-owned, local OpenAI-compatible chat server; never manages model processes."""

import ipaddress
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from scripts.runtime import config
from scripts.runtime.engines.base import InferenceEngine
from scripts.runtime.engines import openai_api
from scripts.runtime.engines.chat_flow import chat_measurement, run_bounded_chat, validate_chat_budget
from scripts.runtime.generation_guard import looks_like_loop
from scripts.runtime.shared import EngineLoopDetected, EngineTimeout, Shared

API_ENV = "LOCAL_AI_BENCH_OPENAI_CONFIG"
KEY_ENV = "LOCAL_AI_BENCH_API_KEY"
SUPPORTED_TESTS = frozenset({"llm", "llm_cached", "conv", "mcq", "math", "reasoning", "code", "tool", "conc_chat", "conc_tool", "sustained"})


def endpoint_config(url: str, model: str = "", label: str = "", context: int = 32768) -> dict:
    parsed = urllib.parse.urlsplit(url.strip())
    host = parsed.hostname or ""
    try:
        local = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = False
    if (parsed.scheme not in {"http", "https"} or not local or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment):
        raise ValueError("OpenAI Compatible requires a local http(s) URL (localhost or loopback), without credentials or query parameters.")
    if type(context) is not int or context <= 0:
        raise ValueError("Server context limit must be a positive integer.")
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path += "/v1"
    return {"url": urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", "")),
            "model": model.strip(), "label": label.strip(), "context": context}


def selection_errors(engines, tests, mtp="off", cpu_only=False) -> list[str]:
    if "openai-compatible" not in engines:
        return []
    errors = []
    if len(engines) != 1:
        errors.append("Run OpenAI Compatible by itself so another engine cannot stop the serving process.")
    unsupported = sorted(set(tests) - SUPPORTED_TESTS)
    if unsupported:
        errors.append("OpenAI Compatible does not support selected tests: " + ", ".join(unsupported))
    if mtp != "off":
        errors.append("OpenAI Compatible cannot switch MTP; leave the benchmark MTP control off and configure the server yourself.")
    if cpu_only:
        errors.append("OpenAI Compatible cannot change the server's device; disable CPU-only.")
    return errors


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("OpenAI Compatible does not follow redirects; enter the final local base URL.")


class OpenAICompatibleEngine(InferenceEngine):
    name = "openai-compatible"

    def __init__(self, settings=None):
        self.settings = settings or json.loads(os.environ.get(API_ENV, "{}"))
        if self.settings:
            self.settings = endpoint_config(**self.settings)
        self._models = None

    def _request(self, path, payload=None, timeout=15):
        if not self.settings:
            raise ValueError("Configure the OpenAI Compatible base URL first.")
        headers = {"Content-Type": "application/json"}
        if key := os.environ.get(KEY_ENV):
            headers["Authorization"] = "Bearer " + key
        request = urllib.request.Request(self.settings["url"] + path,
            data=json.dumps(payload).encode() if payload is not None else None, headers=headers)
        try:
            return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect()).open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"OpenAI Compatible HTTP {exc.code}; check authentication, model, and server settings.") from None
        except urllib.error.URLError:
            raise RuntimeError("OpenAI Compatible server is unreachable at the configured local URL.") from None

    def discover_models(self):
        with self._request("/models") as response:
            data = json.load(response)
        rows = data.get("data") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise ValueError("The server did not return an OpenAI /models list.")
        models = []
        for row in rows:
            tag = row.get("id") if isinstance(row, dict) else None
            if isinstance(tag, str) and tag.strip() and tag not in models:
                models.append(tag)
        if not models:
            raise ValueError("The server advertises no models.")
        self._models = models
        return models

    def list_installed_models(self):
        if not self.settings:
            return []
        models = self.discover_models()
        selected = self.settings["model"]
        if selected and selected not in models:
            raise ValueError("The selected model is no longer advertised by the server.")
        if not selected and len(models) != 1:
            raise ValueError("The server advertises multiple models; select a served model ID.")
        tag = selected or models[0]
        label = self.settings["label"] or tag
        return [{"tag": tag, "label": label, "short": label, "size": None}]

    def model_pulled(self, tag):
        return tag in (self._models if self._models is not None else self.discover_models())

    def is_installed(self):
        return True

    def available(self):
        try:
            self.discover_models()
            return True
        except (RuntimeError, ValueError, OSError):
            return False

    def ensure_running(self):
        return self.available()

    def start(self, *, gpu_visible=True, timeout=15):
        return self.available()

    def stop(self, *, timeout=15):
        pass

    def unload(self, tag):
        pass

    def unload_all(self):
        pass

    def wait_until_unloaded(self, tag, timeout=30):
        return False

    def can_reset_model_state(self):
        return False

    def reachable_or_abort(self):
        return self.available()

    def wait_for_recovery(self, timeout=30):
        return self.available()

    def is_connection_crash(self, exc):
        return isinstance(exc, (ConnectionError, OSError))

    def tail_log(self, n_lines=40):
        return "Server logs are owned by the external runtime."

    def max_context_length(self, tag, default=131072):
        return self.settings.get("context", default)

    def compatibility_metadata(self, tag):
        return {}, None

    def resume_artifact_paths(self, tag):
        return ()

    def resume_runtime_paths(self):
        return {}

    def warmup(self, tag, label, num_ctx, warmup_runs, crash_cache=None, cache_path=None, crash_extra=None):
        try:
            for _ in range(warmup_runs):
                self.generate(tag, "Hello.", timeout=config.RUN_TIMEOUT, num_ctx=num_ctx)
            return True
        except Exception as exc:
            Shared.warn(f"OpenAI Compatible warmup failed: {exc}")
            return False

    def prepare_concurrency(self, tag, n_parallel, per_slot_ctx, warmup_runs=1, timeout=300):
        return self.warmup(tag, tag, per_slot_ctx, warmup_runs)

    def generate(self, tag, prompt, timeout=600, num_ctx=None, n_parallel=1, cache_prompt=False):
        return self.chat(tag, [{"role": "user", "content": prompt}], timeout=timeout,
                         num_ctx=num_ctx, num_predict=config.GENERATE_MAX_TOKENS)

    def _post(self, path, payload, timeout):
        return self._request("/chat/completions", payload, timeout)

    def _chat_request(self, tag: str, messages: list, tools: list | None,
                      deadline: float, num_predict: int,
                      check_loop: bool, budget_nudged: bool,
                      cache_prompt: bool = False) -> dict:
        payload = {
            **self.sampling_payload(),
            "model": tag,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if num_predict is not None and num_predict > 0:
            payload["max_tokens"] = num_predict
        if tools is not None:
            payload.update({"tools": tools, "tool_choice": "auto"})

        request_start = time.perf_counter()
        remaining = deadline - request_start
        if remaining <= 0:
            raise EngineTimeout("openai_chat exceeded its wall-clock deadline",
                                budget_nudged=budget_nudged)

        ttft = None
        tokens = 0
        usage_seen = False
        prompt_eval_count = None
        finish_reason = None
        response_parts = []
        reasoning_parts = []
        tool_fragments: dict[int, dict] = {}
        last_loop_check = request_start

        try:
            with self._post("/v1/chat/completions", payload, remaining) as resp:
                for chunk in openai_api.iter_sse(resp):
                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    reasoning = delta.get("reasoning_content")
                    tool_calls = delta.get("tool_calls")
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
                    if ttft is None and (content or reasoning or tool_calls):
                        ttft = time.perf_counter() - request_start
                    if content:
                        response_parts.append(content)
                    if reasoning:
                        reasoning_parts.append(reasoning)
                    if tool_calls:
                        openai_api.accumulate_tool_fragments(tool_fragments, tool_calls)

                    usage = chunk.get("usage") or {}
                    usage_seen = usage_seen or ("completion_tokens" in usage and "prompt_tokens" in usage)
                    tokens, prompt_eval_count = openai_api.streamed_usage(
                        chunk, tokens, prompt_eval_count,
                    )

                    now = time.perf_counter()
                    response_text = "".join(response_parts) or "".join(reasoning_parts)
                    parsed_calls = openai_api.tool_calls_from_fragments(tool_fragments)
                    partial_text = (json.dumps(parsed_calls) if tools is not None and parsed_calls
                                    else response_text)
                    if now > deadline:
                        raise EngineTimeout("openai_chat exceeded its wall-clock deadline",
                                            partial_text=partial_text, budget_nudged=budget_nudged)
                    if check_loop and now - last_loop_check >= config.LOOP_CHECK_INTERVAL:
                        last_loop_check = now
                        if response_text and looks_like_loop(response_text):
                            raise EngineLoopDetected(
                                f"openai_chat detected a generation loop after "
                                f"{now - request_start:.0f}s",
                                partial_text=response_text, budget_nudged=budget_nudged)

        except TimeoutError:
            partial = "".join(response_parts) or "".join(reasoning_parts)
            calls = openai_api.tool_calls_from_fragments(tool_fragments)
            raise EngineTimeout("OpenAI Compatible stream timed out", partial_text=json.dumps(calls) if calls else partial,
                                budget_nudged=budget_nudged) from None

        if not usage_seen or type(tokens) is not int or tokens < 0 or type(prompt_eval_count) is not int or prompt_eval_count < 0:
            raise ValueError("Server must report integer prompt_tokens and completion_tokens in streamed usage; throughput cannot be inferred.")
        total = time.perf_counter() - request_start
        ttft, decode_seconds, raw_tps, tps = openai_api.stream_timing(total, ttft, tokens)
        return {
            "ttft": ttft,
            "server_prompt_sec": None,
            "wall_seconds": total,
            "tokens": tokens,
            "tps": tps,
            "decode_seconds": decode_seconds,
            "prompt_eval_count": prompt_eval_count,
            "response_text": "".join(response_parts) or "".join(reasoning_parts),
            "tool_calls": openai_api.tool_calls_from_fragments(tool_fragments),
            "finish_reason": finish_reason,
            "server_tps_implausible": tps != raw_tps,
        }


    def _measure(self, tag, messages, tools, timeout, num_predict, check_loop, token_budget):
        validate_chat_budget(num_predict, token_budget)
        if not self.model_pulled(tag):
            raise ValueError("Requested model is not advertised by the server.")
        deadline = time.perf_counter() + timeout
        first, second, nudged = run_bounded_chat(
            lambda messages, tools, deadline, count, check, nudged: self._chat_request(
                tag, messages, tools, deadline, count, check, nudged),
            messages, tools, deadline, num_predict, check_loop, token_budget,
            config.ACC_FINALIZE_FRACTION, config.ACC_FINALIZE_MESSAGE, "openai_chat",
        )
        return chat_measurement(first, second, nudged, 0, openai_api.sanitize_tps, 0)

    def chat(self, tag, messages, timeout=600, num_ctx=None, num_predict=1024,
             check_loop=False, token_budget=None, cache_prompt=False):
        return self._measure(tag, messages, None, timeout, num_predict, check_loop, token_budget)

    def chat_tools(self, tag, messages, tools, timeout=600, num_ctx=None, num_predict=1024,
                   check_loop=False, token_budget=None):
        return self._measure(tag, messages, tools, timeout, num_predict, check_loop, token_budget)

    def embed(self, tag, inputs, timeout=120):
        raise ValueError("OpenAI Compatible currently supports chat models only.")
