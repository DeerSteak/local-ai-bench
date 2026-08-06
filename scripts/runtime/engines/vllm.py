"""vLLM engine — see docs/engines.md. Weights resolve by HuggingFace repo id from
vLLM's own cache, never by path, so a containerised vLLM works unchanged."""

import http.client
import json
import os
import platform
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import requests

from scripts.runtime import config
from scripts.runtime.engines import openai_api
from scripts.runtime.engines.base import (
    ChatMeasurement, EmbeddingMeasurement, GenerationMeasurement, InferenceEngine,
)
from scripts.setup.setup_config import configured_vllm_path, load_setup_config
from scripts.setup.vllm_install import (
    find_vllm_binary, find_vllm_launcher, hf_cache_model_complete, hf_cache_model_dir,
    vllm_cache_home,
)
from scripts.workloads.models import EMBED_MODELS, LLM_MODELS
from scripts.runtime.shared import (
    EngineBudgetExceeded,
    EngineLoopDetected,
    EngineTimeout,
    Shared,
    split_token_budget,
)


class VllmEngine(InferenceEngine):
    name = "vllm"

    # vLLM start-up includes weight load, graph capture, and KV allocation.
    LOAD_TIMEOUT = 900

    def __init__(self):
        setup = load_setup_config(config.SETUP_CONFIG_PATH)
        self._launcher = configured_vllm_path(setup, "launcher") or find_vllm_launcher()
        self._executable = configured_vllm_path(setup, "executable") or find_vllm_binary(
            platform_name=platform.system())
        recorded_home = configured_vllm_path(setup, "hf_home")
        self._cache_home = Path(recorded_home) if recorded_home else vllm_cache_home(self._launcher)

        self._proc: subprocess.Popen | None = None
        self._log_path: Path | None = None
        self._loaded_tag: str | None = None
        self._loaded_num_ctx: int | None = None
        self._loaded_embedding: bool | None = None
        self._loaded_n_parallel: int = 1
        self._loaded_tool_parser: str | None = None
        self._gpu_visible = True
        self._model_lock = threading.RLock()

    # ── model resolution ──

    @staticmethod
    def _catalog_entry(tag: str) -> dict | None:
        for model in LLM_MODELS + EMBED_MODELS:
            if model["tag"] == tag:
                return model
        return None

    @classmethod
    def _tool_parser(cls, tag: str) -> str | None:
        """vLLM's per-model tool-call parser name, or None when the catalog has none."""
        entry = cls._catalog_entry(tag)
        return entry.get("vllm_tool_parser") if entry else None

    @classmethod
    def _repo(cls, tag: str) -> str | None:
        """The HF repo id vLLM serves for `tag`, or None when the catalog has none."""
        entry = cls._catalog_entry(tag)
        return entry.get("vllm_repo") if entry else None

    def _snapshot_dir(self, tag: str) -> Path | None:
        repo = self._repo(tag)
        if repo is None:
            return None
        snapshots = hf_cache_model_dir(self._cache_home, repo) / "snapshots"
        if not snapshots.is_dir():
            return None
        for snapshot in sorted(snapshots.iterdir()):
            if (snapshot / "config.json").is_file():
                return snapshot
        return None

    # ── server/process lifecycle ──

    def available(self) -> bool:  # pragma: no cover — real HTTP call
        try:
            return requests.get(f"{config.VLLM_URL}/health", timeout=5).status_code == 200
        except Exception:
            return False

    def ensure_running(self) -> bool:
        """Preflight only — the real spawn is lazy, per tag, in _ensure_model."""
        if self._launcher is None and self._executable is None:
            Shared.err("No 'vllm' or platform launcher found — run setup_check.py, or "
                       "install vLLM yourself: https://docs.vllm.ai/")
            return False
        if not self._cache_home.exists():
            Shared.err(f"vLLM model cache not found at {self._cache_home} — "
                       "run setup_check.py to download at least one model first")
            return False
        Shared.ok(f"vLLM found at {self._launcher or self._executable} — models load on demand per test")
        return True

    def start(self, *, gpu_visible: bool = True, timeout: int = 15) -> bool:  # pragma: no cover — thin wrapper
        self._gpu_visible = gpu_visible
        return self.ensure_running()

    def stop(self, *, timeout: int = 15) -> None:  # pragma: no cover — kills real processes
        """Stop our subprocess, then any stray vLLM holding the GPU — including a
        preconfigured server this project did not start."""
        self._stop_process(timeout=timeout)
        try:
            if platform.system() == "Windows":
                subprocess.run(["taskkill", "/IM", "vllm.exe", "/F"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.run(["pkill", "-f", "vllm serve"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            pass

    def _stop_process(self, timeout: int = 15) -> None:  # pragma: no cover — kills a real process
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        self._loaded_tag = None
        self._loaded_num_ctx = None
        self._loaded_embedding = None
        self._loaded_n_parallel = 1
        self._loaded_tool_parser = None

    def is_connection_crash(self, e: Exception) -> bool:
        if isinstance(e, (requests.exceptions.ConnectionError, urllib.error.URLError,
                          http.client.IncompleteRead, ConnectionError)):
            return True
        return "actively refused" in str(e).lower()

    def wait_for_recovery(self, timeout: int = 30) -> bool:
        """Always True — recovery happens in _ensure_model on the next call."""
        return True

    def reachable_or_abort(self) -> bool:
        """Always True — _ensure_model is its own per-model health check."""
        return True

    def tail_log(self, n_lines: int = 40) -> str:
        return Shared._tail_log(self._log_path, "vLLM", n_lines)

    # ── model lifecycle ──

    def model_pulled(self, tag: str) -> bool:
        repo = self._repo(tag)
        return repo is not None and hf_cache_model_complete(self._cache_home, repo)

    def list_installed_models(self) -> list[dict]:
        """Catalog tags whose vLLM weights are cached. Non-catalog tags are not
        discoverable here: a cached repo carries no tag of its own."""
        installed = []
        for model in LLM_MODELS + EMBED_MODELS:
            if not self.model_pulled(model["tag"]):
                continue
            blobs = hf_cache_model_dir(self._cache_home, model["vllm_repo"]) / "blobs"
            size = sum(path.stat().st_size for path in blobs.glob("*")) if blobs.is_dir() else None
            installed.append({"tag": model["tag"], "size": size})
        return installed

    def max_context_length(self, tag: str, default: int = 131072) -> int:
        """Read max_position_embeddings from the cached snapshot's config.json."""
        snapshot = self._snapshot_dir(tag)
        if snapshot is None:
            return default
        try:
            data = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default
        for key in ("max_position_embeddings", "max_seq_len", "n_positions"):
            value = data.get(key) or (data.get("text_config") or {}).get(key)
            if isinstance(value, int) and value > 0:
                return value
        return default

    def warmup(self, tag: str, label: str, num_ctx: int, warmup_runs: int,  # pragma: no cover — real model load
               crash_cache: dict | None = None, cache_path: Path | None = None,
               crash_extra: dict | None = None) -> bool:
        Shared.log(f"Warming up {label} at num_ctx={num_ctx} (timeout: {config.RUN_TIMEOUT}s per run) ...")
        for warmup_i in range(warmup_runs):
            t_start = time.perf_counter()
            try:
                self.generate(tag, "Hello.", timeout=config.RUN_TIMEOUT, num_ctx=num_ctx)
            except Exception as e:
                Shared.warn(f"Warmup run {warmup_i+1} failed after {time.perf_counter()-t_start:.0f}s: {e}")
                if crash_cache is not None and cache_path is not None:
                    if self.is_connection_crash(e):
                        self.wait_for_recovery()
                    Shared.record_crash(tag, crash_cache, cache_path,
                                         f"warming up at num_ctx={num_ctx}", extra=crash_extra)
                return False
            Shared.log(f"Warmup run {warmup_i+1}/{warmup_runs} done")
        return True

    def unload(self, tag: str) -> None:
        if self._loaded_tag is not None and tag == self._loaded_tag:
            self._stop_process()
            Shared.ok(f"Unloaded {tag}")

    def unload_all(self) -> None:
        if self._loaded_tag is not None:
            self.unload(self._loaded_tag)
        else:
            Shared.ok("No models currently loaded")

    def wait_until_unloaded(self, tag: str, timeout: int = 30) -> bool:
        return self._loaded_tag is None or tag != self._loaded_tag

    def prepare_concurrency(self, tag: str, n_parallel: int, per_slot_ctx: int,
                             warmup_runs: int = 1, timeout: int = 300) -> bool:  # pragma: no cover — real subprocess
        """Serve `n_parallel` concurrent sequences at `per_slot_ctx` tokens each.
        --max-model-len is per sequence, so it is NOT scaled by n_parallel."""
        try:
            self._ensure_model(tag, per_slot_ctx, n_parallel=n_parallel,
                                deadline=time.perf_counter() + timeout)
            return True
        except Exception as e:
            Shared.warn(f"Failed to load {tag} for {n_parallel}-way concurrency "
                        f"at {per_slot_ctx} tokens/slot: {e}")
            return False

    # ── model process spawn ──

    def server_command(self, repo: str, num_ctx: int | None, *, embedding: bool = False,
                       n_parallel: int = 1, tool_parser: str | None = None) -> list[str]:
        """Argv serving `repo`. A platform launcher (AMD's vllm-launch) is preferred
        over bare `vllm serve` because it carries that platform's environment."""
        options = ["--served-model-name", repo,
                    "--max-num-seqs", str(n_parallel),
                    "--gpu-memory-utilization", str(config.VLLM_GPU_MEMORY_UTILIZATION)]
        if num_ctx is not None:
            options += ["--max-model-len", str(num_ctx)]
        if embedding:
            # --task was replaced by --runner; pooling is the embedding runner.
            options += ["--runner", "pooling"]
        if tool_parser:
            # tool_calls stay empty unless the frontend parser is enabled explicitly.
            options += ["--enable-auto-tool-choice", "--tool-call-parser", tool_parser]
        if not self._launcher and not self._executable:
            raise RuntimeError("no vLLM runtime found — run setup_check.py or install vLLM")
        if self._launcher:
            return [self._launcher, "-p", str(config.VLLM_PORT), "-m", repo, *options]
        return [self._executable, "serve", repo, "--host", "127.0.0.1",
                "--port", str(config.VLLM_PORT), *options]

    def _ensure_model(self, tag: str, num_ctx: int | None, *, embedding: bool = False,
                       n_parallel: int = 1, deadline: float | None = None,
                       tool_parser: str | None = None) -> None:
        """Ensure vLLM is serving `tag`, respawning on any mismatch — one model per process."""
        want = (tag, num_ctx, embedding, n_parallel, tool_parser)

        def ready():
            have = (self._loaded_tag, self._loaded_num_ctx, self._loaded_embedding,
                    self._loaded_n_parallel, self._loaded_tool_parser)
            return want == have and self._proc is not None and self._proc.poll() is None

        if ready():
            return

        with self._model_lock:
            if ready():
                return
            if deadline is not None and time.perf_counter() >= deadline:
                raise EngineTimeout(f"loading {tag} exceeded the request wall-clock timeout")

            if not self._gpu_visible:
                # vLLM has no --device flag; CPU needs a separately built CPU wheel.
                raise RuntimeError("vLLM has no CPU-only mode here — run --cpu-only against llama.cpp")
            repo = self._repo(tag)
            if repo is None:
                raise RuntimeError(f"{tag} has no vLLM weights in the catalog")
            if not self.model_pulled(tag):
                raise RuntimeError(
                    f"{repo} not found in {self._cache_home} — "
                    "download it first with: python -m scripts.setup.setup_check")

            self.stop()
            args = self.server_command(repo, num_ctx, embedding=embedding,
                                        n_parallel=n_parallel, tool_parser=tool_parser)
            log_fh = tempfile.NamedTemporaryFile(mode="w", suffix="-vllm-server.log", delete=False)
            self._log_path = Path(log_fh.name)
            try:
                proc = subprocess.Popen(args, stdout=log_fh, stderr=subprocess.STDOUT,
                                         env={**os.environ, "HF_HOME": str(self._cache_home)})
            except FileNotFoundError:
                log_fh.close()
                raise RuntimeError(f"'{args[0]}' not found in PATH") from None
            log_fh.close()
            self._proc = proc
            Shared._managed_procs.append(proc)

            t0 = time.perf_counter()
            while time.perf_counter() - t0 < self.LOAD_TIMEOUT:
                if deadline is not None and time.perf_counter() >= deadline:
                    self._stop_process()
                    raise EngineTimeout(f"loading {tag} exceeded the request wall-clock timeout")
                if self.available():
                    self._loaded_tag = tag
                    self._loaded_num_ctx = num_ctx
                    self._loaded_embedding = embedding
                    self._loaded_n_parallel = n_parallel
                    self._loaded_tool_parser = tool_parser
                    return
                if proc.poll() is not None:
                    raise RuntimeError(f"vLLM exited unexpectedly (code {proc.returncode}) "
                                        f"loading {tag} — last output:\n{self.tail_log()}")
                time.sleep(1)

            self._stop_process()
            raise RuntimeError(f"vLLM did not become healthy within {self.LOAD_TIMEOUT}s loading {tag}")

    # ── HTTP helpers ──

    @staticmethod
    def _urlopen(req, timeout):
        return openai_api.urlopen_with_detail(req, timeout, "vLLM")

    def _post(self, path: str, payload: dict, timeout: float):
        return self._urlopen(urllib.request.Request(
            f"{config.VLLM_URL}{path}", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        ), timeout)

    # ── inference ──

    def generate(self, tag: str, prompt: str, timeout: int = 600,
                 num_ctx: int | None = None, n_parallel: int = 1) -> GenerationMeasurement:
        """Generate via /v1/completions; n_parallel must match prepare_concurrency."""
        operation_start = time.perf_counter()
        deadline = operation_start + timeout
        self._ensure_model(tag, num_ctx, n_parallel=n_parallel, deadline=deadline)
        model_load_sec = time.perf_counter() - operation_start

        payload = {
            "model": self._repo(tag),
            "prompt": prompt,
            "max_tokens": config.GENERATE_MAX_TOKENS,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        request_start = time.perf_counter()
        ttft = None
        tokens = 0
        prompt_tokens = None
        finish_reason = None
        response_parts = []

        with self._post("/v1/completions", payload, max(deadline - request_start, 0.001)) as resp:
            for chunk in openai_api.iter_sse(resp):
                choice = (chunk.get("choices") or [{}])[0]
                text = choice.get("text")
                if ttft is None and text:
                    ttft = time.perf_counter() - request_start
                if text:
                    response_parts.append(text)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                usage = chunk.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    tokens = usage["completion_tokens"]
                if usage.get("prompt_tokens") is not None:
                    prompt_tokens = usage["prompt_tokens"]
                if time.perf_counter() > deadline:
                    raise EngineTimeout(f"vllm_generate exceeded {timeout}s wall-clock timeout",
                                        partial_text="".join(response_parts))

        total = time.perf_counter() - request_start
        if ttft is None:
            ttft = total
        decode_seconds = max(total - ttft, 0)
        raw_tps = tokens / decode_seconds if decode_seconds else 0
        tps = openai_api.sanitize_tps(raw_tps, tokens, ttft, total)
        return GenerationMeasurement(
            client_ttft_sec=ttft,
            generated_tokens=tokens,
            tokens_per_sec=tps,
            client_wall_sec=total,
            decode_sec=decode_seconds if tps == raw_tps else (tokens / tps if tps else 0),
            server_prompt_sec=None,   # vLLM reports no per-request prompt duration
            prompt_tokens=prompt_tokens,
            response_text="".join(response_parts),
            finish_reason=finish_reason,
            model_load_sec=model_load_sec,
            server_tps_implausible=tps != raw_tps,
        )

    def _chat_request(self, tag: str, messages: list, tools: list | None,
                      deadline: float, num_predict: int,
                      check_loop: bool, budget_nudged: bool) -> dict:
        payload = {
            "model": self._repo(tag),
            "messages": messages,
            "temperature": 0.0,
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
            raise EngineTimeout("vllm_chat exceeded its wall-clock deadline",
                                budget_nudged=budget_nudged)

        ttft = None
        tokens = 0
        prompt_eval_count = 0
        finish_reason = None
        response_parts = []
        reasoning_parts = []
        tool_fragments: dict[int, dict] = {}
        last_loop_check = request_start

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
                if usage.get("completion_tokens") is not None:
                    tokens = usage["completion_tokens"]
                if usage.get("prompt_tokens") is not None:
                    prompt_eval_count = usage["prompt_tokens"]

                now = time.perf_counter()
                response_text = "".join(response_parts) or "".join(reasoning_parts)
                parsed_calls = openai_api.tool_calls_from_fragments(tool_fragments)
                partial_text = (json.dumps(parsed_calls) if tools is not None and parsed_calls
                                else response_text)
                if now > deadline:
                    raise EngineTimeout("vllm_chat exceeded its wall-clock deadline",
                                        partial_text=partial_text, budget_nudged=budget_nudged)
                if check_loop and now - last_loop_check >= config.LOOP_CHECK_INTERVAL:
                    last_loop_check = now
                    if response_text and Shared.looks_like_loop(response_text):
                        raise EngineLoopDetected(
                            f"vllm_chat detected a generation loop after "
                            f"{now - request_start:.0f}s",
                            partial_text=response_text, budget_nudged=budget_nudged)

        total = time.perf_counter() - request_start
        if ttft is None:
            ttft = total
        decode_seconds = max(total - ttft, 0)
        raw_tps = tokens / decode_seconds if decode_seconds else 0
        tps = openai_api.sanitize_tps(raw_tps, tokens, ttft, total)
        if tps != raw_tps:
            decode_seconds = tokens / tps if tps else 0
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

    @staticmethod
    def _graded_response(result: dict, tools: list | None) -> str:
        if tools is not None and result["tool_calls"]:
            return json.dumps(result["tool_calls"])
        return result["response_text"]

    def _chat_with_optional_finalize(self, tag: str, messages: list, tools: list | None,
                                      timeout: int, num_ctx: int | None, num_predict: int,
                                      check_loop: bool, token_budget: int | None):
        if token_budget is not None and num_predict != -1:
            raise ValueError("token_budget cannot be combined with finite num_predict")
        tool_parser = self._tool_parser(tag) if tools is not None else None
        if tools is not None and tool_parser is None:
            raise RuntimeError(
                f"no vLLM tool-call parser is configured for {tag}; vLLM returns no tool_calls "
                "without --tool-call-parser, so a tool result here would be wrong, not zero")
        operation_start = time.perf_counter()
        deadline = operation_start + timeout
        self._ensure_model(tag, num_ctx, deadline=deadline, tool_parser=tool_parser)
        model_load_sec = time.perf_counter() - operation_start

        if token_budget is None:
            return self._chat_request(tag, messages, tools, deadline, num_predict,
                                       check_loop, False), None, False, model_load_sec

        first_budget, second_budget = split_token_budget(
            token_budget, config.ACC_FINALIZE_FRACTION)
        first = self._chat_request(tag, messages, tools, deadline, first_budget, check_loop, False)
        if first["finish_reason"] != "length":
            return first, None, False, model_load_sec
        if second_budget == 0:
            raise EngineBudgetExceeded("vllm_chat exhausted its completion-token budget",
                                        partial_text=self._graded_response(first, tools),
                                        budget_nudged=False)
        first_response = self._graded_response(first, tools)
        if time.perf_counter() >= deadline:
            raise EngineTimeout("vllm_chat exceeded its wall-clock deadline before finalization",
                                 partial_text=first_response)
        followup = [dict(message) for message in messages]
        followup.extend([
            {"role": "assistant", "content": first_response},
            {"role": "user", "content": config.ACC_FINALIZE_MESSAGE},
        ])
        second = self._chat_request(tag, followup, tools, deadline, second_budget, check_loop, True)
        if second["finish_reason"] == "length":
            raise EngineBudgetExceeded("vllm_chat exhausted its completion-token budget",
                                        partial_text=self._graded_response(second, tools))
        return first, second, True, model_load_sec

    @staticmethod
    def _chat_measurement(first: dict, second: dict | None, graded: dict,
                          budget_nudged: bool, model_load_sec: float) -> ChatMeasurement:
        if second is None:
            tokens, decode_seconds, wall_seconds = (
                first["tokens"], first["decode_seconds"], first["wall_seconds"])
        else:
            tokens = first["tokens"] + second["tokens"]
            decode_seconds = first["decode_seconds"] + second["decode_seconds"]
            wall_seconds = first["wall_seconds"] + second["wall_seconds"]
        return ChatMeasurement(
            client_ttft_sec=first["ttft"],
            generated_tokens=tokens,
            tokens_per_sec=tokens / decode_seconds if decode_seconds else 0,
            client_wall_sec=wall_seconds,
            decode_sec=decode_seconds,
            server_prompt_sec=None,
            prompt_tokens=(second or first)["prompt_eval_count"],
            response_text=graded["response_text"],
            finish_reason=graded["finish_reason"],
            tool_calls=graded["tool_calls"],
            budget_nudged=budget_nudged,
            model_load_sec=model_load_sec,
            server_tps_implausible=(first.get("server_tps_implausible", False)
                                    or bool(second and second.get("server_tps_implausible", False))),
        )

    def chat(self, tag: str, messages: list, timeout: int = 600,
             num_ctx: int | None = None, num_predict: int = 1024,
             check_loop: bool = False, token_budget: int | None = None) -> ChatMeasurement:
        first, second, budget_nudged, model_load_sec = self._chat_with_optional_finalize(
            tag, messages, None, timeout, num_ctx, num_predict, check_loop, token_budget)
        return self._chat_measurement(first, second, second or first, budget_nudged, model_load_sec)

    def chat_tools(self, tag: str, messages: list, tools: list, timeout: int = 600,
                   num_ctx: int | None = None, num_predict: int = 1024,
                   check_loop: bool = False, token_budget: int | None = None) -> ChatMeasurement:
        first, second, budget_nudged, model_load_sec = self._chat_with_optional_finalize(
            tag, messages, tools, timeout, num_ctx, num_predict, check_loop, token_budget)
        return self._chat_measurement(first, second, second or first, budget_nudged, model_load_sec)

    def embed(self, tag: str, inputs: list[str], timeout: int = 120) -> EmbeddingMeasurement:
        """Embed every input in one request, serving the model in embedding mode."""
        load_start = time.perf_counter()
        self._ensure_model(tag, num_ctx=None, embedding=True)
        model_load_sec = time.perf_counter() - load_start

        t0 = time.perf_counter()
        resp = requests.post(f"{config.VLLM_URL}/v1/embeddings",
                              json={"model": self._repo(tag), "input": inputs}, timeout=timeout)
        if not resp.ok:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:500]
            raise RuntimeError(f"vLLM rejected embed request (HTTP {resp.status_code}, "
                                f"n_inputs={len(inputs)}): {detail}")
        elapsed = time.perf_counter() - t0
        data = resp.json().get("data", [])
        embeddings = [item["embedding"] for item in sorted(data, key=lambda d: d.get("index", 0))]
        return EmbeddingMeasurement(embeddings=embeddings, client_wall_sec=elapsed,
                                     model_load_sec=model_load_sec)
