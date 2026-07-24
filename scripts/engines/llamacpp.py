"""LlamaCppEngine — a llama.cpp (llama-server) InferenceEngine.
See docs/engines.md#llamacppengine for the full rationale."""

import http.client
import json
import platform
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import gguf
import requests

import config
from engines.base import InferenceEngine
from models import EMBED_MODELS, LLM_MODELS
from shared import (
    EngineBudgetExceeded,
    EngineLoopDetected,
    EngineTimeout,
    Shared,
    split_token_budget,
)


class LlamaCppEngine(InferenceEngine):
    name = "llamacpp"

    BINARY = "llama-server"

    # Model *load* time (disk read + VRAM placement), not inference time — generous since large models can take a while.
    LOAD_TIMEOUT = 300

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._log_path: Path | None = None
        # What llama-server is serving, so _ensure_model knows whether a restart is needed.
        self._loaded_tag: str | None = None
        self._loaded_num_ctx: int | None = None
        self._loaded_embedding: bool | None = None
        self._loaded_n_parallel: int = 1
        # Remembered for the lazy spawn in _ensure_model — no tag yet at start()/ensure_running() time.
        self._gpu_visible = True
        self._cpu_only_active = False
        self._model_lock = threading.RLock()

    # ── binary resolution ──

    @staticmethod
    def _binary_path() -> str | None:
        """Locate llama-server — see docs/engines.md's "Binary resolution"."""
        exe_name = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
        if config.LLAMACPP_DIR.exists():
            match = next(iter(config.LLAMACPP_DIR.rglob(exe_name)), None)
            if match is not None:
                return str(match)
        found = shutil.which("llama-server")
        if found is not None:
            return found
        if platform.system() == "Darwin":
            for prefix in ("/opt/homebrew/bin", "/usr/local/bin"):
                candidate = Path(prefix) / exe_name
                if candidate.exists():
                    return str(candidate)
        return None

    # ── local model-file resolution ──

    @classmethod
    def _models_dir(cls) -> Path:
        """This engine's namespaced model directory — see docs/engines.md."""
        return config.MODELS_DIR / cls.name

    @staticmethod
    def _slug(tag: str) -> str:
        """Filesystem-safe per-tag directory name, e.g. "x:3b" -> "x_3b"."""
        return tag.replace(":", "_").replace("/", "_")

    @staticmethod
    def _catalog_entry(tag: str) -> dict | None:
        """Look up `tag`'s hf_repo/hf_file in models.py's catalog."""
        for model in LLM_MODELS + EMBED_MODELS:
            if model["tag"] == tag:
                return model
        return None

    @classmethod
    def _resolve_model_files(cls, tag: str) -> list[Path] | None:
        """Map a catalog or custom tag to its downloaded GGUF file(s), or None
        if incomplete/ambiguous — see docs/engines.md's model-directory layout."""
        entry = cls._catalog_entry(tag)
        if entry is None:
            if tag != Path(tag).name:
                return None
            paths = sorted((cls._models_dir() / tag).glob("*.gguf"))
            part_re = re.compile(r"^(.*)-(\d+)-of-(\d+)\.gguf$", re.IGNORECASE)
            matches = [part_re.match(path.name) for path in paths]
            if len(paths) == 1 and not matches[0]:
                return paths
            if not paths or not all(matches):
                return None
            prefixes = {match.group(1) for match in matches}
            totals = {int(match.group(3)) for match in matches}
            if len(prefixes) != 1 or len(totals) != 1:
                return None
            total = totals.pop()
            by_part = {int(match.group(2)): path for match, path in zip(matches, paths)}
            expected_parts = set(range(1, total + 1))
            return [by_part[i] for i in range(1, total + 1)] if set(by_part) == expected_parts else None
        hf_files = entry["hf_file"]
        filenames = hf_files if isinstance(hf_files, list) else [hf_files]
        model_dir = cls._models_dir() / cls._slug(tag)
        paths = [model_dir / Path(name).name for name in filenames]
        if all(p.exists() for p in paths):
            return paths
        return None

    # ── server/process lifecycle ──

    def available(self) -> bool:  # pragma: no cover — real HTTP call
        try:
            r = requests.get(f"{config.LLAMACPP_URL}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def ensure_running(self) -> bool:
        """Preflight only (binary + model dir exist) — see docs/engines.md's
        "No standalone up-but-idle state". The real spawn is lazy, per tag, in _ensure_model."""
        if self._binary_path() is None:
            Shared.err(f"'{self.BINARY}' not found — run setup_check.py "
                       "to install it, or build/install llama.cpp yourself: "
                       "https://github.com/ggml-org/llama.cpp")
            return False
        if not self._models_dir().exists():
            Shared.err(f"Model directory not found at {self._models_dir()} — "
                       "run setup_check.py to download at least one model first")
            return False
        Shared.ok(f"{self.BINARY} found at {self._binary_path()} — models load on demand per test")
        return True

    def start(self, *, gpu_visible: bool = True, timeout: int = 15) -> bool:  # pragma: no cover — thin wrapper over ensure_running
        """Remember gpu_visible for the next lazy spawn in _ensure_model."""
        self._gpu_visible = gpu_visible
        self._cpu_only_active = not gpu_visible
        return self.ensure_running()

    def stop(self, *, timeout: int = 15) -> None:  # pragma: no cover — kills real processes
        """Stop this engine's subprocess, then reap any stray llama-server
        from a previous crashed run so a fresh instance can bind the port."""
        self._stop_process(timeout=timeout)
        os_name = platform.system()
        try:
            if os_name == "Windows":
                subprocess.run(["taskkill", "/IM", "llama-server.exe", "/F"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.run(["pkill", "-f", self.BINARY],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            pass
        self._cpu_only_active = False

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

    def is_connection_crash(self, e: Exception) -> bool:
        """True for the exception shapes a dead HTTP server surfaces as."""
        if isinstance(e, (requests.exceptions.ConnectionError, urllib.error.URLError,
                          http.client.IncompleteRead, ConnectionError)):
            return True
        return "actively refused" in str(e).lower()

    def wait_for_recovery(self, timeout: int = 30) -> bool:
        """Always True — see docs/engines.md; recovery happens synchronously
        in _ensure_model on the next call instead."""
        return True

    def reachable_or_abort(self) -> bool:
        """Always True — see docs/engines.md; _ensure_model is its own
        per-model health check, so there's no shared server state to poll here."""
        return True

    def tail_log(self, n_lines: int = 40) -> str:
        return Shared._tail_log(self._log_path, "llama.cpp", n_lines)

    # ── model lifecycle ──

    def model_pulled(self, tag: str) -> bool:
        return self._resolve_model_files(tag) is not None

    def list_installed_models(self) -> list[dict]:
        """Every fully-present catalog tag, plus any non-catalog directory —
        see docs/engines.md's custom-tag resolution."""
        installed = []
        for model in LLM_MODELS + EMBED_MODELS:
            paths = self._resolve_model_files(model["tag"])
            if paths is not None:
                installed.append({"tag": model["tag"], "size": sum(p.stat().st_size for p in paths)})

        models_dir = self._models_dir()
        if models_dir.exists():
            catalog_slugs = {self._slug(model["tag"]) for model in LLM_MODELS + EMBED_MODELS}
            for entry in sorted(p for p in models_dir.iterdir() if p.is_dir()):
                if entry.name in catalog_slugs:
                    continue
                ggufs = self._resolve_model_files(entry.name)
                if ggufs is not None:
                    installed.append({"tag": entry.name, "size": sum(p.stat().st_size for p in ggufs)})
        return installed

    @staticmethod
    def _backend_from_device_listing(output: str) -> str:
        for line in output.splitlines():
            device = line.strip().lower()
            if re.match(r"cuda\d*\s*:", device):
                return "cuda"
            if re.match(r"(?:rocm|hip)\d*\s*:", device):
                return "rocm"
            if re.match(r"(?:metal|mtl)\d*\s*:", device):
                return "metal"
            if re.match(r"(?:sycl|level[- ]?zero)\d*\s*:", device):
                return "xpu"
            if re.match(r"vulkan\d*\s*:", device):
                return "vulkan"
        return "cpu"

    def runtime_backend(self, hardware_backend: str, *, cpu_only: bool = False) -> str:
        if cpu_only:
            return "cpu"
        binary = self._binary_path()
        if binary is None:
            return hardware_backend
        try:
            completed = subprocess.run(
                [binary, "--list-devices"], capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return hardware_backend
        output = f"{completed.stdout}\n{completed.stderr}"
        return self._backend_from_device_listing(output) if completed.returncode == 0 else hardware_backend

    def max_context_length(self, tag: str, default: int = 131072) -> int:
        """Read a model's max context from its GGUF metadata, without loading weights.
        Matches the bare "{arch}.context_length" key, not ".rope.scaling.original_context_length" (YaRN's much smaller pre-scaling base)."""
        paths = self._resolve_model_files(tag)
        if paths is None:
            return default
        try:
            reader = gguf.GGUFReader(str(paths[0]))
            for key, field in reader.fields.items():
                if re.fullmatch(r"[^.]+\.context_length", key):
                    value = field.contents()
                    if isinstance(value, int):
                        return value
        except Exception:
            pass
        return default

    def warmup(self, tag: str, label: str, num_ctx: int, warmup_runs: int,  # pragma: no cover — real model load/inference
               crash_cache: dict | None = None, cache_path: Path | None = None,
               crash_extra: dict | None = None) -> bool:
        """Warm the exact server configuration used by the following calls;
        a timed-out load is synchronously stopped before returning."""
        Shared.log(f"Warming up {label} at num_ctx={num_ctx} (timeout: {config.RUN_TIMEOUT}s per run) ...")
        for warmup_i in range(warmup_runs):
            t_start = time.perf_counter()
            try:
                self.generate(tag, "Hello.", timeout=config.RUN_TIMEOUT, num_ctx=num_ctx)
            except Exception as e:
                elapsed = time.perf_counter() - t_start
                Shared.warn(f"Warmup run {warmup_i+1} failed after {elapsed:.0f}s: {e}")
                # Any warmup exception means this tag failed to load, not just connection-crash shapes.
                if crash_cache is not None and cache_path is not None:
                    if self.is_connection_crash(e):
                        self.wait_for_recovery()
                    Shared.record_crash(tag, crash_cache, cache_path,
                                         f"warming up at num_ctx={num_ctx}", extra=crash_extra)
                return False
            Shared.log(f"Warmup run {warmup_i+1}/{warmup_runs} done")
        return True

    def unload(self, tag: str) -> None:
        """Stop the process if `tag` is the one currently loaded, else no-op."""
        if self._loaded_tag is not None and tag == self._loaded_tag:
            self._stop_process()
            Shared.ok(f"Unloaded {tag}")

    def unload_all(self) -> None:
        if self._loaded_tag is not None:
            self.unload(self._loaded_tag)
        else:
            Shared.ok("No models currently loaded")

    def wait_until_unloaded(self, tag: str, timeout: int = 30) -> bool:
        """unload() is synchronous, so this just reports current state."""
        return self._loaded_tag is None or tag != self._loaded_tag

    def prepare_concurrency(self, tag: str, n_parallel: int, per_slot_ctx: int,
                             warmup_runs: int = 1, timeout: int = 300) -> bool:  # pragma: no cover — spawns a real subprocess
        """(Re)spawn llama-server with --parallel n_parallel slots at
        per_slot_ctx tokens each — see docs/engines.md's "prepare_concurrency"."""
        try:
            self._ensure_model(
                tag, per_slot_ctx, n_parallel=n_parallel,
                deadline=time.perf_counter() + timeout,
            )
            return True
        except Exception as e:
            Shared.warn(f"Failed to load {tag} for {n_parallel}-way concurrency "
                        f"at {per_slot_ctx} tokens/slot: {e}")
            return False

    # ── HTTP streaming helpers (llama-server's SSE protocol) ──

    @staticmethod
    def _urlopen(req, timeout):
        """urlopen wrapper that surfaces llama-server's JSON error body on
        HTTP errors, instead of the bare "HTTP Error 500" HTTPError message."""
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            try:
                detail = json.loads(body).get("error", body)
            except json.JSONDecodeError:
                detail = body
            raise RuntimeError(f"llama-server returned HTTP {e.code}: {str(detail)[:500]}") from None

    @staticmethod
    def _iter_sse(resp):
        """Yield parsed JSON from an SSE response body ('data: {...}' lines).
        Empty dicts for comments/malformed/[DONE] lines, so callers can still enforce a deadline on keepalive-only traffic."""
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

    @staticmethod
    def _sanitize_tps(tps: float, tokens: int, ttft: float, total: float) -> float:
        """Replace an implausible self-reported tps with a wall-clock estimate
        — see docs/engines.md's "_sanitize_tps"."""
        if tps <= config.MAX_PLAUSIBLE_TPS:
            return tps
        decode_elapsed = total - ttft
        return tokens / decode_elapsed if decode_elapsed > 0 else 0

    @staticmethod
    def _warn_tps_sanitized(tag: str, raw_tps: float, sanitized_tps: float,
                             tokens: int, server_predicted_n: int, predicted_ms: float) -> None:
        """Logs the raw server values behind a _sanitize_tps substitution —
        see docs/engines.md's "_warn_tps_sanitized"."""
        Shared.warn(f"{tag}: implausible tps from server (server predicted_n={server_predicted_n}, "
                    f"response tokens={tokens}, predicted_ms={predicted_ms!r}, raw tps={raw_tps:.1f}) — "
                    f"using wall-clock estimate ({sanitized_tps:.1f} tok/s) instead")

    # ── model process spawn ──

    def _ensure_model(self, tag: str, num_ctx: int | None, *, embedding: bool = False,
                       n_parallel: int = 1, deadline: float | None = None) -> None:
        """Ensure llama-server is serving `tag` at `num_ctx`, respawning on any
        mismatch (model/context/mode/parallel-slots) — llama-server is single-model-per-process."""
        want = (tag, num_ctx, embedding, n_parallel)

        def ready():
            have = (self._loaded_tag, self._loaded_num_ctx,
                    self._loaded_embedding, self._loaded_n_parallel)
            return want == have and self._proc is not None and self._proc.poll() is None

        if ready():
            return

        with self._model_lock:
            if ready():
                return
            if deadline is not None and time.perf_counter() >= deadline:
                raise EngineTimeout(f"loading {tag} exceeded the request wall-clock timeout")

            paths = self._resolve_model_files(tag)
            if paths is None:
                raise RuntimeError(
                    f"{tag} not found under {config.MODELS_DIR} — "
                    "download it first with: python setup_check.py"
                )

            self._stop_process()

            binary = self._binary_path()
            if binary is None:
                raise RuntimeError(f"'{self.BINARY}' not found — run setup_check.py to install it")

            args = [
                binary,
                "-m", str(paths[0]),
                "--host", "127.0.0.1",
                "--port", str(config.LLAMACPP_PORT),
                "-ngl", "0" if not self._gpu_visible else "999",
                "--jinja",   # renders the model's own chat template, not llama.cpp's guessing heuristic — see docs/engines.md
                "-b", str(config.LLAMACPP_NUM_BATCH),
            ]
            if num_ctx is not None:
                # -c is a total KV-cache budget split across --parallel slots — see docs/engines.md.
                args += ["-c", str(num_ctx * n_parallel)]
            if embedding:
                args += ["--embeddings", "--pooling", "mean"]
            # Always pinned, even at 1 — see docs/engines.md's "--parallel is always pinned".
            args += ["--parallel", str(n_parallel)]

            log_fh = tempfile.NamedTemporaryFile(mode="w", suffix="-llamacpp-server.log", delete=False)
            self._log_path = Path(log_fh.name)
            try:
                proc = subprocess.Popen(args, stdout=log_fh, stderr=subprocess.STDOUT)
            except FileNotFoundError:
                log_fh.close()
                raise RuntimeError(f"'{self.BINARY}' not found in PATH") from None
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
                    return
                if proc.poll() is not None:
                    raise RuntimeError(f"llama-server exited unexpectedly (code {proc.returncode}) "
                                       f"loading {tag} — last output:\n{self.tail_log()}")
                time.sleep(1)

            self._stop_process()
            raise RuntimeError(f"llama-server did not become healthy within {self.LOAD_TIMEOUT}s loading {tag}")

    # ── inference ──

    def generate(self, tag: str, prompt: str, timeout: int = 600,
                 num_ctx: int | None = None, n_parallel: int = 1) -> tuple[float, int, float]:
        """Generate via /completion. Returns (ttft_sec, tokens_generated,
        tokens_per_sec). n_parallel must match the last prepare_concurrency call."""
        t_start = time.perf_counter()
        deadline = t_start + timeout
        self._ensure_model(tag, num_ctx, n_parallel=n_parallel, deadline=deadline)

        payload = json.dumps({
            "prompt": prompt,
            "n_predict": config.GENERATE_MAX_TOKENS,
            "temperature": 0.0,
            "stream": True,
            "return_tokens": True,
        }).encode()
        req = urllib.request.Request(
            f"{config.LLAMACPP_URL}/completion",
            data=payload, headers={"Content-Type": "application/json"}, method="POST",
        )

        ttft   = None
        tokens = 0
        tps    = 0
        server_predicted_n = 0
        predicted_ms       = 0
        response_parts = []

        remaining = max(deadline - time.perf_counter(), 0.001)
        with self._urlopen(req, remaining) as resp:
            for chunk in self._iter_sse(resp):
                content = chunk.get("content")
                if ttft is None and content:
                    ttft = time.perf_counter() - t_start
                if content:
                    response_parts.append(content)
                tokens += len(chunk.get("tokens") or [])

                if time.perf_counter() > deadline:
                    raise EngineTimeout(f"llamacpp_generate exceeded {timeout}s wall-clock timeout",
                                        partial_text="".join(response_parts))

                timings = chunk.get("timings")
                if timings:
                    server_predicted_n = timings.get("predicted_n", tokens)
                    predicted_ms = timings.get("predicted_ms") or 0

        total = time.perf_counter() - t_start
        if not tokens:
            tokens = server_predicted_n
        if ttft is None:
            ttft = total
        if predicted_ms:
            tps = tokens / (predicted_ms / 1000)
        elif total > ttft:
            tps = tokens / (total - ttft)
        sanitized = self._sanitize_tps(tps, tokens, ttft, total)
        if sanitized != tps:
            self._warn_tps_sanitized(tag, tps, sanitized, tokens, server_predicted_n, predicted_ms)
        return ttft, tokens, sanitized

    def _chat_request(self, tag: str, messages: list, tools: list | None,
                      deadline: float, request_start: float, num_predict: int,
                      check_loop: bool, budget_nudged: bool) -> dict:
        payload = {
            "messages": messages,
            "n_predict": num_predict,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools is not None:
            payload.update({"tools": tools, "tool_choice": "auto"})
        req = urllib.request.Request(
            f"{config.LLAMACPP_URL}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        ttft = None
        tokens = 0
        server_predicted_n = 0
        predicted_ms = 0
        prompt_eval_count = 0
        response_parts = []
        reasoning_parts = []
        tool_fragments: dict[int, dict] = {}
        finish_reason = None
        last_loop_check = request_start

        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise EngineTimeout(
                "llamacpp_chat exceeded its wall-clock deadline",
                budget_nudged=budget_nudged,
            )
        with self._urlopen(req, remaining) as resp:
            for chunk in self._iter_sse(resp):
                choices = chunk.get("choices") or [{}]
                choice = choices[0]
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
                    for call in tool_calls:
                        idx = call.get("index", 0)
                        fragment = tool_fragments.setdefault(
                            idx, {"name": "", "arguments": ""},
                        )
                        function = call.get("function") or {}
                        if function.get("name"):
                            fragment["name"] = function["name"]
                        if function.get("arguments"):
                            fragment["arguments"] += function["arguments"]

                now = time.perf_counter()
                response_text = "".join(response_parts) or "".join(reasoning_parts)
                parsed_calls = self._tool_calls_from_fragments(tool_fragments)
                partial_text = (
                    json.dumps(parsed_calls) if tools is not None and parsed_calls
                    else response_text
                )
                if now > deadline:
                    raise EngineTimeout(
                        "llamacpp_chat exceeded its wall-clock deadline",
                        partial_text=partial_text,
                        budget_nudged=budget_nudged,
                    )
                if check_loop and now - last_loop_check >= config.LOOP_CHECK_INTERVAL:
                    last_loop_check = now
                    if response_text and Shared.looks_like_loop(response_text):
                        raise EngineLoopDetected(
                            f"llamacpp_chat detected a generation loop after "
                            f"{now - request_start:.0f}s",
                            partial_text=response_text,
                            budget_nudged=budget_nudged,
                        )

                timings = chunk.get("timings")
                if timings:
                    server_predicted_n = timings.get("predicted_n", tokens)
                    predicted_ms = timings.get("predicted_ms") or 0
                    prompt_ms = timings.get("prompt_ms")
                    prompt_n = timings.get("prompt_n")
                    if not tokens:
                        tokens = server_predicted_n
                    if prompt_ms is not None and prompt_ms > 0:
                        ttft = prompt_ms / 1000
                    if prompt_n is not None:
                        prompt_eval_count = prompt_n
                usage = chunk.get("usage")
                if usage and usage.get("prompt_tokens") is not None:
                    prompt_eval_count = usage["prompt_tokens"]
                if usage and usage.get("completion_tokens") is not None:
                    tokens = usage["completion_tokens"]

        total = time.perf_counter() - request_start
        if ttft is None:
            ttft = total
        decode_seconds = predicted_ms / 1000 if predicted_ms else max(total - ttft, 0)
        raw_tps = tokens / decode_seconds if decode_seconds else 0
        tps = self._sanitize_tps(raw_tps, tokens, ttft, total)
        if tps != raw_tps:
            self._warn_tps_sanitized(
                tag, raw_tps, tps, tokens, server_predicted_n, predicted_ms,
            )
            decode_seconds = tokens / tps if tps else 0
        return {
            "ttft": ttft,
            "tokens": tokens,
            "tps": tps,
            "decode_seconds": decode_seconds,
            "prompt_eval_count": prompt_eval_count,
            "response_text": "".join(response_parts) or "".join(reasoning_parts),
            "tool_calls": self._tool_calls_from_fragments(tool_fragments),
            "finish_reason": finish_reason,
        }

    @staticmethod
    def _graded_response(result: dict, tools: list | None) -> str:
        if tools is not None and result["tool_calls"]:
            return json.dumps(result["tool_calls"])
        return result["response_text"]

    def _chat_with_optional_finalize(
            self, tag: str, messages: list, tools: list | None, timeout: int,
            num_ctx: int | None, num_predict: int, check_loop: bool,
            token_budget: int | None):
        if token_budget is not None and num_predict != -1:
            raise ValueError("token_budget cannot be combined with finite num_predict")
        t_start = time.perf_counter()
        deadline = t_start + timeout
        self._ensure_model(tag, num_ctx, deadline=deadline)

        if token_budget is None:
            result = self._chat_request(
                tag, messages, tools, deadline, t_start, num_predict, check_loop, False,
            )
            return result, None, False

        first_budget, second_budget = split_token_budget(
            token_budget, config.ACC_FINALIZE_FRACTION,
        )
        first = self._chat_request(
            tag, messages, tools, deadline, t_start, first_budget, check_loop, False,
        )
        if first["finish_reason"] != "length":
            return first, None, False
        if second_budget == 0:
            raise EngineBudgetExceeded(
                "llamacpp_chat exhausted its completion-token budget",
                partial_text=self._graded_response(first, tools),
                budget_nudged=False,
            )

        first_response = self._graded_response(first, tools)
        if time.perf_counter() >= deadline:
            raise EngineTimeout(
                "llamacpp_chat exceeded its wall-clock deadline before finalization",
                partial_text=first_response,
            )
        followup = [dict(message) for message in messages]
        followup.extend([
            {"role": "assistant", "content": first_response},
            {"role": "user", "content": config.ACC_FINALIZE_MESSAGE},
        ])
        second_start = time.perf_counter()
        second = self._chat_request(
            tag, followup, tools, deadline, second_start, second_budget, check_loop, True,
        )
        if second["finish_reason"] == "length":
            raise EngineBudgetExceeded(
                "llamacpp_chat exhausted its completion-token budget",
                partial_text=self._graded_response(second, tools),
            )
        return first, second, True

    @staticmethod
    def _combined_metrics(first: dict, second: dict | None) -> tuple[float, int, float, int]:
        if second is None:
            return (
                first["ttft"], first["tokens"], first["tps"],
                first["prompt_eval_count"],
            )
        tokens = first["tokens"] + second["tokens"]
        decode_seconds = first["decode_seconds"] + second["decode_seconds"]
        return (
            first["ttft"],
            tokens,
            tokens / decode_seconds if decode_seconds else 0,
            second["prompt_eval_count"],
        )

    def chat(self, tag: str, messages: list, timeout: int = 600,
             num_ctx: int | None = None, num_predict: int = 1024,
             check_loop: bool = False, token_budget: int | None = None):
        """Chat once, or use a bounded final-answer pass after a length stop."""
        first, second, budget_nudged = self._chat_with_optional_finalize(
            tag, messages, None, timeout, num_ctx, num_predict, check_loop, token_budget,
        )
        graded = second or first
        metrics = self._combined_metrics(first, second)
        result = (*metrics, graded["response_text"])
        return (*result, budget_nudged) if token_budget is not None else result

    def chat_tools(self, tag: str, messages: list, tools: list, timeout: int = 600,
                   num_ctx: int | None = None, num_predict: int = 1024,
                   check_loop: bool = False, token_budget: int | None = None):
        """Tool chat once, or request one complete replacement after a length stop."""
        first, second, budget_nudged = self._chat_with_optional_finalize(
            tag, messages, tools, timeout, num_ctx, num_predict, check_loop, token_budget,
        )
        graded = second or first
        metrics = self._combined_metrics(first, second)
        result = (*metrics, graded["response_text"], graded["tool_calls"])
        return (*result, budget_nudged) if token_budget is not None else result

    @staticmethod
    def _tool_calls_from_fragments(tool_fragments: dict[int, dict]) -> list[dict]:
        tool_calls_out = []
        for idx in sorted(tool_fragments):
            frag = tool_fragments[idx]
            call = {"name": frag["name"], "arguments": {}}
            try:
                call["arguments"] = json.loads(frag["arguments"]) if frag["arguments"] else {}
            except json.JSONDecodeError:
                call["incomplete"] = True
            tool_calls_out.append(call)
        return tool_calls_out

    def embed(self, tag: str, inputs: list[str], timeout: int = 120) -> tuple[list, float]:
        """Embed every string in `inputs` in one /v1/embeddings call, loading
        the model in embedding mode. Returns (embeddings_list, elapsed_seconds)."""
        self._ensure_model(tag, num_ctx=None, embedding=True)

        t0 = time.perf_counter()
        resp = requests.post(
            f"{config.LLAMACPP_URL}/v1/embeddings",
            json={"input": inputs},
            timeout=timeout,
        )
        if not resp.ok:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:500]
            raise RuntimeError(
                f"llama-server rejected embed request (HTTP {resp.status_code}, "
                f"n_inputs={len(inputs)}): {detail}"
            )
        elapsed = time.perf_counter() - t0
        data = resp.json().get("data", [])
        embeddings = [d["embedding"] for d in sorted(data, key=lambda d: d.get("index", 0))]
        return embeddings, elapsed
