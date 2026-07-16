"""ollama.py — OllamaEngine, the Ollama implementation of InferenceEngine.

The method bodies here moved near-verbatim from Shared.ollama_* — this is the
Ollama REST/process client that used to live on Shared, now behind the engine
interface. State that's genuinely Ollama-process-specific (the server
subprocess and its log path, and whether it's running in forced CPU-only mode)
lives on the instance. The single shared list of processes to clean up on
crash still lives on Shared (Shared._managed_procs) since ComfyUI shares that
shutdown path; this engine registers its server into it.

OllamaTimeout / OllamaLoopDetected stay defined in shared.py (generic
timeout/loop signaling types referenced by name in several files) and are
imported here where the streaming read loop raises them.
"""

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

import config
from engines.base import InferenceEngine
from shared import OllamaLoopDetected, OllamaTimeout, Shared


class OllamaEngine(InferenceEngine):
    name = "ollama"

    def __init__(self):
        # The server subprocess this engine started (None if it attached to a
        # server it didn't start), and the log file capturing its stdout+stderr
        # so a crash's actual message can be surfaced later.
        self._proc: subprocess.Popen | None = None
        self._log_path: Path | None = None
        # True while Ollama is running with GPU devices hidden. shutdown_managed
        # checks this so the script doesn't die leaving a GPU-hidden process
        # running silently in the background.
        self._cpu_only_active = False

    # ── server/process lifecycle ──

    def available(self) -> bool:  # pragma: no cover — real HTTP call; other tests mock this seam directly
        try:
            r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def start(self, *, gpu_visible: bool = True, timeout: int = 15) -> bool:  # pragma: no cover — spawns a real subprocess
        """Start 'ollama serve'. gpu_visible=False forces CPU-only inference by
        hiding every accelerator (HIP/CUDA/ROCR_VISIBLE_DEVICES set empty).
        Tracked in Shared._managed_procs for cleanup on exit. Returns True once
        reachable.

        config.OLLAMA_ENV_DEFAULTS is applied first (setdefault — an operator's
        own shell export still wins) so request-queuing/model-swap/attention
        behavior is pinned instead of left to Ollama's auto-detected defaults,
        which otherwise vary by machine and version and make run-to-run
        timing comparisons unreliable."""
        env = os.environ.copy()
        for k, v in config.OLLAMA_ENV_DEFAULTS.items():
            env.setdefault(k, v)
        if not gpu_visible:
            env["HIP_VISIBLE_DEVICES"] = ""
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["ROCR_VISIBLE_DEVICES"] = ""

        os_name = platform.system()
        try:
            log_fh = tempfile.NamedTemporaryFile(
                mode="w", suffix="-ollama-server.log", delete=False
            )
            self._log_path = Path(log_fh.name)
            kwargs = dict(stdout=log_fh, stderr=subprocess.STDOUT, env=env)
            if os_name == "Windows":
                # On Windows Ollama is a system tray app; 'ollama serve' starts the server
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            proc = subprocess.Popen(["ollama", "serve"], **kwargs)
            log_fh.close()
            self._proc = proc
            Shared._managed_procs.append(proc)
        except FileNotFoundError:
            Shared.err("'ollama' not found in PATH — install from https://ollama.com/download")
            return False

        self._cpu_only_active = not gpu_visible

        for i in range(timeout):
            time.sleep(1)
            if self.available():
                Shared.ok(f"Ollama started (pid {proc.pid}) — log: {self._log_path}")
                return True
            if proc.poll() is not None:
                Shared.err(f"Ollama exited unexpectedly (code {proc.returncode})")
                Shared.err(f"Last output:\n{self.tail_log()}")
                return False

        Shared.err(f"Ollama did not respond within {timeout} seconds")
        return False

    def stop(self, *, timeout: int = 15) -> None:  # pragma: no cover — kills real processes
        """Kill any running Ollama server, including one this script didn't
        start itself, so a fresh instance can be launched with different
        environment variables (e.g. to force CPU-only for embeddings)."""
        os_name = platform.system()
        try:
            if os_name == "Windows":
                subprocess.run(["taskkill", "/IM", "ollama.exe", "/F"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.run(["pkill", "-f", "ollama serve"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            pass

        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if not self.available():
                self._cpu_only_active = False
                return
            time.sleep(1)
        Shared.warn(f"Ollama still reachable {timeout}s after attempting to stop it")

    def ensure_running(self) -> bool:  # pragma: no cover — thin live-server orchestration wrapper
        """Start Ollama if not already running. Returns True if available."""
        if self.available():
            Shared.ok("Ollama already running")
            return True

        Shared.log("Ollama not running — attempting to start ...")
        return self.start()

    def is_connection_crash(self, e: Exception) -> bool:
        """True if `e` looks like Ollama's model runner subprocess died
        (commonly OOM) rather than an ordinary request failure.

        Two timings surface as different exception shapes: connection-refused
        (runner already dead before the request) varies by HTTP client
        (requests vs urllib), so check both; mid-stream death (runner dies
        while streaming — the realistic OOM timing here) surfaces as a
        truncated read — http.client.IncompleteRead, or a builtin
        ConnectionError from the socket.
        """
        if isinstance(e, (requests.exceptions.ConnectionError, urllib.error.URLError,
                          http.client.IncompleteRead, ConnectionError)):
            return True
        return "actively refused" in str(e).lower()

    def wait_for_recovery(self, timeout: int = 30) -> bool:  # pragma: no cover — real polling loop; other tests mock this seam directly
        """Poll until Ollama's main server answers again after its model
        runner subprocess crashed — the main server itself stays up and
        respawns the runner, it just needs a few seconds. Returns False if
        it doesn't come back within `timeout`."""
        wait_t0 = time.perf_counter()
        while time.perf_counter() - wait_t0 < timeout:
            if self.available():
                return True
            time.sleep(2)
        return False

    def reachable_or_abort(self) -> bool:
        """True if Ollama is reachable. Callers looping over models should stop
        when this is False rather than continuing into model_pulled() — which
        swallows connection errors and returns False indistinguishably from
        "genuinely not pulled", so a still-down server would misreport every
        remaining model as not pulled."""
        if self.available():
            return True
        Shared.err("Ollama is not reachable — stopping remaining models in this test")
        return False

    def tail_log(self, n_lines: int = 40) -> str:
        """Return the last n_lines of the current Ollama server's captured
        output, for surfacing the real crash reason instead of guessing."""
        return Shared._tail_log(self._log_path, "Ollama", n_lines)

    # ── model lifecycle ──

    def model_pulled(self, tag: str) -> bool:
        try:
            r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
            models = r.json().get("models", [])
            names = [m["name"] for m in models]
            return tag in names or any(tag in n for n in names)
        except Exception:
            return False

    def list_installed_models(self) -> list[dict]:
        """Every Ollama tag actually pulled locally, straight from /api/tags —
        including ones outside models.py's catalog. Returns [] (not an
        exception) if Ollama isn't reachable, so callers treat "can't reach"
        and "none installed" the same for listing."""
        try:
            r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
            return [{"tag": m["name"], "size": m.get("size")} for m in r.json().get("models", [])]
        except Exception:
            return []

    def max_context_length(self, model_tag: str, default: int = 131072) -> int:
        """Look up a pulled model's real max context length via /api/show.

        Reads manifest metadata only (no model load), so it's cheap. The
        context-length key is architecture-prefixed (llama.context_length,
        qwen35.context_length, gptoss.context_length, ...), so scan for any key
        ending in that suffix rather than guessing the prefix. Falls back to
        `default` if the model isn't found, the field is missing, or the
        request fails — callers then behave as if it supports exactly `default`.
        """
        try:
            r = requests.post(f"{config.OLLAMA_URL}/api/show",
                               json={"model": model_tag}, timeout=15)
            r.raise_for_status()
            info = r.json().get("model_info", {})
            for key, value in info.items():
                if key.endswith(".context_length") and isinstance(value, int):
                    return value
        except Exception:
            pass
        return default

    def warmup(self, tag: str, label: str, num_ctx: int, warmup_runs: int,  # pragma: no cover — real threaded/watchdogged model load
               crash_cache: dict | None = None, cache_path: Path | None = None,
               crash_extra: dict | None = None) -> bool:
        """
        Load `tag` into memory with `warmup_runs` blocking generate calls, each
        watchdogged by a daemon thread so a hung load (model too large for
        memory) times out after config.RUN_TIMEOUT instead of hanging the whole
        run. Returns False on the first hung or failed run so the caller can
        skip this model.

        A hang or a runner-crash exception is exactly the deterministic failure
        the crash cache exists to remember — warmup is often where an oversized
        model's OOM first shows up. Pass `crash_cache`/`cache_path` to record
        that case here too. `crash_extra` (e.g. {"bank_hash": ...}) is forwarded
        to Shared.record_crash so a bank-aware caller's check_crash_cache can
        tell a stale record apart from a current one.
        """
        Shared.log(f"Warming up {label} at num_ctx={num_ctx} (timeout: {config.RUN_TIMEOUT}s per run) ...")
        for warmup_i in range(warmup_runs):
            exc_box = [None]

            def _warmup():
                try:
                    self.generate(tag, "Hello.", timeout=config.RUN_TIMEOUT, num_ctx=num_ctx)
                except Exception as e:
                    exc_box[0] = e

            t = threading.Thread(target=_warmup, daemon=True)
            t_start = time.perf_counter()
            t.start()
            t.join(timeout=config.RUN_TIMEOUT)

            if t.is_alive():
                elapsed = time.perf_counter() - t_start
                Shared.warn(f"{label}: warmup run {warmup_i+1} did not complete within {elapsed:.0f}s")
                Shared.warn(f"{label}: model is likely too large for available memory — skipping")
                if crash_cache is not None and cache_path is not None:
                    Shared.record_crash(tag, crash_cache, cache_path,
                                         f"warming up (hung past {config.RUN_TIMEOUT}s at num_ctx={num_ctx})",
                                         extra=crash_extra)
                return False
            elif exc_box[0] is not None:
                Shared.warn(f"Warmup run {warmup_i+1} failed: {exc_box[0]}")
                if crash_cache is not None and cache_path is not None and self.is_connection_crash(exc_box[0]):
                    self.wait_for_recovery()
                    Shared.record_crash(tag, crash_cache, cache_path,
                                         f"warming up at num_ctx={num_ctx}", extra=crash_extra)
                return False
            else:
                Shared.log(f"Warmup run {warmup_i+1}/{warmup_runs} done")
        return True

    def unload(self, model_tag: str) -> None:
        """Force Ollama to evict a model from memory immediately."""
        try:
            requests.post(
                f"{config.OLLAMA_URL}/api/generate",
                json={"model": model_tag, "keep_alive": 0},
                timeout=30,
            )
            Shared.ok(f"Unloaded {model_tag}")
        except Exception as e:
            Shared.warn(f"Could not unload {model_tag}: {e}")

    def unload_all(self) -> None:
        """Unload every model currently loaded in Ollama."""
        try:
            r = requests.get(f"{config.OLLAMA_URL}/api/ps", timeout=10)
            loaded = r.json().get("models", [])
            if not loaded:
                Shared.ok("No models currently loaded in Ollama")
                return
            for m in loaded:
                self.unload(m["name"])
        except Exception as e:
            Shared.warn(f"Could not query loaded models: {e}")

    def wait_until_unloaded(self, model_tag: str, timeout: int = 30) -> None:
        """Poll /api/ps until the model no longer appears."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"{config.OLLAMA_URL}/api/ps", timeout=5)
                loaded = [m["name"] for m in r.json().get("models", [])]
                if not any(model_tag in name for name in loaded):
                    return True
            except Exception:
                pass
            time.sleep(1)
        Shared.warn(f"{model_tag} still appears loaded after {timeout}s")
        return False

    # ── NDJSON / HTTP-error parsing helpers (Ollama-protocol-specific) ──

    @staticmethod
    def _ollama_urlopen(req, timeout):
        """urlopen wrapper that surfaces the response body on HTTP error status.

        Ollama puts the actual failure reason (OOM, backend/driver crash, model
        load failure, etc.) in a JSON body even on a 500 — the bare HTTPError
        only says "HTTP Error 500: Internal Server Error" and hides it.
        """
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            try:
                detail = json.loads(body).get("error", body)
            except json.JSONDecodeError:
                detail = body
            raise RuntimeError(f"Ollama returned HTTP {e.code}: {detail[:500]}") from None

    @staticmethod
    def _iter_ndjson(resp):
        """Yield parsed JSON objects from a streaming NDJSON response body,
        skipping blank or malformed lines."""
        for raw_line in resp:
            if not raw_line.strip():
                continue
            try:
                yield json.loads(raw_line)
            except json.JSONDecodeError:
                continue

    @staticmethod
    def _final_chunk_metrics(chunk: dict, tokens: int):
        """Extract (ttft_override, eval_count, tps) from Ollama's final
        ("done") stream chunk. ttft_override is None when the server didn't
        report prompt_eval_duration, so the caller should keep its own ttft."""
        eval_count    = chunk.get("eval_count", tokens)
        eval_duration = chunk.get("eval_duration", 0)        # nanoseconds
        prompt_dur    = chunk.get("prompt_eval_duration", 0) # nanoseconds

        ttft_override = prompt_dur / 1e9 if prompt_dur and prompt_dur > 0 else None
        tps = eval_count / (eval_duration / 1e9) if eval_duration and eval_duration > 0 else 0
        return ttft_override, eval_count, tps

    # ── inference ──

    def generate(self, model_tag: str, prompt: str, timeout: int = 600,
                 num_ctx: int | None = None):
        """
        Generate via Ollama and return timing metrics.
        Returns: (ttft_sec, tokens_generated, tokens_per_sec)

        Uses urllib rather than requests so streaming isn't TCP-buffered (which
        batches chunks and inflates TTFT). Ollama's final chunk carries
        server-side timing (prompt_eval_duration, eval_count, eval_duration in
        ns), preferred over wall-clock where available.

        num_ctx must match the context length being tested — without it Ollama
        uses the model default, and a prompt longer than that triggers a full
        model reload, inflating TTFT by minutes.
        """
        options: dict = {"num_predict": 512, "temperature": 0.0, "num_batch": config.OLLAMA_NUM_BATCH}
        if num_ctx is not None:
            options["num_ctx"] = num_ctx

        payload = json.dumps({
            "model":  model_tag,
            "prompt": prompt,
            "stream": True,
            "options": options,
        }).encode()

        req = urllib.request.Request(
            f"{config.OLLAMA_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t_start = time.perf_counter()
        ttft    = None
        tokens  = 0
        tps     = 0
        eval_count = 0

        with self._ollama_urlopen(req, timeout) as resp:
            for chunk in self._iter_ndjson(resp):
                # First token received — record TTFT from wall clock
                if ttft is None and chunk.get("response"):
                    ttft = time.perf_counter() - t_start

                if chunk.get("response"):
                    tokens += 1

                if chunk.get("done"):
                    # Prefer server-side TTFT (prompt processing time) if available
                    ttft_override, eval_count, tps = self._final_chunk_metrics(chunk, tokens)
                    if ttft_override is not None:
                        ttft = ttft_override
                    break

        total = time.perf_counter() - t_start
        if ttft is None:
            ttft = total
        return ttft, eval_count, tps

    def chat(self, model_tag: str, messages: list, timeout: int = 600,
             num_ctx: int | None = None, num_predict: int = 1024,
             check_loop: bool = False):
        """
        Generate via Ollama's /api/chat and return timing metrics plus the reply text.
        Returns: (ttft_sec, tokens_generated, tokens_per_sec, prompt_eval_count, response_text)

        prompt_eval_count is the *total* prompt token count for this call
        (ground truth for context depth), not just the new suffix — even when
        `messages` shares a prefix with a prior call and llama.cpp's slot cache
        skips re-processing it (that shows up as low prompt_eval_duration/ttft
        instead).

        check_loop opts into re-checking the accumulated text for a degenerate
        generation loop (Shared.looks_like_loop) every config.LOOP_CHECK_INTERVAL
        seconds instead of only once `timeout` is fully exhausted — lets the
        accuracy tests (mcq/math/code) bail out of an obviously stuck response
        well before burning the full per-question timeout. Off by default since
        it's only meaningful for single-shot, unbounded-reasoning callers.
        """
        options: dict = {"num_predict": num_predict, "temperature": 0.0, "num_batch": config.OLLAMA_NUM_BATCH}
        if num_ctx is not None:
            options["num_ctx"] = num_ctx

        payload = json.dumps({
            "model":    model_tag,
            "messages": messages,
            "stream":   True,
            "options":  options,
        }).encode()

        req = urllib.request.Request(
            f"{config.OLLAMA_URL}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t_start = time.perf_counter()
        ttft   = None
        tokens = 0
        tps    = 0
        eval_count        = 0
        prompt_eval_count = 0
        response_parts    = []
        thinking_parts    = []
        last_loop_check   = t_start

        with self._ollama_urlopen(req, timeout) as resp:
            for chunk in self._iter_ndjson(resp):
                message  = chunk.get("message", {})
                content  = message.get("content")
                thinking = message.get("thinking")

                if ttft is None and (content or thinking):
                    ttft = time.perf_counter() - t_start

                if content:
                    tokens += 1
                    response_parts.append(content)
                if thinking:
                    thinking_parts.append(thinking)

                now = time.perf_counter()

                # urlopen()'s timeout is per-read, not total duration — it
                # resets on every token. Enforce the real wall-clock deadline.
                if now - t_start > timeout:
                    partial_text = "".join(response_parts) or "".join(thinking_parts)
                    raise OllamaTimeout(f"ollama_chat exceeded {timeout}s wall-clock timeout",
                                        partial_text=partial_text)

                # Re-check for a degenerate loop every LOOP_CHECK_INTERVAL
                # seconds rather than waiting for the full timeout above —
                # a model spinning in circles is usually detectable within
                # a fraction of ACC_TIMEOUT, and there's no reason to keep
                # streaming (and burning GPU time) once it's obvious.
                if check_loop and now - last_loop_check >= config.LOOP_CHECK_INTERVAL:
                    last_loop_check = now
                    partial_text = "".join(response_parts) or "".join(thinking_parts)
                    if partial_text and Shared.looks_like_loop(partial_text):
                        raise OllamaLoopDetected(
                            f"ollama_chat detected a generation loop after {now - t_start:.0f}s",
                            partial_text=partial_text)

                if chunk.get("done"):
                    ttft_override, eval_count, tps = self._final_chunk_metrics(chunk, tokens)
                    if ttft_override is not None:
                        ttft = ttft_override
                    prompt_eval_count = chunk.get("prompt_eval_count", 0)
                    break

        total = time.perf_counter() - t_start
        if ttft is None:
            ttft = total
        # Reasoning models (Qwen3.x, DeepSeek-R1, Gemma-thinking, ...) can stream
        # their whole turn through message.thinking with message.content empty.
        # Fall back to the thinking text so the history we feed back next turn
        # isn't an empty assistant message — otherwise the growth loop overcounts
        # how much context actually persists.
        response_text = "".join(response_parts) or "".join(thinking_parts)
        return ttft, eval_count, tps, prompt_eval_count, response_text

    def embed(self, tag: str, inputs: list[str], timeout: int = 120) -> tuple[list, float]:
        """Embed every string in `inputs` in a single /api/embed call, the way
        an app ingests one document. Returns (embeddings_list, elapsed_seconds).

        Raises RuntimeError with the response detail on a non-ok status (same
        surfacing as the measured embedding call it was factored out of), so a
        rejected request carries Ollama's actual reason instead of a bare status.
        """
        t0 = time.perf_counter()
        resp = requests.post(
            f"{config.OLLAMA_URL}/api/embed",
            json={"model": tag, "input": inputs,
                  "options": {"num_batch": config.OLLAMA_NUM_BATCH}},
            timeout=timeout,
        )
        if not resp.ok:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:500]
            raise RuntimeError(
                f"Ollama rejected embed request (HTTP {resp.status_code}, "
                f"n_inputs={len(inputs)}): {detail}"
            )
        elapsed = time.perf_counter() - t0
        embeddings = resp.json().get("embeddings", [])
        return embeddings, elapsed
