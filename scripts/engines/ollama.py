"""ollama.py — OllamaEngine, the Ollama REST/process client implementation of
InferenceEngine. See docs/engines.md#ollamaengine.
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
        self._active_num_parallel = 1  # see prepare_concurrency

    # ── server/process lifecycle ──

    def available(self) -> bool:  # pragma: no cover — real HTTP call; other tests mock this seam directly
        try:
            r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def start(self, *, gpu_visible: bool = True, timeout: int = 15,
              env_overrides: dict | None = None) -> bool:  # pragma: no cover — spawns a real subprocess
        """Start 'ollama serve'. gpu_visible=False forces CPU-only by hiding
        every accelerator env var. Tracked in Shared._managed_procs. Returns
        True once reachable. config.OLLAMA_ENV_DEFAULTS applies first
        (setdefault — a shell export still wins); env_overrides applies after
        (env.update — always wins), used only by prepare_concurrency."""
        env = os.environ.copy()
        for k, v in config.OLLAMA_ENV_DEFAULTS.items():
            env.setdefault(k, v)
        if not gpu_visible:
            env["HIP_VISIBLE_DEVICES"] = ""
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["ROCR_VISIBLE_DEVICES"] = ""
        if env_overrides:
            env.update(env_overrides)
        self._active_num_parallel = int(env.get("OLLAMA_NUM_PARALLEL", 1))

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
        self._warn_if_systemd_managed(os_name)

    def prepare_concurrency(self, tag: str, n_parallel: int, per_slot_ctx: int,
                             warmup_runs: int = 1, timeout: int = 300) -> bool:  # pragma: no cover — restarts a real server + loads a real model
        """Restart Ollama with OLLAMA_NUM_PARALLEL=n_parallel if it isn't
        already running with that value, then warm up `tag` at per_slot_ctx
        to confirm it fits. Unlike llama-server, Ollama allocates a full
        per_slot_ctx KV cache per slot rather than dividing a shared total."""
        if self._active_num_parallel != n_parallel or not self.available():
            self.stop()
            if not self.start(timeout=timeout, env_overrides={"OLLAMA_NUM_PARALLEL": str(n_parallel)}):
                return False
        return self.warmup(tag, tag, per_slot_ctx, warmup_runs)

    @staticmethod
    def _warn_if_systemd_managed(os_name: str) -> None:  # pragma: no cover — real systemctl call
        """If ollama.service is systemd-managed with auto-restart, pkill's
        "still reachable" above is really systemd relaunching it, not a slow
        shutdown — surface that distinction (read-only check, no root)."""
        if os_name != "Linux":
            return
        try:
            result = subprocess.run(
                ["systemctl", "is-active", "ollama"],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip() == "active":
                Shared.warn(
                    "Ollama is managed by a systemd service (ollama.service) that "
                    "auto-restarts it — that's very likely why it's still reachable "
                    "above, not a slow shutdown. Every --engine switch and --cpu-only "
                    "run will keep fighting this. Disable it once so this script owns "
                    "Ollama's lifecycle instead: sudo systemctl disable --now ollama"
                )
        except Exception:
            pass

    def ensure_running(self) -> bool:  # pragma: no cover — thin live-server orchestration wrapper
        """Start Ollama if not already running. Returns True if available."""
        if self.available():
            Shared.ok("Ollama already running")
            return True

        Shared.log("Ollama not running — attempting to start ...")
        return self.start()

    def is_connection_crash(self, e: Exception) -> bool:
        """True if `e` looks like Ollama's model runner died (commonly OOM)
        rather than an ordinary request failure: connection-refused (varies
        by HTTP client, so check both requests/urllib) or a mid-stream death
        surfacing as a truncated read."""
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
        """Look up a pulled model's real max context length via /api/show —
        manifest metadata only, no model load, so it's cheap. The key is
        architecture-prefixed (llama.context_length, qwen35.context_length,
        ...), so scan for any key ending in that suffix. Falls back to
        `default` if not found/missing/failed."""
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
        """Load `tag` with `warmup_runs` blocking generate calls, each
        watchdogged so a hung load (model too large) times out after
        config.RUN_TIMEOUT rather than hanging the run. Returns False on the
        first hung/failed run. A hang or runner-crash is recorded to
        `crash_cache`/`cache_path` — warmup is often where an oversized
        model's OOM first shows up. `crash_extra` forwards to
        Shared.record_crash (e.g. {"bank_hash": ...})."""
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
                 num_ctx: int | None = None, n_parallel: int = 1):
        """Generate via Ollama. Returns (ttft_sec, tokens_generated,
        tokens_per_sec). Uses urllib, not requests, so streaming isn't
        TCP-buffered (which would inflate TTFT); prefers the final chunk's
        server-side timing over wall-clock. num_ctx must match the tested
        context length, or a longer prompt triggers a full model reload.
        n_parallel is accepted for interface parity with LlamaCppEngine but
        ignored — Ollama's concurrency is server-level (prepare_concurrency),
        not per-request."""
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
        """Generate via Ollama's /api/chat. Returns (ttft_sec,
        tokens_generated, tokens_per_sec, prompt_eval_count, response_text).
        prompt_eval_count is the *total* prompt token count (ground truth for
        context depth), even when a prefix cache hit skips re-processing it
        (that shows up as a low prompt_eval_duration/ttft instead).
        check_loop re-checks for a degenerate generation loop every
        LOOP_CHECK_INTERVAL seconds instead of waiting for the full timeout —
        off by default, only meaningful for unbounded-reasoning callers."""
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

                # Re-check every LOOP_CHECK_INTERVAL rather than waiting for the full timeout — no reason to keep streaming once it's obviously stuck.
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
        # A reasoning model can stream its whole turn via message.thinking with content empty;
        # falling back avoids an empty assistant turn corrupting the next turn's history.
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
