"""llamacpp.py — LlamaCppEngine, a llama.cpp (llama-server) implementation of
InferenceEngine. Resolves each tag to GGUF file(s) downloaded ahead of time by
setup_check.py into config.MODELS_DIR, and restarts its single-model-per-
process server whenever the requested (tag, num_ctx, embedding-mode) changes.
See docs/engines.md#llamacppengine for the full rationale.
"""

import http.client
import json
import platform
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
from shared import EngineLoopDetected, EngineTimeout, Shared


class LlamaCppEngine(InferenceEngine):
    name = "llamacpp"

    BINARY = "llama-server"

    # Seconds to wait for llama-server's /health once a model's subprocess is
    # spawned. Generous — this is model *load* time (disk read + VRAM
    # placement), not inference time, and large models can take a while.
    LOAD_TIMEOUT = 300

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._log_path: Path | None = None
        # What llama-server is currently serving, so _ensure_model knows
        # whether a restart is needed. None until the first model loads.
        self._loaded_tag: str | None = None
        self._loaded_num_ctx: int | None = None
        self._loaded_embedding: bool | None = None
        self._loaded_n_parallel: int = 1
        # Remembered across calls so a lazily-spawned server (there's no
        # tag at start()/ensure_running() time to load yet) still launches
        # in the right mode.
        self._gpu_visible = True
        self._cpu_only_active = False

    # ── binary resolution ──

    @staticmethod
    def _binary_path() -> str | None:
        """Locate llama-server: config.LLAMACPP_DIR (vendored installs) first,
        then PATH, then (macOS) the two well-known Homebrew prefixes directly
        — a brew install may not be on PATH yet in another already-open shell.
        See docs/engines.md#llamacppengine."""
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
        """This engine's namespaced model directory — config.MODELS_DIR/llamacpp/
        — so a future engine with its own model format/layout (e.g. MLX)
        gets its own subtree instead of colliding with this one's."""
        return config.MODELS_DIR / cls.name

    @staticmethod
    def _slug(tag: str) -> str:
        """Filesystem-safe per-tag directory name under _models_dir(),
        e.g. "llama3.2:3b-instruct-q4_K_M" -> "llama3.2_3b-instruct-q4_K_M"."""
        return tag.replace(":", "_").replace("/", "_")

    @staticmethod
    def _catalog_entry(tag: str) -> dict | None:
        """Look up `tag` in models.py's LLM_MODELS/EMBED_MODELS catalog to
        find its hf_repo/hf_file — the only place that mapping lives."""
        for model in LLM_MODELS + EMBED_MODELS:
            if model["tag"] == tag:
                return model
        return None

    @classmethod
    def _resolve_model_files(cls, tag: str) -> list[Path] | None:
        """Map a catalog tag to its downloaded GGUF file(s) under
        _models_dir()/<slug>/, as placed there by setup_check.py's
        HuggingFace download step. `hf_file` in models.py is a single
        filename, or a list for a model split across multiple GGUF parts
        (large-tier models) — every listed file must exist locally for the
        model to count as resolved. None if `tag` isn't in the catalog or
        any expected file is missing."""
        entry = cls._catalog_entry(tag)
        if entry is None:
            return None
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
        """llama-server has no standalone "up but no model loaded" state to
        start — a process needs a model to launch with. This is the preflight
        instead: confirm the binary and model storage directory both exist,
        so a caller (e.g. --list-models, or the top of a run) gets a clear
        error before wasting time on the first per-model load. The actual
        subprocess spawns lazily per tag in _ensure_model."""
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
        """Remember gpu_visible for the next lazy spawn in _ensure_model —
        there's no tag yet to actually load a model with here. gpu_visible is
        the interface-level version of llama-server's -ngl flag (see
        InferenceEngine's docstring)."""
        self._gpu_visible = gpu_visible
        self._cpu_only_active = not gpu_visible
        return self.ensure_running()

    def stop(self, *, timeout: int = 15) -> None:  # pragma: no cover — kills real processes
        """Stop this engine's own subprocess, then also reap any stray
        llama-server left behind by a previous crashed run, so a fresh
        instance can bind the port again."""
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
        """True for the exception shapes a dead HTTP server surfaces as
        (requests/urllib connection errors, or a refused connection)."""
        if isinstance(e, (requests.exceptions.ConnectionError, urllib.error.URLError,
                          http.client.IncompleteRead, ConnectionError)):
            return True
        return "actively refused" in str(e).lower()

    def wait_for_recovery(self, timeout: int = 30) -> bool:
        """Always True: llama-server's whole process *is* the model runner,
        so there's no separate daemon to poll for recovery. Recovery instead
        happens synchronously on the next generate/chat/embed call, via
        _ensure_model respawning; an unrecoverable model still gets caught
        there, just on that attempt."""
        return True

    def reachable_or_abort(self) -> bool:
        """Always True: there's no shared server that stays up between models
        to check here — llama-server is spawned fresh per model in
        _ensure_model, which is its own health check for that specific
        model. model_pulled() reads local GGUF files straight off disk, no
        server involved, so it's never at risk of a down server making
        'reachable' and 'not downloaded' indistinguishable."""
        return True

    def tail_log(self, n_lines: int = 40) -> str:
        return Shared._tail_log(self._log_path, "llama.cpp", n_lines)

    # ── model lifecycle ──

    def model_pulled(self, tag: str) -> bool:
        return self._resolve_model_files(tag) is not None

    def list_installed_models(self) -> list[dict]:
        """Every catalog tag whose GGUF file(s) are fully present under
        _models_dir(), plus any extra subdirectory there that doesn't match a
        catalog slug — lets someone benchmark a model they dropped in
        manually without adding it to models.py first."""
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
                ggufs = sorted(entry.glob("*.gguf"))
                if ggufs:
                    installed.append({"tag": entry.name, "size": sum(p.stat().st_size for p in ggufs)})
        return installed

    def max_context_length(self, tag: str, default: int = 131072) -> int:
        """Read a downloaded model's real max context length straight from
        its GGUF metadata. GGUFReader memory-maps the file and only walks its
        key/value header section, so this never loads the model's weights.
        Scans every architecture-prefixed key convention GGUF uses
        (llama.context_length, qwen35.context_length, gptoss.context_length,
        ...) since the prefix varies by model family.
        """
        paths = self._resolve_model_files(tag)
        if paths is None:
            return default
        try:
            reader = gguf.GGUFReader(str(paths[0]))
            for key, field in reader.fields.items():
                if key.endswith(".context_length"):
                    value = field.contents()
                    if isinstance(value, int):
                        return value
        except Exception:
            pass
        return default

    def warmup(self, tag: str, label: str, num_ctx: int, warmup_runs: int,  # pragma: no cover — real threaded/watchdogged model load
               crash_cache: dict | None = None, cache_path: Path | None = None,
               crash_extra: dict | None = None) -> bool:
        """Watchdog-threaded warmup — the first call here is what actually
        spawns and loads the llama-server subprocess, via generate() ->
        _ensure_model(), so a model too large for available memory times out
        here rather than hanging the whole run."""
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
                # Every warmup exception here (not just connection-crash shapes) means this tag
                # failed to load — llama-server is freshly spawned per model, so it's as
                # deterministic as a hang.
                if crash_cache is not None and cache_path is not None:
                    if self.is_connection_crash(exc_box[0]):
                        self.wait_for_recovery()
                    Shared.record_crash(tag, crash_cache, cache_path,
                                         f"warming up at num_ctx={num_ctx}", extra=crash_extra)
                return False
            else:
                Shared.log(f"Warmup run {warmup_i+1}/{warmup_runs} done")
        return True

    def unload(self, tag: str) -> None:
        """llama-server serves one model per process, so "unload" just means
        stopping that process — a no-op if `tag` isn't the one currently
        loaded."""
        if self._loaded_tag is not None and tag == self._loaded_tag:
            self._stop_process()
            Shared.ok(f"Unloaded {tag}")

    def unload_all(self) -> None:
        if self._loaded_tag is not None:
            self.unload(self._loaded_tag)
        else:
            Shared.ok("No models currently loaded")

    def wait_until_unloaded(self, tag: str, timeout: int = 30) -> bool:
        """unload() is synchronous (it terminates and waits on the
        subprocess), so by the time either it or _stop_process returns
        there's nothing left to poll for — this just reports the current
        state."""
        return self._loaded_tag is None or tag != self._loaded_tag

    def prepare_concurrency(self, tag: str, n_parallel: int, per_slot_ctx: int,
                             warmup_runs: int = 1, timeout: int = 300) -> bool:  # pragma: no cover — spawns a real subprocess
        """(Re)spawn llama-server with --parallel n_parallel slots at
        per_slot_ctx tokens each. `warmup_runs` is accepted for interface
        parity with other engines but unused *here* — this only blocks until
        the process is up and the KV cache is allocated (process-level
        readiness), not until a real decode has run at this concurrent
        shape. ConcurrencyBenchmark.run fires the actual throwaway warmup
        batches itself, after this returns, since every level respawns the
        process (n_parallel is part of _ensure_model's want/have check) —
        each level's first real inference is on a fresh process, so it
        genuinely needs its own warmup."""
        try:
            self._ensure_model(tag, per_slot_ctx, n_parallel=n_parallel)
            return True
        except Exception as e:
            Shared.warn(f"Failed to load {tag} for {n_parallel}-way concurrency "
                        f"at {per_slot_ctx} tokens/slot: {e}")
            return False

    # ── HTTP streaming helpers (llama-server's SSE protocol) ──

    @staticmethod
    def _urlopen(req, timeout):
        """urlopen wrapper that surfaces the response body on HTTP error
        status — the bare HTTPError only says "HTTP Error 500: Internal
        Server Error" and hides llama-server's actual JSON error detail."""
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
        """Yield parsed JSON objects from a streaming Server-Sent-Events
        response body (llama-server's /completion and /v1/chat/completions
        both stream this way: 'data: {...}' lines, terminated by 'data:
        [DONE]'), skipping blank lines, non-data lines, and the [DONE]
        sentinel itself."""
        for raw_line in resp:
            line = raw_line.decode(errors="replace") if isinstance(raw_line, bytes) else raw_line
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                continue
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue

    @staticmethod
    def _sanitize_tps(tps: float, tokens: int, ttft: float, total: float) -> float:
        """Replace a self-reported tps with a wall-clock estimate whenever it
        exceeds config.MAX_PLAUSIBLE_TPS — see that constant's docstring for
        why llama-server's own numbers occasionally aren't trustworthy.
        `tokens` is our own locally-counted content-chunk count, not the
        server's predicted_n, so this fallback doesn't share the server-side
        bug it's guarding against."""
        if tps <= config.MAX_PLAUSIBLE_TPS:
            return tps
        decode_elapsed = total - ttft
        return tokens / decode_elapsed if decode_elapsed > 0 else 0

    @staticmethod
    def _warn_tps_sanitized(tag: str, raw_tps: float, sanitized_tps: float,
                             eval_count: int, predicted_ms: float) -> None:
        """Surfaces the raw server values behind a _sanitize_tps substitution
        — without this, the only trace of the bad reading is the corrected
        number, which is useless for tracking down what llama-server
        actually reported. predicted_ms is printed at full precision (not
        rounded) since the anomaly is specifically that it's an implausibly
        tiny fraction of a millisecond."""
        Shared.warn(f"{tag}: implausible tps from server (predicted_n={eval_count}, "
                    f"predicted_ms={predicted_ms!r}, raw tps={raw_tps:.1f}) — "
                    f"using wall-clock estimate ({sanitized_tps:.1f} tok/s) instead")

    # ── model process spawn ──

    def _ensure_model(self, tag: str, num_ctx: int | None, *, embedding: bool = False,
                       n_parallel: int = 1) -> None:
        """Make sure llama-server is up and serving `tag` at `num_ctx`,
        (re)spawning the subprocess if a different model, context size,
        embedding-vs-chat mode, or parallel-slot count is requested. See the
        module docstring for why every mismatch means a full restart
        (llama-server is single-model-per-process).
        `n_parallel` > 1 is only used by the concurrency test."""
        want = (tag, num_ctx, embedding, n_parallel)
        have = (self._loaded_tag, self._loaded_num_ctx, self._loaded_embedding, self._loaded_n_parallel)
        if want == have and self._proc is not None and self._proc.poll() is None and self.available():
            return

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
            # -c is a total KV-cache budget split across --parallel slots, so
            # scale it up here; num_ctx stays the per-slot value everywhere
            # else (self._loaded_num_ctx below, the want/have check above).
            args += ["-c", str(num_ctx * n_parallel)]
        if embedding:
            args += ["--embeddings", "--pooling", "mean"]
        if n_parallel > 1:
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
        """Generate via llama-server's native /completion endpoint. Returns
        (ttft_sec, tokens_generated, tokens_per_sec), preferring the server's
        own reported timings over wall clock. n_parallel must match the last
        prepare_concurrency call (default 1 elsewhere); concurrent callers
        passing the same values is safe without locking since
        _ensure_model's want == have check is then read-only."""
        self._ensure_model(tag, num_ctx, n_parallel=n_parallel)

        payload = json.dumps({
            "prompt": prompt,
            "n_predict": 512,
            "temperature": 0.0,
            "stream": True,
        }).encode()
        req = urllib.request.Request(
            f"{config.LLAMACPP_URL}/completion",
            data=payload, headers={"Content-Type": "application/json"}, method="POST",
        )

        t_start = time.perf_counter()
        ttft   = None
        tokens = 0
        tps    = 0
        eval_count   = 0
        predicted_ms = 0

        with self._urlopen(req, timeout) as resp:
            for chunk in self._iter_sse(resp):
                content = chunk.get("content")
                if ttft is None and content:
                    ttft = time.perf_counter() - t_start
                if content:
                    tokens += 1

                timings = chunk.get("timings")
                if timings:
                    eval_count = timings.get("predicted_n", tokens)
                    predicted_ms = timings.get("predicted_ms") or 0
                    prompt_ms = timings.get("prompt_ms")
                    if predicted_ms:
                        tps = eval_count / (predicted_ms / 1000)
                    if prompt_ms is not None and prompt_ms > 0:
                        ttft = prompt_ms / 1000

        total = time.perf_counter() - t_start
        if ttft is None:
            ttft = total
        sanitized = self._sanitize_tps(tps, tokens, ttft, total)
        if sanitized != tps:
            self._warn_tps_sanitized(tag, tps, sanitized, eval_count, predicted_ms)
        return ttft, eval_count, sanitized

    def chat(self, tag: str, messages: list, timeout: int = 600,
             num_ctx: int | None = None, num_predict: int = 1024,
             check_loop: bool = False):
        """Generate via llama-server's OpenAI-compatible /v1/chat/completions.
        Returns (ttft_sec, tokens_generated, tokens_per_sec, prompt_eval_count,
        response_text). n_predict is passed straight through as an extension
        field (-1 = unbounded). check_loop, when set, polls the streaming
        response for a degenerate generation loop (see Shared.looks_like_loop)
        and raises EngineLoopDetected early rather than waiting out the full
        timeout. stream_options.include_usage asks for a trailing usage
        chunk — prompt_eval_count reads its prompt_tokens (the true running
        total), not timings.prompt_n (only newly-prefilled tokens this call,
        which under-counts once the prefix cache kicks in)."""
        self._ensure_model(tag, num_ctx)

        payload = json.dumps({
            "messages":       messages,
            "n_predict":      num_predict,
            "temperature":    0.0,
            "stream":         True,
            "stream_options": {"include_usage": True},
        }).encode()
        req = urllib.request.Request(
            f"{config.LLAMACPP_URL}/v1/chat/completions",
            data=payload, headers={"Content-Type": "application/json"}, method="POST",
        )

        t_start = time.perf_counter()
        ttft   = None
        tokens = 0
        tps    = 0
        eval_count        = 0
        predicted_ms      = 0
        prompt_eval_count = 0
        response_parts    = []
        reasoning_parts    = []
        last_loop_check   = t_start

        with self._urlopen(req, timeout) as resp:
            for chunk in self._iter_sse(resp):
                choices   = chunk.get("choices") or [{}]
                delta     = choices[0].get("delta", {})
                content   = delta.get("content")
                reasoning = delta.get("reasoning_content")

                if ttft is None and (content or reasoning):
                    ttft = time.perf_counter() - t_start

                if content:
                    tokens += 1
                    response_parts.append(content)
                if reasoning:
                    reasoning_parts.append(reasoning)

                now = time.perf_counter()

                # urlopen()'s timeout is per-read, not total duration — it
                # resets on every token. Enforce the real wall-clock deadline.
                if now - t_start > timeout:
                    partial_text = "".join(response_parts) or "".join(reasoning_parts)
                    raise EngineTimeout(f"llamacpp_chat exceeded {timeout}s wall-clock timeout",
                                        partial_text=partial_text)

                if check_loop and now - last_loop_check >= config.LOOP_CHECK_INTERVAL:
                    last_loop_check = now
                    partial_text = "".join(response_parts) or "".join(reasoning_parts)
                    if partial_text and Shared.looks_like_loop(partial_text):
                        raise EngineLoopDetected(
                            f"llamacpp_chat detected a generation loop after {now - t_start:.0f}s",
                            partial_text=partial_text)

                timings = chunk.get("timings")
                if timings:
                    eval_count   = timings.get("predicted_n", tokens)
                    predicted_ms = timings.get("predicted_ms") or 0
                    prompt_ms    = timings.get("prompt_ms")
                    prompt_n     = timings.get("prompt_n")
                    if predicted_ms:
                        tps = eval_count / (predicted_ms / 1000)
                    if prompt_ms is not None and prompt_ms > 0:
                        ttft = prompt_ms / 1000
                    if prompt_n is not None:
                        prompt_eval_count = prompt_n

                # Trailing chunk, so this overrides prompt_n above with the true total.
                usage = chunk.get("usage")
                if usage and usage.get("prompt_tokens") is not None:
                    prompt_eval_count = usage["prompt_tokens"]

        total = time.perf_counter() - t_start
        if ttft is None:
            ttft = total
        sanitized = self._sanitize_tps(tps, tokens, ttft, total)
        if sanitized != tps:
            self._warn_tps_sanitized(tag, tps, sanitized, eval_count, predicted_ms)
        # A reasoning model can stream its whole turn via reasoning_content with content empty;
        # falling back avoids an empty assistant turn corrupting history.
        response_text = "".join(response_parts) or "".join(reasoning_parts)
        return ttft, eval_count, sanitized, prompt_eval_count, response_text

    def embed(self, tag: str, inputs: list[str], timeout: int = 120) -> tuple[list, float]:
        """Embed every string in `inputs` in a single /v1/embeddings call.
        Returns (embeddings_list, elapsed_seconds).

        Loads the model in embedding mode (--embeddings --pooling mean) —
        embedding models need this for the endpoint to be enabled at all.
        """
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
