"""llamacpp.py — LlamaCppEngine, a llama.cpp (llama-server) implementation of
InferenceEngine.

Reuses models already pulled via `ollama pull` instead of downloading its own
copy: Ollama stores every pulled model as a content-addressed GGUF blob (no
file extension) under its models directory, referenced by a manifest JSON
that maps a tag to a sha256 digest. llama.cpp identifies GGUF files by their
magic bytes, not filename, so that blob is a valid `-m` target for
llama-server as-is — see _resolve_blob_path. This only resolves tags pulled
from the standard Ollama library namespace (registry.ollama.ai/library/...),
which is every tag in this project's catalog (models.py); a custom/private
registry pull isn't handled.

llama-server serves exactly one model per process (unlike Ollama, which swaps
models in and out of one long-running server on demand), so this engine
manages its own subprocess and restarts it whenever the requested (tag,
num_ctx, embedding-mode) combination changes — see _ensure_model. That
mirrors Ollama's own num_ctx-mismatch-triggers-reload behavior (see
OllamaEngine.chat's docstring), so the two engines pay the same kind of
cold-swap cost, just via a different mechanism, keeping cross-engine timing
comparisons meaningful.

Requires the llama.cpp 'llama-server' binary on PATH and a recent enough
build to support --jinja (renders the model's own embedded chat template
rather than llama.cpp's built-in template-guessing heuristics, for closer
parity with what Ollama renders from the same GGUF metadata).
"""

import http.client
import json
import os
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
from shared import OllamaLoopDetected, OllamaTimeout, Shared


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
        # Remembered across calls so a lazily-spawned server (there's no
        # tag at start()/ensure_running() time to load yet) still launches
        # in the right mode.
        self._gpu_visible = True
        self._cpu_only_active = False

    # ── binary resolution ──

    @staticmethod
    def _binary_path() -> str | None:
        """Locate llama-server: setup_check.py's Linux (source build) and
        Windows (prebuilt zip) install paths both vendor it somewhere under
        config.LLAMACPP_DIR rather than putting it on PATH — searched
        recursively since the exact layout varies by install method/zip
        structure. Falls back to PATH for the macOS (brew) install path,
        which does put it there, or a manual install — and, if PATH doesn't
        have it, the two well-known Homebrew prefixes directly, since a brew
        install done in one shell (e.g. by setup.sh) doesn't update PATH in
        any other already-open shell until its rc file is re-sourced. This
        keeps engine resolution independent of shell state: no `source` or
        terminal restart ever required after setup."""
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

    # ── Ollama blob-store resolution ──

    # Linux install methods that don't put models under the current user's
    # ~/.ollama/models: Ubuntu's `snap install ollama` (runs as a snap-confined
    # 'ollama' user, common data under /var/snap/...) and the official
    # curl|sh installer's systemd service (runs as a dedicated 'ollama' system
    # user, home /usr/share/ollama). 'ollama list' works fine from any user's
    # shell either way — it just talks to the running server over HTTP — but a
    # file-level reader like this one needs the real on-disk path.
    _LINUX_SERVICE_MODEL_DIRS = (
        Path("/var/snap/ollama/common/models"),
        Path("/usr/share/ollama/.ollama/models"),
    )

    @classmethod
    def _ollama_models_dir(cls) -> Path:
        """Resolve Ollama's model store. $OLLAMA_MODELS wins outright if set —
        an operator's own override always wins. Otherwise prefer the current
        user's ~/.ollama/models (manual/dev installs, and the common case),
        falling back to known service-managed locations (see
        _LINUX_SERVICE_MODEL_DIRS) if that doesn't exist. Returns the ~/.ollama
        default even when nothing was found, so callers' "not found at {path}"
        errors point at the most common location rather than an obscure one."""
        env = os.environ.get("OLLAMA_MODELS")
        if env:
            return Path(env)
        if platform.system() == "Windows":
            return Path(os.environ.get("USERPROFILE", "")) / ".ollama" / "models"

        home_dir = Path.home() / ".ollama" / "models"
        if home_dir.exists():
            return home_dir
        for candidate in cls._LINUX_SERVICE_MODEL_DIRS:
            if candidate.exists():
                return candidate
        return home_dir

    @staticmethod
    def _split_tag(tag: str) -> tuple[str, str]:
        """"llama3.2:3b-instruct-q4_K_M" -> ("llama3.2", "3b-instruct-q4_K_M");
        a bare "phi4-mini" implies ":latest", matching Ollama's own convention
        for an untagged pull."""
        name, _, version = tag.partition(":")
        return name, version or "latest"

    @classmethod
    def _manifest_path(cls, tag: str) -> Path:
        name, version = cls._split_tag(tag)
        return (cls._ollama_models_dir() / "manifests" / "registry.ollama.ai"
                 / "library" / name / version)

    @classmethod
    def _resolve_blob_path(cls, tag: str) -> Path | None:
        """Map an Ollama tag straight to the on-disk GGUF blob backing it,
        reading Ollama's own manifest/blob store — the same layout `ollama
        pull` writes — without calling Ollama at all. None if the manifest or
        its model-weight layer (as opposed to the Modelfile/params/license
        layers a manifest also lists) isn't found."""
        manifest_path = cls._manifest_path(tag)
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

        for layer in manifest.get("layers", []):
            if layer.get("mediaType") == "application/vnd.ollama.image.model":
                digest = layer.get("digest", "")
                if digest.startswith("sha256:"):
                    blob = cls._ollama_models_dir() / "blobs" / f"sha256-{digest[7:]}"
                    if blob.exists():
                        return blob
        return None

    # ── server/process lifecycle ──

    def available(self) -> bool:  # pragma: no cover — real HTTP call
        try:
            r = requests.get(f"{config.LLAMACPP_URL}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def ensure_running(self) -> bool:
        """Unlike Ollama, llama-server has no standalone "up but no model
        loaded" state to start — a process needs a model to launch with. This
        is the preflight instead: confirm the binary and Ollama's model store
        both exist, so a caller (e.g. --list-models, or the top of a run) gets
        a clear error before wasting time on the first per-model load. The
        actual subprocess spawns lazily per tag in _ensure_model."""
        if self._binary_path() is None:
            Shared.err(f"'{self.BINARY}' not found — run setup_check.py --engine llamacpp "
                       "to install it, or build/install llama.cpp yourself: "
                       "https://github.com/ggml-org/llama.cpp")
            return False
        if not self._ollama_models_dir().exists():
            Shared.err(f"Ollama's model store not found at {self._ollama_models_dir()} — "
                       "the llamacpp engine reuses models pulled via 'ollama pull', "
                       "pull at least one model first")
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
        llama-server left behind by a previous crashed run (the same
        belt-and-suspenders OllamaEngine.stop does for 'ollama serve'), so a
        fresh instance can bind the port again."""
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

    def is_connection_crash(self, e: Exception) -> bool:
        """Same exception shapes as OllamaEngine.is_connection_crash — both
        engines are hit over HTTP with the same requests/urllib clients, so a
        dead server surfaces identically regardless of which engine it is."""
        if isinstance(e, (requests.exceptions.ConnectionError, urllib.error.URLError,
                          http.client.IncompleteRead, ConnectionError)):
            return True
        return "actively refused" in str(e).lower()

    def wait_for_recovery(self, timeout: int = 30) -> bool:
        """Always True: unlike Ollama's main server (which survives a
        model-runner crash and just respawns the runner, so a caller polls
        available() waiting for that self-heal), llama-server's whole process
        is the model runner — a crash takes the server down with it, and
        nothing brings it back up on its own to poll for. Recovery instead
        happens synchronously on the caller's next generate/chat/embed call,
        via _ensure_model respawning the process for that tag — if that
        respawn itself fails, it raises on that call and this same
        crash-handling loop (Shared.run_measured_calls) runs again, so a
        model that's actually unrecoverable still gets caught, just by the
        next real attempt instead of a passive wait beforehand."""
        return True

    def reachable_or_abort(self) -> bool:
        """Always True: unlike Ollama's always-on daemon, there's no shared
        server that stays up between models to check here — llama-server is
        spawned fresh per model in _ensure_model, which is its own health
        check for that specific model. model_pulled() also isn't at risk of
        the failure mode this guards against on OllamaEngine (a down server
        making 'reachable' and 'not pulled' indistinguishable over HTTP): it
        reads Ollama's manifest/blob files straight off disk, no server
        involved."""
        return True

    def tail_log(self, n_lines: int = 40) -> str:
        return Shared._tail_log(self._log_path, "llama.cpp", n_lines)

    # ── model lifecycle ──

    def model_pulled(self, tag: str) -> bool:
        return self._resolve_blob_path(tag) is not None

    def list_installed_models(self) -> list[dict]:
        """Every tag with a resolvable model-weight blob under Ollama's
        standard library namespace — walks the manifest tree directly rather
        than calling Ollama, so this works even if Ollama itself isn't
        running. Returns [] if the store doesn't exist."""
        library = self._ollama_models_dir() / "manifests" / "registry.ollama.ai" / "library"
        if not library.exists():
            return []
        installed = []
        for model_dir in sorted(p for p in library.iterdir() if p.is_dir()):
            for version_file in sorted(p for p in model_dir.iterdir() if p.is_file()):
                tag = f"{model_dir.name}:{version_file.name}"
                blob = self._resolve_blob_path(tag)
                if blob is not None:
                    installed.append({"tag": tag, "size": blob.stat().st_size})
        return installed

    def max_context_length(self, tag: str, default: int = 131072) -> int:
        """Read a pulled model's real max context length straight from its
        GGUF metadata. GGUFReader memory-maps the file and only walks its
        key/value header section, so — like OllamaEngine's /api/show version
        — this never loads the model's weights. Same architecture-prefixed-key
        scan as OllamaEngine.max_context_length (llama.context_length,
        qwen35.context_length, gptoss.context_length, ...), since both read
        the same underlying GGUF metadata convention.
        """
        blob = self._resolve_blob_path(tag)
        if blob is None:
            return default
        try:
            reader = gguf.GGUFReader(str(blob))
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
        """Identical shape to OllamaEngine.warmup (same watchdog-thread
        pattern) — the first call here is what actually spawns and loads the
        llama-server subprocess, via generate() -> _ensure_model(), so a
        model too large for available memory times out here exactly the same
        way an oversized Ollama pull would."""
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
                # Unlike Ollama (where a non-crash warmup exception can be an
                # ordinary transient request failure against an always-on
                # daemon), llama-server is spawned fresh per model here, so
                # *any* warmup exception — not just a connection-crash shape —
                # means this tag failed to load under llamacpp (bad GGUF,
                # unsupported architecture, exited on startup, etc.) and is
                # just as deterministic as a hang. Recording it regardless of
                # is_connection_crash means a later --engine both pass over
                # ollama (which reads this same cache) skips it too instead of
                # re-discovering the same failure.
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
        if self._loaded_tag is not None and self._split_tag(tag) == self._split_tag(self._loaded_tag):
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
        return self._loaded_tag is None or self._split_tag(tag) != self._split_tag(self._loaded_tag)

    # ── HTTP streaming helpers (llama-server's SSE protocol) ──

    @staticmethod
    def _urlopen(req, timeout):
        """urlopen wrapper that surfaces the response body on HTTP error
        status, same reasoning as OllamaEngine._ollama_urlopen: the bare
        HTTPError only says "HTTP Error 500: Internal Server Error" and hides
        llama-server's actual JSON error detail."""
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

    # ── model process spawn ──

    def _ensure_model(self, tag: str, num_ctx: int | None, *, embedding: bool = False) -> None:
        """Make sure llama-server is up and serving `tag` at `num_ctx`,
        (re)spawning the subprocess if a different model, a different context
        size, or an embedding-vs-chat mode switch is requested. See the
        module docstring for why every mismatch here means a full restart
        (llama-server is single-model-per-process, unlike Ollama)."""
        want = (tag, num_ctx, embedding)
        have = (self._loaded_tag, self._loaded_num_ctx, self._loaded_embedding)
        if want == have and self._proc is not None and self._proc.poll() is None and self.available():
            return

        blob = self._resolve_blob_path(tag)
        if blob is None:
            raise RuntimeError(
                f"{tag} not found in Ollama's model store ({self._ollama_models_dir()}) — "
                f"pull it first with: ollama pull {tag}"
            )

        self._stop_process()

        binary = self._binary_path()
        if binary is None:
            raise RuntimeError(f"'{self.BINARY}' not found — run setup_check.py --engine llamacpp to install it")

        args = [
            binary,
            "-m", str(blob),
            "--host", "127.0.0.1",
            "--port", str(config.LLAMACPP_PORT),
            "-ngl", "0" if not self._gpu_visible else "999",
            # Render the model's own embedded Jinja chat template (from its
            # GGUF tokenizer.chat_template metadata) instead of llama.cpp's
            # built-in template-guessing — closer parity with what Ollama
            # renders from the same underlying metadata.
            "--jinja",
            # Same prompt-processing batch size pinned for the Ollama engine,
            # so timing numbers are comparable across engines.
            "-b", str(config.OLLAMA_NUM_BATCH),
        ]
        if num_ctx is not None:
            args += ["-c", str(num_ctx)]
        if embedding:
            args += ["--embeddings", "--pooling", "mean"]

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
                return
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited unexpectedly (code {proc.returncode}) "
                                   f"loading {tag} — last output:\n{self.tail_log()}")
            time.sleep(1)

        self._stop_process()
        raise RuntimeError(f"llama-server did not become healthy within {self.LOAD_TIMEOUT}s loading {tag}")

    # ── inference ──

    def generate(self, tag: str, prompt: str, timeout: int = 600,
                 num_ctx: int | None = None) -> tuple[float, int, float]:
        """Generate via llama-server's native /completion endpoint and return
        timing metrics. Returns: (ttft_sec, tokens_generated, tokens_per_sec)

        Prefers llama-server's own reported timings (predicted_n,
        predicted_ms, prompt_ms in its final streamed chunk) over wall clock,
        same reasoning as OllamaEngine.generate.
        """
        self._ensure_model(tag, num_ctx)

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
        eval_count = 0

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
        return ttft, eval_count, tps

    def chat(self, tag: str, messages: list, timeout: int = 600,
             num_ctx: int | None = None, num_predict: int = 1024,
             check_loop: bool = False):
        """Generate via llama-server's OpenAI-compatible /v1/chat/completions
        (so the model's chat template gets applied for us) and return timing
        metrics plus the reply text.
        Returns: (ttft_sec, tokens_generated, tokens_per_sec, prompt_eval_count, response_text)

        n_predict is llama.cpp's own sampling-length parameter (what Ollama's
        num_predict is itself forwarded to under the hood) — passed straight
        through as an extension field alongside the OpenAI-shaped body;
        -1 means unbounded, same convention the accuracy tests already rely on
        for Ollama. See OllamaEngine.chat's docstring for check_loop.
        """
        self._ensure_model(tag, num_ctx)

        payload = json.dumps({
            "messages":    messages,
            "n_predict":   num_predict,
            "temperature": 0.0,
            "stream":      True,
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
                    raise OllamaTimeout(f"llamacpp_chat exceeded {timeout}s wall-clock timeout",
                                        partial_text=partial_text)

                if check_loop and now - last_loop_check >= config.LOOP_CHECK_INTERVAL:
                    last_loop_check = now
                    partial_text = "".join(response_parts) or "".join(reasoning_parts)
                    if partial_text and Shared.looks_like_loop(partial_text):
                        raise OllamaLoopDetected(
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

        total = time.perf_counter() - t_start
        if ttft is None:
            ttft = total
        # Reasoning models can stream their whole turn through
        # delta.reasoning_content with delta.content empty — same fallback
        # OllamaEngine.chat applies to message.thinking, for the same reason
        # (an empty assistant turn would corrupt the next turn's history).
        response_text = "".join(response_parts) or "".join(reasoning_parts)
        return ttft, eval_count, tps, prompt_eval_count, response_text

    def embed(self, tag: str, inputs: list[str], timeout: int = 120) -> tuple[list, float]:
        """Embed every string in `inputs` in a single /v1/embeddings call.
        Returns (embeddings_list, elapsed_seconds).

        Loads the model in embedding mode (--embeddings --pooling mean) —
        embedding models need this for the endpoint to be enabled at all, and
        the mean-pooling default matches what Ollama uses for the same
        nomic-embed-text / mxbai-embed-large models.
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
