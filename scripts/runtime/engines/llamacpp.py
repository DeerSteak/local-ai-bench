"""LlamaCppEngine — a llama.cpp (llama-server) InferenceEngine.
See docs/engines.md#llamacppengine for the full rationale."""

import http.client
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import cast

import gguf
import requests

from scripts.runtime import config
from scripts.runtime.hardware import gpu_device_selection, gpu_tensor_split
from scripts.runtime.llamacpp_tools import find_llamacpp_tool, probe_llamacpp_backend
from scripts.runtime.engines.base import ChatMeasurement, EmbeddingMeasurement, GenerationMeasurement, InferenceEngine
from scripts.runtime.engines import openai_api
from scripts.runtime.engines.chat_flow import chat_measurement, run_bounded_chat, validate_chat_budget
from scripts.workloads.models import EMBED_MODELS, LLM_MODELS
from scripts.workloads.model_variants import expanded_variant_catalog
from scripts.setup.custom_models import custom_model
from scripts.setup.intel_xpu_install import oneapi_environment
from scripts.setup.setup_config import configured_gpu_devices, load_setup_config
from scripts.runtime.model_identity import model_tag_slug
from scripts.runtime.mtp import native_mtp_config
from scripts.runtime.generation_guard import looks_like_loop
from scripts.runtime.crash_cache import record_crash
from scripts.runtime.shared import (
    EngineLoopDetected,
    EngineTimeout,
    Shared,
)


ACCELERATOR_BACKENDS = {"cuda", "metal", "rocm", "vulkan", "xpu"}


def model_placement_error(backend: str | None, placement: dict) -> str | None:
    if backend not in ACCELERATOR_BACKENDS:
        return None
    gpu_layers = placement.get("gpu_layers")
    if not isinstance(gpu_layers, int) or isinstance(gpu_layers, bool):
        return f"{backend} model load did not report GPU layer placement"
    if gpu_layers <= 0:
        return f"{backend} model load offloaded zero layers to the GPU"
    return None


class LlamaCppEngine(InferenceEngine):
    name = "llamacpp"

    BINARY = "llama-server"
    REQUIRED_BACKEND: str | None = None

    # Model *load* time (disk read + VRAM placement), not inference time. Matches VllmEngine's
    # LOAD_TIMEOUT — large catalog entries (e.g. 120B split GGUFs) can still be loading at 300s.
    LOAD_TIMEOUT = 900
    SPAWN_LOG_LINES = 200
    _GPU_LAYERS_RE = re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", re.I)
    _MODEL_BUFFER_RE = re.compile(
        r"\bload_tensors:\s+(.+?)\s+model buffer size\s*=\s*([\d.]+)\s*MiB",
        re.I,
    )

    @staticmethod
    def gpu_split_args(*, include_cache: bool = False, cpu_only: bool = False) -> list[str]:
        configured_mode = config.LLAMACPP_GPU_SPLIT_MODE
        mode = "none" if cpu_only or configured_mode == "single" else configured_mode
        args = ["--split-mode", mode]
        setup = load_setup_config(config.SETUP_CONFIG_PATH)
        devices = configured_gpu_devices(setup)
        if configured_mode == "single" and not cpu_only:
            if selection := gpu_device_selection(devices):
                args += ["--device", selection.split(",", 1)[0]]
        elif mode != "none":
            if selection := gpu_device_selection(devices):
                args += ["--device", selection]
            if tensor_split := gpu_tensor_split(devices):
                args += ["--tensor-split", tensor_split]
        if include_cache:
            cache_type = "f16" if mode == "tensor" else config.LLAMACPP_KV_CACHE_TYPE
            args += ["--cache-type-k", cache_type, "--cache-type-v", cache_type]
        return args

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._log_path: Path | None = None
        # What llama-server is serving, so _ensure_model knows whether a restart is needed.
        self._loaded_tag: str | None = None
        self._loaded_num_ctx: int | None = None
        self._loaded_embedding: bool | None = None
        self._loaded_n_parallel: int = 1
        self._loaded_mtp_config: dict | None = None
        self._loaded_model_placement: dict = {}
        # Remembered for the lazy spawn in _ensure_model — no tag yet at start()/ensure_running() time.
        self._gpu_visible = True
        self._cpu_only_active = False
        self._mtp_enabled = False
        self._process_env: dict[str, str] | None = None
        self._expected_backend: str | None = None
        self._model_lock = threading.RLock()

    @classmethod
    def parse_model_placement(cls, log_text: str) -> dict:
        """Parse llama.cpp's actual layer and model-buffer placement."""
        layer_matches = cls._GPU_LAYERS_RE.findall(log_text or "")
        buffers = cls._MODEL_BUFFER_RE.findall(log_text or "")
        placement = {}
        if layer_matches:
            gpu_layers, total_layers = layer_matches[-1]
            placement.update({"gpu_layers": int(gpu_layers), "total_layers": int(total_layers)})
        cpu_mib = sum(float(size) for backend, size in buffers
                      if "cpu" in backend.lower() or "host" in backend.lower())
        if cpu_mib:
            placement["cpu_model_buffer_gb"] = round(cpu_mib / 1024, 3)
        return placement

    # ── binary resolution ──

    @classmethod
    def _runtime_dir(cls) -> Path:
        return config.LLAMACPP_DIR

    @classmethod
    def tool_path(cls, name: str) -> str | None:
        return find_llamacpp_tool(
            name, vendored_dir=cls._runtime_dir(),
            platform_name=platform.system(), which_fn=shutil.which,
            engine_name=cls.name,
        )

    @classmethod
    def _binary_path(cls) -> str | None:
        """Locate llama-server — see docs/engines.md's "Binary resolution"."""
        return cls.tool_path("llama-server")

    def runtime_location(self) -> str | None:
        return self._binary_path()

    def model_paths(self, tag: str) -> tuple[Path, ...]:
        return tuple(self._resolve_model_files(tag) or ())

    def model_artifacts_are_local(self) -> bool:
        return True

    def set_mtp_enabled(self, enabled: bool) -> None:
        self._mtp_enabled = bool(enabled)

    def _native_mtp_config(self, tag: str, *, embedding: bool = False) -> dict | None:
        if not self._mtp_enabled or embedding:
            return None
        model = next((
            model for model in expanded_variant_catalog(LLM_MODELS) if model["tag"] == tag
        ), None)
        if model is None:
            raise RuntimeError(f"{tag} has no cataloged native MTP configuration for llama.cpp")
        config = native_mtp_config(model, self.family)
        if config is None:
            raise RuntimeError(f"{tag} does not support native MTP with llama.cpp")
        return config

    def _mtp_draft_path(self, tag: str, mtp_config: dict | None) -> Path | None:
        if mtp_config is None or "draft_file" not in mtp_config:
            return None
        path = self._models_dir() / self._slug(tag) / Path(mtp_config["draft_file"]).name
        if not path.is_file():
            raise RuntimeError(
                f"{tag} native MTP predictor is missing at {path} — rerun setup to download it"
            )
        return path

    def compatibility_metadata(self, tag: str) -> tuple[dict, str | None]:
        from scripts.setup.model_compatibility import gguf_metadata
        paths = self.model_paths(tag)
        return gguf_metadata(paths[0]) if paths else ({}, "Model weight files are incomplete.")

    @staticmethod
    def repack_args() -> list[str]:
        return ["--no-repack"] if config.LLAMACPP_NO_REPACK else []

    @staticmethod
    def no_host_args(*, cpu_only: bool = False, value_required: bool = False) -> list[str]:
        if cpu_only or not config.LLAMACPP_NO_HOST:
            return []
        return ["--no-host", "1"] if value_required else ["--no-host"]

    # ── local model-file resolution ──

    @classmethod
    def _models_dir(cls) -> Path:
        """This engine's namespaced model directory — see docs/engines.md."""
        from scripts.runtime.engine_identity import engine_family
        return config.MODELS_DIR / engine_family(cls.name)

    @staticmethod
    def _slug(tag: str) -> str:
        """Filesystem-safe per-tag directory name, e.g. "x:3b" -> "x_3b"."""
        return model_tag_slug(tag)

    @staticmethod
    def _catalog_entry(tag: str) -> dict | None:
        """Look up `tag`'s hf_repo/hf_file in models.py's catalog."""
        for model in expanded_variant_catalog(LLM_MODELS) + EMBED_MODELS:
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
            confirmed = cast(list[re.Match], matches)
            prefixes = {match.group(1) for match in confirmed}
            totals = {int(match.group(3)) for match in confirmed}
            if len(prefixes) != 1 or len(totals) != 1:
                return None
            total = totals.pop()
            by_part = {int(match.group(2)): path for match, path in zip(confirmed, paths)}
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

    @staticmethod
    def serving_model_file(props: dict | None) -> str | None:
        """Model filename llama-server reports on /props, across the keys it has used.
        None when the running server cannot be identified — see docs/engines.md."""
        if not isinstance(props, dict):
            return None
        candidates = (
            props.get("model_path"),
            (props.get("default_generation_settings") or {}).get("model"),
            props.get("model"),
        )
        for value in candidates:
            if isinstance(value, str) and value.strip():
                return PurePosixPath(value.replace("\\", "/")).name
        return None

    def _fetch_props(self) -> dict | None:  # pragma: no cover — real HTTP call
        try:
            response = requests.get(f"{config.LLAMACPP_URL}/props", timeout=5)
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None

    def is_installed(self) -> bool:
        binary = self._binary_path()
        if binary is None:
            return False
        return self.REQUIRED_BACKEND is None or probe_llamacpp_backend(binary) == self.REQUIRED_BACKEND

    def ensure_running(self) -> bool:
        """Preflight only (binary + model dir exist) — see docs/engines.md's
        "No standalone up-but-idle state". The real spawn is lazy, per tag, in _ensure_model."""
        binary = self._binary_path()
        if binary is None:
            Shared.err(f"'{self.BINARY}' not found — run setup_check.py "
                       "to install it, or build/install llama.cpp yourself: "
                       "https://github.com/ggml-org/llama.cpp")
            return False
        required_backend = self.REQUIRED_BACKEND or self._expected_backend
        if required_backend in ACCELERATOR_BACKENDS:
            backend = probe_llamacpp_backend(binary, env=self._process_env)
            if backend != required_backend:
                Shared.err(f"{self.name} requires {required_backend}, but its runtime "
                           f"exposes {backend or 'no backend'}")
                return False
        if not self._models_dir().exists():
            Shared.err(f"Model directory not found at {self._models_dir()} — "
                       "run setup_check.py to download at least one model first")
            return False
        Shared.ok(f"{self.BINARY} found at {binary} — models load on demand per test")
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
        self._loaded_mtp_config = None
        self._loaded_model_placement = {}

    def is_connection_crash(self, exc: Exception) -> bool:
        """True for the exception shapes a dead HTTP server surfaces as."""
        if isinstance(exc, (requests.exceptions.ConnectionError, urllib.error.URLError,
                            http.client.IncompleteRead, ConnectionError)):
            return True
        return "actively refused" in str(exc).lower()

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

    def _full_log(self) -> str:
        try:
            return self._log_path.read_text(encoding="utf-8", errors="replace") \
                if self._log_path else ""
        except OSError:
            return ""

    # ── model lifecycle ──

    def model_pulled(self, tag: str) -> bool:
        return self._resolve_model_files(tag) is not None

    def resume_artifact_paths(self, tag: str) -> tuple[Path, ...]:
        paths = self._resolve_model_files(tag)
        if paths is None:
            raise ValueError(f"cannot identify local model artifact for resume: {tag}")
        mtp_config = self._native_mtp_config(tag)
        draft_path = self._mtp_draft_path(tag, mtp_config)
        if draft_path is not None:
            paths.append(draft_path)
        return tuple(path.resolve() for path in paths)

    def resume_runtime_paths(self) -> dict[str, Path]:
        binary = self._binary_path()
        if binary is None:
            raise ValueError("cannot identify llama-server runtime for resume")
        return {"llama-server": Path(binary).resolve()}

    def list_installed_models(self) -> list[dict]:
        """Every fully-present catalog tag, plus any non-catalog directory —
        see docs/engines.md's custom-tag resolution."""
        installed = []
        catalog = expanded_variant_catalog(LLM_MODELS) + EMBED_MODELS
        for model in catalog:
            paths = self._resolve_model_files(model["tag"])
            if paths is not None:
                installed.append({"tag": model["tag"], "size": sum(p.stat().st_size for p in paths)})

        models_dir = self._models_dir()
        if models_dir.exists():
            catalog_slugs = {self._slug(model["tag"]) for model in catalog}
            for entry in sorted(p for p in models_dir.iterdir() if p.is_dir()):
                if entry.name in catalog_slugs:
                    continue
                ggufs = self._resolve_model_files(entry.name)
                if ggufs is not None:
                    imported = custom_model(self.family, entry.name)
                    item = {"tag": entry.name, "size": sum(p.stat().st_size for p in ggufs)}
                    if imported and imported.get("label"):
                        item["label"] = imported["label"]
                    installed.append(item)
        return installed

    def runtime_backend(self, hardware_backend: str, *, cpu_only: bool = False) -> str:
        self._process_env = (
            oneapi_environment()
            if hardware_backend == "xpu" and self.REQUIRED_BACKEND is None else None
        )
        if cpu_only:
            self._expected_backend = "cpu"
            return self._expected_backend
        binary = self._binary_path()
        if binary is None:
            detected_backend = hardware_backend
        else:
            detected_backend = (
                probe_llamacpp_backend(binary, env=self._process_env) or hardware_backend
            )
        self._expected_backend = self.REQUIRED_BACKEND or (
            detected_backend if detected_backend in ACCELERATOR_BACKENDS else
            hardware_backend if hardware_backend in ACCELERATOR_BACKENDS else detected_backend
        )
        return detected_backend

    def process_environment(self) -> dict[str, str] | None:
        return self._process_env

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
                    record_crash(tag, crash_cache, cache_path,
                                 f"warming up at num_ctx={num_ctx}",
                                 extra=crash_extra, engine_name=self.name)
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
        return openai_api.urlopen_with_detail(req, timeout, "llama-server")

    @staticmethod
    def _iter_sse(resp):
        return openai_api.iter_sse(resp)

    @staticmethod
    def _sanitize_tps(tps: float, tokens: int, ttft: float, total: float) -> float:
        """See docs/engines.md's "_sanitize_tps"."""
        return openai_api.sanitize_tps(tps, tokens, ttft, total)

    @staticmethod
    def _warn_tps_sanitized(tag: str, raw_tps: float, sanitized_tps: float,
                             tokens: int, server_predicted_n: int, predicted_ms: float) -> None:
        """Logs the raw server values behind a _sanitize_tps substitution —
        see docs/engines.md's "_warn_tps_sanitized"."""
        Shared.warn(f"{tag}: implausible tps from server (server predicted_n={server_predicted_n}, "
                    f"response tokens={tokens}, predicted_ms={predicted_ms!r}, raw tps={raw_tps:.1f}) — "
                    f"wall-clock diagnostic is {sanitized_tps:.1f} tok/s; marking measurement invalid")

    # ── model process spawn ──

    def _ensure_model(self, tag: str, num_ctx: int | None, *, embedding: bool = False,
                       n_parallel: int = 1, deadline: float | None = None) -> None:
        """Ensure llama-server is serving `tag` at `num_ctx`, respawning on any
        mismatch (model/context/mode/parallel-slots) — llama-server is single-model-per-process."""
        mtp_config = self._native_mtp_config(tag, embedding=embedding)
        draft_path = self._mtp_draft_path(tag, mtp_config)
        want = (tag, num_ctx, embedding, n_parallel, mtp_config)

        def ready():
            have = (self._loaded_tag, self._loaded_num_ctx,
                    self._loaded_embedding, self._loaded_n_parallel,
                    self._loaded_mtp_config)
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
                # Current llama.cpp hides placement summaries at its default verbosity.
                "-lv", "4",
                # "auto" lets llama-server's own --fit logic offload as many layers as fit in
                # free VRAM and run the rest on CPU, instead of forcing all layers and OOM-ing.
                "-ngl", "0" if not self._gpu_visible else (
                    "all" if config.LLAMACPP_GPU_SPLIT_MODE == "tensor" else "auto"
                ),
                "--jinja",   # renders the model's own chat template, not llama.cpp's guessing heuristic — see docs/engines.md
                "-b", str(config.LLAMACPP_NUM_BATCH),
                # Quantized KV cache needs flash attention explicitly on — see config.LLAMACPP_KV_CACHE_TYPE.
                "--flash-attn", "on",
                *self.repack_args(),
                *self.no_host_args(cpu_only=not self._gpu_visible),
                *self.gpu_split_args(include_cache=True, cpu_only=not self._gpu_visible),
            ]
            if num_ctx is not None:
                # -c is a total KV-cache budget split across --parallel slots — see docs/engines.md.
                args += ["-c", str(num_ctx * n_parallel)]
            if embedding:
                args += ["--embeddings", "--pooling", "mean"]
            if mtp_config is not None:
                args += [
                    "--spec-type", "draft-mtp",
                    "--spec-draft-n-max", str(mtp_config["num_speculative_tokens"]),
                ]
                if draft_path is not None:
                    args += ["--spec-draft-model", str(draft_path)]
            # Always pinned, even at 1 — see docs/engines.md's "--parallel is always pinned".
            args += ["--parallel", str(n_parallel)]

            log_fh = tempfile.NamedTemporaryFile(mode="w", suffix="-llamacpp-server.log", delete=False)
            self._log_path = Path(log_fh.name)
            try:
                proc = subprocess.Popen(
                    args, stdout=log_fh, stderr=subprocess.STDOUT, env=self._process_env,
                )
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
                # Before health: a bind failure exits fast, and /health cannot tell our
                # server from one another process already has on the port.
                if proc.poll() is not None:
                    raise RuntimeError(f"llama-server exited unexpectedly (code {proc.returncode}) "
                                       f"loading {tag} — last output:\n{self.tail_log()}")
                if self.available():
                    serving = self.serving_model_file(self._fetch_props())
                    if serving is not None and serving != paths[0].name:
                        self._stop_process()
                        raise RuntimeError(
                            f"port {config.LLAMACPP_PORT} is serving {serving}, not {paths[0].name} "
                            f"— another llama-server owns it; stop it before loading {tag}"
                        )
                    placement = self.parse_model_placement(self._full_log())
                    if placement_error := model_placement_error(
                        self._expected_backend, placement,
                    ):
                        log_tail = self.tail_log()
                        self._stop_process()
                        raise RuntimeError(
                            f"{placement_error}; refusing silent CPU fallback loading {tag}"
                            f" — last output:\n{log_tail}"
                        )
                    self._loaded_tag = tag
                    self._loaded_num_ctx = num_ctx
                    self._loaded_embedding = embedding
                    self._loaded_n_parallel = n_parallel
                    self._loaded_mtp_config = mtp_config
                    self._loaded_model_placement = placement
                    return
                time.sleep(1)

            self._stop_process()
            raise RuntimeError(f"llama-server did not become healthy within {self.LOAD_TIMEOUT}s loading {tag}")

    # ── inference ──

    def generate(self, tag: str, prompt: str, timeout: int = 600,
                 num_ctx: int | None = None, n_parallel: int = 1,
                 cache_prompt: bool = False) -> GenerationMeasurement:
        """Generate via /completion; n_parallel must match prepare_concurrency."""
        operation_start = time.perf_counter()
        deadline = operation_start + timeout
        self._ensure_model(tag, num_ctx, n_parallel=n_parallel, deadline=deadline)
        model_load_sec = time.perf_counter() - operation_start

        payload = json.dumps({
            **self.sampling_payload(),
            "prompt": prompt,
            "n_predict": config.GENERATE_MAX_TOKENS,
            "stream": True,
            "return_tokens": True,
            "cache_prompt": cache_prompt,
        }).encode()
        req = urllib.request.Request(
            f"{config.LLAMACPP_URL}/completion",
            data=payload, headers={"Content-Type": "application/json"}, method="POST",
        )
        request_start = time.perf_counter()

        ttft   = None
        tokens = 0
        tps    = 0
        server_predicted_n = 0
        predicted_ms       = 0
        prompt_ms = None
        prompt_tokens = None
        response_parts = []
        finish_reason = None

        remaining = max(deadline - time.perf_counter(), 0.001)
        with self._urlopen(req, remaining) as resp:
            for chunk in self._iter_sse(resp):
                content = chunk.get("content")
                if ttft is None and content:
                    ttft = time.perf_counter() - request_start
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
                    prompt_ms = timings.get("prompt_ms")
                    prompt_tokens = timings.get("prompt_n")
                if chunk.get("stop"):
                    finish_reason = chunk.get("stop_type") or "stop"

        total = time.perf_counter() - request_start
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
        decode_seconds = (total - ttft) if sanitized != tps else (
            predicted_ms / 1000 if predicted_ms else max(total - ttft, 0)
        )
        return GenerationMeasurement(
            client_ttft_sec=ttft,
            generated_tokens=tokens,
            tokens_per_sec=sanitized,
            client_wall_sec=total,
            decode_sec=decode_seconds,
            server_prompt_sec=prompt_ms / 1000 if prompt_ms is not None else None,
            prompt_tokens=prompt_tokens,
            response_text="".join(response_parts),
            finish_reason=finish_reason,
            model_load_sec=model_load_sec,
            server_tps_implausible=sanitized != tps,
            gpu_layers=self._loaded_model_placement.get("gpu_layers"),
            total_layers=self._loaded_model_placement.get("total_layers"),
            cpu_model_buffer_gb=self._loaded_model_placement.get("cpu_model_buffer_gb"),
        )

    def _chat_request(self, tag: str, messages: list, tools: list | None,
                      deadline: float, num_predict: int,
                      check_loop: bool, budget_nudged: bool) -> dict:
        payload = {
            **self.sampling_payload(),
            "messages": messages,
            "n_predict": num_predict,
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
        server_prompt_sec = None
        response_parts = []
        reasoning_parts = []
        tool_fragments: dict[int, dict] = {}
        finish_reason = None
        request_start = time.perf_counter()
        last_loop_check = request_start
        remaining = deadline - request_start
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
                    if response_text and looks_like_loop(response_text):
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
                    if prompt_ms is not None and prompt_ms >= 0:
                        server_prompt_sec = prompt_ms / 1000
                    if prompt_n is not None:
                        prompt_eval_count = prompt_n
                tokens, prompt_eval_count = openai_api.streamed_usage(
                    chunk, tokens, prompt_eval_count,
                )

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
            "server_prompt_sec": server_prompt_sec,
            "wall_seconds": total,
            "tokens": tokens,
            "tps": tps,
            "decode_seconds": decode_seconds,
            "prompt_eval_count": prompt_eval_count,
            "response_text": "".join(response_parts) or "".join(reasoning_parts),
            "tool_calls": self._tool_calls_from_fragments(tool_fragments),
            "finish_reason": finish_reason,
            "server_tps_implausible": tps != raw_tps,
        }

    def _chat_with_optional_finalize(
            self, tag: str, messages: list, tools: list | None, timeout: int,
            num_ctx: int | None, num_predict: int, check_loop: bool,
            token_budget: int | None):
        validate_chat_budget(num_predict, token_budget)
        operation_start = time.perf_counter()
        deadline = operation_start + timeout
        self._ensure_model(tag, num_ctx, deadline=deadline)
        model_load_sec = time.perf_counter() - operation_start
        first, second, budget_nudged = run_bounded_chat(
            lambda req_messages, req_tools, req_deadline, req_predict, req_check, nudged:
                self._chat_request(
                    tag, req_messages, req_tools, req_deadline, req_predict, req_check, nudged,
                ),
            messages, tools, deadline, num_predict, check_loop, token_budget,
            config.ACC_FINALIZE_FRACTION, config.ACC_FINALIZE_MESSAGE, "llamacpp_chat",
        )
        return first, second, budget_nudged, model_load_sec

    def chat(self, tag: str, messages: list, timeout: int = 600,
             num_ctx: int | None = None, num_predict: int = 1024,
             check_loop: bool = False, token_budget: int | None = None) -> ChatMeasurement:
        """Chat once, or use a bounded final-answer pass after a length stop."""
        first, second, budget_nudged, model_load_sec = self._chat_with_optional_finalize(
            tag, messages, None, timeout, num_ctx, num_predict, check_loop, token_budget,
        )
        return chat_measurement(
            first, second, budget_nudged, model_load_sec,
            model_placement=self._loaded_model_placement,
        )

    def chat_tools(self, tag: str, messages: list, tools: list, timeout: int = 600,
                   num_ctx: int | None = None, num_predict: int = 1024,
                   check_loop: bool = False, token_budget: int | None = None) -> ChatMeasurement:
        """Tool chat once, or request one complete replacement after a length stop."""
        first, second, budget_nudged, model_load_sec = self._chat_with_optional_finalize(
            tag, messages, tools, timeout, num_ctx, num_predict, check_loop, token_budget,
        )
        return chat_measurement(
            first, second, budget_nudged, model_load_sec,
            model_placement=self._loaded_model_placement,
        )

    @staticmethod
    def _tool_calls_from_fragments(tool_fragments: dict[int, dict]) -> list[dict]:
        return openai_api.tool_calls_from_fragments(tool_fragments)

    def embed(self, tag: str, inputs: list[str], timeout: int = 120) -> EmbeddingMeasurement:
        """Embed every input in one request, loading in embedding mode."""
        load_start = time.perf_counter()
        self._ensure_model(tag, num_ctx=None, embedding=True)
        model_load_sec = time.perf_counter() - load_start

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
        return EmbeddingMeasurement(
            embeddings=embeddings, client_wall_sec=elapsed, model_load_sec=model_load_sec,
        )
