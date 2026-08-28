"""vLLM engine — see docs/engines.md. Weights resolve by HuggingFace repo id from
vLLM's own cache, never by path, so a containerised vLLM works unchanged."""

import http.client
import json
import math
import os
import platform
import re
import shutil
import secrets
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import requests
import psutil

from scripts.runtime import config
from scripts.runtime.engines import openai_api
from scripts.runtime.engines.chat_flow import chat_measurement, run_bounded_chat, validate_chat_budget
from scripts.runtime.engines.base import (
    ChatMeasurement, EmbeddingMeasurement, GenerationMeasurement, InferenceEngine,
)
from scripts.setup.setup_config import (
    configured_gpu_devices, configured_vllm_launcher_args, configured_vllm_path,
    load_setup_config,
)
from scripts.setup.custom_models import custom_model, load_custom_models
from scripts.setup.vllm_install import (
    find_vllm_binary, find_vllm_launcher, hf_cache_model_complete, hf_cache_model_dir,
    hf_cache_snapshot_dir, vllm_cache_home,
)
from scripts.workloads.models import EMBED_MODELS, LLM_MODELS
from scripts.runtime.generation_guard import looks_like_loop
from scripts.runtime.crash_cache import record_crash
from scripts.runtime.mtp import native_mtp_config
from scripts.runtime.shared import EngineLoopDetected, EngineTimeout, Shared


_KV_MEMORY_RE = re.compile(r"Available KV cache memory:\s*(-?\d+(?:\.\d+)?)\s*GiB", re.I)
_KV_REQUIRED_RE = re.compile(
    r"\(?([\d.]+)\s*GiB KV cache is needed, which is larger than "
    r"the available KV cache memory \(([\d.]+)\s*GiB\)", re.I,
)


def available_kv_cache_gib(log_text: str) -> float | None:
    """Return the last vLLM KV-cache memory profile reading."""
    matches = _KV_MEMORY_RE.findall(log_text or "")
    return float(matches[-1]) if matches else None


def next_cpu_offload_gb(log_text: str, current_gb: int = 0) -> int | None:
    """Choose a 2 GiB-aligned retry from vLLM's measured KV-cache deficit."""
    available = available_kv_cache_gib(log_text)
    required = _KV_REQUIRED_RE.findall(log_text or "")
    if required:
        needed_gib, available_gib = map(float, required[-1])
        shortfall = max(0.0, needed_gib - available_gib)
    elif available is not None and "No available memory for the cache blocks" in log_text:
        shortfall = max(0.0, -available)
    else:
        return None
    needed = shortfall + (config.VLLM_OFFLOAD_RESERVE_GB if current_gb == 0 else 0)
    calculated = math.ceil(needed / config.VLLM_OFFLOAD_STEP_GB) * config.VLLM_OFFLOAD_STEP_GB
    return current_gb + max(config.VLLM_OFFLOAD_STEP_GB, calculated)


def tensor_parallel_size(args: list[str]) -> int:
    """Read vLLM's per-worker CPU-offload multiplier from launcher arguments."""
    for index, value in enumerate(args):
        if value in {"--tensor-parallel-size", "-tp"} and index + 1 < len(args):
            try:
                return max(1, int(args[index + 1]))
            except ValueError:
                return 1
        for prefix in ("--tensor-parallel-size=", "-tp="):
            if value.startswith(prefix):
                try:
                    return max(1, int(value.removeprefix(prefix)))
                except ValueError:
                    return 1
    return 1


def offload_retry_allowed(retry_gb: int | None, host_limit_gb: int, attempts: int) -> bool:
    """Bound adaptive retries by current host capacity and attempt count."""
    return offload_stop_reason(retry_gb, host_limit_gb, attempts) is None


def offload_stop_reason(retry_gb: int | None, host_limit_gb: int, attempts: int) -> str | None:
    """Explain why an adaptive retry is unsafe or inapplicable."""
    if retry_gb is None:
        return "failure was not a recognized KV-cache memory shortage"
    if retry_gb > host_limit_gb:
        return (f"CPU offload retry needs {retry_gb} GiB per worker, "
                f"above the {host_limit_gb} GiB host-RAM limit")
    if attempts >= config.VLLM_OFFLOAD_MAX_ATTEMPTS:
        return f"CPU offload calibration reached its {config.VLLM_OFFLOAD_MAX_ATTEMPTS}-retry limit"
    return None


def offload_timeout_message(tag: str, cpu_offload_gb: int, attempts: int) -> str:
    """Describe which calibration load exhausted its caller budget."""
    total = config.VLLM_OFFLOAD_MAX_ATTEMPTS + 1
    return (f"loading {tag} exceeded the request wall-clock timeout during CPU-offload "
            f"calibration attempt {attempts + 1}/{total} "
            f"(--cpu-offload-gb {cpu_offload_gb})")


def load_attempt_deadline(now: float, caller_deadline: float | None, timeout: int) -> float:
    """Bound one load attempt by both its own window and its caller's budget."""
    attempt_deadline = now + timeout
    return min(attempt_deadline, caller_deadline) if caller_deadline is not None else attempt_deadline


def load_timeout_error(tag: str, cpu_offload_gb: int, attempts: int,
                       timeout: int, caller_expired: bool) -> Exception:
    """Keep caller exhaustion distinct from an unhealthy server."""
    if caller_expired:
        return EngineTimeout(offload_timeout_message(tag, cpu_offload_gb, attempts))
    return RuntimeError(f"vLLM did not become healthy within {timeout}s loading {tag}")


def offload_calibration_timeout(load_timeout: int, requested_timeout: int) -> int:
    """Budget concurrency startup for the initial load and every bounded retry."""
    attempts = config.VLLM_OFFLOAD_MAX_ATTEMPTS + 1
    return max(requested_timeout, load_timeout * attempts)


def vllm_gpu_memory_utilization(machine: str, devices: list[dict]) -> float:
    names = " ".join(str(device.get("name", "")) for device in devices).casefold()
    if machine.casefold() in {"arm64", "aarch64"} and "gb10" in names:
        return 0.70
    return config.VLLM_GPU_MEMORY_UTILIZATION


class VllmEngine(InferenceEngine):
    name = "vllm"

    # vLLM start-up includes weight load, graph capture, and KV allocation.
    LOAD_TIMEOUT = 900
    LAUNCHER_STOP_TIMEOUT = 300

    # vLLM's startup traceback is long and its root cause precedes it, so a short tail hides it.
    SPAWN_LOG_LINES = 200

    def __init__(self):
        setup = load_setup_config(config.SETUP_CONFIG_PATH)
        self._launcher = configured_vllm_path(setup, "launcher") or find_vllm_launcher()
        self._executable = (
            configured_vllm_path(setup, "executable") or find_vllm_binary(
                platform_name=platform.system())
        )
        # Set when setup_check.py found a reachable vLLM with no local binary/launcher —
        # an externally-managed server we talk to but never spawn or stop ourselves.
        configured_server_url = configured_vllm_path(setup, "server_url")
        self._server_url = configured_server_url if self._local_runtime is None else None
        self._launcher_extra_args = configured_vllm_launcher_args(setup)
        gpu_devices = configured_gpu_devices(setup)
        self._gpu_fingerprint = json.dumps(gpu_devices, sort_keys=True)
        self._gpu_memory_utilization = vllm_gpu_memory_utilization(
            platform.machine(), gpu_devices,
        )
        recorded_home = configured_vllm_path(setup, "hf_home")
        self._cache_home = Path(recorded_home) if recorded_home else vllm_cache_home(self._launcher)

        self._proc: subprocess.Popen | None = None
        self._log_path: Path | None = None
        self._loaded_tag: str | None = None
        self._loaded_model_id: str | None = None
        self._loaded_num_ctx: int | None = None
        self._loaded_embedding: bool | None = None
        self._loaded_n_parallel: int = 1
        self._loaded_tool_parser: str | None = None
        self._loaded_mtp_config: dict | None = None
        self._loaded_cpu_offload_gb = 0
        self._gpu_visible = True
        self._kv_cache_dtype = "auto"
        self._cpu_offload_gb: dict[str, int] = self._load_offload_cache()
        self._model_lock = threading.RLock()

    def set_mtp_enabled(self, enabled: bool) -> None:
        self._mtp_enabled = bool(enabled)

    def _native_mtp_config(self, tag: str, *, embedding: bool = False) -> dict | None:
        if not getattr(self, "_mtp_enabled", False) or embedding:
            return None
        model = next((model for model in LLM_MODELS if model["tag"] == tag), None)
        if model is None:
            raise RuntimeError(f"{tag} has no cataloged native MTP configuration for vLLM")
        mtp_config = native_mtp_config(model, self.name)
        if mtp_config is None:
            raise RuntimeError(f"{tag} does not support native MTP with vLLM")
        return {"method": "mtp", **mtp_config}

    @property
    def _local_runtime(self) -> str | None:
        return self._executable or self._launcher

    @property
    def _uses_launcher(self) -> bool:
        return self._executable is None and self._launcher is not None

    @property
    def _offload_cache_path(self) -> Path:
        return self._cache_home / "local-ai-bench-vllm-offload.json"

    def _load_offload_cache(self) -> dict[str, int]:
        try:
            payload = json.loads(self._offload_cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        values = payload.get("offload_gb") if isinstance(payload, dict) else None
        if not isinstance(values, dict):
            return {}
        return {
            key: value for key, value in values.items()
            if isinstance(key, str) and isinstance(value, int) and value > 0
        }

    def _save_offload_cache(self) -> None:
        self._cache_home.mkdir(parents=True, exist_ok=True)
        target = self._offload_cache_path
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps({"offload_gb": self._cpu_offload_gb}, indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(target)

    def _offload_key(self, tag: str, repo: str) -> str:
        runtime = self._local_runtime or "external"
        try:
            runtime_mtime = Path(runtime).stat().st_mtime_ns
        except OSError:
            runtime_mtime = 0
        snapshot = self._snapshot_dir(tag)
        revision = snapshot.name if snapshot else "unknown"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        return "|".join((repo, revision, self._kv_cache_dtype, str(runtime),
                         str(runtime_mtime), self._gpu_fingerprint, visible))

    def _host_offload_limit_gb(self) -> int:
        available = psutil.virtual_memory().available / (1024 ** 3)
        usable = max(0, int(available - config.VLLM_OFFLOAD_HOST_RESERVE_GB))
        launcher_args = self._launcher_extra_args if self._uses_launcher else []
        per_worker = usable // tensor_parallel_size(launcher_args)
        return per_worker // config.VLLM_OFFLOAD_STEP_GB * config.VLLM_OFFLOAD_STEP_GB

    @staticmethod
    def supported_kv_cache_dtype(runtime_backend: str) -> str:
        """Use FP8 only on accelerator backends where vLLM explicitly supports it."""
        return "fp8" if runtime_backend in {"cuda", "rocm"} else "auto"

    def configure_kv_cache(self, runtime_backend: str) -> str:
        """Select one cache policy for every locally managed vLLM workload."""
        self._kv_cache_dtype = (
            "auto" if self._server_url else self.supported_kv_cache_dtype(runtime_backend)
        )
        return self._kv_cache_dtype

    @property
    def kv_cache_dtype(self) -> str:
        return self._kv_cache_dtype

    @property
    def launcher_extra_args(self) -> list[str]:
        return list(self._launcher_extra_args) if self._uses_launcher else []

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

    def supports_tool_calls(self, tag: str) -> bool:
        """vLLM emits no tool_calls without --tool-call-parser, and its parsers are
        per-model, so a tag with none configured cannot be measured."""
        return self._tool_parser(tag) is not None

    @classmethod
    def _repo(cls, tag: str) -> str | None:
        """The HF repo id vLLM serves for `tag`, or None when the catalog has none."""
        entry = cls._catalog_entry(tag)
        imported = custom_model(cls.name, tag)
        return entry.get("vllm_repo") if entry else imported.get("repo") if imported else None

    def _snapshot_dir(self, tag: str) -> Path | None:
        repo = self._repo(tag)
        return hf_cache_snapshot_dir(self._cache_home, repo) if repo else None

    def _standalone_chat_template(self, tag: str) -> tuple[str | None, str | None]:
        snapshot = self._snapshot_dir(tag)
        path = snapshot / "chat_template.jinja" if snapshot else None
        if path is None or not path.is_file():
            return None, None
        try:
            template = path.read_text(encoding="utf-8")
        except OSError as exc:
            return None, str(exc)
        return template, None

    def _chat_template_argument(self, tag: str) -> str | None:
        snapshot = self._snapshot_dir(tag)
        try:
            tokenizer_data = json.loads(
                (snapshot / "tokenizer_config.json").read_text(encoding="utf-8")
            ) if snapshot else {}
        except (OSError, json.JSONDecodeError):
            return None
        if tokenizer_data.get("chat_template"):
            return None
        template, error = self._standalone_chat_template(tag)
        if error is not None or template is None:
            return None
        path = snapshot / "chat_template.jinja" if snapshot else None
        if self._uses_launcher:
            return template
        return str(path.resolve()) if path else None

    # ── per-request prefill timing ──

    # vLLM exposes no prompt duration on the response itself, but it records one
    # per request into this histogram. See docs/engines.md#prefill-timing.
    PREFILL_METRIC = "vllm:request_prefill_time_seconds"

    @staticmethod
    def parse_prefill_metric(text: str, metric: str = PREFILL_METRIC) -> tuple[float, int] | None:
        """The histogram's running (sum, count) from a Prometheus /metrics body, summed
        across label sets — one model is ever served at a time, so this is one series."""
        totals = {}
        patterns = {
            suffix: re.compile(r"^" + re.escape(metric + suffix) + r"(?:\{[^}]*\})?\s+(\S+)$")
            for suffix in ("_sum", "_count")
        }
        for line in (text or "").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for suffix, pattern in patterns.items():
                match = pattern.match(line)
                if match:
                    try:
                        totals[suffix] = totals.get(suffix, 0.0) + float(match.group(1))
                    except ValueError:
                        return None
                    break
        if "_sum" not in totals or "_count" not in totals:
            return None
        return totals["_sum"], int(totals["_count"])

    @staticmethod
    def prefill_seconds_from_delta(before, after) -> float | None:
        """Prefill seconds for one request, from readings taken around it. Attributable
        only when the histogram advanced by exactly one — see docs/engines.md#prefill-timing."""
        if before is None or after is None:
            return None
        if after[1] - before[1] != 1:
            return None
        seconds = after[0] - before[0]
        return seconds if seconds >= 0 else None

    def _prefill_reading(self):  # pragma: no cover — real HTTP call
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=5)
        except Exception:
            return None
        return self.parse_prefill_metric(response.text) if response.status_code == 200 else None

    # ── server/process lifecycle ──

    @property
    def base_url(self) -> str:
        """`server_url` when talking to an externally-managed vLLM, else our own port."""
        return self._server_url or config.VLLM_URL

    def available(self) -> bool:  # pragma: no cover — real HTTP call
        try:
            return requests.get(f"{self.base_url}/health", timeout=5).status_code == 200
        except Exception:
            return False

    def _served_model_ids(self) -> set[str] | None:
        """Model IDs advertised by an external server, or None when unavailable."""
        if not self._server_url:
            return None
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            payload = response.json() if response.status_code == 200 else None
        except Exception:
            return None
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            return None
        return {
            entry["id"] for entry in payload["data"]
            if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        }

    def _external_model_id(self, tag: str, model_ids: set[str] | None = None) -> str | None:
        repo = self._repo(tag)
        if repo is None:
            return None
        served = self._served_model_ids() if model_ids is None else model_ids
        if served is None:
            return None
        return next((model_id for model_id in (repo, tag) if model_id in served), None)

    def _external_server_has_tag(self, tag: str, model_ids: set[str] | None = None) -> bool:
        return self._external_model_id(tag, model_ids) is not None

    def is_installed(self) -> bool:
        return (self._launcher or self._executable or self._server_url) is not None

    def cache_home(self) -> Path:
        return self._cache_home

    def runtime_location(self) -> str | None:
        return self._executable

    def runtime_launcher(self) -> str | None:
        return self._launcher if self._uses_launcher else None

    def external_server_url(self) -> str | None:
        return self._server_url

    def model_snapshot(self, tag: str) -> Path | None:
        return self._snapshot_dir(tag)

    def supports_model_import(self) -> bool:
        return self._server_url is None

    def bench_executable(self) -> str | None:
        """`vllm bench` needs the real binary — a launcher only wraps `vllm serve`."""
        return self._executable

    def bench_gpu_memory_utilization(self) -> float:
        if self._gpu_memory_utilization == 0.70:
            return config.VLLMBENCH_GB10_GPU_MEMORY_UTILIZATION
        return min(self._gpu_memory_utilization, config.VLLMBENCH_GPU_MEMORY_UTILIZATION)

    def ensure_running(self) -> bool:
        """Preflight only — the real spawn is lazy, per tag, in _ensure_model."""
        if self._server_url:
            if not self.available():
                Shared.err(f"vLLM server configured at {self._server_url} is not reachable")
                return False
            Shared.ok(f"vLLM server found at {self._server_url} — using whatever model it has loaded")
            return True
        if self._launcher is None and self._executable is None:
            Shared.err("No 'vllm' or platform launcher found — run setup_check.py, or "
                       "install vLLM yourself: https://docs.vllm.ai/")
            return False
        if not self._cache_home.exists():
            Shared.err(f"vLLM model cache not found at {self._cache_home} — "
                       "run setup_check.py to download at least one model first")
            return False
        Shared.ok(f"vLLM found at {self._local_runtime} — models load on demand per test")
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
                # EngineCore is a separate process whose command line is not "vllm serve",
                # so an orphan from an earlier crash needs its own pattern.
                for pattern in ("vllm serve", "VLLM::EngineCore", "from_engine_args"):
                    subprocess.run(["pkill", "-f", pattern],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            pass

    def _stop_process(self, timeout: int = 15) -> None:  # pragma: no cover — kills real processes
        """Signal the whole process group, so the EngineCore child dies with the server."""
        proc = self._proc
        launcher_was_running = self._uses_launcher and proc is not None
        if proc is not None and proc.poll() is None:
            # Container launchers handle an interactive interrupt by stopping their container.
            self._signal_group(signal.SIGINT if self._uses_launcher else signal.SIGTERM)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._signal_group(signal.SIGKILL)
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
        if launcher_was_running:
            self._wait_for_launcher_shutdown(self.LAUNCHER_STOP_TIMEOUT)
        self._proc = None
        self._loaded_tag = None
        self._loaded_model_id = None
        self._loaded_num_ctx = None
        self._loaded_embedding = None
        self._loaded_n_parallel = 1
        self._loaded_tool_parser = None
        self._loaded_mtp_config = None
        self._loaded_cpu_offload_gb = 0

    def _wait_for_launcher_shutdown(self, timeout: int) -> None:
        """Wait for a launcher-owned container to stop serving after its wrapper exits."""
        deadline = time.perf_counter() + timeout
        while self.available():
            if time.perf_counter() >= deadline:
                raise RuntimeError(
                    f"vLLM is still reachable after waiting {timeout}s for its container to stop")
            time.sleep(1)

    def _signal_group(self, sig) -> None:  # pragma: no cover — signals real processes
        """Signal the server's whole group, falling back to the process itself."""
        proc = self._proc
        if proc is None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (AttributeError, OSError, ProcessLookupError):
            try:
                proc.send_signal(sig)
            except (OSError, ProcessLookupError):
                pass

    def is_connection_crash(self, exc: Exception) -> bool:
        if isinstance(exc, (requests.exceptions.ConnectionError, urllib.error.URLError,
                            http.client.IncompleteRead, ConnectionError)):
            return True
        return "actively refused" in str(exc).lower()

    def wait_for_recovery(self, timeout: int = 30) -> bool:
        """Always True — recovery happens in _ensure_model on the next call."""
        return True

    def reachable_or_abort(self) -> bool:
        """Always True — _ensure_model is its own per-model health check."""
        return True

    def tail_log(self, n_lines: int = 40) -> str:
        return Shared._tail_log(self._log_path, "vLLM", n_lines)

    # ── model lifecycle ──

    def resume_artifact_paths(self, tag: str) -> tuple[Path, ...]:
        """Weights and config of the cached snapshot, resolved through the HF cache's
        symlinks so identity follows the blob rather than the link."""
        snapshot = self._snapshot_dir(tag)
        if snapshot is None:
            raise ValueError(f"cannot identify local model artifact for resume: {tag}")
        paths = sorted(snapshot.glob("*.safetensors")) + [
            snapshot / "config.json", snapshot / "tokenizer_config.json",
            snapshot / "chat_template.jinja",
        ]
        return tuple(path.resolve() for path in paths if path.exists())

    def resume_runtime_paths(self) -> dict[str, Path]:
        runtime = self._local_runtime
        if runtime is None:
            raise ValueError("cannot identify vLLM runtime for resume")
        return {"vllm": Path(runtime).resolve()}

    def model_pulled(self, tag: str) -> bool:
        if self._server_url:
            return self._external_server_has_tag(tag)
        repo = self._repo(tag)
        return repo is not None and hf_cache_model_complete(self._cache_home, repo)

    def model_paths(self, tag: str) -> tuple[Path, ...]:
        snapshot = self._snapshot_dir(tag)
        if snapshot is None:
            return ()
        patterns = ("*.safetensors", "*.bin", "*.pt")
        return tuple(sorted(path for pattern in patterns for path in snapshot.glob(pattern)))

    def model_artifacts_are_local(self) -> bool:
        return self._server_url is None

    def can_reset_model_state(self) -> bool:
        return self._server_url is None

    def compatibility_metadata(self, tag: str) -> tuple[dict, str | None]:
        snapshot = self._snapshot_dir(tag)
        if snapshot is None:
            return {}, "The local model snapshot was not found."
        try:
            config_data = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
            tokenizer_data = json.loads(
                (snapshot / "tokenizer_config.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            return {}, str(exc)
        chat_template = tokenizer_data.get("chat_template")
        if not chat_template:
            chat_template, template_error = self._standalone_chat_template(tag)
            if template_error is not None:
                return {}, template_error
        architecture = next(iter(config_data.get("architectures") or []), None)
        context = next((
            config_data.get(key) or (config_data.get("text_config") or {}).get(key)
            for key in ("max_position_embeddings", "max_seq_len", "n_positions")
            if config_data.get(key) or (config_data.get("text_config") or {}).get(key)
        ), None)
        return {
            "general.architecture": architecture,
            "tokenizer.chat_template": chat_template,
            "model.context_length": context,
        }, None

    def list_installed_models(self) -> list[dict]:
        """Catalog and registered custom tags available from the server or cache."""
        served = self._served_model_ids() if self._server_url else None
        if self._server_url and served is None:
            return []
        installed = []
        for model in LLM_MODELS + EMBED_MODELS:
            if self._server_url:
                if not self._external_server_has_tag(model["tag"], served):
                    continue
                installed.append({"tag": model["tag"], "size": None})
                continue
            if not self.model_pulled(model["tag"]):
                continue
            repo = model.get("vllm_repo")
            if repo is None:
                continue
            blobs = hf_cache_model_dir(self._cache_home, repo) / "blobs"
            size = sum(path.stat().st_size for path in blobs.glob("*")) if blobs.is_dir() else None
            installed.append({"tag": model["tag"], "size": size})
        if not self._server_url:
            for model in load_custom_models():
                if model.get("engine") != self.name or not self.model_pulled(str(model.get("tag", ""))):
                    continue
                repo = str(model["repo"])
                blobs = hf_cache_model_dir(self._cache_home, repo) / "blobs"
                size = sum(path.stat().st_size for path in blobs.glob("*")) if blobs.is_dir() else None
                installed.append({"tag": model["tag"], "label": model.get("label"), "size": size})
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
                    record_crash(tag, crash_cache, cache_path,
                                 f"warming up at num_ctx={num_ctx}",
                                 extra=crash_extra, engine_name=self.name)
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
            load_timeout = offload_calibration_timeout(self.LOAD_TIMEOUT, timeout)
            self._ensure_model(tag, per_slot_ctx, n_parallel=n_parallel,
                                deadline=time.perf_counter() + load_timeout)
            return True
        except Exception as e:
            Shared.warn(f"Failed to load {tag} for {n_parallel}-way concurrency "
                        f"at {per_slot_ctx} tokens/slot: {e}")
            return False

    # ── model process spawn ──

    def context_limit(self, tag: str, num_ctx: int | None) -> int | None:
        """--max-model-len for a padded-to-num_ctx prompt. Character-based padding can
        overshoot, and vLLM rejects prompt+max_tokens over the limit outright."""
        if num_ctx is None:
            return None
        return min(num_ctx + config.VLLM_CTX_TOLERANCE, self.max_context_length(tag))

    def server_command(self, repo: str, num_ctx: int | None, *, embedding: bool = False,
                       n_parallel: int = 1, tool_parser: str | None = None,
                       cpu_offload_gb: int = 0,
                       chat_template: str | None = None,
                       mtp_config: dict | None = None) -> list[str]:
        """Argv serving `repo` from the managed runtime, with a platform launcher fallback."""
        options = ["--served-model-name", repo,
                    "--generation-config", "vllm",
                    "--max-num-seqs", str(n_parallel),
                    "--gpu-memory-utilization", str(self._gpu_memory_utilization)]
        if not embedding:
            options.append("--enable-prefix-caching")
        if self._kv_cache_dtype != "auto" and not embedding:
            options += ["--kv-cache-dtype", self._kv_cache_dtype]
        if num_ctx is not None:
            options += ["--max-model-len", str(num_ctx)]
        if cpu_offload_gb:
            options += ["--cpu-offload-gb", str(cpu_offload_gb)]
        if chat_template:
            options += ["--chat-template", chat_template]
        if embedding:
            # --task was replaced by --runner; pooling is the embedding runner.
            options += ["--runner", "pooling"]
        if tool_parser:
            # tool_calls stay empty unless the frontend parser is enabled explicitly.
            options += ["--enable-auto-tool-choice", "--tool-call-parser", tool_parser]
        if mtp_config:
            options += ["--speculative-config", json.dumps(mtp_config, separators=(",", ":"))]
        if self._executable:
            return [self._executable, "serve", repo, "--host", "127.0.0.1",
                    "--port", str(config.VLLM_PORT), *options]
        if self._launcher:
            return [self._launcher, "-p", str(config.VLLM_PORT), "-m", repo, *options]
        raise RuntimeError("no vLLM runtime found — run setup_check.py or install vLLM")

    def _ensure_model(self, tag: str, num_ctx: int | None, *, embedding: bool = False,
                       n_parallel: int = 1, deadline: float | None = None,
                       tool_parser: str | None = None) -> None:
        """Ensure vLLM is serving `tag`, respawning on any mismatch — one model per process."""
        mtp_config = self._native_mtp_config(tag, embedding=embedding)
        want = (tag, num_ctx, embedding, n_parallel, tool_parser, mtp_config)

        def ready():
            have = (self._loaded_tag, self._loaded_num_ctx, self._loaded_embedding,
                    self._loaded_n_parallel, self._loaded_tool_parser,
                    self._loaded_mtp_config)
            return want == have and self._proc is not None and self._proc.poll() is None

        if ready():
            return

        with self._model_lock:
            if ready():
                return
            if deadline is not None and time.perf_counter() >= deadline:
                raise EngineTimeout(f"loading {tag} exceeded the request wall-clock timeout")

            if self._server_url:
                # We never spawn or reconfigure an externally-managed server — it already
                # serves one fixed model, so reject requests that would mislabel its results.
                if not self.available():
                    raise RuntimeError(f"vLLM server at {self._server_url} is not reachable")
                if mtp_config:
                    raise RuntimeError(
                        "native MTP cannot be verified or configured on an external vLLM server"
                    )
                model_ids = self._served_model_ids()
                if model_ids is None:
                    raise RuntimeError(
                        f"vLLM server at {self._server_url} did not report its loaded model")
                model_id = self._external_model_id(tag, model_ids)
                if model_id is None:
                    served = ", ".join(sorted(model_ids)) or "(none)"
                    raise RuntimeError(
                        f"vLLM server at {self._server_url} serves {served}, not {tag}")
                self._loaded_tag, self._loaded_num_ctx = tag, num_ctx
                self._loaded_model_id = model_id
                self._loaded_embedding, self._loaded_n_parallel = embedding, n_parallel
                self._loaded_tool_parser = tool_parser
                self._loaded_mtp_config = None
                self._loaded_cpu_offload_gb = 0
                return

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

            context_limit = self.context_limit(tag, num_ctx)
            offload_key = self._offload_key(tag, repo)
            cpu_offload_gb = self._cpu_offload_gb.get(offload_key, 0)
            host_limit_gb = self._host_offload_limit_gb()
            if cpu_offload_gb > host_limit_gb:
                raise RuntimeError(
                    f"cached vLLM CPU offload for {tag} is {cpu_offload_gb} GiB per worker, "
                    f"but current free host RAM permits at most {host_limit_gb} GiB")
            offload_attempts = 0
            while True:
                self.stop()
                if deadline is not None and time.perf_counter() >= deadline:
                    raise EngineTimeout(
                        offload_timeout_message(tag, cpu_offload_gb, offload_attempts))
                if cpu_offload_gb:
                    Shared.warn(f"Loading {tag} with {cpu_offload_gb} GiB CPU offload")
                args = self.server_command(
                    repo, context_limit, embedding=embedding, n_parallel=n_parallel,
                    tool_parser=tool_parser, cpu_offload_gb=cpu_offload_gb,
                    chat_template=self._chat_template_argument(tag),
                    mtp_config=mtp_config,
                )
                log_fh = tempfile.NamedTemporaryFile(
                    mode="w", suffix="-vllm-server.log", delete=False)
                self._log_path = Path(log_fh.name)
                try:
                    # Own process group: the EngineCore child holds the weights and cache.
                    popen_kwargs = {
                        "stdout": log_fh, "stderr": subprocess.STDOUT,
                        "env": self.runtime_environment(),
                    }
                    if os.name == "nt":
                        popen_kwargs["creationflags"] = getattr(
                            subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    else:
                        popen_kwargs["start_new_session"] = True
                    proc = subprocess.Popen(args, **popen_kwargs)
                except FileNotFoundError:
                    log_fh.close()
                    raise RuntimeError(f"'{args[0]}' not found in PATH") from None
                log_fh.close()
                setattr(proc, "own_process_group", True)   # see Shared.shutdown_managed
                self._proc = proc
                Shared._managed_procs.append(proc)

                attempt_started = time.perf_counter()
                attempt_timeout = attempt_started + self.LOAD_TIMEOUT
                attempt_deadline = load_attempt_deadline(
                    attempt_started, deadline, self.LOAD_TIMEOUT)
                while time.perf_counter() < attempt_deadline:
                    if self.available():
                        self._loaded_tag = tag
                        self._loaded_model_id = repo
                        self._loaded_num_ctx = num_ctx
                        self._loaded_embedding = embedding
                        self._loaded_n_parallel = n_parallel
                        self._loaded_tool_parser = tool_parser
                        self._loaded_mtp_config = mtp_config
                        self._loaded_cpu_offload_gb = cpu_offload_gb
                        if cpu_offload_gb and self._cpu_offload_gb.get(offload_key) != cpu_offload_gb:
                            self._cpu_offload_gb[offload_key] = cpu_offload_gb
                            self._save_offload_cache()
                        return
                    if proc.poll() is not None:
                        output = self.tail_log(self.SPAWN_LOG_LINES)
                        retry_gb = next_cpu_offload_gb(output, cpu_offload_gb)
                        reason = offload_stop_reason(
                            retry_gb, host_limit_gb, offload_attempts)
                        if reason is not None:
                            raise RuntimeError(
                                f"vLLM exited unexpectedly (code {proc.returncode}) loading {tag}; "
                                f"{reason} — last output:\n{output}")
                        assert retry_gb is not None
                        Shared.warn(
                            f"vLLM measured insufficient KV-cache memory; retrying {tag} "
                            f"with {retry_gb} GiB CPU offload")
                        cpu_offload_gb = retry_gb
                        offload_attempts += 1
                        break
                    time.sleep(1)
                else:
                    self._stop_process()
                    raise load_timeout_error(
                        tag, cpu_offload_gb, offload_attempts, self.LOAD_TIMEOUT,
                        deadline is not None and deadline <= attempt_timeout)

    def runtime_environment(self) -> dict:
        """Environment shared by serving and offline vLLM commands."""
        env = {**os.environ, "HF_HOME": str(self._cache_home)}
        # The venv's bin holds ninja, which FlashInfer shells out to when JIT-building kernels.
        venv_bin = config.VLLM_VENV / ("Scripts" if os.name == "nt" else "bin")
        if venv_bin.is_dir():
            env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
        token_file = config.SCRIPT_DIR / "hf.txt"
        if not env.get("HF_TOKEN") and token_file.is_file():
            token = token_file.read_text(encoding="utf-8").strip()
            if token:
                env["HF_TOKEN"] = token
        if (not self._uses_launcher and self._server_url is None
                and Shared.detect_wsl(platform.system(), platform.release())):
            env.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
        return env

    # ── HTTP helpers ──

    @staticmethod
    def _urlopen(req, timeout):
        return openai_api.urlopen_with_detail(req, timeout, "vLLM")

    def _post(self, path: str, payload: dict, timeout: float):
        return self._urlopen(urllib.request.Request(
            f"{self.base_url}{path}", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        ), timeout)

    # ── inference ──

    def generate(self, tag: str, prompt: str, timeout: int = 600,
                 num_ctx: int | None = None, n_parallel: int = 1,
                 cache_prompt: bool = False) -> GenerationMeasurement:
        """Generate via /v1/completions; n_parallel must match prepare_concurrency."""
        operation_start = time.perf_counter()
        load_timeout = offload_calibration_timeout(self.LOAD_TIMEOUT, self.LOAD_TIMEOUT)
        self._ensure_model(
            tag, num_ctx, n_parallel=n_parallel,
            deadline=operation_start + load_timeout,
        )
        model_load_sec = time.perf_counter() - operation_start

        payload = {
            **self.sampling_payload(),
            "model": self._loaded_model_id or self._repo(tag),
            "prompt": prompt,
            "max_tokens": config.GENERATE_MAX_TOKENS,
            "stream": True,
            "stream_options": {"include_usage": True},
            "cache_salt": tag if cache_prompt else secrets.token_urlsafe(32),
        }
        prefill_before = self._prefill_reading()
        request_start = time.perf_counter()
        deadline = request_start + timeout
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
                tokens, prompt_tokens = openai_api.streamed_usage(
                    chunk, tokens, prompt_tokens,
                )
                if time.perf_counter() > deadline:
                    raise EngineTimeout(f"vllm_generate exceeded {timeout}s wall-clock timeout",
                                        partial_text="".join(response_parts))

        total = time.perf_counter() - request_start
        prefill_sec = self.prefill_seconds_from_delta(prefill_before, self._prefill_reading())
        ttft, decode_seconds, raw_tps, tps = openai_api.stream_timing(total, ttft, tokens)
        return GenerationMeasurement(
            client_ttft_sec=ttft,
            generated_tokens=tokens,
            tokens_per_sec=tps,
            client_wall_sec=total,
            decode_sec=decode_seconds,
            server_prompt_sec=prefill_sec,
            prompt_tokens=prompt_tokens,
            response_text="".join(response_parts),
            finish_reason=finish_reason,
            model_load_sec=model_load_sec,
            server_tps_implausible=tps != raw_tps,
            cpu_offload_gb=self._loaded_cpu_offload_gb,
        )

    def _chat_request(self, tag: str, messages: list, tools: list | None,
                      deadline: float, num_predict: int,
                      check_loop: bool, budget_nudged: bool) -> dict:
        payload = {
            **self.sampling_payload(),
            "model": self._loaded_model_id or self._repo(tag),
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

                tokens, prompt_eval_count = openai_api.streamed_usage(
                    chunk, tokens, prompt_eval_count,
                )

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
                    if response_text and looks_like_loop(response_text):
                        raise EngineLoopDetected(
                            f"vllm_chat detected a generation loop after "
                            f"{now - request_start:.0f}s",
                            partial_text=response_text, budget_nudged=budget_nudged)

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

    def _chat_with_optional_finalize(self, tag: str, messages: list, tools: list | None,
                                      timeout: int, num_ctx: int | None, num_predict: int,
                                      check_loop: bool, token_budget: int | None):
        validate_chat_budget(num_predict, token_budget)
        tool_parser = self._tool_parser(tag) if tools is not None else None
        if tools is not None and tool_parser is None:
            raise RuntimeError(
                f"no vLLM tool-call parser is configured for {tag}; vLLM returns no tool_calls "
                "without --tool-call-parser, so a tool result here would be wrong, not zero")
        operation_start = time.perf_counter()
        load_timeout = offload_calibration_timeout(self.LOAD_TIMEOUT, self.LOAD_TIMEOUT)
        self._ensure_model(
            tag, num_ctx, deadline=operation_start + load_timeout,
            tool_parser=tool_parser,
        )
        model_load_sec = time.perf_counter() - operation_start
        deadline = time.perf_counter() + timeout

        first, second, budget_nudged = run_bounded_chat(
            lambda req_messages, req_tools, req_deadline, req_predict, req_check, nudged:
                self._chat_request(
                    tag, req_messages, req_tools, req_deadline, req_predict, req_check, nudged,
                ),
            messages, tools, deadline, num_predict, check_loop, token_budget,
            config.ACC_FINALIZE_FRACTION, config.ACC_FINALIZE_MESSAGE, "vllm_chat",
        )
        return first, second, budget_nudged, model_load_sec

    def _chat_measurement(self, tag: str, messages: list, tools: list | None,
                          timeout: int, num_ctx: int | None, num_predict: int,
                          check_loop: bool, token_budget: int | None) -> ChatMeasurement:
        first, second, budget_nudged, model_load_sec = self._chat_with_optional_finalize(
            tag, messages, tools, timeout, num_ctx, num_predict, check_loop, token_budget)
        return chat_measurement(
            first, second, budget_nudged, model_load_sec, openai_api.sanitize_tps,
            self._loaded_cpu_offload_gb,
        )

    def chat(self, tag: str, messages: list, timeout: int = 600,
             num_ctx: int | None = None, num_predict: int = 1024,
             check_loop: bool = False, token_budget: int | None = None) -> ChatMeasurement:
        return self._chat_measurement(
            tag, messages, None, timeout, num_ctx, num_predict, check_loop, token_budget,
        )

    def chat_tools(self, tag: str, messages: list, tools: list, timeout: int = 600,
                   num_ctx: int | None = None, num_predict: int = 1024,
                   check_loop: bool = False, token_budget: int | None = None) -> ChatMeasurement:
        return self._chat_measurement(
            tag, messages, tools, timeout, num_ctx, num_predict, check_loop, token_budget,
        )

    def embed(self, tag: str, inputs: list[str], timeout: int = 120) -> EmbeddingMeasurement:
        """Embed every input in one request, serving the model in embedding mode."""
        load_start = time.perf_counter()
        self._ensure_model(tag, num_ctx=None, embedding=True)
        model_load_sec = time.perf_counter() - load_start

        t0 = time.perf_counter()
        resp = requests.post(f"{self.base_url}/v1/embeddings",
                              json={"model": self._loaded_model_id or self._repo(tag),
                                    "input": inputs}, timeout=timeout)
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
