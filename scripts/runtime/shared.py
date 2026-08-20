"""Cross-workload logging, ComfyUI, profiling, and orchestration helpers."""

import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import psutil
import requests

from scripts.runtime import config
from scripts.runtime.comfyui_installation import (
    checkpoint_names_from_object_info,
    find_comfyui_python,
    managed_checkpoints_visible,
    stop_running_comfyui,
    write_extra_model_paths,
)
from scripts.runtime import hardware
from scripts.runtime.log_redaction import redact_log_text
from scripts.workloads.models import IMAGE_MODELS
from scripts.runtime.pause_control import wait_if_paused
from scripts.results.result_store import atomic_write_json
from scripts.runtime.progress_events import emit_model_finished, emit_progress
from scripts.runtime.failure_handling import unexpected_model_failure
from scripts.runtime.generation_guard import looks_like_loop

if TYPE_CHECKING:
    from scripts.runtime.engines.base import InferenceEngine


RUN_LOG_UTC_OFFSET_ENV = "LOCAL_AI_BENCH_RUN_LOG_UTC_OFFSET_MINUTES"
WINDOWS_DISPLAY_ADAPTERS = {
    "microsoft basic display adapter", "microsoft remote display adapter",
}


def _windows_gpu_names(output: str) -> list[str]:
    return [
        name for line in output.splitlines() if (name := line.strip())
        and name.casefold() not in WINDOWS_DISPLAY_ADAPTERS
    ]


def _console_timezone(environment=None):
    raw = (os.environ if environment is None else environment).get(RUN_LOG_UTC_OFFSET_ENV)
    try:
        minutes = int(raw) if raw is not None else None
    except ValueError:
        return None
    if minutes is None or not -14 * 60 <= minutes <= 14 * 60:
        return None
    return timezone(timedelta(minutes=minutes))


def _console_now():
    local_timezone = _console_timezone()
    return datetime.now(local_timezone) if local_timezone else datetime.now()


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _console_supports_ansi(stream=None, environment=None):
    stream = stream or sys.stdout
    environment = os.environ if environment is None else environment
    return "NO_COLOR" not in environment and bool(getattr(stream, "isatty", lambda: False)())


def _console_safe_text(value):
    text = str(value)
    if not _console_supports_ansi():
        text = ANSI_ESCAPE_RE.sub("", text)
    encoding = getattr(sys.stdout, "encoding", None)
    if not encoding:
        return text
    return text.encode(encoding, errors="replace").decode(encoding)


def _nvidia_gpu_summary(output):
    devices = hardware.parse_nvidia_gpus(output)
    if not devices:
        return None, None
    capacities = [device["vram_gb"] for device in devices if device["vram_gb"] is not None]
    return devices[0]["name"], sum(capacities) if len(capacities) == len(devices) else None


def _rocm_gpu_summary(output):
    names = hardware.rocminfo_gpu_names(output)
    return names[0] if names else None


def _machine_identity(cpu, gpu, ram_gb, total_vram_gb=None):
    lines = []
    if cpu:
        lines.append(f"{cpu} / {ram_gb:g} GB RAM")
    elif ram_gb is not None:
        lines.append(f"{ram_gb:g} GB RAM")
    if gpu:
        suffix = f" / {total_vram_gb:g} GB VRAM" if total_vram_gb is not None else ""
        lines.append(f"{gpu}{suffix}")
    return "\n".join(lines)


class EngineTimeout(TimeoutError):
    """Raised when chat() exceeds its wall-clock timeout. Carries whatever text
    had streamed before the cutoff — see docs/workloads.md#timeouts-and-loop-detection."""

    def __init__(self, message: str, partial_text: str = "",
                 budget_nudged: bool = False):
        super().__init__(message)
        self.partial_text = partial_text
        self.budget_nudged = budget_nudged


class EngineLoopDetected(EngineTimeout):
    """A degenerate-loop cutoff before the timeout elapsed — kept distinct
    from EngineTimeout so callers keep "timed out" vs. "looped" counts apart."""


class EngineBudgetExceeded(Exception):
    """Raised when the finalize pass consumes the remaining token budget."""

    def __init__(self, message: str, partial_text: str = "",
                 budget_nudged: bool = True):
        super().__init__(message)
        self.partial_text = partial_text
        self.budget_nudged = budget_nudged


def split_token_budget(token_budget: int, first_pass_fraction: float) -> tuple[int, int]:
    if isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget <= 0:
        raise ValueError("token_budget must be a positive integer")
    if not 0 < first_pass_fraction < 1:
        raise ValueError("first_pass_fraction must be between 0 and 1")
    first_pass = max(1, math.floor(token_budget * first_pass_fraction))
    return first_pass, token_budget - first_pass


class Shared:
    # Both the inference engine's server and ComfyUI register here so shutdown_managed() can clean up everything at once.
    _managed_procs: list[subprocess.Popen] = []

    # Set once by benchmark.py so shutdown_managed() can reach the engine without every caller threading it through.
    _active_engine: "InferenceEngine | None" = None

    # Kept for the process's life so a later crash still has a log to inspect.
    _comfyui_log_path: Path | None = None

    # A deterministic crash (e.g. OOM) would otherwise recur identically forever.
    CRASH_RETRY_MAX = 2

    # ── logging ──
    @staticmethod
    def plain_output(msg="", *, end="\n"):
        print(_console_safe_text(redact_log_text(msg)), end=end)

    @staticmethod
    def clear_terminal():
        if platform.system() == "Windows":
            os.system("cls")
        else:
            Shared.plain_output("\033[2J\033[H", end="")

    @staticmethod
    def output(msg, *, leading_blank=False, timestamp_newline=False, end="\n"):
        if leading_blank:
            print()
        separator = "\n" if timestamp_newline else " "
        safe_message = redact_log_text(msg)
        line = f"[{_console_now().strftime('%H:%M:%S')}]{separator}{safe_message}"
        print(_console_safe_text(line), end=end)

    @staticmethod
    def log(msg):   Shared.output(f"  {config.CYAN}→{config.RESET}  {msg}")
    @staticmethod
    def ok(msg):    Shared.output(f"  {config.GREEN}✓{config.RESET}  {msg}")
    @staticmethod
    def warn(msg):  Shared.output(f"  {config.YELLOW}!{config.RESET}  {msg}")
    @staticmethod
    def err(msg):   Shared.output(f"  {config.RED}✗{config.RESET}  {msg}")
    @staticmethod
    def section(t): Shared.output(
        f"{config.BOLD}{'─'*50}\n  {t}\n{'─'*50}{config.RESET}",
        leading_blank=True, timestamp_newline=True,
    )

    # ── stats ──
    @staticmethod
    def mean(vals):   return statistics.mean(vals) if vals else 0
    @staticmethod
    def stdev(vals):  return statistics.stdev(vals) if len(vals) >= 2 else 0
    @staticmethod
    def median(vals): return statistics.median(vals) if vals else 0
    @staticmethod
    def coefficient_of_variation(vals):
        return Shared.stdev(vals) / Shared.mean(vals) if len(vals) >= 2 and Shared.mean(vals) else 0

    @staticmethod
    def context_label(tokens: int) -> str:
        return f"{tokens / 1024:g}K"

    @staticmethod
    def system_ram_gb():
        return psutil.virtual_memory().total / (1024 ** 3)

    @staticmethod
    def sample_memory_gb() -> dict:  # pragma: no cover — shells out to GPU tools + psutil
        """System RAM always; GPU VRAM only when nvidia-smi/rocm-smi answers
        (rocm-smi gated to a confirmed discrete AMD card, same as setup_check.py's check_rocm)."""
        vm = psutil.virtual_memory()
        snapshot = {
            "system_ram_used_gb":  round(vm.used / (1024 ** 3), 2),
            "system_ram_total_gb": round(vm.total / (1024 ** 3), 2),
            "gpu_vram_used_gb":  None,
            "gpu_vram_total_gb": None,
        }

        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL, timeout=10,
            )
            used_mib = total_mib = 0.0
            for line in out.strip().splitlines():
                used, total = line.split(",")
                used_mib += float(used.strip())
                total_mib += float(total.strip())
            snapshot["gpu_vram_used_gb"]  = round(used_mib / 1024, 2)
            snapshot["gpu_vram_total_gb"] = round(total_mib / 1024, 2)
            return snapshot
        except Exception:
            pass

        try:
            info_out = subprocess.check_output(
                ["rocminfo"], text=True, stderr=subprocess.DEVNULL,
            )
            gpu_names = hardware.rocminfo_gpu_names(info_out)
            if any(hardware.classify_gpu(name) == "discrete" for name in gpu_names):
                mem_out = subprocess.check_output(
                    ["rocm-smi", "--showmeminfo", "vram", "--json"],
                    text=True, stderr=subprocess.DEVNULL, timeout=10,
                )
                mem_data = json.loads(mem_out)
                used_bytes  = sum(int(c.get("VRAM Total Used Memory (B)", 0)) for c in mem_data.values())
                total_bytes = sum(int(c.get("VRAM Total Memory (B)", 0)) for c in mem_data.values())
                if total_bytes > 0:
                    snapshot["gpu_vram_used_gb"]  = round(used_bytes / (1024 ** 3), 2)
                    snapshot["gpu_vram_total_gb"] = round(total_bytes / (1024 ** 3), 2)
        except Exception:
            pass

        return snapshot

    # ── server management ──

    @staticmethod
    def shutdown_managed(engine: "InferenceEngine | None" = None):  # pragma: no cover — manages real subprocesses
        """Terminate any servers we started. A forced-CPU-only engine is
        stopped first so it doesn't linger with GPU devices hidden."""
        engine = engine or Shared._active_engine
        if engine is not None and getattr(engine, "_cpu_only_active", False):
            Shared.warn("Exiting while the engine is in forced CPU-only mode — killing it "
                        "rather than leaving GPU devices hidden in the background")
            engine.stop()
        for proc in Shared._managed_procs:
            if proc.poll() is None:
                Shared.log(f"Stopping managed process (pid {proc.pid}) ...")
                # A server started in its own group (vLLM, whose EngineCore child holds
                # the weights) must be signalled as a group or the child is orphaned.
                if getattr(proc, "own_process_group", False):
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        except (OSError, ProcessLookupError):
                            proc.terminate()
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        Shared._managed_procs.clear()

    @staticmethod
    def _tail_log(path: Path | None, service_name: str, n_lines: int = 40) -> str:
        """Return the last n_lines of a server's captured output."""
        if path is None:
            return f"(no {service_name} log captured this session)"
        try:
            lines = path.read_text(errors="replace").splitlines()
            return "\n".join(lines[-n_lines:]) or "(log file is empty)"
        except Exception as e:
            return f"(failed to read {service_name} log: {e})"

    @staticmethod
    def tail_comfyui_log(n_lines: int = 40) -> str:
        """Return the last n_lines of the current ComfyUI server's captured
        output, for surfacing the real crash reason instead of guessing."""
        return Shared._tail_log(Shared._comfyui_log_path, "ComfyUI", n_lines)

    @staticmethod
    def find_comfyui_python(comfyui_dir: Path) -> str:
        """Return the selected installation's Python environment."""
        return find_comfyui_python(comfyui_dir)

    @staticmethod
    def ensure_comfyui(comfyui_dir: Path) -> bool:  # pragma: no cover — spawns a real subprocess and polls a live server
        """Start ComfyUI if not already running. Returns whether it's now available."""
        if Shared.comfyui_available():
            try:
                response = requests.get(f"{config.COMFYUI_URL}/object_info/CheckpointLoaderSimple", timeout=5)
                available = checkpoint_names_from_object_info(response.json())
                managed = {
                    model["checkpoint"] for model in IMAGE_MODELS
                    if (config.COMFYUI_MODELS_DIR / "checkpoints" / model["checkpoint"]).is_file()
                }
                if not managed_checkpoints_visible(available, managed):
                    Shared.warn("ComfyUI is running but has not loaded Local AI Bench's managed model path")
                    if not stop_running_comfyui(comfyui_dir):
                        Shared.warn("The running server belongs to another ComfyUI installation; stop it and retry")
                        return False
                    if Shared.comfyui_available():
                        Shared.warn("ComfyUI is still reachable after stopping the selected installation")
                        return False
                    Shared.log("Restarting the selected ComfyUI installation with the managed model path ...")
            except Exception as exc:
                Shared.warn(f"Could not verify checkpoints in the running ComfyUI server: {exc}")
                return False
            Shared.ok("ComfyUI already running")
            return True

        if not comfyui_dir.exists():
            Shared.warn(f"ComfyUI directory not found at {comfyui_dir}")
            Shared.warn("Clone it with: git clone https://github.com/comfyanonymous/ComfyUI")
            return False

        main_py = comfyui_dir / "main.py"
        if not main_py.exists():
            Shared.warn(f"main.py not found in {comfyui_dir}")
            return False

        # Check at least one image model checkpoint is present
        checkpoints_dir = config.COMFYUI_MODELS_DIR / "checkpoints"
        known = [m["checkpoint"] for m in IMAGE_MODELS]
        found = [c for c in known if (checkpoints_dir / c).exists()]
        if not found:
            Shared.warn("No image model checkpoints found in " + str(checkpoints_dir))
            Shared.warn("Expected one of: " + ", ".join(known))
            Shared.warn("Run setup_check.py to download Flux models automatically")
            return False
        Shared.log(f"Found {len(found)}/{len(known)} image checkpoints: {found}")

        python_exe = Shared.find_comfyui_python(comfyui_dir)

        # Windows portable builds: python_embeded is a sibling of ComfyUI/, cwd must be the parent
        portable_windows = (comfyui_dir.parent / "python_embeded" / "python.exe").exists()
        if portable_windows:
            cmd = [python_exe, "-s", str(main_py), "--windows-standalone-build", "--listen"]
            launch_cwd = str(comfyui_dir.parent)
        else:
            cmd = [python_exe, str(main_py), "--listen"]
            launch_cwd = str(comfyui_dir)

        write_extra_model_paths(config.COMFYUI_EXTRA_MODEL_PATHS, config.COMFYUI_MODELS_DIR)
        cmd.extend(["--extra-model-paths-config", str(config.COMFYUI_EXTRA_MODEL_PATHS)])

        # Dynamic VRAM has an unresolved upstream bug streaming combined checkpoint
        # files like SDXL's (Comfy-Org/ComfyUI#14239, #14281) — disabled globally.
        cmd.append("--disable-dynamic-vram")

        Shared.log(f"Starting ComfyUI from {comfyui_dir} using {python_exe} ...")

        env = os.environ.copy()
        # AMD on Windows: Triton JIT compilation fails; interpreter mode works around it
        if portable_windows and Shared.detect_backend() == "rocm":
            env["TRITON_INTERPRET"] = "1"

        # Capture stdout+stderr to a log kept for the whole process lifetime,
        # so a crash later in the run still has real output to inspect.
        try:
            log_fh = tempfile.NamedTemporaryFile(
                mode="w", suffix="-comfyui-server.log", delete=False
            )
            Shared._comfyui_log_path = Path(log_fh.name)
            proc = subprocess.Popen(
                cmd,
                cwd=launch_cwd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=env,
            )
            log_fh.close()
            Shared._managed_procs.append(proc)
        except Exception as e:
            Shared.err(f"Failed to start ComfyUI: {e}")
            return False

        # Wait up to 60s — model loading takes time
        Shared.log("Waiting for ComfyUI to be ready (up to 60s) ...")
        for i in range(60):
            time.sleep(1)
            if Shared.comfyui_available():
                Shared.ok(f"ComfyUI started (pid {proc.pid}) — log: {Shared._comfyui_log_path}")
                return True
            if proc.poll() is not None:
                Shared.err(f"ComfyUI exited unexpectedly (code {proc.returncode})")
                Shared.err(f"Last output from ComfyUI:\n{Shared.tail_comfyui_log()}")
                Shared.err(f"Try starting manually: cd {comfyui_dir} && python main.py {' '.join(cmd[2:])}")
                return False
            if (i + 1) % 10 == 0:
                Shared.log(f"Still waiting ... ({i+1}s)")

        Shared.err("ComfyUI did not respond within 60 seconds")
        return False

    # ── machine profile ──

    @staticmethod
    def detect_wsl(os_name: str, release: str) -> bool:
        """WSL reports itself as Linux; only the kernel release distinguishes it."""
        return hardware.detect_wsl(os_name, release)

    @staticmethod
    def get_hostname():  # pragma: no cover — shells out to OS-specific hardware profiling tools
        system = platform.system()
        ram_gb = round(Shared.system_ram_gb())

        if system == "Darwin":
            try:
                sp = subprocess.run(
                    ["system_profiler", "SPHardwareDataType"],
                    capture_output=True, text=True, timeout=10,
                )
                model = chip = ram = None
                for line in sp.stdout.splitlines():
                    if "Model Name:" in line:
                        model = line.split(":", 1)[1].strip()
                    elif "Chip:" in line:
                        chip = line.split(":", 1)[1].strip().removeprefix("Apple ").strip()
                    elif "Memory:" in line:
                        ram = line.split(":", 1)[1].strip()
                if model and chip and ram:
                    return f"{model}\n{chip} {ram}"
            except Exception:
                pass

        elif system == "Windows":
            cpu = gpu = None
            total_vram_gb = None

            def _ps_names(cim_class):
                try:
                    out = subprocess.run(
                        ["powershell", "-NoProfile", "-Command",
                         f"(Get-CimInstance {cim_class}).Name"],
                        capture_output=True, text=True, timeout=10,
                    ).stdout
                    return [n.strip() for n in out.splitlines() if n.strip()]
                except Exception:
                    return []

            cpu_names = _ps_names("Win32_Processor")
            if cpu_names:
                cpu = cpu_names[0]

            gpus = _windows_gpu_names("\n".join(_ps_names("Win32_VideoController")))
            if gpus:
                gpu = gpus[0]
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10,
                ).stdout
                nvidia_gpu, total_vram_gb = _nvidia_gpu_summary(out)
                gpu = nvidia_gpu or gpu
            except Exception:
                pass
            if cpu or gpu:
                return _machine_identity(cpu, gpu, ram_gb, total_vram_gb)

        elif system == "Linux":
            cpu = gpu = None
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("model name"):
                            cpu = line.split(":", 1)[1].strip()
                            break
            except Exception:
                pass
            # NVIDIA first, then AMD via rocminfo, then lspci fallback
            try:
                out = subprocess.run(
                    [hardware.nvidia_smi_executable(), "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip()
                if out:
                    gpu = out.splitlines()[0].strip()
            except Exception:
                pass
            if not gpu:
                try:
                    out = subprocess.run(
                        ["rocminfo"], capture_output=True, text=True, timeout=10,
                    ).stdout
                    gpu = _rocm_gpu_summary(out)
                except Exception:
                    pass
            if not gpu and Shared.detect_wsl(system, platform.release()):
                try:
                    out = subprocess.run(
                        ["powershell.exe", "-NoProfile", "-Command",
                         "(Get-CimInstance Win32_VideoController).Name"],
                        capture_output=True, text=True, timeout=10,
                    ).stdout
                    gpus = _windows_gpu_names(out)
                    gpu = gpus[0] if gpus else None
                except Exception:
                    pass
            if not gpu:
                try:
                    out = subprocess.run(
                        ["lspci", "-nn"], capture_output=True, text=True, timeout=10,
                    ).stdout
                    for line in out.splitlines():
                        if any(k in line for k in ("VGA", "3D controller", "Display")):
                            gpu = line.split(":", 2)[-1].strip()
                            break
                except Exception:
                    pass
            if cpu and gpu:
                return f"{cpu}\n{gpu} {ram_gb} GB"
            elif cpu:
                return f"{cpu}\n{ram_gb} GB"
            elif gpu:
                return f"{gpu} {ram_gb} GB"

        return platform.node()

    @staticmethod
    def build_profile():  # pragma: no cover — thin wrapper around get_hostname/detect_backend
        os_name, release = platform.system(), platform.release()
        profile = {
            "hostname":   Shared.get_hostname(),
            "os":         f"{os_name} {release}",
            "arch":       platform.machine(),
            "python":     sys.version.split()[0],
            "ram_gb":     round(Shared.system_ram_gb(), 1),
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "backend":    Shared.detect_backend(),
        }
        if Shared.detect_wsl(os_name, release):
            profile["wsl"] = True
        return profile

    @staticmethod
    def detect_backend():  # pragma: no cover — shells out to GPU-detection tools
        # Nvidia
        try:
            subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
            return "cuda"
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        # ROCm (Linux)
        try:
            out = subprocess.check_output(["rocminfo"], text=True,
                                           stderr=subprocess.DEVNULL)
            if _rocm_gpu_summary(out):
                return "rocm"
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        # AMD/Intel on Windows — rocminfo/xpu-smi don't exist; detect via PowerShell
        if platform.system() == "Windows":
            try:
                out = subprocess.check_output(
                    ["powershell", "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_VideoController).Name"],
                    text=True, stderr=subprocess.DEVNULL,
                )
                names = [n.strip() for n in out.splitlines() if n.strip()]
                if any("AMD" in n or "Radeon" in n for n in names):
                    return "rocm"
                if any(hardware.is_intel_xpu_display(n) for n in names):
                    return "xpu"
            except Exception:
                pass
        # lspci may report Arc Pro B-series by its Battlemage codename.
        if platform.system() == "Linux":
            try:
                out = subprocess.check_output(
                    ["lspci", "-nn"], text=True, stderr=subprocess.DEVNULL,
                )
                for line in out.splitlines():
                    if (any(k in line for k in ("VGA", "3D controller", "Display"))
                            and hardware.is_intel_xpu_display(line)):
                        return "xpu"
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass
        # Metal
        if platform.system() == "Darwin":
            return "metal"
        return "cpu"

    # ── prompt builders ──

    LONG_DOCUMENT_PATH = Path(__file__).resolve().parents[1] / "workloads" / "data" / "long_document.txt"
    _long_document_cache: str | None = None

    @classmethod
    def _long_document(cls) -> str:
        """Lazily load and cache the real, non-repetitive source text used to pad prompts."""
        if cls._long_document_cache is None:
            cls._long_document_cache = cls.LONG_DOCUMENT_PATH.read_text(encoding="utf-8")
        return cls._long_document_cache

    @staticmethod
    def ctx_with_headroom(base_ctx: int, headroom: int, model_max: int) -> int:
        """base_ctx plus generation headroom, clamped to the model's real max."""
        return min(base_ctx + headroom, model_max)

    @staticmethod
    def build_prompt_for_context(target_tokens: int, variant: int = 0) -> str:
        """Build stable real-text prompt content for a context and variant."""
        if variant < 0:
            raise ValueError("prompt variant must be non-negative")
        prefix = f"[single-shot {target_tokens}:{variant}] "
        chars_needed = target_tokens * 4
        body_needed = max(0, chars_needed - len(prefix))

        document = Shared._long_document()
        if body_needed > len(document):
            document = document * (body_needed // len(document) + 1)
        available = max(0, len(document) - body_needed)
        start = ((target_tokens * 2654435761 + variant * 2246822519) % (available + 1)
                 if available else 0)
        body = document[start:start + body_needed]

        return (prefix + body)[:chars_needed]

    @staticmethod
    def stratified_sample(questions: list[dict], n: int, seed: int = 1337) -> list[dict]:
        """Deterministically pick `n` questions round-robin by category — see `--sample` in docs/cli-reference.md."""
        if n >= len(questions):
            return list(questions)
        by_category: dict[str, list[dict]] = {}
        for q in questions:
            by_category.setdefault(q["category"], []).append(q)
        rng = random.Random(seed)
        for group in by_category.values():
            rng.shuffle(group)
        categories = sorted(by_category)
        picked = []
        idx = 0
        while len(picked) < n:
            progressed = False
            for cat in categories:
                if idx < len(by_category[cat]):
                    picked.append(by_category[cat][idx])
                    progressed = True
                    if len(picked) == n:
                        break
            if not progressed:
                break
            idx += 1
        return picked

    @staticmethod
    def file_hash(path: Path | str) -> str:
        """First 12 hex chars of `path`'s sha256 — a bank-version fingerprint
        that catches whitespace/key-order changes a question count wouldn't."""
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]

    @staticmethod
    def run_measured_calls(n_runs: int, call, tag: str, crash_cache: dict, cache_path: Path,
                            what: str, engine: "InferenceEngine",
                            crash_extra: dict | None = None) -> tuple[list, str, str, dict]:
        """Shared "N measured runs" loop — see docs/workloads.md#timeouts-and-loop-detection
        and docs/project-structure.md's *_crash_cache.json entries. Returns (samples, status, partial_text, metadata)."""
        from scripts.runtime.crash_cache import record_crash

        samples = []
        run_i = 0
        crash_retries = 0
        while run_i < n_runs:
            wait_if_paused()
            try:
                samples.append(call(run_i))
                run_i += 1
            except Exception as e:
                metadata = {"budget_nudged": getattr(e, "budget_nudged", False)}
                if isinstance(e, EngineBudgetExceeded):
                    Shared.err(f"{tag}: exhausted its generation budget {what} (run {run_i+1})")
                    return samples, "budget_exceeded", e.partial_text, metadata
                if isinstance(e, EngineLoopDetected):
                    Shared.err(f"{tag}: detected a generation loop {what} (run {run_i+1})")
                    return samples, "loop_detected", e.partial_text, metadata
                is_timeout = isinstance(e, TimeoutError) or "timed out" in str(e).lower()
                if is_timeout:
                    Shared.err(f"{tag}: timed out {what} (run {run_i+1})")
                    partial_text = getattr(e, "partial_text", "")
                    return samples, "timed_out", partial_text, metadata
                Shared.err(f"Run {run_i+1} failed: {e}")
                if not engine.is_connection_crash(e):
                    run_i += 1
                    continue
                crash_retries += 1
                Shared.err(f"The engine's model runner appears to have crashed {what} "
                           f"— last server output:\n{engine.tail_log()}")
                if crash_retries > Shared.CRASH_RETRY_MAX:
                    Shared.err(f"The engine's model runner crashed {crash_retries} times — giving up on {tag}")
                    record_crash(tag, crash_cache, cache_path, what,
                                 extra=crash_extra, engine_name=engine.name)
                    return samples, "crashed", "", metadata
                Shared.warn(f"Waiting for recovery, retry {crash_retries}/{Shared.CRASH_RETRY_MAX} ...")
                if not engine.wait_for_recovery():
                    Shared.warn("The engine did not become reachable again within 30s — giving up on this model")
                    record_crash(tag, crash_cache, cache_path, what,
                                 extra=crash_extra, engine_name=engine.name)
                    return samples, "crashed", "", metadata
                # don't advance run_i — retry the same run now that the engine is back
        return samples, "ok", "", {"budget_nudged": False}

    @staticmethod
    def retry_implausible_tps(call, tag: str, progress_stage: str | None = None):
        measurement = call()
        if not measurement.server_tps_implausible:
            return measurement
        Shared.warn(f"{tag}: retrying once after an implausible server TPS report")
        if progress_stage:
            emit_progress("measurement", progress_stage, "retrying", tag)
        measurement = call()
        if measurement.server_tps_implausible:
            Shared.warn(f"{tag}: retry also reported implausible TPS; dropping that measurement")
            if progress_stage:
                emit_progress("measurement", progress_stage, "invalid", tag)
        elif progress_stage:
            emit_progress("measurement", progress_stage, "valid", tag)
        return measurement

    @staticmethod
    def write_answers_sidecar(path: Path, data: dict) -> None:
        """Write an accuracy test's raw-answer sidecar — see docs/project-structure.md's "answers_*.json" section."""
        atomic_write_json(path, data)

    @staticmethod
    def run_accuracy_benchmark(section_label: str, skip_label: str, question_noun: str,
                                data_path: Path, crash_cache_path: Path, models, questions,
                                warmup_runs: int, engine: "InferenceEngine",
                                ask_fn, rescore_partial_fn, score_fn,
                                save_fn=None, answers_path: Path | None = None,
                                progress_stage: str | None = None,
                                requires_tool_calls: bool = False,
                                telemetry=None,
                                journal=None,
                                ) -> dict:
        """Shared run() body for the MCQ/Math/Reasoning/Code/Tool accuracy tests,
        parameterized by `ask_fn`/`rescore_partial_fn`/`score_fn` (see callers)."""
        from scripts.runtime.crash_cache import check_crash_cache, load_crash_cache

        results = journal.export_results() if journal else {}
        answers_out: dict = journal.export_answers() if journal else {}

        if not engine.ensure_running():
            Shared.err(f"Inference engine not reachable — skipping {skip_label} benchmark")
            if journal:
                raise RuntimeError("inference engine unavailable in supervised accuracy runner")
            return results

        crash_cache = load_crash_cache(crash_cache_path)
        bank_hash = Shared.file_hash(data_path)

        for model in models:
            tag   = model["tag"]
            label = model["label"]
            short = model["short"]

            Shared.section(f"{section_label} ({engine.name}): {label}")

            if not engine.reachable_or_abort():
                break

            if progress_stage:
                emit_progress("model", progress_stage, "running", label, model_id=tag)
            telemetry_active = False
            try:
                pending_questions = journal.pending_questions(model) if journal else questions
                if journal and not pending_questions:
                    results = journal.export_results()
                    answers_out = journal.export_answers()
                    continue
                if not engine.model_pulled(tag):
                    Shared.warn(f"{tag} not downloaded — skipping")
                    Shared.warn("Download it with: python setup_check.py")
                    if journal:
                        journal.record_model_state(model, "skipped", {
                            "skipped": True, "skip_reason": "model_not_downloaded",
                        })
                    continue

                if requires_tool_calls and not engine.supports_tool_calls(tag):
                    Shared.warn(f"{tag}: {engine.name} cannot return parsed tool calls for "
                                "this model — skipping rather than scoring unparsed calls wrong")
                    results[short] = {
                        "label": label,
                        "skipped": True,
                        "skip_reason": "tool_calls_unsupported",
                        "skip_detail": f"No {engine.name} tool-call parser for this model",
                    }
                    if journal:
                        journal.record_model_state(model, "skipped", results[short])
                    continue

                skip_entry = check_crash_cache(
                    tag, label, crash_cache, crash_cache_path,
                    expected_bank_hash=bank_hash, engine_name=engine.name,
                )
                if skip_entry is not None:
                    results[short] = skip_entry
                    if journal:
                        journal.record_model_state(model, "skipped", skip_entry)
                    continue

                if telemetry:
                    telemetry.begin_model_load()
                    telemetry_active = True
                if not engine.warmup(tag, label, config.ACCURACY_CONTEXT, warmup_runs,
                                     crash_cache, crash_cache_path,
                                     crash_extra={"bank_hash": bank_hash}):
                    engine.unload(tag)
                    continue

                Shared.log(
                    f"Answering {len(pending_questions)} {question_noun} "
                    f"({config.ACC_TIMEOUT}s and {config.ACC_TOKEN_BUDGET} completion tokens each) ..."
                )
                answers: dict = {}
                raw_responses: dict[str, str] = {}
                timed_out_ids: list[str] = []
                likely_loop_ids: list[str] = []
                budget_nudged_ids: list[str] = []
                budget_exceeded_ids: list[str] = []
                stopped_early = None
                answers_out[short] = {"label": label, "answers": [], "partial": True}
                results[short] = {"label": label, "partial": True, "answered": 0, "total": len(questions)}

                for i, q in enumerate(pending_questions):
                    if telemetry:
                        telemetry.begin_measured(f"measured:{q['id']}")
                    samples, status, partial_text, metadata = Shared.run_measured_calls(
                        1, lambda run_i, q=q: ask_fn(tag, q), tag, crash_cache,
                        crash_cache_path, f"answering {q['id']}", engine,
                        crash_extra={"bank_hash": bank_hash})
                    budget_nudged = metadata["budget_nudged"]
                    if samples:
                        given, raw, budget_nudged = samples[0]
                    elif partial_text:
                        # Score whatever streamed before the cutoff rather than treating it as blank.
                        given, raw = rescore_partial_fn(q, partial_text), partial_text
                    else:
                        given, raw = None, ""
                    answers[q["id"]] = given
                    raw_responses[q["id"]] = raw
                    answers_out[short]["answers"].append({
                        "id": q["id"], "given": given, "raw_response": raw,
                    })
                    results[short]["answered"] = len(answers)

                    if journal:
                        attempt_number = journal.next_attempt(model, q["id"])
                        if attempt_number is None:
                            raise ValueError(f"accuracy question already completed: {q['id']}")
                        journal.record_question(
                            model, q["id"], given, raw, status,
                            budget_nudged=budget_nudged,
                            likely_loop=(status == "loop_detected" or (
                                status == "timed_out" and bool(partial_text)
                                and looks_like_loop(partial_text)
                            )),
                            attempt_number=attempt_number,
                        )
                        results = journal.export_results()
                        answers_out = journal.export_answers()

                    if budget_nudged:
                        budget_nudged_ids.append(q["id"])
                    if status == "budget_exceeded":
                        Shared.warn(
                            f"{q['id']} exhausted its {config.ACC_TOKEN_BUDGET}-token "
                            "generation budget — scoring the final partial response and continuing"
                        )
                        budget_exceeded_ids.append(q["id"])
                    elif status == "timed_out":
                        # The partial response, if any, was rescored above; continue the bank either way.
                        Shared.warn(f"{q['id']} timed out after {config.ACC_TIMEOUT}s — "
                                    "scoring the partial response and continuing")
                        timed_out_ids.append(q["id"])
                        if partial_text and looks_like_loop(partial_text):
                            likely_loop_ids.append(q["id"])
                    elif status == "loop_detected":
                        Shared.warn(f"{q['id']}: response looks like a generation loop")
                        likely_loop_ids.append(q["id"])
                    if status == "crashed":
                        stopped_early = "crashed"
                        break

                    if (i + 1) % 10 == 0:
                        Shared.log(f"  {i+1}/{len(questions)} answered ...")

                if journal:
                    scored_result = results.get(short, {})
                    if scored_result.get("accuracy_pct") is not None:
                        Shared.ok(f"{label}: {scored_result['accuracy_pct']:.1f}% "
                                  f"({scored_result['correct']}/{scored_result['total']})")
                    Shared.log(f"Unloading {label} ...")
                    engine.unload(tag)
                    engine.wait_until_unloaded(tag)
                    continue

                scored = score_fn(questions, answers)
                answers_out[short] = {
                    "label": label,
                    "answers": [
                        {**entry, "raw_response": raw_responses.get(entry["id"], "")}
                        for entry in scored["all"]
                    ],
                }
                scored.pop("all", None)
                results[short] = {"label": label, **scored}

                if timed_out_ids:
                    results[short]["timed_out_count"] = len(timed_out_ids)
                    results[short]["timed_out_ids"] = timed_out_ids
                if budget_nudged_ids:
                    results[short]["budget_nudged_count"] = len(budget_nudged_ids)
                    results[short]["budget_nudged_ids"] = budget_nudged_ids
                if budget_exceeded_ids:
                    results[short]["budget_exceeded_count"] = len(budget_exceeded_ids)
                    results[short]["budget_exceeded_ids"] = budget_exceeded_ids
                if likely_loop_ids:
                    # Only list a flagged question as a loop if it also scored wrong.
                    incorrect_ids = {entry["id"] for entry in scored["incorrect"]}
                    likely_loop_ids = [qid for qid in likely_loop_ids if qid in incorrect_ids]
                if likely_loop_ids:
                    results[short]["likely_loop_count"] = len(likely_loop_ids)
                    results[short]["likely_loop_ids"] = likely_loop_ids
                if stopped_early == "crashed":
                    crashed_at = crash_cache.get(tag, {}).get("crashed_at", "an earlier run")
                    results[short]["crashed"] = True
                    results[short]["crashed_at"] = crashed_at

                Shared.ok(f"{label}: {scored['accuracy_pct']:.1f}% "
                          f"({scored['correct']}/{scored['total']})")

                Shared.log(f"Unloading {label} ...")
                engine.unload(tag)
                engine.wait_until_unloaded(tag)
            except Exception as exc:
                Shared.err(f"{label}: unexpected error running the {skip_label} benchmark — {exc} — "
                           "skipping remaining work for this model")
                results.setdefault(short, {}).update(
                    unexpected_model_failure(label, exc, crashed=True))
                if journal:
                    journal.record_model_state(model, "failed", results[short])
                    results = journal.export_results()
                    answers_out = journal.export_answers()
            finally:
                if telemetry_active and telemetry:
                    memory = telemetry.finish_case()
                    if journal:
                        journal.record_model_evidence(
                            model, memory, getattr(telemetry, "last_power", None),
                        )
                        results = journal.export_results()
                        answers_out = journal.export_answers()
                    elif isinstance(results.get(short), dict):
                        results[short]["memory"] = memory
                        if (power := getattr(telemetry, "last_power", None)) is not None:
                            results[short]["power"] = power
                if save_fn and not journal:
                    save_fn(results)
                if answers_path and not journal:
                    Shared.write_answers_sidecar(answers_path, answers_out)
                if progress_stage:
                    emit_model_finished(progress_stage, label, results.get(short), model_id=tag)

        if journal:
            journal.finish()
            return journal.export_results()
        return results


    # ── ComfyUI ──

    @staticmethod
    def comfyui_available():  # pragma: no cover — real HTTP call
        try:
            r = requests.get(f"{config.COMFYUI_URL}/system_stats", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    # ── shared across LLM prefill + conversation tests ──

    @staticmethod
    def slow_tps_early_exit(results, short, label, label_ctx, is_first_ctx, tps_list, force_all):
        """Shared by the LLM prefill/conversation tests — see docs/workloads.md's slow-model check."""
        if not (is_first_ctx and tps_list and Shared.mean(tps_list) < config.SLOW_MODEL_MIN_TPS):
            return False
        if force_all:
            Shared.warn(f"{label}: {Shared.mean(tps_list):.1f} tok/s at {label_ctx} context is below "
                        f"{config.SLOW_MODEL_MIN_TPS:.0f} tok/s cutoff — --force-all set, continuing anyway")
            return False
        Shared.warn(f"{label}: {Shared.mean(tps_list):.1f} tok/s at {label_ctx} context is below "
                    f"{config.SLOW_MODEL_MIN_TPS:.0f} tok/s cutoff — marking slow, skipping deeper contexts")
        results[short]["slow_tps"] = label_ctx
        return True
