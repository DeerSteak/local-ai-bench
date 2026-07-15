"""
shared.py — cross-cutting helpers used by more than one test: logging, server
lifecycle management, machine profiling, and the low-level Ollama/ComfyUI HTTP
clients. Everything here is stateless-per-call except the managed-process
bookkeeping (used to shut down servers this script itself started), so
methods are static and state lives on the class rather than an instance.
"""

import hashlib
import http.client
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psutil
import requests

import config
from models import IMAGE_MODELS


class OllamaTimeout(TimeoutError):
    """Raised when ollama_chat exceeds its wall-clock timeout. Carries whatever
    text had streamed in before the deadline hit, so callers can tell a bare
    timeout (no text at all) apart from a timeout that cut off a response the
    model had already started writing — which might have been a wrong-format
    answer regardless of the timeout, or might have been about to be correct."""

    def __init__(self, message: str, partial_text: str = ""):
        super().__init__(message)
        self.partial_text = partial_text


class Shared:
    # Tracks processes we started so we can shut them down cleanly
    _managed_procs: list[subprocess.Popen] = []

    # True while Ollama is running with GPU devices hidden (--cpu-only). If
    # the script dies before it restores normal mode, shutdown_managed() must
    # not leave that GPU-hidden process running silently in the background.
    _cpu_only_active = False

    # Path to the log file capturing the current/most recent Ollama server's
    # stdout+stderr — set by start_ollama() so a crash's actual message can be
    # surfaced later instead of silently discarded.
    _ollama_log_path: Path | None = None

    # Same as _ollama_log_path but for the ComfyUI server process. Kept for the
    # life of the process (not deleted on successful startup) so a crash later
    # in the run — e.g. an OOM while loading a large checkpoint — still has a
    # log to inspect instead of going silent.
    _comfyui_log_path: Path | None = None

    # Cap on how many times a benchmark retries a request after Ollama's model
    # runner subprocess crashes (commonly OOM) before giving up on that model —
    # a deterministic crash would otherwise recur identically forever.
    CRASH_RETRY_MAX = 2

    # ── logging ──
    @staticmethod
    def log(msg):   print(f"  {config.CYAN}→{config.RESET}  {msg}")
    @staticmethod
    def ok(msg):    print(f"  {config.GREEN}✓{config.RESET}  {msg}")
    @staticmethod
    def warn(msg):  print(f"  {config.YELLOW}!{config.RESET}  {msg}")
    @staticmethod
    def err(msg):   print(f"  {config.RED}✗{config.RESET}  {msg}")
    @staticmethod
    def section(t): print(f"\n{config.BOLD}{'─'*50}\n  {t}\n{'─'*50}{config.RESET}")

    # ── stats ──
    @staticmethod
    def mean(vals):   return statistics.mean(vals) if vals else 0
    @staticmethod
    def stdev(vals):  return statistics.stdev(vals) if len(vals) >= 2 else 0

    @staticmethod
    def system_ram_gb():
        return psutil.virtual_memory().total / (1024 ** 3)

    # ── server management ──

    @staticmethod
    def shutdown_managed():  # pragma: no cover — manages real subprocesses
        """Terminate any servers we started."""
        if Shared._cpu_only_active:
            Shared.warn("Exiting while Ollama is in forced CPU-only mode — killing it "
                        "rather than leaving GPU devices hidden in the background")
            Shared.stop_all_ollama()
        for proc in Shared._managed_procs:
            if proc.poll() is None:
                Shared.log(f"Stopping managed process (pid {proc.pid}) ...")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        Shared._managed_procs.clear()

    @staticmethod
    def tail_ollama_log(n_lines: int = 40) -> str:
        """Return the last n_lines of the current Ollama server's captured
        output, for surfacing the real crash reason instead of guessing."""
        if Shared._ollama_log_path is None:
            return "(no Ollama log captured this session)"
        try:
            lines = Shared._ollama_log_path.read_text(errors="replace").splitlines()
            return "\n".join(lines[-n_lines:]) or "(log file is empty)"
        except Exception as e:
            return f"(failed to read Ollama log: {e})"

    @staticmethod
    def tail_comfyui_log(n_lines: int = 40) -> str:
        """Return the last n_lines of the current ComfyUI server's captured
        output, for surfacing the real crash reason instead of guessing."""
        if Shared._comfyui_log_path is None:
            return "(no ComfyUI log captured this session)"
        try:
            lines = Shared._comfyui_log_path.read_text(errors="replace").splitlines()
            return "\n".join(lines[-n_lines:]) or "(log file is empty)"
        except Exception as e:
            return f"(failed to read ComfyUI log: {e})"

    @staticmethod
    def start_ollama(extra_env: dict | None = None, timeout: int = 15) -> bool:  # pragma: no cover — spawns a real subprocess
        """Start 'ollama serve', optionally with extra/overridden environment
        variables (e.g. HIP_VISIBLE_DEVICES="" to force CPU-only). Tracked in
        _managed_procs for cleanup on exit. Returns True once reachable."""
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)

        os_name = platform.system()
        try:
            log_fh = tempfile.NamedTemporaryFile(
                mode="w", suffix="-ollama-server.log", delete=False
            )
            Shared._ollama_log_path = Path(log_fh.name)
            kwargs = dict(stdout=log_fh, stderr=subprocess.STDOUT, env=env)
            if os_name == "Windows":
                # On Windows Ollama is a system tray app; 'ollama serve' starts the server
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            proc = subprocess.Popen(["ollama", "serve"], **kwargs)
            log_fh.close()
            Shared._managed_procs.append(proc)
        except FileNotFoundError:
            Shared.err("'ollama' not found in PATH — install from https://ollama.com/download")
            return False

        for i in range(timeout):
            time.sleep(1)
            if Shared.ollama_available():
                Shared.ok(f"Ollama started (pid {proc.pid}) — log: {Shared._ollama_log_path}")
                return True
            if proc.poll() is not None:
                Shared.err(f"Ollama exited unexpectedly (code {proc.returncode})")
                Shared.err(f"Last output:\n{Shared.tail_ollama_log()}")
                return False

        Shared.err(f"Ollama did not respond within {timeout} seconds")
        return False

    @staticmethod
    def stop_all_ollama(timeout: int = 15) -> None:  # pragma: no cover — kills real processes
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
            if not Shared.ollama_available():
                return
            time.sleep(1)
        Shared.warn(f"Ollama still reachable {timeout}s after attempting to stop it")

    @staticmethod
    def ensure_ollama():  # pragma: no cover — thin live-server orchestration wrapper
        """Start Ollama if not already running. Returns True if available."""
        if Shared.ollama_available():
            Shared.ok("Ollama already running")
            return True

        Shared.log("Ollama not running — attempting to start ...")
        return Shared.start_ollama()

    @staticmethod
    def find_comfyui_python(comfyui_dir: Path) -> str:
        """
        Return the Python executable to use for ComfyUI.
        Prefers a venv inside comfyui_dir, then the venv running this script,
        then whatever 'python' resolves to.
        """
        # Official AMD portable build: python_embeded sits next to ComfyUI/, not inside it
        for candidate in [
            comfyui_dir.parent / "python_embeded" / "python.exe",
            comfyui_dir / "python_env" / "python.exe",
            comfyui_dir / "venv" / "bin" / "python",
            comfyui_dir / ".venv" / "bin" / "python",
            comfyui_dir / "venv" / "Scripts" / "python.exe",
        ]:
            if candidate.exists():
                return str(candidate)

        # The venv currently running this script (most likely on Mac/Linux)
        current_venv = os.environ.get("VIRTUAL_ENV")
        if current_venv:
            for rel in ["bin/python", "Scripts/python.exe"]:
                p = Path(current_venv) / rel
                if p.exists():
                    return str(p)

        return sys.executable

    @staticmethod
    def ensure_comfyui(comfyui_dir: Path) -> bool:  # pragma: no cover — spawns a real subprocess and polls a live server
        """
        Start ComfyUI if not already running.
        Returns True if ComfyUI is available (either was already running or we started it).
        """
        if Shared.comfyui_available():
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
        checkpoints_dir = comfyui_dir / "models" / "checkpoints"
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

        # ComfyUI's Dynamic VRAM (comfy-aimdo) has an unresolved upstream bug that
        # raises "hostbuf_file_reader_read failed" while streaming weights straight
        # from combined checkpoint files (e.g. SDXL's CheckpointLoaderSimple, which
        # packs unet+clip+vae into one .safetensors) — see Comfy-Org/ComfyUI#14239
        # and #14281. Flux/Flux2 load CLIP/VAE from separate files and are unaffected,
        # so disabling it globally trades away Dynamic VRAM's memory savings for
        # correctness across all image models rather than only the ones that need it.
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

            _skip = {"microsoft basic display adapter", "microsoft remote display adapter"}
            gpus = [n for n in _ps_names("Win32_VideoController") if n and n.lower() not in _skip]
            if gpus:
                gpu = gpus[0]
            if cpu and gpu:
                return f"{cpu}\n{gpu} {ram_gb} GB"
            elif cpu:
                return f"{cpu}\n{ram_gb} GB"
            elif gpu:
                return f"{gpu} {ram_gb} GB"

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
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
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
                    for line in out.splitlines():
                        if "Marketing Name:" in line:
                            gpu = line.split(":", 1)[1].strip()
                            break
                except Exception:
                    pass
            if not gpu:
                try:
                    out = subprocess.run(
                        ["lspci"], capture_output=True, text=True, timeout=10,
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
        os_name = platform.system()
        profile = {
            "hostname":   Shared.get_hostname(),
            "os":         f"{os_name} {platform.release()}",
            "arch":       platform.machine(),
            "python":     sys.version.split()[0],
            "ram_gb":     round(Shared.system_ram_gb(), 1),
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "backend":    Shared.detect_backend(),
        }
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
            if "Marketing Name" in out:
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
                if any("Intel" in n and "Arc" in n for n in names):
                    return "xpu"
            except Exception:
                pass
        # Intel Arc on Linux — no guaranteed xpu-smi, so reuse the "Intel"+"Arc"
        # name heuristic from the Windows check on lspci's GPU line. Requiring
        # "Arc", not just "Intel", avoids misreporting integrated graphics
        # (e.g. "Intel Iris Xe") with no discrete acceleration path.
        if platform.system() == "Linux":
            try:
                out = subprocess.check_output(["lspci"], text=True, stderr=subprocess.DEVNULL)
                for line in out.splitlines():
                    if (any(k in line for k in ("VGA", "3D controller", "Display"))
                            and "Intel" in line and "Arc" in line):
                        return "xpu"
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass
        # Metal
        if platform.system() == "Darwin":
            return "metal"
        return "cpu"

    # ── prompt builders ──

    SHORT_PROMPT = (
        "You are a hardware reviewer for HotHardware.com. Compare the performance, "
        "power efficiency, and value proposition of the latest GPU architectures "
        "for gaming and content creation workloads. Discuss thermal design, memory "
        "bandwidth, ray tracing capabilities, and how driver maturity affects "
        "real-world performance across AAA titles and professional applications."
    )

    _PADDING_UNIT = (
        " Additionally, analyze how CPU and platform choices — including chiplet "
        "designs, memory controllers, and PCIe bandwidth — interact with GPU "
        "performance, and what this means for system builders choosing between "
        "competing platforms at different price points."
    )

    @staticmethod
    def build_prompt_for_context(target_tokens: int) -> str:
        """
        Pad a prompt to approximate a target context length (1 token ≈ 4 chars).

        Prepends a unique per-call nonce so repeated calls at the same length
        don't share a prefix — without it Ollama's slot cache serves a cache
        hit on every rerun, so every run after the first measures cache-hit
        latency rather than real prompt-processing time.
        """
        nonce = uuid.uuid4().hex
        prefix = f"[run {nonce}] "
        chars_needed = target_tokens * 4
        parts = [prefix, Shared.SHORT_PROMPT]
        total = len(prefix) + len(Shared.SHORT_PROMPT)
        while total < chars_needed:
            parts.append(Shared._PADDING_UNIT)
            total += len(Shared._PADDING_UNIT)
        return "".join(parts)[:chars_needed]

    # ── Ollama HTTP client ──

    @staticmethod
    def ollama_available():  # pragma: no cover — real HTTP call; other tests mock this seam directly
        try:
            r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    @staticmethod
    def is_connection_crash(e: Exception) -> bool:
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

    @staticmethod
    def wait_for_ollama_recovery(timeout: int = 30) -> bool:  # pragma: no cover — real polling loop; other tests mock this seam directly
        """Poll until Ollama's main server answers again after its model
        runner subprocess crashed — the main server itself stays up and
        respawns the runner, it just needs a few seconds. Returns False if
        it doesn't come back within `timeout`."""
        wait_t0 = time.perf_counter()
        while time.perf_counter() - wait_t0 < timeout:
            if Shared.ollama_available():
                return True
            time.sleep(2)
        return False

    @staticmethod
    def ollama_reachable_or_abort() -> bool:
        """True if Ollama is reachable. Callers looping over models should stop
        when this is False rather than continuing into model_pulled() — which
        swallows connection errors and returns False indistinguishably from
        "genuinely not pulled", so a still-down server would misreport every
        remaining model as not pulled."""
        if Shared.ollama_available():
            return True
        Shared.err("Ollama is not reachable — stopping remaining models in this test")
        return False

    @staticmethod
    def stratified_sample(questions: list[dict], n: int, seed: int = 1337) -> list[dict]:
        """Deterministically picks `n` questions out of `questions`, touching
        every category present rather than risking a run of unlucky luck that
        skips a whole category. For fast local dev iteration against the full
        accuracy banks (mcq/math/code) — never used for a full/published run.

        Groups by `category`, shuffles each group with a seeded RNG (so the
        same (bank, n) always yields the same sample — reproducible and
        diffable across runs), then round-robins across categories in sorted
        order, one question at a time, until `n` are collected or the bank is
        exhausted. Round-robin naturally gives larger categories more picks
        without needing an explicit proportional-allocation step. Returns
        `questions` unchanged (not even reordered) if `n >= len(questions)`.
        """
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
    def file_hash(path: Path) -> str:
        """First 12 hex chars of the sha256 of `path`'s raw bytes — a short,
        stable fingerprint for a question bank so results can record exactly
        which version of the data they were scored against. Doesn't parse
        the JSON, so it also catches whitespace-only or key-order changes
        that wouldn't show up in the question count."""
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]

    @staticmethod
    def load_crash_cache(path: Path) -> dict:
        """Load a benchmark's cache of tag -> crash record, so a model that
        deterministically crashes Ollama's runner on a given test isn't
        retried forever across separate script invocations."""
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}

    @staticmethod
    def save_crash_cache(path: Path, cache: dict) -> None:
        try:
            path.write_text(json.dumps(cache, indent=2))
        except Exception as e:
            Shared.warn(f"Failed to save crash cache to {path}: {e}")

    @staticmethod
    def check_crash_cache(tag: str, label: str, crash_cache: dict, cache_path: Path,
                           expected_bank_hash: str | None = None) -> dict | None:
        """Returns a skip-result dict if `tag` is a known repeat-crasher on
        this test, else None.

        `expected_bank_hash`, when given (accuracy benchmarks backed by a
        versioned question bank), invalidates a cached crash recorded against
        a different bank version — a model that crashed on the old 185-
        question bank shouldn't be silently skipped forever on the new
        360-question one just because the tag matches. The stale entry is
        left in place (it'll be overwritten if this tag crashes again on the
        current bank) rather than deleted, so this stays a pure read."""
        detail = crash_cache.get(tag)
        if detail is None:
            return None
        if expected_bank_hash is not None and detail.get("bank_hash") != expected_bank_hash:
            Shared.warn(f"{tag}'s recorded crash is for a different question-bank version "
                        "— ignoring stale entry and retrying")
            return None
        crashed_at = detail.get("crashed_at", "an earlier run")
        Shared.warn(f"{tag} previously crashed Ollama's runner repeatedly on "
                    f"{crashed_at} — skipping (delete {cache_path} to retry)")
        return {
            "label": label,
            "skipped": True,
            "skip_reason": "known_crash",
            "skip_detail": f"Crashed Ollama's runner repeatedly on {crashed_at}",
        }

    @staticmethod
    def record_crash(tag: str, crash_cache: dict, cache_path: Path, what: str,
                      extra: dict | None = None) -> str:
        """Records a deterministic crash for `tag` in the cache. Returns the
        crash timestamp so callers can fold it into their own result detail.
        `extra` is merged into the stored record — accuracy benchmarks pass
        {"bank_hash": ...} so check_crash_cache can tell a stale crash record
        (from a since-changed question bank) apart from a current one."""
        crashed_at = datetime.now().isoformat(timespec="seconds")
        crash_cache[tag] = {"crashed_at": crashed_at, **(extra or {})}
        Shared.save_crash_cache(cache_path, crash_cache)
        Shared.err(f"Ollama's runner crashed repeatedly {what} — recorded to {cache_path}")
        return crashed_at

    @staticmethod
    def run_measured_calls(n_runs: int, call, tag: str, crash_cache: dict, cache_path: Path,
                            what: str, crash_extra: dict | None = None) -> tuple[list, str, str]:
        """
        Call `call(run_i)` up to `n_runs` times, collecting each return value —
        the shared shape behind every benchmark's "N measured runs" loop.

        A timeout stops immediately. A crash (Shared.is_connection_crash)
        retries the *same* run without counting it, after waiting for Ollama to
        respawn the runner — up to Shared.CRASH_RETRY_MAX attempts, since a
        deterministic crash would recur forever. Any other exception counts as
        a failed run and moves on. A crash that exhausts its retries is recorded
        to `cache_path` so future invocations skip this tag/test. `crash_extra`
        (e.g. {"bank_hash": ...}) is passed through to Shared.record_crash.

        Returns (samples, status, partial_text) where status is "ok",
        "timed_out", or "crashed"; `samples` may be non-empty even when
        status != "ok". `partial_text` is whatever text had streamed in before
        a timeout hit (empty otherwise) — a timed-out run that had already
        written a wrong-format (or even correct) answer is a different failure
        than one that produced nothing at all, and callers that score
        free-form answers need to tell them apart rather than treating every
        timeout as a blank.
        """
        samples = []
        run_i = 0
        crash_retries = 0
        while run_i < n_runs:
            try:
                samples.append(call(run_i))
                run_i += 1
            except Exception as e:
                is_timeout = isinstance(e, TimeoutError) or "timed out" in str(e).lower()
                if is_timeout:
                    # What happens next (abandon the rest of this tag vs. score this
                    # one attempt wrong and move on) is caller-specific, so this only
                    # reports the timeout itself — not what the caller does about it.
                    Shared.err(f"{tag}: timed out {what} (run {run_i+1})")
                    partial_text = getattr(e, "partial_text", "")
                    return samples, "timed_out", partial_text
                Shared.err(f"Run {run_i+1} failed: {e}")
                if not Shared.is_connection_crash(e):
                    run_i += 1
                    continue
                crash_retries += 1
                Shared.err(f"Ollama's model runner appears to have crashed {what} "
                           f"— last server output:\n{Shared.tail_ollama_log()}")
                if crash_retries > Shared.CRASH_RETRY_MAX:
                    Shared.err(f"Ollama's model runner crashed {crash_retries} times — giving up on {tag}")
                    Shared.record_crash(tag, crash_cache, cache_path, what, extra=crash_extra)
                    return samples, "crashed", ""
                Shared.warn(f"Waiting for recovery, retry {crash_retries}/{Shared.CRASH_RETRY_MAX} ...")
                if not Shared.wait_for_ollama_recovery():
                    Shared.warn("Ollama did not become reachable again within 30s — giving up on this model")
                    Shared.record_crash(tag, crash_cache, cache_path, what, extra=crash_extra)
                    return samples, "crashed", ""
                # don't advance run_i — retry the same run now that Ollama is back
        return samples, "ok", ""

    @staticmethod
    def model_pulled(tag):
        try:
            r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
            models = r.json().get("models", [])
            names = [m["name"] for m in models]
            return tag in names or any(tag in n for n in names)
        except Exception:
            return False

    @staticmethod
    def list_installed_models() -> list[dict]:
        """Every Ollama tag actually pulled locally, straight from /api/tags —
        including ones outside models.py's catalog. Returns [] (not an
        exception) if Ollama isn't reachable, so callers treat "can't reach"
        and "none installed" the same for listing."""
        try:
            r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
            return [{"tag": m["name"], "size": m.get("size")} for m in r.json().get("models", [])]
        except Exception:
            return []

    # Self-correction/hedging markers a model repeats when it's spinning in
    # place on the same reasoning (re-deriving, "catching" the same "mistake",
    # apologizing) without ever landing on an answer — a paraphrased loop, not
    # a verbatim one, so _has_repeated_verbatim_ngram alone won't catch it.
    # Lowercase substrings, checked against lowercased text.
    _LOOP_HEDGE_PHRASES = [
        "wait,", "wait -", "actually,", "let me reconsider", "let me recalculate",
        "let me re-check", "let me recheck", "let me recompute", "let me redo",
        "let me try again", "let's try again", "let's recalculate",
        "on second thought", "hold on,", "there seems to be a mistake",
        "there seems to have been", "i made an error", "i made a mistake",
        "that's not right", "this is incorrect", "let's start over",
        "apolog",  # apologize / apologies / apologizing
        "correcting myself", "let me reevaluate", "let me re-evaluate",
    ]

    @staticmethod
    def _has_repeated_verbatim_ngram(text: str, ngram_words: int = 12, min_repeats: int = 3) -> bool:
        """Flags `text` if any run of `ngram_words` consecutive words recurs
        at least `min_repeats` times — the signature of a model restating the
        same reasoning block (or the same code block, one indentation level
        deeper each time) verbatim until the wall-clock cutoff hits.
        Requiring a dozen-word run to repeat three times is deliberately
        conservative: a real answer might reuse a short phrase a couple
        times, but a 12+ word verbatim run recurring 3+ times essentially
        never happens outside an actual loop. Word-level rather than
        character-level so it isn't thrown off by minor whitespace/formatting
        differences between repeats."""
        words = text.split()
        if len(words) < ngram_words * min_repeats:
            return False
        seen: dict[str, int] = {}
        for i in range(len(words) - ngram_words + 1):
            gram = " ".join(words[i:i + ngram_words])
            count = seen.get(gram, 0) + 1
            if count >= min_repeats:
                return True
            seen[gram] = count
        return False

    @staticmethod
    def _has_repeated_hedging_phrase(text: str, min_repeats: int = 3) -> bool:
        """Flags `text` if any single phrase from _LOOP_HEDGE_PHRASES occurs
        at least `min_repeats` times — catches a loop that paraphrases each
        pass (re-deriving the same result, repeatedly "catching" and
        "correcting" the same mistake) rather than repeating verbatim."""
        lowered = text.lower()
        return any(lowered.count(phrase) >= min_repeats for phrase in Shared._LOOP_HEDGE_PHRASES)

    @staticmethod
    def looks_like_loop(text: str, ngram_words: int = 12, min_repeats: int = 3,
                         hedge_min_repeats: int = 3) -> bool:
        """Heuristic for a degenerate generation loop in a timed-out accuracy-
        test response: true if the model either repeated a substantial chunk
        of text verbatim, or repeatedly hedged/self-corrected without ever
        landing on an answer. See _has_repeated_verbatim_ngram and
        _has_repeated_hedging_phrase for the two signals."""
        return (Shared._has_repeated_verbatim_ngram(text, ngram_words, min_repeats)
                or Shared._has_repeated_hedging_phrase(text, hedge_min_repeats))

    @staticmethod
    def write_answers_sidecar(path: Path, data: dict) -> None:
        """Write an accuracy test's per-model raw-answer sidecar (wrong answers'
        full raw_response text) to `path`, overwriting each call so it updates
        incrementally as models finish — same checkpoint-as-you-go as the main
        results JSON, so a crash mid-run doesn't lose collected answers. Kept
        out of that JSON since raw model output is large and bloats it fast."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    @staticmethod
    def run_accuracy_benchmark(section_label: str, skip_label: str, question_noun: str,
                                data_path: Path, crash_cache_path: Path, models, questions,
                                warmup_runs: int, ask_fn, rescore_partial_fn, score_fn,
                                save_fn=None, answers_path: Path | None = None
                                ) -> dict:  # pragma: no cover — orchestrates real Ollama runs
        """Shared run() body for the MCQ/Math/Code accuracy benchmarks: per-model
        warmup, crash-cache check, one-question-at-a-time timeout/loop-detection
        handling, and result/sidecar saving — identical across all three, which
        only differ in how a question is asked (`ask_fn`), a partial (timed-out)
        response is rescored (`rescore_partial_fn`), and a completed answer set
        is tallied (`score_fn`).

        `ask_fn(tag, question) -> (parsed_answer, raw_text)` and
        `rescore_partial_fn(question, partial_text) -> parsed_answer` mirror each
        benchmark's own `_ask`/`parse_answer`-or-`evaluate_question` shape.
        `score_fn(questions, answers) -> dict` is each benchmark's `score()`.
        `section_label`/`skip_label`/`question_noun` preserve each benchmark's
        own wording ("MCQ"/"MCQ"/"MCQ questions", "Math"/"math"/"math questions",
        "Code"/"code"/"coding problems") rather than deriving one from another.
        """
        results = {}
        answers_out: dict = {}

        if not Shared.ollama_available():
            Shared.err(f"Ollama server not reachable — skipping {skip_label} benchmark")
            Shared.err("Start with: ollama serve")
            return results

        crash_cache = Shared.load_crash_cache(crash_cache_path)
        bank_hash = Shared.file_hash(data_path)

        for model in models:
            tag   = model["tag"]
            label = model["label"]
            short = model["short"]

            Shared.section(f"{section_label}: {label}")

            if not Shared.ollama_reachable_or_abort():
                break

            try:
                if not Shared.model_pulled(tag):
                    Shared.warn(f"{tag} not pulled — skipping")
                    Shared.warn(f"Pull with: ollama pull {tag}")
                    continue

                skip_entry = Shared.check_crash_cache(tag, label, crash_cache, crash_cache_path,
                                                       expected_bank_hash=bank_hash)
                if skip_entry is not None:
                    results[short] = skip_entry
                    continue

                if not Shared.warmup_model(tag, label, config.CONTEXT_LENGTHS[0], warmup_runs,
                                           crash_cache, crash_cache_path,
                                           crash_extra={"bank_hash": bank_hash}):
                    Shared.unload_model(tag)
                    continue

                Shared.log(f"Answering {len(questions)} {question_noun} "
                           f"({config.ACC_TIMEOUT}s timeout each) ...")
                answers: dict = {}
                raw_responses: dict[str, str] = {}
                timed_out_ids: list[str] = []
                likely_loop_ids: list[str] = []
                stopped_early = None

                for i, q in enumerate(questions):
                    samples, status, partial_text = Shared.run_measured_calls(
                        1, lambda run_i, q=q: ask_fn(tag, q), tag, crash_cache,
                        crash_cache_path, f"answering {q['id']}",
                        crash_extra={"bank_hash": bank_hash})
                    if samples:
                        given, raw = samples[0]
                    elif status == "timed_out" and partial_text:
                        # The model had already started answering when the wall-clock
                        # timeout hit. Score whatever it wrote instead of a blank —
                        # this is either a genuinely correct/incorrect answer cut off
                        # right at the end, or unparseable (wrong-format) text, not
                        # necessarily "the model produced nothing."
                        given, raw = rescore_partial_fn(q, partial_text), partial_text
                    else:
                        given, raw = None, ""
                    answers[q["id"]] = given
                    raw_responses[q["id"]] = raw

                    if status == "timed_out":
                        # A single stuck question is scored wrong and the run moves
                        # on — with ACC_TIMEOUT this short, a model that reliably
                        # gets stuck could otherwise rack up timeouts on a sizeable
                        # fraction of the bank, but that's still cheaper than the
                        # old behavior of abandoning the rest of the bank outright
                        # (and would incorrectly zero out everything after one bad
                        # question for a model that's merely slow, not stuck).
                        Shared.warn(f"{q['id']} timed out after {config.ACC_TIMEOUT}s — "
                                    "scoring as wrong and continuing")
                        timed_out_ids.append(q["id"])
                        if partial_text and Shared.looks_like_loop(partial_text):
                            Shared.warn(f"{q['id']}: response looks like a generation loop")
                            likely_loop_ids.append(q["id"])
                    if status == "crashed":
                        stopped_early = "crashed"
                        break

                    if (i + 1) % 10 == 0:
                        Shared.log(f"  {i+1}/{len(questions)} answered ...")

                scored = score_fn(questions, answers)
                answers_out[short] = {
                    "label": label,
                    "incorrect": [
                        {**entry, "raw_response": raw_responses.get(entry["id"], "")}
                        for entry in scored["incorrect"]
                    ],
                }
                results[short] = {"label": label, **scored}

                if timed_out_ids:
                    results[short]["timed_out_count"] = len(timed_out_ids)
                    results[short]["timed_out_ids"] = timed_out_ids
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
                Shared.unload_model(tag)
                Shared.wait_until_unloaded(tag)
            finally:
                if save_fn:
                    save_fn(results)
                if answers_path:
                    Shared.write_answers_sidecar(answers_path, answers_out)

        return results

    @staticmethod
    def ollama_model_max_ctx(model_tag, default=131072):
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
    def ollama_generate(model_tag: str, prompt: str, timeout: int = 600,
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
        options: dict = {"num_predict": 512, "temperature": 0.0}
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

        with Shared._ollama_urlopen(req, timeout) as resp:
            for raw_line in resp:
                if not raw_line.strip():
                    continue
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                # First token received — record TTFT from wall clock
                if ttft is None and chunk.get("response"):
                    ttft = time.perf_counter() - t_start

                if chunk.get("response"):
                    tokens += 1

                if chunk.get("done"):
                    eval_count    = chunk.get("eval_count", tokens)
                    eval_duration = chunk.get("eval_duration", 0)      # nanoseconds
                    prompt_dur    = chunk.get("prompt_eval_duration", 0) # nanoseconds

                    # Prefer server-side TTFT (prompt processing time) if available
                    if prompt_dur and prompt_dur > 0:
                        ttft = prompt_dur / 1e9

                    if eval_duration and eval_duration > 0:
                        tps = eval_count / (eval_duration / 1e9)
                    break

        total = time.perf_counter() - t_start
        if ttft is None:
            ttft = total
        return ttft, eval_count, tps

    @staticmethod
    def warmup_model(tag: str, label: str, num_ctx: int, warmup_runs: int,  # pragma: no cover — real threaded/watchdogged model load
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
                    Shared.ollama_generate(tag, "Hello.", timeout=config.RUN_TIMEOUT, num_ctx=num_ctx)
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
                if crash_cache is not None and cache_path is not None and Shared.is_connection_crash(exc_box[0]):
                    Shared.wait_for_ollama_recovery()
                    Shared.record_crash(tag, crash_cache, cache_path,
                                         f"warming up at num_ctx={num_ctx}", extra=crash_extra)
                return False
            else:
                Shared.log(f"Warmup run {warmup_i+1}/{warmup_runs} done")
        return True

    @staticmethod
    def ollama_chat(model_tag: str, messages: list, timeout: int = 600,
                     num_ctx: int | None = None, num_predict: int = 1024):
        """
        Generate via Ollama's /api/chat and return timing metrics plus the reply text.
        Returns: (ttft_sec, tokens_generated, tokens_per_sec, prompt_eval_count, response_text)

        prompt_eval_count is the *total* prompt token count for this call
        (ground truth for context depth), not just the new suffix — even when
        `messages` shares a prefix with a prior call and llama.cpp's slot cache
        skips re-processing it (that shows up as low prompt_eval_duration/ttft
        instead).
        """
        options: dict = {"num_predict": num_predict, "temperature": 0.0}
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

        with Shared._ollama_urlopen(req, timeout) as resp:
            for raw_line in resp:
                if not raw_line.strip():
                    continue
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

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

                # urlopen()'s timeout is per-read, not total duration — it
                # resets on every token. Enforce the real wall-clock deadline.
                if time.perf_counter() - t_start > timeout:
                    partial_text = "".join(response_parts) or "".join(thinking_parts)
                    raise OllamaTimeout(f"ollama_chat exceeded {timeout}s wall-clock timeout",
                                        partial_text=partial_text)

                if chunk.get("done"):
                    eval_count        = chunk.get("eval_count", tokens)
                    eval_duration      = chunk.get("eval_duration", 0)      # nanoseconds
                    prompt_dur         = chunk.get("prompt_eval_duration", 0)
                    prompt_eval_count  = chunk.get("prompt_eval_count", 0)

                    if prompt_dur and prompt_dur > 0:
                        ttft = prompt_dur / 1e9

                    if eval_duration and eval_duration > 0:
                        tps = eval_count / (eval_duration / 1e9)
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

    # ── Ollama model loading/unloading ──

    @staticmethod
    def unload_model(model_tag: str):
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

    @staticmethod
    def unload_all_models():
        """Unload every model currently loaded in Ollama."""
        try:
            r = requests.get(f"{config.OLLAMA_URL}/api/ps", timeout=10)
            loaded = r.json().get("models", [])
            if not loaded:
                Shared.ok("No models currently loaded in Ollama")
                return
            for m in loaded:
                Shared.unload_model(m["name"])
        except Exception as e:
            Shared.warn(f"Could not query loaded models: {e}")

    @staticmethod
    def wait_until_unloaded(model_tag: str, timeout: int = 30):
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
        """Shared by the LLM prefill and conversation tests: if the first context
        depth's decode speed is below SLOW_MODEL_MIN_TPS, mark the model slow and
        tell the caller to stop testing deeper contexts (unless force_all)."""
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
