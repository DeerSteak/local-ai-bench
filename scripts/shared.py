"""
shared.py — cross-cutting helpers used by more than one test: logging, server
lifecycle management, machine profiling, and the low-level Ollama/ComfyUI HTTP
clients. Everything here is stateless-per-call except the managed-process
bookkeeping (used to shut down servers this script itself started), so
methods are static and state lives on the class rather than an instance.
"""

import http.client
import json
import os
import platform
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


class Shared:
    # Tracks processes we started so we can shut them down cleanly
    _managed_procs: list[subprocess.Popen] = []

    # True while Ollama is running with GPU devices hidden (--emb-cpu-only). If
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

        # Capture stdout+stderr to a log file kept for the whole process lifetime
        # (not just startup) so a crash later in the run — e.g. an OOM while
        # loading a large checkpoint — still has real output to inspect instead
        # of going silent.
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

        Prepends a unique per-call nonce so repeated calls at the same context
        length don't share an identical prefix — without it, Ollama's slot cache
        recognizes the exact same prompt on every subsequent run and serves a
        cache hit instead of genuinely reprocessing it, making every run after
        the first measure cache-hit latency rather than real prompt-processing
        time (the one real cold-prefill measurement then gets discarded as the
        "slow outlier").
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

        Two distinct timings surface as different exception shapes:
        - Connection-refused (the runner is already dead before the request
          starts) surfaces as a different exception type depending on which
          HTTP client made the call (requests vs urllib), so check both.
        - Mid-stream (the runner dies while a response is being streamed,
          e.g. partway through generating tokens at a large context depth —
          the realistic OOM timing these long-running tests actually hit)
          surfaces as a truncated/reset read instead: http.client.IncompleteRead,
          or a builtin ConnectionError (BrokenPipeError/ConnectionResetError/
          ConnectionAbortedError) from the socket itself.
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
        processing the remaining ones when this is False, rather than
        continuing into model_pulled() — which swallows connection errors and
        returns False indistinguishably from "genuinely not pulled", so a
        server that's still down after a failed wait_for_ollama_recovery()
        would otherwise get every remaining model misreported as not pulled
        instead of surfacing the real problem."""
        if Shared.ollama_available():
            return True
        Shared.err("Ollama is not reachable — stopping remaining models in this test")
        return False

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
    def check_crash_cache(tag: str, label: str, crash_cache: dict, cache_path: Path) -> dict | None:
        """Returns a skip-result dict if `tag` is a known repeat-crasher on
        this test, else None."""
        detail = crash_cache.get(tag)
        if detail is None:
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
    def record_crash(tag: str, crash_cache: dict, cache_path: Path, what: str) -> str:
        """Records a deterministic crash for `tag` in the cache. Returns the
        crash timestamp so callers can fold it into their own result detail."""
        crashed_at = datetime.now().isoformat(timespec="seconds")
        crash_cache[tag] = {"crashed_at": crashed_at}
        Shared.save_crash_cache(cache_path, crash_cache)
        Shared.err(f"Ollama's runner crashed repeatedly {what} — recorded to {cache_path}")
        return crashed_at

    @staticmethod
    def run_measured_calls(n_runs: int, call, tag: str, crash_cache: dict, cache_path: Path,
                            what: str) -> tuple[list, str]:
        """
        Call `call(run_i)` up to `n_runs` times, collecting each return value
        into a list — the shared shape behind every benchmark's "N measured
        runs" loop (embedding throughput, LLM prefill TTFT/TPS, ...).

        A timeout stops immediately (the caller decides what a partial result
        means). A crash (Shared.is_connection_crash) retries the *same* run,
        without counting it, after waiting for Ollama's main server to notice
        and respawn the runner — up to Shared.CRASH_RETRY_MAX attempts, since
        a deterministic crash on this tag/workload would just recur
        identically forever. Any other exception counts as a failed run and
        moves on to the next one. A crash that exhausts its retries is
        recorded to `cache_path` so future invocations skip this tag/test
        combo instead of rediscovering the same crash.

        Returns (samples, status) where status is "ok", "timed_out", or
        "crashed" — `samples` may be non-empty even when status != "ok" (e.g.
        earlier runs succeeded before a later one crashed or timed out).
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
                    Shared.err(f"Run {run_i+1} timed out — stopping remaining runs for {tag}")
                    return samples, "timed_out"
                Shared.err(f"Run {run_i+1} failed: {e}")
                if not Shared.is_connection_crash(e):
                    run_i += 1
                    continue
                crash_retries += 1
                Shared.err(f"Ollama's model runner appears to have crashed {what} "
                           f"— last server output:\n{Shared.tail_ollama_log()}")
                if crash_retries > Shared.CRASH_RETRY_MAX:
                    Shared.err(f"Ollama's model runner crashed {crash_retries} times — giving up on {tag}")
                    Shared.record_crash(tag, crash_cache, cache_path, what)
                    return samples, "crashed"
                Shared.warn(f"Waiting for recovery, retry {crash_retries}/{Shared.CRASH_RETRY_MAX} ...")
                if not Shared.wait_for_ollama_recovery():
                    Shared.warn("Ollama did not become reachable again within 30s — giving up on this model")
                    Shared.record_crash(tag, crash_cache, cache_path, what)
                    return samples, "crashed"
                # don't advance run_i — retry the same run now that Ollama is back
        return samples, "ok"

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
    def ollama_model_max_ctx(model_tag, default=131072):
        """Look up a pulled model's real max context length via /api/show.

        Reads manifest metadata only — doesn't load the model into memory, so
        this is a cheap call. The context-length key is prefixed by
        architecture (llama.context_length, phi3.context_length,
        qwen35.context_length, gemma4.context_length, gptoss.context_length,
        ...), so scan for any key ending in that suffix rather than guessing
        the prefix from the tag. Falls back to `default` if the model isn't
        found, the field is missing, or the request fails for any reason —
        callers then behave as if the model supports exactly `default`.
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

        Uses urllib instead of requests for streaming to avoid TCP buffering
        that causes iter_lines() to batch all chunks and inflate TTFT.

        Ollama's final chunk includes server-side timing fields:
          prompt_eval_duration  — time to process the prompt (nanoseconds)
          eval_count            — tokens generated
          eval_duration         — time spent generating (nanoseconds)
        These are authoritative and used in preference to wall-clock where available.

        num_ctx must match the context length being tested. Without it, Ollama uses
        the model's default, and sending a prompt longer than that default triggers a
        full model reload — inflating TTFT by minutes rather than seconds.
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
                      crash_cache: dict | None = None, cache_path: Path | None = None) -> bool:
        """
        Load `tag` into memory with `warmup_runs` blocking generate calls, each
        watchdogged by a daemon thread so a hung load (model too large for
        available memory) times out after config.RUN_TIMEOUT instead of hanging
        the whole benchmark run. Returns False on the first hung or failed run,
        so the caller can skip this model.

        A hang, or an exception that looks like Ollama's runner crashing
        outright, is exactly the kind of deterministic failure the crash cache
        exists to remember — warmup is often where an oversized model's OOM
        first shows up, before a single measured run ever gets a chance to
        record it. Pass `crash_cache`/`cache_path` (from Shared.load_crash_cache)
        to have that case recorded here too.
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
                                         f"warming up (hung past {config.RUN_TIMEOUT}s at num_ctx={num_ctx})")
                return False
            elif exc_box[0] is not None:
                Shared.warn(f"Warmup run {warmup_i+1} failed: {exc_box[0]}")
                if crash_cache is not None and cache_path is not None and Shared.is_connection_crash(exc_box[0]):
                    Shared.wait_for_ollama_recovery()
                    Shared.record_crash(tag, crash_cache, cache_path,
                                         f"warming up at num_ctx={num_ctx}")
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

        prompt_eval_count is the *total* number of tokens in this call's prompt
        (ground truth for how deep the context is), not just the new suffix — even
        when `messages` shares a prefix with a prior call and llama.cpp's slot
        cache skips re-processing it. The cache reuse shows up as a low
        prompt_eval_duration/ttft instead, not a smaller prompt_eval_count.
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
        # their entire turn through message.thinking with message.content left
        # empty. Fall back to the thinking text so the conversation history we
        # feed back on the next turn isn't an empty assistant message — otherwise
        # the growth loop below wildly overcounts how much context actually
        # persists (eval_count reflects thinking tokens that never make it into
        # the next turn's prompt at all).
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
