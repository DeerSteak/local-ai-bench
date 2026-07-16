"""
shared.py — cross-cutting helpers used by more than one test: logging, the
ComfyUI server lifecycle, machine profiling, crash-cache bookkeeping, and the
engine-agnostic benchmark orchestration (run_measured_calls,
run_accuracy_benchmark, loop detection). The Ollama-specific HTTP/process
client moved out to engines/ollama.py behind the InferenceEngine interface;
what stays here is driven through that interface, not tied to Ollama. Most
helpers are stateless-per-call, so methods are static and the little state
there is (managed-process bookkeeping) lives on the class.
"""

import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import psutil
import requests

import config
from models import IMAGE_MODELS

if TYPE_CHECKING:
    from engines.base import InferenceEngine


class OllamaTimeout(TimeoutError):
    """Raised when ollama_chat exceeds its wall-clock timeout. Carries whatever
    text had streamed in before the deadline hit, so callers can tell a bare
    timeout (no text at all) apart from a timeout that cut off a response the
    model had already started writing — which might have been a wrong-format
    answer regardless of the timeout, or might have been about to be correct."""

    def __init__(self, message: str, partial_text: str = ""):
        super().__init__(message)
        self.partial_text = partial_text


class OllamaLoopDetected(OllamaTimeout):
    """Raised when ollama_chat's check_loop polling flags a degenerate
    generation loop *before* the wall-clock timeout elapses. Deliberately a
    distinct type from a bare OllamaTimeout (though still a TimeoutError
    subclass, so generic timeout handling elsewhere keeps working): the model
    didn't run out of its time budget here, it was cut off early because the
    stream already looked pointless — callers that count "timed out" vs.
    "looped" need to tell those apart rather than lumping every early loop
    catch into the timeout bucket."""


class Shared:
    # Tracks processes we started so we can shut them down cleanly. Both the
    # inference engine's server (Ollama today) and ComfyUI register here, so
    # shutdown_managed() can clean up everything from one list on crash/exit.
    _managed_procs: list[subprocess.Popen] = []

    # The live inference engine for this run, set once by benchmark.py. Held so
    # shutdown_managed() can ask it (e.g. whether it's in forced CPU-only mode)
    # without the caller having to thread the instance into every cleanup path.
    _active_engine: "InferenceEngine | None" = None

    # Path to the log file capturing the ComfyUI server process's stdout+stderr.
    # Kept for the life of the process (not deleted on successful startup) so a
    # crash later in the run — e.g. an OOM while loading a large checkpoint —
    # still has a log to inspect instead of going silent.
    _comfyui_log_path: Path | None = None

    # Cap on how many times a benchmark retries a request after the engine's
    # model runner subprocess crashes (commonly OOM) before giving up on that
    # model — a deterministic crash would otherwise recur identically forever.
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
    def shutdown_managed(engine: "InferenceEngine | None" = None):  # pragma: no cover — manages real subprocesses
        """Terminate any servers we started. If the inference engine is running
        in forced CPU-only mode, stop it first so the script doesn't exit
        leaving a GPU-hidden server running silently in the background."""
        engine = engine or Shared._active_engine
        if engine is not None and getattr(engine, "_cpu_only_active", False):
            Shared.warn("Exiting while the engine is in forced CPU-only mode — killing it "
                        "rather than leaving GPU devices hidden in the background")
            engine.stop()
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
                            what: str, engine: "InferenceEngine",
                            crash_extra: dict | None = None) -> tuple[list, str, str]:
        """
        Call `call(run_i)` up to `n_runs` times, collecting each return value —
        the shared shape behind every benchmark's "N measured runs" loop.

        A timeout stops immediately. A crash (engine.is_connection_crash)
        retries the *same* run without counting it, after waiting for the engine
        to respawn the runner — up to Shared.CRASH_RETRY_MAX attempts, since a
        deterministic crash would recur forever. Any other exception counts as
        a failed run and moves on. A crash that exhausts its retries is recorded
        to `cache_path` so future invocations skip this tag/test. `crash_extra`
        (e.g. {"bank_hash": ...}) is passed through to Shared.record_crash.

        Returns (samples, status, partial_text) where status is "ok",
        "timed_out", "loop_detected", or "crashed"; `samples` may be non-empty
        even when status != "ok". `partial_text` is whatever text had streamed
        in before a timeout/loop-detection hit (empty otherwise) — a cut-off
        run that had already written a wrong-format (or even correct) answer
        is a different failure than one that produced nothing at all, and
        callers that score free-form answers need to tell them apart rather
        than treating every cutoff as a blank.

        "timed_out" means the call's full wall-clock budget was exhausted;
        "loop_detected" means the engine's chat check_loop polling recognized a
        degenerate generation loop and cut the call short *before* that
        budget ran out (see OllamaLoopDetected) — kept distinct so a caller
        counting timeouts doesn't also double-count early loop catches, and
        vice versa.
        """
        samples = []
        run_i = 0
        crash_retries = 0
        while run_i < n_runs:
            try:
                samples.append(call(run_i))
                run_i += 1
            except Exception as e:
                if isinstance(e, OllamaLoopDetected):
                    Shared.err(f"{tag}: detected a generation loop {what} (run {run_i+1})")
                    return samples, "loop_detected", e.partial_text
                is_timeout = isinstance(e, TimeoutError) or "timed out" in str(e).lower()
                if is_timeout:
                    # What happens next (abandon the rest of this tag vs. score this
                    # one attempt wrong and move on) is caller-specific, so this only
                    # reports the timeout itself — not what the caller does about it.
                    Shared.err(f"{tag}: timed out {what} (run {run_i+1})")
                    partial_text = getattr(e, "partial_text", "")
                    return samples, "timed_out", partial_text
                Shared.err(f"Run {run_i+1} failed: {e}")
                if not engine.is_connection_crash(e):
                    run_i += 1
                    continue
                crash_retries += 1
                Shared.err(f"The engine's model runner appears to have crashed {what} "
                           f"— last server output:\n{engine.tail_log()}")
                if crash_retries > Shared.CRASH_RETRY_MAX:
                    Shared.err(f"The engine's model runner crashed {crash_retries} times — giving up on {tag}")
                    Shared.record_crash(tag, crash_cache, cache_path, what, extra=crash_extra)
                    return samples, "crashed", ""
                Shared.warn(f"Waiting for recovery, retry {crash_retries}/{Shared.CRASH_RETRY_MAX} ...")
                if not engine.wait_for_recovery():
                    Shared.warn("The engine did not become reachable again within 30s — giving up on this model")
                    Shared.record_crash(tag, crash_cache, cache_path, what, extra=crash_extra)
                    return samples, "crashed", ""
                # don't advance run_i — retry the same run now that Ollama is back
        return samples, "ok", ""

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
                                warmup_runs: int, engine: "InferenceEngine",
                                ask_fn, rescore_partial_fn, score_fn,
                                save_fn=None, answers_path: Path | None = None
                                ) -> dict:
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

        if not engine.available():
            Shared.err(f"Inference engine not reachable — skipping {skip_label} benchmark")
            Shared.err("Start with: ollama serve")
            return results

        crash_cache = Shared.load_crash_cache(crash_cache_path)
        bank_hash = Shared.file_hash(data_path)

        for model in models:
            tag   = model["tag"]
            label = model["label"]
            short = model["short"]

            Shared.section(f"{section_label}: {label}")

            if not engine.reachable_or_abort():
                break

            try:
                if not engine.model_pulled(tag):
                    Shared.warn(f"{tag} not pulled — skipping")
                    Shared.warn(f"Pull with: ollama pull {tag}")
                    continue

                skip_entry = Shared.check_crash_cache(tag, label, crash_cache, crash_cache_path,
                                                       expected_bank_hash=bank_hash)
                if skip_entry is not None:
                    results[short] = skip_entry
                    continue

                if not engine.warmup(tag, label, config.CONTEXT_LENGTHS[0], warmup_runs,
                                     crash_cache, crash_cache_path,
                                     crash_extra={"bank_hash": bank_hash}):
                    engine.unload(tag)
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
                        crash_cache_path, f"answering {q['id']}", engine,
                        crash_extra={"bank_hash": bank_hash})
                    if samples:
                        given, raw = samples[0]
                    elif status in ("timed_out", "loop_detected") and partial_text:
                        # The model had already started answering when the wall-clock
                        # timeout (or the loop check) cut it off. Score whatever it
                        # wrote instead of a blank — this is either a genuinely
                        # correct/incorrect answer cut off right at the end, or
                        # unparseable (wrong-format) text, not necessarily "the model
                        # produced nothing."
                        given, raw = rescore_partial_fn(q, partial_text), partial_text
                    else:
                        given, raw = None, ""
                    answers[q["id"]] = given
                    raw_responses[q["id"]] = raw

                    # timed_out_ids and likely_loop_ids are independent buckets, not
                    # a superset/subset: timed_out_ids only counts questions that
                    # actually burned the full ACC_TIMEOUT (status == "timed_out"),
                    # while likely_loop_ids only counts questions whose text pattern-
                    # matches a degenerate loop (Shared.looks_like_loop), whether that
                    # was caught early (status == "loop_detected", well under
                    # ACC_TIMEOUT) or only became visible once the full timeout's
                    # partial text came back. A run can be slow-but-not-looping
                    # (timed_out only), looping-and-caught-early (loop only), or —
                    # rarely, if looks_like_loop's heuristic didn't fire until the
                    # last chunk — both.
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
                            likely_loop_ids.append(q["id"])
                    elif status == "loop_detected":
                        Shared.warn(f"{q['id']}: response looks like a generation loop — "
                                    "scoring as wrong and continuing")
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
                engine.unload(tag)
                engine.wait_until_unloaded(tag)
            finally:
                if save_fn:
                    save_fn(results)
                if answers_path:
                    Shared.write_answers_sidecar(answers_path, answers_out)

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
