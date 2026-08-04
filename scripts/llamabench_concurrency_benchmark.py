"""llama-batched-bench parallel-sequence throughput sweep — see docs/workloads.md#llama-bench-concurrency.
Talks to LlamaCppEngine directly rather than the InferenceEngine interface — llama-batched-bench has no cross-engine equivalent."""

import json
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path

import config
from engines.llamacpp import LlamaCppEngine
from shared import Shared


class LlamaBenchConcurrencyBenchmark:
    IDLE_POLL_INTERVAL = 1.0   # how often run_one checks the idle-output watchdog below

    @staticmethod
    def find_binary() -> str | None:
        """Mirrors LlamaCppEngine._binary_path but for llama-batched-bench instead of
        llama-server — see docs/engines.md's "Binary resolution"."""
        exe_name = "llama-batched-bench.exe" if platform.system() == "Windows" else "llama-batched-bench"
        if config.LLAMACPP_DIR.exists():
            match = next((p for p in config.LLAMACPP_DIR.rglob(exe_name) if p.is_file()), None)
            if match is not None:
                return str(match)
        found = shutil.which("llama-batched-bench")
        if found is not None:
            return found
        if platform.system() == "Darwin":
            for prefix in ("/opt/homebrew/bin", "/usr/local/bin"):
                candidate = Path(prefix) / exe_name
                if candidate.exists():
                    return str(candidate)
        return None

    @staticmethod
    def fit_npl(pp: int, tg_values: list[int], npl_values: list[int],
                model_max: int) -> tuple[int, list[int]]:
        """Clamps prompt depth and drops concurrency levels whose unified KV need
        (pl * (pp + max_tg)) exceeds the model's real context — see docs/workloads.md#llama-bench-concurrency."""
        max_tg = max(tg_values)
        effective_pp = max(1, min(pp, model_max - max_tg))
        fitted = [npl for npl in npl_values if npl * (effective_pp + max_tg) <= model_max]
        return effective_pp, fitted or [1]

    @staticmethod
    def build_command(binary: str, model_path: Path, ctx_size: int, pp: int, tg: list[int],
                      npl: list[int], batch_size: int, ubatch_size: int, ngl: int) -> list[str]:
        """Builds the llama-batched-bench argv — see docs/workloads.md#llama-bench-concurrency."""
        return [
            binary,
            "-m", str(model_path),
            "-c", str(ctx_size),
            "-npp", str(pp),
            "-ntg", ",".join(str(v) for v in tg),
            "-npl", ",".join(str(v) for v in npl),
            "-b", str(batch_size),
            "-ub", str(ubatch_size),
            "-ngl", str(ngl),
            "--output-format", "jsonl",
        ]

    @classmethod
    def run_one(cls, binary: str, model_path: Path, ctx_size: int, pp: int, tg: list[int],
                npl: list[int], batch_size: int, ubatch_size: int, ngl: int, timeout: int,
                on_progress=None, on_entry=None) -> list[dict]:
        """Parses each stdout JSONL row as it arrives (this build is silent on stderr, so
        that's the only progress signal). `timeout` is an idle timeout, not a wall-clock cap."""
        cmd = cls.build_command(binary, model_path, ctx_size, pp, tg, npl, batch_size, ubatch_size, ngl)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        Shared._managed_procs.append(proc)

        entries: list[dict] = []
        stderr_chunks: list[str] = []
        callback_errors: list[BaseException] = []
        activity_lock = threading.Lock()
        last_activity = [time.monotonic()]

        def _touch():
            with activity_lock:
                last_activity[0] = time.monotonic()

        def _drain_stdout():
            assert proc.stdout is not None
            for line in proc.stdout:
                _touch()
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                entries.append(entry)
                if on_entry:
                    try:
                        on_entry(entry)
                    except BaseException as exc:
                        callback_errors.append(exc)
                        proc.kill()
                        return
                if on_progress:
                    on_progress(cls.format_entry(entry))

        def _drain_stderr():
            assert proc.stderr is not None
            for line in proc.stderr:
                stderr_chunks.append(line)
                _touch()

        stdout_thread = threading.Thread(target=_drain_stdout, daemon=True)
        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        idle_timed_out = False
        while proc.poll() is None:
            with activity_lock:
                idle = time.monotonic() - last_activity[0]
            if idle > timeout:
                idle_timed_out = True
                proc.kill()
                break
            time.sleep(cls.IDLE_POLL_INTERVAL)

        proc.wait()
        stdout_thread.join()
        stderr_thread.join()
        if proc in Shared._managed_procs:
            Shared._managed_procs.remove(proc)

        if callback_errors:
            raise callback_errors[0]

        if idle_timed_out:
            error = subprocess.TimeoutExpired(cmd, timeout)
            error.partial_entries = list(entries)
            raise error

        if proc.returncode != 0:
            raise RuntimeError(
                f"llama-batched-bench exited {proc.returncode}: {''.join(stderr_chunks).strip()[-2000:]}")
        if not entries:
            raise RuntimeError("llama-batched-bench produced no parseable JSONL output")
        return entries

    @staticmethod
    def format_entry(entry: dict) -> str:
        return (f"pp{entry.get('pp', 0)}+tg{entry.get('tg', 0)} @ {entry.get('pl', 0)}-way: "
                f"{entry.get('speed_tg', 0.0):.1f} tok/s aggregate")

    def run(self, engine, models, cpu_only=False, save_fn=None):
        results = {}

        if not isinstance(engine, LlamaCppEngine):
            Shared.warn(f"llama-batched-bench only supports the llamacpp engine — skipping for {engine.name}")
            return results

        binary = self.find_binary()
        if binary is None:
            Shared.err("llama-batched-bench not found — run setup.sh/setup.bat to install it, or build it "
                       "yourself: https://github.com/ggml-org/llama.cpp")
            return results

        ngl = 0 if cpu_only else config.LLAMABENCH_FULL_OFFLOAD_NGL

        for model in models:
            tag, label, short = model["tag"], model["label"], model["short"]
            Shared.section(f"llama-batched-bench ({engine.name}): {label}")

            try:
                if not engine.model_pulled(tag):
                    Shared.warn(f"{tag} not pulled — skipping")
                    Shared.warn("Download it with: python setup_check.py")
                    continue

                paths = LlamaCppEngine._resolve_model_files(tag)
                if paths is None:
                    Shared.err(f"{tag}: model files went missing between listing and run — skipping")
                    results[short] = {"error": "model files not found"}
                    continue

                model_max = engine.max_context_length(tag)
                effective_pp, npl = self.fit_npl(
                    config.LLAMABENCH_CONC_PP, config.LLAMABENCH_CONC_TG,
                    config.LLAMABENCH_CONC_NPL, model_max,
                )
                ctx_size = max(npl) * (effective_pp + max(config.LLAMABENCH_CONC_TG))
                if effective_pp != config.LLAMABENCH_CONC_PP:
                    Shared.warn(f"{label}: prompt depth clamped to {effective_pp} tokens to fit the "
                                f"model's {model_max}-token context")

                entries = []
                results[short] = {
                    "entries": entries, "pp": effective_pp, "ctx_size": ctx_size,
                    "requested_cases": len(config.LLAMABENCH_CONC_TG) * len(npl),
                    "completed_cases": 0,
                }

                def _record_entry(entry):
                    entries.append(entry)
                    results[short]["completed_cases"] = len(entries)
                    if save_fn:
                        save_fn(results)

                try:
                    returned_entries = self.run_one(
                        binary, paths[0], ctx_size, effective_pp, config.LLAMABENCH_CONC_TG, npl,
                        config.LLAMABENCH_BATCH_SIZE, config.LLAMABENCH_UBATCH_SIZE,
                        ngl, config.LLAMABENCH_TIMEOUT,
                        on_progress=Shared.log, on_entry=_record_entry,
                    )
                    if not entries:
                        for entry in returned_entries:
                            _record_entry(entry)
                except subprocess.TimeoutExpired:
                    Shared.err(f"{label}: llama-batched-bench stopped with {len(entries)} completed entries")
                    results[short]["timed_out"] = True
                    results[short]["error"] = f"no output for {config.LLAMABENCH_TIMEOUT}s (idle timeout)"
                    continue
                except Exception as e:
                    Shared.err(f"{label}: {e}")
                    results[short]["error"] = str(e)
                    continue
                for entry in entries:
                    Shared.ok(self.format_entry(entry))
            finally:
                if save_fn:
                    save_fn(results)

        return results
