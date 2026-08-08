"""llama-bench pp/tg throughput sweep across installed models — see docs/workloads.md#llama-bench.
Talks to LlamaCppEngine directly rather than the InferenceEngine interface — llama-bench has no cross-engine equivalent."""

import json
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import NotRequired, TypedDict

from scripts.runtime import config
from scripts.runtime.llamacpp_tools import find_llamacpp_tool
from scripts.runtime.engines.llamacpp import LlamaCppEngine
from scripts.runtime.shared import Shared
from scripts.app.progress_events import emit_model_finished, emit_progress
from scripts.runtime.pause_control import wait_if_paused


class LlamaBenchModelResult(TypedDict):
    """One model's llama-bench payload in the results JSON."""
    prefill_entries: list
    decode_entries: list
    requested_cases: int
    completed_cases: int
    requested_repetitions: int
    completed_repetitions: int
    error: NotRequired[str]
    timed_out: NotRequired[bool]
    timed_out_at: NotRequired[str]


class LlamaBenchBenchmark:
    IDLE_POLL_INTERVAL = 1.0   # how often run_one checks the idle-output watchdog below

    @staticmethod
    def find_binary() -> str | None:
        """Locate llama-bench with the same policy as llama-server."""
        return find_llamacpp_tool(
            "llama-bench", vendored_dir=config.LLAMACPP_DIR,
            platform_name=platform.system(), which_fn=shutil.which,
        )

    @staticmethod
    def _base_command(binary: str, model_path: Path, batch_size: int, ubatch_size: int,
                      reps: int, ngl: int) -> list[str]:
        return [
            binary,
            "-m", str(model_path),
            "-b", str(batch_size),
            "-ub", str(ubatch_size),
            "-ngl", str(ngl),
            *LlamaCppEngine.gpu_split_args(cpu_only=ngl == 0),
            "-r", str(reps),
            "-o", "jsonl",
            "--progress",
        ]

    @classmethod
    def build_prefill_command(cls, binary: str, model_path: Path, pp: list[int],
                              batch_size: int, ubatch_size: int, reps: int, ngl: int) -> list[str]:
        """Builds standalone prompt-processing tests so avg_ts is true prefill throughput."""
        return [
            *cls._base_command(binary, model_path, batch_size, ubatch_size, reps, ngl),
            "-p", ",".join(str(v) for v in pp),
            "-n", "0",
            "-d", "0",
        ]

    @classmethod
    def build_decode_command(cls, binary: str, model_path: Path, pp: list[int], tg: list[int],
                             batch_size: int, ubatch_size: int, reps: int, ngl: int) -> list[str]:
        """Builds generation tests at each prefilled depth so avg_ts is true decode throughput."""
        return [
            *cls._base_command(binary, model_path, batch_size, ubatch_size, reps, ngl),
            "-p", "0",
            "-n", ",".join(str(v) for v in tg),
            "-d", ",".join(str(v) for v in pp),
        ]

    @classmethod
    def run_one(cls, cmd: list[str], timeout: int, on_progress=None, on_result=None) -> list[dict]:
        """Streams progress and parses one llama-bench JSONL pass; timeout measures idle output,
        not total wall-clock duration."""
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        Shared._managed_procs.append(proc)

        rows: list[dict] = []
        stdout_errors: list[str] = []
        callback_errors: list[BaseException] = []
        stderr_chunks: list[str] = []
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
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    stdout_errors.append(stripped)
                    continue
                if not isinstance(row, dict):
                    stdout_errors.append(stripped)
                    continue
                rows.append(row)
                if on_result:
                    try:
                        on_result(row)
                    except BaseException as exc:
                        callback_errors.append(exc)
                        return

        def _drain_stderr():
            assert proc.stderr is not None
            for line in proc.stderr:
                stderr_chunks.append(line)
                _touch()
                stripped = line.strip()
                if stripped and on_progress:
                    on_progress(stripped)

        stdout_thread = threading.Thread(target=_drain_stdout, daemon=True)
        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        idle_timed_out = False
        while proc.poll() is None:
            if callback_errors:
                proc.kill()
                break
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

        if idle_timed_out:
            raise subprocess.TimeoutExpired(cmd, timeout)
        if callback_errors:
            raise callback_errors[0]

        if proc.returncode != 0:
            raise RuntimeError(f"llama-bench exited {proc.returncode}: {''.join(stderr_chunks).strip()[-2000:]}")
        if stdout_errors:
            raise RuntimeError(f"llama-bench produced unparseable JSONL: {stdout_errors[-1][-1000:]}")
        return rows

    @staticmethod
    def format_entry(entry: dict) -> str:
        n_prompt = entry.get("n_prompt", 0)
        n_gen = entry.get("n_gen", 0)
        n_depth = entry.get("n_depth", 0)
        label = (f"pp{n_prompt}" if n_gen == 0 else
                 f"tg{n_gen} @ pp{n_depth}" if n_prompt == 0 else
                 f"pp{n_prompt}+tg{n_gen}")
        avg_ts = entry.get("avg_ts", 0.0)
        stddev_ts = entry.get("stddev_ts", 0.0)
        return f"{label} @ ngl={entry.get('n_gpu_layers')}: {avg_ts:.1f} ± {stddev_ts:.1f} tok/s"

    @staticmethod
    def normalize_streamed_entry(entry: dict, requested_reps: int) -> dict:
        speeds = entry.get("samples_ts") or [entry["avg_ts"]]
        return {
            **entry,
            "ts_runs": speeds,
            "requested_reps": requested_reps,
            "completed_reps": len(speeds),
        }

    def run(self, engine, models, reps, cpu_only=False, save_fn=None, journal=None):
        results = journal.export() if journal else {}

        if not isinstance(engine, LlamaCppEngine):
            Shared.warn(f"llama-bench only supports the llamacpp engine — skipping for {engine.name}")
            if journal:
                journal.finish()
            return results

        binary = self.find_binary()
        if binary is None:
            Shared.err("llama-bench not found — run setup.sh/setup.bat to install it, or build it "
                       "yourself: https://github.com/ggml-org/llama.cpp")
            if journal:
                journal.finish()
            return results

        ngl = 0 if cpu_only else config.LLAMABENCH_FULL_OFFLOAD_NGL

        for model in models:
            tag, label, short = model["tag"], model["label"], model["short"]
            Shared.section(f"llama-bench ({engine.name}): {label}")

            emit_progress("model", "llamabench", "running", label)
            try:
                if not engine.model_pulled(tag):
                    Shared.warn(f"{tag} not pulled — skipping")
                    Shared.warn("Download it with: python setup_check.py")
                    if journal:
                        journal.record_model_state(model, "skipped", {
                            "skipped": True, "skip_reason": "not_installed",
                        })
                    continue

                paths = LlamaCppEngine._resolve_model_files(tag)
                if paths is None:
                    Shared.err(f"{tag}: model files went missing between listing and run — skipping")
                    results[short] = {"error": "model files not found"}
                    if journal:
                        journal.record_model_state(model, "failed", results[short])
                    continue

                prefill_entries, decode_entries = [], []
                requested_cases = len(config.LLAMABENCH_PP) * (1 + len(config.LLAMABENCH_TG))
                model_result: LlamaBenchModelResult = {
                    "prefill_entries": prefill_entries, "decode_entries": decode_entries,
                    "requested_cases": requested_cases, "completed_cases": 0,
                    "requested_repetitions": requested_cases * reps, "completed_repetitions": 0,
                }
                results[short] = model_result
                if journal:
                    journal.record_model_plan(model, requested_cases, reps)
                stopped = False

                sweeps = (journal.pending_sweeps(
                    model, config.LLAMABENCH_PP, config.LLAMABENCH_TG,
                ) if journal else [
                    ("prefill", config.LLAMABENCH_PP, []),
                    ("decode", config.LLAMABENCH_PP, config.LLAMABENCH_TG),
                ])
                for sweep, pending_pp, pending_tg in sweeps:
                    wait_if_paused()
                    command = (
                        self.build_prefill_command(
                            binary, paths[0], pending_pp,
                            config.LLAMABENCH_BATCH_SIZE, config.LLAMABENCH_UBATCH_SIZE,
                            reps, ngl,
                        ) if sweep == "prefill" else self.build_decode_command(
                            binary, paths[0], pending_pp, pending_tg,
                            config.LLAMABENCH_BATCH_SIZE, config.LLAMABENCH_UBATCH_SIZE,
                            reps, ngl,
                        )
                    )

                    def record_row(row):
                        entry = self.normalize_streamed_entry(row, reps)
                        target = prefill_entries if row.get("n_gen", 0) == 0 else decode_entries
                        target.append(entry)
                        model_result["completed_repetitions"] += entry["completed_reps"]
                        if entry["completed_reps"] == reps:
                            model_result["completed_cases"] += 1
                        Shared.ok(self.format_entry(entry))
                        if journal:
                            journal.record_entry(model, entry)
                        elif save_fn:
                            save_fn(results)

                    try:
                        self.run_one(
                            command, config.LLAMABENCH_TIMEOUT,
                            on_progress=Shared.log, on_result=record_row,
                        )
                    except subprocess.TimeoutExpired:
                        model_result.update(
                            timed_out=True,
                            timed_out_at=sweep,
                            error=f"no output for {config.LLAMABENCH_TIMEOUT}s (idle timeout)",
                        )
                        if journal:
                            journal.record_model_state(model, "timed_out", {
                                "timed_out": True, "timed_out_at": sweep,
                                "error": model_result.get("error"),
                            })
                        stopped = True
                        break
                    except Exception as e:
                        model_result["error"] = str(e)
                        if journal:
                            journal.record_model_state(
                                model, "failed", {"error": model_result.get("error")},
                            )
                        stopped = True
                        break
                if stopped:
                    Shared.err(f"{label}: native benchmark stopped with partial results")
            finally:
                if save_fn:
                    save_fn(journal.export() if journal else results)
                emit_model_finished("llamabench", label, results.get(short))

        if journal:
            journal.finish()
            return journal.export()
        return results
