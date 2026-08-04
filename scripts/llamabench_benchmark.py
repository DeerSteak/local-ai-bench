"""llama-bench pp/tg throughput sweep across installed models — see docs/workloads.md#llama-bench.
Talks to LlamaCppEngine directly rather than the InferenceEngine interface — llama-bench has no cross-engine equivalent."""

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


class LlamaBenchBenchmark:
    IDLE_POLL_INTERVAL = 1.0   # how often run_one checks the idle-output watchdog below

    @staticmethod
    def find_binary() -> str | None:
        """Mirrors LlamaCppEngine._binary_path but for llama-bench instead of
        llama-server — see docs/engines.md's "Binary resolution"."""
        exe_name = "llama-bench.exe" if platform.system() == "Windows" else "llama-bench"
        if config.LLAMACPP_DIR.exists():
            match = next((p for p in config.LLAMACPP_DIR.rglob(exe_name) if p.is_file()), None)
            if match is not None:
                return str(match)
        found = shutil.which("llama-bench")
        if found is not None:
            return found
        if platform.system() == "Darwin":
            for prefix in ("/opt/homebrew/bin", "/usr/local/bin"):
                candidate = Path(prefix) / exe_name
                if candidate.exists():
                    return str(candidate)
        return None

    @staticmethod
    def _base_command(binary: str, model_path: Path, batch_size: int, ubatch_size: int,
                      reps: int, ngl: int) -> list[str]:
        return [
            binary,
            "-m", str(model_path),
            "-b", str(batch_size),
            "-ub", str(ubatch_size),
            "-ngl", str(ngl),
            "-r", str(reps),
            "-o", "json",
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
    def run_one(cls, cmd: list[str], timeout: int, on_progress=None) -> list[dict]:
        """Streams progress and parses one llama-bench JSON pass; timeout measures idle output,
        not total wall-clock duration."""
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        Shared._managed_procs.append(proc)

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        activity_lock = threading.Lock()
        last_activity = [time.monotonic()]

        def _touch():
            with activity_lock:
                last_activity[0] = time.monotonic()

        def _drain_stdout():
            assert proc.stdout is not None
            for line in proc.stdout:
                stdout_chunks.append(line)
                _touch()

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

        if proc.returncode != 0:
            raise RuntimeError(f"llama-bench exited {proc.returncode}: {''.join(stderr_chunks).strip()[-2000:]}")
        try:
            return json.loads("".join(stdout_chunks))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"llama-bench produced unparseable JSON: {e}") from None

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
    def aggregate_repetitions(entries: list[dict], requested_reps: int) -> dict:
        speeds = [entry["avg_ts"] for entry in entries]
        return {
            **entries[-1],
            "avg_ts": Shared.mean(speeds),
            "stddev_ts": Shared.stdev(speeds),
            "ts_runs": speeds,
            "requested_reps": requested_reps,
            "completed_reps": len(speeds),
        }

    def run(self, engine, models, reps, cpu_only=False, save_fn=None):
        results = {}

        if not isinstance(engine, LlamaCppEngine):
            Shared.warn(f"llama-bench only supports the llamacpp engine — skipping for {engine.name}")
            return results

        binary = self.find_binary()
        if binary is None:
            Shared.err("llama-bench not found — run setup.sh/setup.bat to install it, or build it "
                       "yourself: https://github.com/ggml-org/llama.cpp")
            return results

        ngl = 0 if cpu_only else config.LLAMABENCH_FULL_OFFLOAD_NGL

        for model in models:
            tag, label, short = model["tag"], model["label"], model["short"]
            Shared.section(f"llama-bench ({engine.name}): {label}")

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

                prefill_entries, decode_entries = [], []
                model_result = {"prefill_entries": prefill_entries, "decode_entries": decode_entries}
                results[short] = model_result
                timed_out = False

                cases = [
                    ("prefill", pp, None) for pp in config.LLAMABENCH_PP
                ] + [
                    ("decode", pp, tg) for pp in config.LLAMABENCH_PP for tg in config.LLAMABENCH_TG
                ]
                model_result["requested_cases"] = len(cases)
                model_result["completed_cases"] = 0
                model_result["requested_repetitions"] = len(cases) * reps
                model_result["completed_repetitions"] = 0
                for case_kind, pp, tg in cases:
                    repetitions = []
                    for _rep in range(reps):
                        command = (
                            self.build_prefill_command(
                                binary, paths[0], [pp], config.LLAMABENCH_BATCH_SIZE,
                                config.LLAMABENCH_UBATCH_SIZE, 1, ngl,
                            ) if case_kind == "prefill" else
                            self.build_decode_command(
                                binary, paths[0], [pp], [tg], config.LLAMABENCH_BATCH_SIZE,
                                config.LLAMABENCH_UBATCH_SIZE, 1, ngl,
                            )
                        )
                        try:
                            rows = self.run_one(command, config.LLAMABENCH_TIMEOUT, on_progress=Shared.log)
                        except subprocess.TimeoutExpired:
                            timed_out = True
                            model_result.update(
                                timed_out=True,
                                timed_out_at=(f"{case_kind}:pp{pp}" + (f":tg{tg}" if tg else "")
                                              + f":rep{len(repetitions) + 1}"),
                                error=f"no output for {config.LLAMABENCH_TIMEOUT}s (idle timeout)",
                            )
                            break
                        except Exception as e:
                            model_result["error"] = str(e)
                            timed_out = True
                            break
                        if not rows:
                            model_result["error"] = "llama-bench produced no result rows"
                            timed_out = True
                            break
                        repetitions.append(rows[0])
                        model_result["completed_repetitions"] += 1
                        aggregate = self.aggregate_repetitions(repetitions, reps)
                        target = prefill_entries if case_kind == "prefill" else decode_entries
                        if target and target[-1].get("n_prompt") == aggregate.get("n_prompt") \
                                and target[-1].get("n_depth") == aggregate.get("n_depth") \
                                and target[-1].get("n_gen") == aggregate.get("n_gen"):
                            target[-1] = aggregate
                        else:
                            target.append(aggregate)
                        if save_fn:
                            save_fn(results)
                    if timed_out:
                        break
                    model_result["completed_cases"] += 1
                    Shared.ok(self.format_entry(aggregate))
                if timed_out:
                    Shared.err(f"{label}: native benchmark stopped with partial results")
            finally:
                if save_fn:
                    save_fn(results)

        return results
