"""llama-bench pp/tg throughput sweep across installed models — see docs/workloads.md#llama-bench.
Talks to LlamaCppEngine directly rather than the InferenceEngine interface — llama-bench has no cross-engine equivalent."""

import json
import platform
import shutil
import subprocess
import threading
from pathlib import Path

import config
from engines.llamacpp import LlamaCppEngine
from shared import Shared


class LlamaBenchBenchmark:
    @staticmethod
    def find_binary() -> str | None:
        """Mirrors LlamaCppEngine._binary_path but for llama-bench instead of
        llama-server — see docs/engines.md's "Binary resolution"."""
        exe_name = "llama-bench.exe" if platform.system() == "Windows" else "llama-bench"
        if config.LLAMACPP_DIR.exists():
            match = next(iter(config.LLAMACPP_DIR.rglob(exe_name)), None)
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
    def build_command(binary: str, model_path: Path, pp: list[int], tg: list[int],
                      batch_size: int, ubatch_size: int, reps: int, ngl: int) -> list[str]:
        """Builds the llama-bench argv — see docs/workloads.md#llama-bench for what each flag means."""
        return [
            binary,
            "-m", str(model_path),
            "-p", ",".join(str(v) for v in pp),
            "-n", ",".join(str(v) for v in tg),
            "-b", str(batch_size),
            "-ub", str(ubatch_size),
            "-ngl", str(ngl),
            "-r", str(reps),
            "-o", "json",
            "--progress",
        ]

    @classmethod
    def run_one(cls, binary: str, model_path: Path, pp: list[int], tg: list[int],
               batch_size: int, ubatch_size: int, reps: int, ngl: int, timeout: int,
               on_progress=None) -> list[dict]:
        """Runs one llama-bench pass, streaming stderr progress lines to on_progress as they arrive
        rather than buffering until exit, and returns the parsed stdout JSON entries."""
        cmd = cls.build_command(binary, model_path, pp, tg, batch_size, ubatch_size, reps, ngl)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        def _drain_stdout():
            for line in proc.stdout:
                stdout_chunks.append(line)

        def _drain_stderr():
            for line in proc.stderr:
                stderr_chunks.append(line)
                stripped = line.strip()
                if stripped and on_progress:
                    on_progress(stripped)

        stdout_thread = threading.Thread(target=_drain_stdout, daemon=True)
        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
        finally:
            stdout_thread.join()
            stderr_thread.join()

        if returncode != 0:
            raise RuntimeError(f"llama-bench exited {returncode}: {''.join(stderr_chunks).strip()[-2000:]}")
        try:
            return json.loads("".join(stdout_chunks))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"llama-bench produced unparseable JSON: {e}") from None

    @staticmethod
    def format_entry(entry: dict) -> str:
        n_prompt = entry.get("n_prompt", 0)
        n_gen = entry.get("n_gen", 0)
        label = f"pp{n_prompt}" if n_gen == 0 else f"tg{n_gen}" if n_prompt == 0 else f"pp{n_prompt}+tg{n_gen}"
        avg_ts = entry.get("avg_ts", 0.0)
        stddev_ts = entry.get("stddev_ts", 0.0)
        return f"{label} @ ngl={entry.get('n_gpu_layers')}: {avg_ts:.1f} ± {stddev_ts:.1f} tok/s"

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

                try:
                    entries = self.run_one(
                        binary, paths[0], config.LLAMABENCH_PP, config.LLAMABENCH_TG,
                        config.LLAMABENCH_BATCH_SIZE, config.LLAMABENCH_UBATCH_SIZE,
                        reps, ngl, config.LLAMABENCH_TIMEOUT,
                        on_progress=Shared.log,
                    )
                except subprocess.TimeoutExpired:
                    Shared.err(f"{label}: llama-bench exceeded {config.LLAMABENCH_TIMEOUT}s — skipping")
                    results[short] = {"error": f"timed out after {config.LLAMABENCH_TIMEOUT}s"}
                    continue
                except Exception as e:
                    Shared.err(f"{label}: {e}")
                    results[short] = {"error": str(e)}
                    continue

                results[short] = {"entries": entries}
                for entry in entries:
                    Shared.ok(self.format_entry(entry))
            finally:
                if save_fn:
                    save_fn(results)

        return results
