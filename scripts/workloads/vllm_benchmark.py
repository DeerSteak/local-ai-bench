"""`vllm bench` latency/throughput sweep across installed models — see docs/workloads.md#vllm-bench.
Talks to VllmEngine directly rather than the InferenceEngine interface, mirroring llamabench_benchmark."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import NotRequired, TypedDict

from scripts.runtime import config
from scripts.runtime.engines.vllm import VllmEngine
from scripts.runtime.shared import Shared
from scripts.app.progress_events import emit_model_finished, emit_progress
from scripts.runtime.pause_control import wait_if_paused


class VllmBenchModelResult(TypedDict):
    """One model's `vllm bench` payload in the results JSON."""
    latency_entries: list
    throughput_entries: list
    requested_cases: int
    completed_cases: int
    error: NotRequired[str]
    timed_out: NotRequired[bool]
    timed_out_at: NotRequired[str]


class VllmBenchBenchmark:
    @staticmethod
    def bench_command(executable: str, subcommand: str, repo: str, output_json: Path,
                      *, input_len: int, output_len: int, extra: list[str]) -> list[str]:
        """`vllm bench <subcommand>` for one size. The launcher is never used: it wraps
        `vllm serve`, and these subcommands load the weights themselves."""
        return [
            executable, "bench", subcommand,
            "--model", repo,
            "--input-len", str(input_len),
            "--output-len", str(output_len),
            *extra,
            "--output-json", str(output_json),
        ]

    @classmethod
    def build_latency_command(cls, executable: str, repo: str, output_json: Path,
                              input_len: int, output_len: int) -> list[str]:
        return cls.bench_command(
            executable, "latency", repo, output_json,
            input_len=input_len, output_len=output_len,
            extra=[
                "--batch-size", str(config.VLLMBENCH_BATCH_SIZE),
                "--num-iters", str(config.VLLMBENCH_ITERS),
                "--num-iters-warmup", str(config.VLLMBENCH_WARMUP_ITERS),
            ],
        )

    @classmethod
    def build_throughput_command(cls, executable: str, repo: str, output_json: Path,
                                 input_len: int, output_len: int) -> list[str]:
        return cls.bench_command(
            executable, "throughput", repo, output_json,
            input_len=input_len, output_len=output_len,
            extra=["--num-prompts", str(config.VLLMBENCH_NUM_PROMPTS)],
        )

    @staticmethod
    def parse_latency_result(payload: dict | None, input_len: int, output_len: int) -> dict | None:
        """`vllm bench latency` reports seconds; percentile keys are strings of the percent."""
        if not isinstance(payload, dict):
            return None
        average = payload.get("avg_latency")
        if not isinstance(average, (int, float)) or isinstance(average, bool) or average <= 0:
            return None
        samples = [v for v in (payload.get("latencies") or [])
                   if isinstance(v, (int, float)) and not isinstance(v, bool)]
        percentiles = {
            str(name): value
            for name, value in (payload.get("percentiles") or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        return {
            "input_len": input_len,
            "output_len": output_len,
            "avg_latency_sec": round(average, 4),
            "latency_runs_sec": [round(v, 4) for v in samples],
            "completed_iters": len(samples),
            "percentiles_sec": {k: round(v, 4) for k, v in percentiles.items()},
            # Output tokens per second for one batch, the closest analogue to a decode rate.
            "output_tps": round(output_len * config.VLLMBENCH_BATCH_SIZE / average, 2),
        }

    @staticmethod
    def parse_throughput_result(payload: dict | None, input_len: int, output_len: int) -> dict | None:
        """`total_num_tokens` counts prompt plus output, so output-only rate is derived."""
        if not isinstance(payload, dict):
            return None
        elapsed = payload.get("elapsed_time")
        requests = payload.get("num_requests")
        if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or elapsed <= 0:
            return None
        if not isinstance(requests, int) or isinstance(requests, bool) or requests <= 0:
            return None
        total_tokens = payload.get("total_num_tokens")
        entry = {
            "input_len": input_len,
            "output_len": output_len,
            "elapsed_sec": round(elapsed, 4),
            "num_requests": requests,
            "requests_per_sec": round(payload.get("requests_per_second") or requests / elapsed, 3),
            "output_tps": round(requests * output_len / elapsed, 2),
        }
        if isinstance(total_tokens, int) and not isinstance(total_tokens, bool):
            entry["total_num_tokens"] = total_tokens
            entry["total_tps"] = round(payload.get("tokens_per_second") or total_tokens / elapsed, 2)
        return entry

    @staticmethod
    def format_entry(kind: str, entry: dict) -> str:
        shape = f"in{entry['input_len']}/out{entry['output_len']}"
        if kind == "latency":
            return (f"{shape}: {entry['avg_latency_sec']:.3f}s per batch "
                    f"({entry['output_tps']:.1f} out tok/s)")
        return (f"{shape}: {entry['requests_per_sec']:.2f} req/s "
                f"({entry['output_tps']:.1f} out tok/s)")

    @staticmethod
    def sweep_sizes(inputs: list[int], outputs: list[int], context_limit: int | None) -> list[tuple]:
        """(input, output) pairs the model can actually hold — vLLM rejects a request whose
        prompt plus generation exceeds --max-model-len outright."""
        pairs = []
        for input_len in inputs:
            for output_len in outputs:
                if context_limit is None or input_len + output_len <= context_limit:
                    pairs.append((input_len, output_len))
        return pairs

    def run_one(self, command: list[str], output_json: Path, timeout: int,
                env: dict) -> dict | None:  # pragma: no cover — spawns a real subprocess
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=env)
        Shared._managed_procs.append(proc)
        try:
            output, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise
        finally:
            if proc in Shared._managed_procs:
                Shared._managed_procs.remove(proc)
        if proc.returncode != 0:
            raise RuntimeError(f"vllm bench exited {proc.returncode}: {(output or '').strip()[-2000:]}")
        try:
            return json.loads(output_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"vllm bench wrote no readable JSON: {exc}") from None

    def run(self, engine, models, save_fn=None):  # pragma: no cover — spawns real subprocesses
        results = {}
        if not isinstance(engine, VllmEngine):
            Shared.warn(f"vllm bench only supports the vllm engine — skipping for {engine.name}")
            return results
        executable = engine.bench_executable()
        if executable is None:
            Shared.err("vllm bench not available — install the extra with: "
                       "pip install 'vllm[bench]'")
            return results

        # The offline subcommands load the weights themselves, so nothing else may hold the GPU.
        engine.stop()
        env = {**os.environ, "HF_HOME": str(engine.cache_home())}

        for model in models:
            tag, label, short = model["tag"], model["label"], model["short"]
            Shared.section(f"vllm bench ({engine.name}): {label}")
            emit_progress("model", "vllmbench", "running", label)
            try:
                if not engine.model_pulled(tag):
                    Shared.warn(f"{tag} not pulled — skipping")
                    continue
                repo = engine._repo(tag)
                if repo is None:
                    Shared.err(f"{tag}: no vLLM weights in the catalog — skipping")
                    results[short] = {"error": "no vllm_repo for tag"}
                    continue

                latency_entries, throughput_entries = [], []
                sizes = self.sweep_sizes(
                    config.VLLMBENCH_INPUT, config.VLLMBENCH_OUTPUT,
                    engine.max_context_length(tag),
                )
                model_result: VllmBenchModelResult = {
                    "latency_entries": latency_entries,
                    "throughput_entries": throughput_entries,
                    "requested_cases": len(sizes) * 2,
                    "completed_cases": 0,
                }
                results[short] = model_result

                for input_len, output_len in sizes:
                    wait_if_paused()
                    for kind, builder, parser, bucket in (
                        ("latency", self.build_latency_command,
                         self.parse_latency_result, latency_entries),
                        ("throughput", self.build_throughput_command,
                         self.parse_throughput_result, throughput_entries),
                    ):
                        with tempfile.TemporaryDirectory() as workdir:
                            out = Path(workdir) / f"{kind}.json"
                            command = builder(executable, repo, out, input_len, output_len)
                            try:
                                payload = self.run_one(
                                    command, out, config.VLLMBENCH_TIMEOUT, env,
                                )
                            except subprocess.TimeoutExpired:
                                model_result.update(
                                    timed_out=True, timed_out_at=f"{kind} in{input_len}",
                                    error=f"no result within {config.VLLMBENCH_TIMEOUT}s",
                                )
                                Shared.err(f"{label}: {kind} timed out at in{input_len}")
                                break
                            except Exception as exc:
                                model_result["error"] = str(exc)
                                Shared.err(f"{label}: {kind} failed at in{input_len}: {exc}")
                                break
                        entry = parser(payload, input_len, output_len)
                        if entry is None:
                            Shared.warn(f"{label}: {kind} at in{input_len} reported no usable result")
                            continue
                        bucket.append(entry)
                        model_result["completed_cases"] += 1
                        Shared.ok(self.format_entry(kind, entry))
                        if save_fn:
                            save_fn(results)
                    if model_result.get("error") or model_result.get("timed_out"):
                        break
            finally:
                if save_fn:
                    save_fn(results)
                emit_model_finished("vllmbench", label, results.get(short))
        return results
