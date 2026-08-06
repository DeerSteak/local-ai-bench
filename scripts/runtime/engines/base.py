"""base.py — the engine-neutral inference interface used by every workload.
See docs/engines.md.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
import statistics
from pathlib import Path


@dataclass(frozen=True)
class GenerationMeasurement:
    client_ttft_sec: float
    generated_tokens: int
    tokens_per_sec: float
    client_wall_sec: float
    decode_sec: float
    server_prompt_sec: float | None = None
    prompt_tokens: int | None = None
    response_text: str = ""
    finish_reason: str | None = None
    model_load_sec: float = 0
    server_tps_implausible: bool = False


@dataclass(frozen=True)
class ChatMeasurement(GenerationMeasurement):
    tool_calls: list[dict] = field(default_factory=list)
    budget_nudged: bool = False


@dataclass(frozen=True)
class EmbeddingMeasurement:
    embeddings: list
    client_wall_sec: float
    model_load_sec: float = 0


def measurement_validation_errors(measurement: GenerationMeasurement) -> list[str]:
    errors = []
    if measurement.server_tps_implausible:
        errors.append("implausible_server_tps")
    durations = {
        "client_ttft_sec": measurement.client_ttft_sec,
        "client_wall_sec": measurement.client_wall_sec,
        "decode_sec": measurement.decode_sec,
    }
    if measurement.server_prompt_sec is not None:
        durations["server_prompt_sec"] = measurement.server_prompt_sec
    durations["model_load_sec"] = measurement.model_load_sec
    errors.extend(name for name, value in durations.items()
                  if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0)
    if isinstance(measurement.generated_tokens, bool) or not isinstance(measurement.generated_tokens, int) \
            or measurement.generated_tokens < 0:
        errors.append("generated_tokens")
    if not isinstance(measurement.tokens_per_sec, (int, float)) \
            or not math.isfinite(measurement.tokens_per_sec) or measurement.tokens_per_sec < 0:
        errors.append("tokens_per_sec")
    if all(isinstance(value, (int, float)) and math.isfinite(value)
           for value in (measurement.client_ttft_sec, measurement.client_wall_sec)) \
            and measurement.client_ttft_sec > measurement.client_wall_sec + 1e-6:
        errors.append("client_ttft_after_wall")
    if isinstance(measurement.decode_sec, (int, float)) and measurement.decode_sec > 0 \
            and isinstance(measurement.tokens_per_sec, (int, float)) \
            and math.isfinite(measurement.tokens_per_sec) \
            and isinstance(measurement.generated_tokens, int) \
            and not isinstance(measurement.generated_tokens, bool):
        expected_tps = measurement.generated_tokens / measurement.decode_sec
        if abs(measurement.tokens_per_sec - expected_tps) > max(expected_tps * 0.01, 0.01):
            errors.append("decode_rate_mismatch")
    return errors


def is_valid_measurement(measurement: GenerationMeasurement) -> bool:
    return not measurement_validation_errors(measurement)


def embedding_validation_errors(measurement: EmbeddingMeasurement) -> list[str]:
    errors = []
    if not isinstance(measurement.client_wall_sec, (int, float)) \
            or not math.isfinite(measurement.client_wall_sec) or measurement.client_wall_sec <= 0:
        errors.append("client_wall_sec")
    if not isinstance(measurement.embeddings, list):
        errors.append("embeddings")
    return errors


def aggregate_generation_measurements(samples: list[GenerationMeasurement],
                                      requested_runs: int) -> dict:
    valid = [sample for sample in samples if not measurement_validation_errors(sample)]
    invalid = [
        {"run": index + 1, "errors": measurement_validation_errors(sample)}
        for index, sample in enumerate(samples) if measurement_validation_errors(sample)
    ]
    result = {
        "n_runs": len(samples),
        "requested_runs": requested_runs,
        "completed_runs": len(samples),
        "valid_runs": len(valid),
        "invalid_runs": invalid,
        "valid_samples": [
            {
                "client_ttft_sec": round(sample.client_ttft_sec, 3),
                "server_prompt_sec": round(sample.server_prompt_sec, 3)
                if sample.server_prompt_sec is not None else None,
                "client_wall_sec": round(sample.client_wall_sec, 3),
                "decode_sec": round(sample.decode_sec, 3),
                "generated_tokens": sample.generated_tokens,
                "tokens_per_sec": round(sample.tokens_per_sec, 2),
                "finish_reason": sample.finish_reason,
                "model_load_sec": round(sample.model_load_sec, 3),
            }
            for sample in valid
        ],
    }
    if not valid:
        return result
    ttfts = [sample.client_ttft_sec for sample in valid]
    tps_values = [sample.tokens_per_sec for sample in valid]
    result.update({
        "client_ttft_mean_sec": round(statistics.mean(ttfts), 3),
        "client_ttft_stdev_sec": round(statistics.stdev(ttfts), 3) if len(ttfts) >= 2 else 0,
        "client_ttft_runs_sec": [round(value, 3) for value in ttfts],
        "server_prompt_runs_sec": [
            round(sample.server_prompt_sec, 3) for sample in valid
            if sample.server_prompt_sec is not None
        ],
        "client_wall_runs_sec": [round(sample.client_wall_sec, 3) for sample in valid],
    })
    server_times = [sample.server_prompt_sec for sample in valid
                    if sample.server_prompt_sec is not None]
    if server_times:
        result["server_prompt_mean_sec"] = round(statistics.mean(server_times), 3)
    if len(valid) >= 2:
        result.update({
            "client_ttft_median_sec": round(statistics.median(ttfts), 3),
            "client_ttft_cv": round(statistics.stdev(ttfts) / statistics.mean(ttfts), 4)
            if statistics.mean(ttfts) else 0,
            "tps_median": round(statistics.median(tps_values), 2),
            "tps_cv": round(statistics.stdev(tps_values) / statistics.mean(tps_values), 4)
            if statistics.mean(tps_values) else 0,
        })
    return result


class InferenceEngine(ABC):
    name: str  # e.g. "llamacpp"

    # ── server / process lifecycle ──

    def is_installed(self) -> bool:
        """Whether this engine's runtime is present, without logging or starting anything.
        Frontends use it to offer only engines that can actually run."""
        return True

    @abstractmethod
    def ensure_running(self) -> bool:
        """Start the engine's server if it isn't already up. Returns True if
        it's available afterwards."""

    @abstractmethod
    def start(self, *, gpu_visible: bool = True, timeout: int = 15) -> bool:
        """Start the engine's server; gpu_visible=False forces CPU-only. Returns True once reachable."""

    @abstractmethod
    def stop(self, *, timeout: int = 15) -> None:
        """Stop any running server for this engine, including one this script
        didn't start."""

    @abstractmethod
    def available(self) -> bool:
        """True if the engine's server is reachable."""

    @abstractmethod
    def reachable_or_abort(self) -> bool:
        """True if reachable; otherwise log an error so a caller looping over
        models can stop rather than misreporting every remaining model."""

    @abstractmethod
    def wait_for_recovery(self, timeout: int = 30) -> bool:
        """Poll until the server answers again after a model-runner crash.
        Returns False if it doesn't come back within `timeout`."""

    @abstractmethod
    def is_connection_crash(self, exc: Exception) -> bool:
        """True if `exc` looks like the model runner died (commonly OOM)
        rather than an ordinary request failure."""

    @abstractmethod
    def tail_log(self, n_lines: int = 40) -> str:
        """Return the last n_lines of the server's captured output, to surface
        a real crash reason instead of guessing."""

    def runtime_backend(self, hardware_backend: str, *, cpu_only: bool = False) -> str:
        """Backend the engine will actually use, falling back to the detected
        hardware classification when an engine has no stronger signal."""
        return "cpu" if cpu_only else hardware_backend

    def resume_artifact_paths(self, tag: str) -> tuple[Path, ...]:
        """Return local model files whose bytes must match before safe resume."""
        raise NotImplementedError(f"{self.name} does not expose resume artifact identity")

    def resume_runtime_paths(self) -> dict[str, Path]:
        """Return stable runtime names and executable files required for safe resume."""
        raise NotImplementedError(f"{self.name} does not expose resume runtime identity")

    # ── model lifecycle ──

    @abstractmethod
    def model_pulled(self, tag: str) -> bool:
        """True if `tag` is installed locally."""

    @abstractmethod
    def list_installed_models(self) -> list[dict]:
        """Every model installed locally, as [{"tag": ..., "size": ...}]."""

    @abstractmethod
    def max_context_length(self, tag: str, default: int = 131072) -> int:
        """A pulled model's real max context length, or `default` on failure."""

    @abstractmethod
    def warmup(self, tag: str, label: str, num_ctx: int, warmup_runs: int,
               crash_cache: dict | None = None, cache_path: Path | None = None,
               crash_extra: dict | None = None) -> bool:
        """Load `tag` with `warmup_runs` watchdogged calls. Returns False on the first hung/failed run."""

    @abstractmethod
    def unload(self, tag: str) -> None:
        """Force the engine to evict `tag` from memory immediately."""

    @abstractmethod
    def unload_all(self) -> None:
        """Unload every model currently loaded."""

    @abstractmethod
    def wait_until_unloaded(self, tag: str, timeout: int = 30) -> None:
        """Poll until `tag` is no longer loaded."""

    @abstractmethod
    def prepare_concurrency(self, tag: str, n_parallel: int, per_slot_ctx: int,
                             warmup_runs: int = 1, timeout: int = 300) -> bool:
        """Load `tag` to serve `n_parallel` requests at `per_slot_ctx` tokens
        each; subsequent generate() calls must pass the same n_parallel back."""

    # ── inference ──

    @abstractmethod
    def generate(self, tag: str, prompt: str, timeout: int = 600,
                 num_ctx: int | None = None, n_parallel: int = 1) -> GenerationMeasurement:
        """Return one named single-shot measurement."""

    @abstractmethod
    def chat(self, tag: str, messages: list, timeout: int = 600,
             num_ctx: int | None = None, num_predict: int = 1024,
             check_loop: bool = False, token_budget: int | None = None) -> ChatMeasurement:
        """Return one named chat measurement."""

    def supports_tool_calls(self, tag: str) -> bool:
        """Whether this engine can return parsed tool_calls for `tag`. False makes the
        tool workload skip the model rather than score unparsed calls as wrong answers."""
        return True

    @abstractmethod
    def chat_tools(self, tag: str, messages: list, tools: list, timeout: int = 600,
                   num_ctx: int | None = None, num_predict: int = 1024,
                   check_loop: bool = False, token_budget: int | None = None) -> ChatMeasurement:
        """Return one named tool-chat measurement."""

    @abstractmethod
    def embed(self, tag: str, inputs: list[str], timeout: int = 120) -> EmbeddingMeasurement:
        """Return one named embedding measurement."""
