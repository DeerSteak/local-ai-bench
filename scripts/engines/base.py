"""base.py — the InferenceEngine interface. Derived directly from the
Ollama-specific surface that used to live on Shared, so a new engine drops
into the existing orchestration untouched. See docs/engines.md.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class InferenceEngine(ABC):
    name: str  # e.g. "ollama"

    # ── server / process lifecycle ──

    @abstractmethod
    def ensure_running(self) -> bool:
        """Start the engine's server if it isn't already up. Returns True if
        it's available afterwards."""

    @abstractmethod
    def start(self, *, gpu_visible: bool = True, timeout: int = 15) -> bool:
        """Start the engine's server. gpu_visible=False forces CPU-only
        inference (the engine picks the right knob). Returns True once
        reachable."""

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
        """Load `tag` into memory with `warmup_runs` blocking calls, each
        watchdogged so a hung load times out. Returns False on the first hung
        or failed run."""

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
        """Load `tag` configured to serve `n_parallel` simultaneous requests
        at `per_slot_ctx` tokens each. Call once before any concurrent
        generate() calls, which must pass the same n_parallel back. Returns
        False on a load failure (a hard stop, no retry at a lower level)."""

    # ── inference ──

    @abstractmethod
    def generate(self, tag: str, prompt: str, timeout: int = 600,
                 num_ctx: int | None = None, n_parallel: int = 1) -> tuple[float, int, float]:
        """Single-shot generate. Returns (ttft_sec, tokens_generated,
        tokens_per_sec). n_parallel must match the last prepare_concurrency
        call (default 1 everywhere outside the concurrency test)."""

    @abstractmethod
    def chat(self, tag: str, messages: list, timeout: int = 600,
             num_ctx: int | None = None, num_predict: int = 1024,
             check_loop: bool = False) -> tuple[float, int, float, int, str]:
        """Multi-turn chat. Returns (ttft_sec, tokens_generated,
        tokens_per_sec, prompt_eval_count, response_text)."""

    @abstractmethod
    def embed(self, tag: str, inputs: list[str], timeout: int = 120) -> tuple[list, float]:
        """Embed `inputs` in a single call. Returns (embeddings_list,
        elapsed_seconds)."""
