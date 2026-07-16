"""Tests for Shared.run_accuracy_benchmark driven by a fake InferenceEngine.

This function was previously `pragma: no cover` because it was entangled with
real Ollama network/process calls. Now that those live behind the
InferenceEngine interface, a fully in-memory test double implementing the
interface with canned per-question responses can drive it through every branch
that matters: a normal run scored correctly, a question that times out with
partial text that gets rescored, a question caught in a generation loop, and a
run that crashes and stops early — the actual payoff of the engine refactor.

The double drives the real MCQBenchmark ask/rescore/score functions (the
thinnest accuracy benchmark), so the orchestration is exercised end-to-end,
not against a stubbed scorer.
"""

import json

import pytest

import config
from engines.base import InferenceEngine
from mcq_benchmark import MCQBenchmark
from shared import OllamaLoopDetected, OllamaTimeout, Shared


class FakeEngine(InferenceEngine):
    """In-memory InferenceEngine. chat() dispatches on a per-question behavior
    keyed by a marker embedded in the question prompt (so the fake needs no
    knowledge of message formatting): "ok" returns canned text, "timeout"
    raises OllamaTimeout with partial text, "loop" raises OllamaLoopDetected,
    "crash" raises a ConnectionError (the runner-died shape). Everything else
    is a no-op or trivially-true, since run_accuracy_benchmark only needs the
    server/model to look available and to load."""

    name = "fake"

    def __init__(self, behaviors: dict[str, tuple[str, str]]):
        # marker -> (kind, text): kind in {"ok","timeout","loop","crash"}
        self._behaviors = behaviors
        self.unloaded: list[str] = []

    # server/process lifecycle
    def ensure_running(self) -> bool: return True
    def start(self, *, gpu_visible: bool = True, timeout: int = 15) -> bool: return True
    def stop(self, *, timeout: int = 15) -> None: pass
    def available(self) -> bool: return True
    def reachable_or_abort(self) -> bool: return True
    def wait_for_recovery(self, timeout: int = 30) -> bool: return True
    def is_connection_crash(self, exc: Exception) -> bool:
        return isinstance(exc, ConnectionError)
    def tail_log(self, n_lines: int = 40) -> str: return "(fake log)"

    # model lifecycle
    def model_pulled(self, tag: str) -> bool: return True
    def list_installed_models(self) -> list[dict]: return []
    def max_context_length(self, tag: str, default: int = 131072) -> int: return default
    def warmup(self, tag, label, num_ctx, warmup_runs, crash_cache=None,
               cache_path=None, crash_extra=None) -> bool:
        return True
    def unload(self, tag: str) -> None: self.unloaded.append(tag)
    def unload_all(self) -> None: pass
    def wait_until_unloaded(self, tag: str, timeout: int = 30) -> None: pass

    # inference
    def generate(self, tag, prompt, timeout=600, num_ctx=None):
        return 0.1, 1, 1.0

    def chat(self, tag, messages, timeout=600, num_ctx=None, num_predict=1024, check_loop=False):
        content = messages[-1]["content"]
        for marker, (kind, text) in self._behaviors.items():
            if marker in content:
                if kind == "ok":
                    return 0.1, len(text.split()), 5.0, 10, text
                if kind == "timeout":
                    raise OllamaTimeout("timed out", partial_text=text)
                if kind == "loop":
                    raise OllamaLoopDetected("generation loop", partial_text=text)
                if kind == "crash":
                    raise ConnectionError("actively refused")
        raise AssertionError(f"no canned behavior matched prompt: {content!r}")

    def embed(self, tag, inputs, timeout=120):
        return [], 0.0


def _question(qid: str, answer: str) -> dict:
    # The prompt carries the qid so FakeEngine.chat can dispatch on it.
    return {
        "id": qid,
        "category": "general",
        "prompt": f"[{qid}] What is the answer?",
        "choices": {"A": "a", "B": "b", "C": "c", "D": "d"},
        "answer": answer,
    }


def _run(tmp_path, questions, behaviors):
    data_path = tmp_path / "bank.json"
    data_path.write_text(json.dumps(questions))
    engine = FakeEngine(behaviors)
    models = [{"tag": "fake:tag", "label": "Fake Model", "short": "fake"}]
    results = Shared.run_accuracy_benchmark(
        section_label="MCQ", skip_label="MCQ", question_noun="MCQ questions",
        data_path=data_path, crash_cache_path=tmp_path / "crash.json",
        models=models, questions=questions, warmup_runs=1, engine=engine,
        ask_fn=lambda tag, q: MCQBenchmark._ask(engine, tag, q),
        rescore_partial_fn=lambda q, text: MCQBenchmark.parse_answer(text, q["choices"].keys()),
        score_fn=MCQBenchmark.score,
    )
    return results, engine


def test_normal_run_scores_correctly(tmp_path):
    questions = [_question("q1", "B"), _question("q2", "A")]
    behaviors = {
        "q1": ("ok", "The answer is B"),   # correct
        "q2": ("ok", "The answer is D"),   # wrong (expected A)
    }
    results, engine = _run(tmp_path, questions, behaviors)

    assert results["fake"]["correct"] == 1
    assert results["fake"]["total"] == 2
    assert results["fake"]["accuracy_pct"] == 50.0
    assert "timed_out_count" not in results["fake"]
    assert "likely_loop_count" not in results["fake"]
    assert "crashed" not in results["fake"]
    assert engine.unloaded == ["fake:tag"]  # unloaded when the model finished


def test_timeout_with_partial_text_gets_rescored(tmp_path):
    questions = [_question("q1", "B")]
    # Times out, but had already written a parseable (wrong) answer: it's
    # rescored from the partial text rather than treated as a blank, and still
    # counted as a timeout.
    behaviors = {"q1": ("timeout", "I think the answer is C")}
    results, _ = _run(tmp_path, questions, behaviors)

    assert results["fake"]["correct"] == 0        # C != B, scored wrong
    assert results["fake"]["answered"] == 1       # rescored, not blank
    assert results["fake"]["timed_out_count"] == 1
    assert results["fake"]["timed_out_ids"] == ["q1"]


def test_loop_detected_question_is_flagged(tmp_path):
    questions = [_question("q1", "B")]
    behaviors = {"q1": ("loop", "wait, wait, wait, still stuck")}
    results, _ = _run(tmp_path, questions, behaviors)

    assert results["fake"]["correct"] == 0
    assert results["fake"]["likely_loop_count"] == 1
    assert results["fake"]["likely_loop_ids"] == ["q1"]
    # A loop caught early is not a wall-clock timeout — the buckets are distinct.
    assert "timed_out_count" not in results["fake"]


def test_crashed_run_stops_early(tmp_path):
    questions = [_question("q1", "B"), _question("q2", "A"), _question("q3", "C")]
    # q2 crashes the runner deterministically; the run should stop and never
    # reach q3.
    behaviors = {
        "q1": ("ok", "The answer is B"),
        "q2": ("crash", ""),
        "q3": ("ok", "The answer is C"),
    }
    results, _ = _run(tmp_path, questions, behaviors)

    assert results["fake"]["crashed"] is True
    assert "crashed_at" in results["fake"]
    # q1 was scored before the crash; q3 was never reached (still counts toward
    # total as unanswered, not correct).
    assert results["fake"]["correct"] == 1
    assert results["fake"]["total"] == 3
    assert results["fake"]["answered"] == 1
