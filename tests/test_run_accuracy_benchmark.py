"""Tests for Shared.run_accuracy_benchmark driven by a fake InferenceEngine.

This function was previously `pragma: no cover` because it was entangled with
real network/process calls. Now that those live behind the
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
from tool_benchmark import ToolBenchmark
from shared import EngineLoopDetected, EngineTimeout, Shared


class FakeEngine(InferenceEngine):
    """In-memory InferenceEngine. chat() dispatches on a per-question behavior
    keyed by a marker embedded in the question prompt (so the fake needs no
    knowledge of message formatting): "ok" returns canned text, "timeout"
    raises EngineTimeout with partial text, "loop" raises EngineLoopDetected,
    "crash" raises a ConnectionError (the runner-died shape). Everything else
    is a no-op or trivially-true, since run_accuracy_benchmark only needs the
    server/model to look available and to load."""

    name = "fake"

    def __init__(self, behaviors: dict[str, tuple[str, str]],
                 tool_behaviors: dict[str, tuple] | None = None):
        # marker -> (kind, text): kind in {"ok","timeout","loop","crash"}
        self._behaviors = behaviors
        # marker -> (kind, payload): for chat_tools. "ok" payload is a
        # tool_calls list; timeout/loop payload is partial text.
        self._tool_behaviors = tool_behaviors or {}
        self.unloaded: list[str] = []
        self.warmup_contexts: list[int] = []
        self.chat_contexts: list[int | None] = []
        self.tool_contexts: list[int | None] = []

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
        self.warmup_contexts.append(num_ctx)
        return True
    def unload(self, tag: str) -> None: self.unloaded.append(tag)
    def unload_all(self) -> None: pass
    def wait_until_unloaded(self, tag: str, timeout: int = 30) -> None: pass
    def prepare_concurrency(self, tag, n_parallel, per_slot_ctx, warmup_runs=1, timeout=300) -> bool:
        return True

    # inference
    def generate(self, tag, prompt, timeout=600, num_ctx=None, n_parallel=1):
        return 0.1, 1, 1.0

    def chat(self, tag, messages, timeout=600, num_ctx=None, num_predict=1024, check_loop=False):
        self.chat_contexts.append(num_ctx)
        content = messages[-1]["content"]
        for marker, (kind, text) in self._behaviors.items():
            if marker in content:
                if kind == "ok":
                    return 0.1, len(text.split()), 5.0, 10, text
                if kind == "timeout":
                    raise EngineTimeout("timed out", partial_text=text)
                if kind == "loop":
                    raise EngineLoopDetected("generation loop", partial_text=text)
                if kind == "crash":
                    raise ConnectionError("actively refused")
        raise AssertionError(f"no canned behavior matched prompt: {content!r}")

    def chat_tools(self, tag, messages, tools, timeout=600, num_ctx=None,
                   num_predict=1024, check_loop=False):
        self.tool_contexts.append(num_ctx)
        content = messages[-1]["content"]
        for marker, (kind, payload) in self._tool_behaviors.items():
            if marker in content:
                if kind == "ok":
                    return 0.1, 1, 5.0, 10, json.dumps(payload), payload
                if kind == "timeout":
                    raise EngineTimeout("timed out", partial_text=payload)
                if kind == "loop":
                    raise EngineLoopDetected("generation loop", partial_text=payload)
                if kind == "crash":
                    raise ConnectionError("actively refused")
        raise AssertionError(f"no canned tool behavior matched prompt: {content!r}")

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
    assert engine.warmup_contexts == [config.ACCURACY_CONTEXT]
    assert engine.chat_contexts == [config.ACCURACY_CONTEXT] * len(questions)
    assert engine.runtime_backend("cuda") == "cuda"
    assert engine.runtime_backend("cuda", cpu_only=True) == "cpu"


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


def test_loop_detected_but_correct_answer_is_not_flagged(tmp_path):
    # The model got caught in a loop-detection cutoff, but the partial text
    # already contained a correct, parseable answer — it should be scored
    # correct and never show up in likely_loop_ids, since that list is a
    # diagnostic of wrong answers, not of raw detector hits.
    questions = [_question("q1", "B")]
    behaviors = {"q1": ("loop", "The answer is B, wait, wait, wait, let me restate: B")}
    results, _ = _run(tmp_path, questions, behaviors)

    assert results["fake"]["correct"] == 1
    assert "likely_loop_count" not in results["fake"]
    assert "likely_loop_ids" not in results["fake"]


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


# ── Tool-calling accuracy path (ToolBenchmark through the same orchestration) ──


def _tool_question(qid: str, expected: dict) -> dict:
    return {
        "id": qid,
        "category": "single_tool_call",
        "prompt": f"[{qid}] do the thing",
        "tools": [{"type": "function", "function": {"name": "do_it"}}],
        "expected": expected,
    }


def _run_tool(tmp_path, questions, tool_behaviors):
    data_path = tmp_path / "tool_bank.json"
    data_path.write_text(json.dumps(questions))
    engine = FakeEngine({}, tool_behaviors=tool_behaviors)
    models = [{"tag": "fake:tag", "label": "Fake Model", "short": "fake"}]
    results = Shared.run_accuracy_benchmark(
        section_label="Tool", skip_label="tool", question_noun="tool question",
        data_path=data_path, crash_cache_path=tmp_path / "tool_crash.json",
        models=models, questions=questions, warmup_runs=1, engine=engine,
        ask_fn=lambda tag, q: ToolBenchmark._ask(engine, tag, q),
        rescore_partial_fn=ToolBenchmark.rescore_partial_fn,
        score_fn=ToolBenchmark.score,
    )
    return results, engine


def test_tool_normal_run_scores_correctly(tmp_path):
    questions = [
        _tool_question("q1", {"call": True, "name": "do_it", "arguments": {"x": 1}}),
        _tool_question("q2", {"call": True, "name": "do_it", "arguments": {"x": 2}}),
    ]
    tool_behaviors = {
        "q1": ("ok", [{"name": "do_it", "arguments": {"x": 1}}]),  # correct
        "q2": ("ok", [{"name": "do_it", "arguments": {"x": 99}}]),  # wrong argument
    }
    results, engine = _run_tool(tmp_path, questions, tool_behaviors)
    assert results["fake"]["correct"] == 1
    assert results["fake"]["total"] == 2
    assert engine.unloaded == ["fake:tag"]
    assert engine.warmup_contexts == [config.ACCURACY_CONTEXT]
    assert engine.tool_contexts == [config.ACCURACY_CONTEXT] * len(questions)


def test_tool_timeout_with_partial_text_gets_rescored(tmp_path):
    questions = [_tool_question("q1", {"call": True, "name": "do_it", "arguments": {"x": 1}})]
    # Times out, but the partial text is a parseable (correct) tool-call list.
    partial = json.dumps([{"name": "do_it", "arguments": {"x": 1}}])
    results, _ = _run_tool(tmp_path, questions, {"q1": ("timeout", partial)})
    assert results["fake"]["correct"] == 1        # rescored from partial text
    assert results["fake"]["answered"] == 1
    assert results["fake"]["timed_out_count"] == 1
    assert results["fake"]["timed_out_ids"] == ["q1"]


def test_tool_crashed_run_stops_early(tmp_path):
    questions = [
        _tool_question("q1", {"call": True, "name": "do_it", "arguments": {}}),
        _tool_question("q2", {"call": True, "name": "do_it", "arguments": {}}),
        _tool_question("q3", {"call": True, "name": "do_it", "arguments": {}}),
    ]
    tool_behaviors = {
        "q1": ("ok", [{"name": "do_it", "arguments": {}}]),
        "q2": ("crash", ""),
        "q3": ("ok", [{"name": "do_it", "arguments": {}}]),
    }
    results, _ = _run_tool(tmp_path, questions, tool_behaviors)
    assert results["fake"]["crashed"] is True
    assert results["fake"]["correct"] == 1
    assert results["fake"]["total"] == 3
    assert results["fake"]["answered"] == 1
