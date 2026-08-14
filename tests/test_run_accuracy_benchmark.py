"""Tests for Shared.run_accuracy_benchmark driven by a fake InferenceEngine —
see docs/testing.md for the coverage breakdown."""

import json
from typing import cast

import pytest

from scripts.runtime import config
from scripts.runtime.engines.base import ChatMeasurement, EmbeddingMeasurement, GenerationMeasurement, InferenceEngine
from scripts.workloads.mcq_benchmark import MCQBenchmark
from scripts.workloads.tool_benchmark import ToolBenchmark
from scripts.runtime.shared import EngineBudgetExceeded, EngineLoopDetected, EngineTimeout, Shared


class FakeEngine(InferenceEngine):
    """In-memory InferenceEngine; chat() dispatches on a marker embedded in
    the question prompt: "ok"/"timeout"/"loop"/"crash"."""

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
    def wait_until_unloaded(self, tag: str, timeout: int = 30) -> bool: return True
    def prepare_concurrency(self, tag, n_parallel, per_slot_ctx, warmup_runs=1, timeout=300) -> bool:
        return True

    # inference
    def generate(self, tag, prompt, timeout=600, num_ctx=None, n_parallel=1):
        return GenerationMeasurement(0.1, 1, 1.0, 1.1, 1.0)

    def chat(self, tag, messages, timeout=600, num_ctx=None, num_predict=1024,
             check_loop=False, token_budget=None):
        self.chat_contexts.append(num_ctx)
        content = messages[-1]["content"]
        for marker, (kind, text) in self._behaviors.items():
            if marker in content:
                if kind == "ok":
                    return ChatMeasurement(0.1, len(text.split()), 5.0, 1.1, 1.0,
                                           prompt_tokens=10, response_text=text)
                if kind == "nudged":
                    return ChatMeasurement(0.1, len(text.split()), 5.0, 1.1, 1.0,
                                           prompt_tokens=10, response_text=text,
                                           budget_nudged=True)
                if kind == "budget":
                    raise EngineBudgetExceeded("budget exhausted", partial_text=text)
                if kind == "nudged_timeout":
                    raise EngineTimeout("timed out", partial_text=text, budget_nudged=True)
                if kind == "timeout":
                    raise EngineTimeout("timed out", partial_text=text)
                if kind == "loop":
                    raise EngineLoopDetected("generation loop", partial_text=text)
                if kind == "crash":
                    raise ConnectionError("actively refused")
        raise AssertionError(f"no canned behavior matched prompt: {content!r}")

    def chat_tools(self, tag, messages, tools, timeout=600, num_ctx=None,
                   num_predict=1024, check_loop=False, token_budget=None):
        self.tool_contexts.append(num_ctx)
        content = messages[-1]["content"]
        for marker, (kind, payload) in self._tool_behaviors.items():
            if marker in content:
                if kind == "ok":
                    return ChatMeasurement(0.1, 1, 5.0, 0.3, 0.2,
                                           prompt_tokens=10, response_text=json.dumps(payload),
                                           tool_calls=payload)
                if kind == "timeout":
                    raise EngineTimeout("timed out", partial_text=payload)
                if kind == "loop":
                    raise EngineLoopDetected("generation loop", partial_text=payload)
                if kind == "crash":
                    raise ConnectionError("actively refused")
        raise AssertionError(f"no canned tool behavior matched prompt: {content!r}")

    def embed(self, tag, inputs, timeout=120):
        return EmbeddingMeasurement([], 0.1)


def _question(qid: str, answer: str) -> dict:
    # The prompt carries the qid so FakeEngine.chat can dispatch on it.
    return {
        "id": qid,
        "category": "general",
        "prompt": f"[{qid}] What is the answer?",
        "choices": {"A": "a", "B": "b", "C": "c", "D": "d"},
        "answer": answer,
    }


def _run(tmp_path, questions, behaviors, telemetry=None):
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
        telemetry=telemetry,
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


def test_accuracy_telemetry_retains_question_subwindows(tmp_path):
    class Telemetry:
        def __init__(self): self.calls = []
        def begin_model_load(self): self.calls.append("load")
        def begin_measured(self, name): self.calls.append(name)
        def finish_case(self):
            self.calls.append("finish")
            return {"windows": [{"name": name} for name in self.calls if name.startswith("measured:")]}

    telemetry = Telemetry()
    results, _ = _run(
        tmp_path, [_question("q1", "B"), _question("q2", "A")],
        {"q1": ("ok", "B"), "q2": ("ok", "A")}, telemetry,
    )
    assert telemetry.calls == ["load", "measured:q1", "measured:q2", "finish"]
    assert [window["name"] for window in results["fake"]["memory"]["windows"]] == [
        "measured:q1", "measured:q2",
    ]


def test_timeout_with_partial_text_gets_rescored(tmp_path):
    questions = [_question("q1", "B")]
    # Rescored from the partial text rather than treated as blank, and still counted as a timeout.
    behaviors = {"q1": ("timeout", "I think the answer is C")}
    results, _ = _run(tmp_path, questions, behaviors)

    assert results["fake"]["correct"] == 0        # C != B, scored wrong
    assert results["fake"]["answered"] == 1       # rescored, not blank
    assert results["fake"]["timed_out_count"] == 1
    assert results["fake"]["timed_out_ids"] == ["q1"]


def test_successful_and_exceptional_retries_record_independent_diagnostics(tmp_path):
    questions = [_question("q1", "B"), _question("q2", "A"), _question("q3", "C")]
    behaviors = {
        "q1": ("nudged", "The answer is B"),
        "q2": ("budget", "The answer is A"),
        "q3": ("nudged_timeout", "The answer is C"),
    }
    results, _ = _run(tmp_path, questions, behaviors)
    model = results["fake"]
    assert model["correct"] == 3
    assert model["budget_nudged_ids"] == ["q1", "q2", "q3"]
    assert model["budget_nudged_count"] == 3
    assert model["budget_exceeded_ids"] == ["q2"]
    assert model["budget_exceeded_count"] == 1
    assert model["timed_out_ids"] == ["q3"]
    assert model["timed_out_count"] == 1


def test_budget_exhaustion_continues_bank_and_sidecar_uses_graded_partial(tmp_path):
    questions = [_question("q1", "B"), _question("q2", "A")]
    data_path = tmp_path / "bank.json"
    data_path.write_text(json.dumps(questions))
    answers_path = tmp_path / "answers.json"
    engine = FakeEngine({
        "q1": ("budget", "The answer is B"),
        "q2": ("ok", "The answer is A"),
    })
    results = Shared.run_accuracy_benchmark(
        section_label="MCQ", skip_label="MCQ", question_noun="questions",
        data_path=data_path, crash_cache_path=tmp_path / "crash.json",
        models=[{"tag": "fake:tag", "label": "Fake Model", "short": "fake"}],
        questions=questions, warmup_runs=1, engine=engine,
        ask_fn=lambda tag, q: MCQBenchmark._ask(engine, tag, q),
        rescore_partial_fn=lambda q, text: MCQBenchmark.parse_answer(text, q["choices"].keys()),
        score_fn=MCQBenchmark.score, answers_path=answers_path,
    )
    assert results["fake"]["correct"] == 2
    sidecar = json.loads(answers_path.read_text())
    assert [entry["raw_response"] for entry in sidecar["fake"]["answers"]] == [
        "The answer is B", "The answer is A",
    ]


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
    # likely_loop_ids is a diagnostic of wrong answers, not raw detector hits.
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


def test_answers_sidecar_includes_correct_and_incorrect_responses(tmp_path):
    questions = [_question("q1", "B"), _question("q2", "A")]
    behaviors = {
        "q1": ("ok", "The answer is B"),   # correct
        "q2": ("ok", "The answer is D"),   # wrong (expected A)
    }
    data_path = tmp_path / "bank.json"
    data_path.write_text(json.dumps(questions))
    answers_path = tmp_path / "answers.json"
    engine = FakeEngine(behaviors)
    models = [{"tag": "fake:tag", "label": "Fake Model", "short": "fake"}]
    results = Shared.run_accuracy_benchmark(
        section_label="MCQ", skip_label="MCQ", question_noun="MCQ questions",
        data_path=data_path, crash_cache_path=tmp_path / "crash.json",
        models=models, questions=questions, warmup_runs=1, engine=engine,
        ask_fn=lambda tag, q: MCQBenchmark._ask(engine, tag, q),
        rescore_partial_fn=lambda q, text: MCQBenchmark.parse_answer(text, q["choices"].keys()),
        score_fn=MCQBenchmark.score,
        answers_path=answers_path,
    )

    sidecar = json.loads(answers_path.read_text())
    entries = sidecar["fake"]["answers"]
    assert {e["id"] for e in entries} == {"q1", "q2"}  # every question, not just wrong ones

    q1_entry = next(e for e in entries if e["id"] == "q1")
    assert q1_entry["correct"] is True
    assert q1_entry["raw_response"] == "The answer is B"

    q2_entry = next(e for e in entries if e["id"] == "q2")
    assert q2_entry["correct"] is False
    assert q2_entry["raw_response"] == "The answer is D"

    # The main results JSON is a separate schema — it must keep its existing
    # incorrect-only shape and never pick up the sidecar's "all" list.
    assert "all" not in results["fake"]
    assert {e["id"] for e in results["fake"]["incorrect"]} == {"q2"}
    assert results["fake"]["correct"] == 1


def test_answers_sidecar_rejects_non_finite_numbers(tmp_path):
    path = tmp_path / "answers.json"
    with pytest.raises(ValueError, match="non-finite numeric value"):
        Shared.write_answers_sidecar(path, {"given": float("inf")})
    assert not path.exists()


def test_answers_sidecar_preserves_pending_completed_answers(tmp_path):
    path = tmp_path / "answers.json"
    pending = {"m": {"label": "Model", "partial": True, "answers": [
        {"id": "q1", "given": "A", "raw_response": "Answer: A"},
    ]}}
    Shared.write_answers_sidecar(path, pending)
    assert json.loads(path.read_text()) == pending


def test_interrupt_flushes_completed_questions_from_current_model(tmp_path):
    questions = [_question("q1", "A"), _question("q2", "B")]
    data_path = tmp_path / "bank.json"
    data_path.write_text(json.dumps(questions))
    answers_path = tmp_path / "answers.json"
    calls = iter([("A", "Answer: A", False)])

    def ask(_tag, _question):
        try:
            return next(calls)
        except StopIteration:
            raise KeyboardInterrupt from None

    with pytest.raises(KeyboardInterrupt):
        Shared.run_accuracy_benchmark(
            "MCQ", "MCQ", "questions", data_path, tmp_path / "crash.json",
            [{"tag": "fake", "label": "Fake", "short": "fake"}], questions, 0,
            FakeEngine({}), ask, lambda q, text: text, MCQBenchmark.score,
            answers_path=answers_path,
        )
    sidecar = json.loads(answers_path.read_text())
    assert sidecar["fake"]["partial"] is True
    assert sidecar["fake"]["answers"] == [
        {"id": "q1", "given": "A", "raw_response": "Answer: A"},
    ]


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


def test_tool_workload_skips_a_model_the_engine_cannot_parse(monkeypatch, tmp_path):
    """Without a parser vLLM returns no tool_calls; scoring that as wrong would publish
    0% for a model that was never actually measured."""
    from scripts.runtime.shared import Shared

    class Engine:
        name = "vllm"
        def ensure_running(self): return True
        def reachable_or_abort(self): return True
        def model_pulled(self, tag): return True
        def supports_tool_calls(self, tag): return False
        def warmup(self, *a, **k): raise AssertionError("must not warm up an unmeasurable model")

    bank = tmp_path / "bank.json"
    bank.write_text("[]")
    results = Shared.run_accuracy_benchmark(
        section_label="Tool", skip_label="tool", question_noun="tool question",
        data_path=bank, crash_cache_path=tmp_path / "crash.json",
        models=[{"tag": "qwen3.5:9b-q4_K_M", "label": "Qwen", "short": "qwen"}],
        questions=[{"id": "q1"}], warmup_runs=0, engine=cast(InferenceEngine, Engine()),
        ask_fn=lambda *a: None, rescore_partial_fn=lambda *a: None, score_fn=lambda *a: {},
        requires_tool_calls=True,
    )
    entry = results["qwen"]
    assert entry["skipped"] is True
    assert entry["skip_reason"] == "tool_calls_unsupported"
    assert entry["skip_detail"] == "No vllm tool-call parser for this model"
    assert "score" not in entry, "a skipped model must not publish a score"
