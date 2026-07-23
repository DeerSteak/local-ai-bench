import json
from pathlib import Path

import pytest

import config
import regrade
from shared import Shared


def score_block(label="Model"):
    return {
        "label": label,
        "correct": 0,
        "total": 1,
        "answered": 0,
        "accuracy_pct": 0.0,
        "by_category": {"test": {"correct": 0, "total": 1, "accuracy_pct": 0.0}},
        "incorrect": [{"id": "old"}],
    }


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value))


def test_answer_sidecar_path_preserves_results_suffix():
    path = Path("results/results_Mac_mini_20260721.json")
    assert regrade.answer_sidecar_path(path, "math") == Path(
        "results/answers_math_Mac_mini_20260721.json"
    )


def test_answer_sidecar_path_handles_custom_result_name():
    path = Path("results/custom.json")
    assert regrade.answer_sidecar_path(path, "tool") == Path(
        "results/answers_tool_custom.json"
    )


def test_regraded_path_only_prefixes_filename():
    assert regrade.regraded_path(Path("results/results_host.json")) == Path(
        "results/regraded_results_host.json"
    )


@pytest.mark.parametrize(("contents", "message"), [
    ("not json", "Could not read"),
    ("[]", "Expected a JSON object"),
])
def test_load_json_object_rejects_invalid_inputs(tmp_path, contents, message):
    path = tmp_path / "invalid.json"
    path.write_text(contents)
    with pytest.raises(regrade.RegradeError, match=message):
        regrade.load_json_object(path)


@pytest.mark.parametrize(("raw", "expected"), [
    ('[{"name": "weather", "arguments": {}}]', [{"name": "weather", "arguments": {}}]),
    (
        '{"tool_calls": [{"name": "weather", "arguments": {}}], "text": "Done"}',
        [{"name": "weather", "arguments": {}}],
    ),
    ("No tool is appropriate.", []),
    ('{"tool_calls": "not a list"}', []),
    ('{"other": []}', []),
])
def test_tool_calls_from_raw_handles_stored_response_shapes(raw, expected):
    assert regrade.tool_calls_from_raw(raw) == expected


def test_evaluate_raw_reasoning_uses_strict_choice_parsing():
    question = {"choices": {"A": "one", "B": "two", "C": "three", "D": "four"}}
    assert regrade.evaluate_raw_answer("reasoning", question, "Final answer: C") == "C"
    assert regrade.evaluate_raw_answer(
        "reasoning", question, "Considering C carefully, C remains plausible.",
    ) is None


def test_validate_bank_hashes_reports_every_mismatch(tmp_path, monkeypatch):
    bank = tmp_path / "bank.json"
    bank.write_text("[]")
    monkeypatch.setattr(regrade, "WORKLOADS", {"mcq": (bank, lambda: [])})

    with pytest.raises(regrade.RegradeError, match=r"stored old, current"):
        regrade.validate_bank_hashes(
            {"bank_versions": {"mcq": "old"}}, tmp_path / "results.json",
        )


def test_validate_bank_hashes_requires_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(regrade, "WORKLOADS", {})
    with pytest.raises(regrade.RegradeError, match="has no bank_versions object"):
        regrade.validate_bank_hashes({}, tmp_path / "results.json")


def test_recorded_workloads_keeps_legacy_results_regradeable(monkeypatch):
    monkeypatch.setattr(regrade, "WORKLOADS", {
        "mcq": (Path("mcq"), lambda: []),
        "reasoning": (Path("reasoning"), lambda: []),
    })
    results = {"bank_versions": {"mcq": "hash"}, "mcq": {}}
    assert regrade.recorded_workloads(results, Path("legacy.json")) == ["mcq"]


def test_recorded_workloads_rejects_unrecognized_bank_metadata(monkeypatch):
    monkeypatch.setattr(regrade, "WORKLOADS", {"reasoning": (Path("bank"), lambda: [])})
    with pytest.raises(regrade.RegradeError, match="no recognized accuracy bank versions"):
        regrade.recorded_workloads(
            {"bank_versions": {"removed_workload": "hash"}}, Path("results.json"),
        )


def test_questions_for_result_uses_recorded_sample_order(monkeypatch):
    questions = [{"id": "q1"}, {"id": "q2"}, {"id": "q3"}]
    monkeypatch.setattr(
        regrade, "WORKLOADS", {"mcq": (Path("unused"), lambda: questions)},
    )
    results = {"sample_ids": {"mcq": ["q3", "q1"]}}
    assert regrade.questions_for_result(results, "mcq") == [questions[2], questions[0]]


@pytest.mark.parametrize("sample_ids", [["missing"], ["q1", "q1"], "q1"])
def test_questions_for_result_rejects_invalid_sample_ids(sample_ids, monkeypatch):
    monkeypatch.setattr(
        regrade, "WORKLOADS",
        {"mcq": (Path("unused"), lambda: [{"id": "q1"}])},
    )
    with pytest.raises(regrade.RegradeError):
        regrade.questions_for_result({"sample_ids": {"mcq": sample_ids}}, "mcq")


def test_questions_for_result_rejects_non_object_sample_ids(monkeypatch):
    monkeypatch.setattr(
        regrade, "WORKLOADS",
        {"mcq": (Path("unused"), lambda: [{"id": "q1"}])},
    )
    with pytest.raises(regrade.RegradeError, match="sample_ids must be a JSON object"):
        regrade.questions_for_result({"sample_ids": []}, "mcq")


def test_validate_answer_entries_requires_exact_unique_bank_ids():
    questions = [{"id": "q1"}, {"id": "q2"}]
    with pytest.raises(regrade.RegradeError, match="repeats answer ID q1"):
        regrade.validate_answer_entries(
            "mcq", "model",
            [
                {"id": "q1", "raw_response": "A"},
                {"id": "q1", "raw_response": "B"},
            ],
            questions,
        )
    with pytest.raises(regrade.RegradeError, match="do not match the bank"):
        regrade.validate_answer_entries(
            "mcq", "model", [{"id": "q1", "raw_response": "A"}], questions,
        )


@pytest.mark.parametrize("entries", [None, {}, "answers"])
def test_validate_answer_entries_requires_list(entries):
    with pytest.raises(regrade.RegradeError, match="answers must be a list"):
        regrade.validate_answer_entries("mcq", "model", entries, [])


@pytest.mark.parametrize("entry", [{}, {"id": 1}, {"id": "q1", "raw_response": None}])
def test_validate_answer_entries_requires_id_and_raw_response(entry):
    with pytest.raises(regrade.RegradeError):
        regrade.validate_answer_entries("mcq", "model", [entry], [{"id": "q1"}])


def test_regrade_workload_reparses_raw_mcq_and_rebuilds_sidecar():
    questions = [{
        "id": "mcq_001",
        "category": "test",
        "answer": "B",
        "choices": {"A": "Wrong", "B": "Right"},
    }]
    original = score_block()
    original["timed_out_count"] = 1
    original["timed_out_ids"] = ["mcq_001"]
    original["likely_loop_count"] = 1
    original["likely_loop_ids"] = ["mcq_001"]
    sidecar = {
        "label": "Model",
        "answers": [{
            "id": "mcq_001", "category": "test", "given": "A", "expected": "B",
            "correct": False, "raw_response": "The answer is B.",
        }],
    }

    results, answers = regrade.regrade_workload(
        "mcq", questions, {"model": original}, {"model": sidecar},
    )

    assert results["model"]["correct"] == 1
    assert results["model"]["answered"] == 1
    assert results["model"]["timed_out_ids"] == ["mcq_001"]
    assert "likely_loop_ids" not in results["model"]
    assert answers["model"]["answers"] == [{
        "id": "mcq_001", "category": "test", "given": "B", "expected": "B",
        "correct": True, "raw_response": "The answer is B.",
    }]


def test_regrade_workload_rejects_model_set_mismatch():
    with pytest.raises(regrade.RegradeError, match="model keys differ"):
        regrade.regrade_workload("math", [], {"one": {}}, {"two": {}})


def test_regrade_workload_requires_object_blocks():
    with pytest.raises(regrade.RegradeError, match="must be JSON objects"):
        regrade.regrade_workload("math", [], [], {})
    with pytest.raises(regrade.RegradeError, match="must be a JSON object"):
        regrade.regrade_workload("math", [], {"model": []}, {"model": {}})


def test_merge_diagnostics_keeps_only_still_incorrect_loop_ids():
    original = {"likely_loop_ids": ["q1", "q2"], "likely_loop_count": 2}
    scored = {"incorrect": [{"id": "q2"}], "all": []}
    assert regrade.merge_diagnostics(original, scored) == {
        "incorrect": [{"id": "q2"}],
        "likely_loop_ids": ["q2"],
        "likely_loop_count": 1,
    }


def test_regrade_workload_wraps_evaluator_failure_with_question_context(monkeypatch):
    questions = [{"id": "math_1", "category": "test", "answer": 1}]
    monkeypatch.setattr(
        regrade, "evaluate_raw_answer",
        lambda *args: (_ for _ in ()).throw(RuntimeError("evaluation failed")),
    )
    with pytest.raises(
            regrade.RegradeError,
            match=r"Failed to regrade math/model/math_1: evaluation failed"):
        regrade.regrade_workload(
            "math", questions, {"model": score_block()},
            {"model": {"answers": [{"id": "math_1", "raw_response": "1"}]}},
        )


def make_result_set(tmp_path, monkeypatch):
    questions = {
        "mcq": [{
            "id": "mcq_1", "category": "test", "answer": "B",
            "choices": {"A": "Wrong", "B": "Right"},
        }],
        "math": [{"id": "math_1", "category": "test", "answer": 5, "tolerance": 0}],
        "code": [{
            "id": "code_1", "category": "test", "function_name": "answer",
            "visible_tests": [{"args": [], "expected": 1}], "hidden_tests": [],
        }],
        "tool": [{
            "id": "tool_1", "category": "test", "expected": {"call": False},
            "tools": [],
        }],
    }
    banks = {}
    workloads = {}
    for workload, workload_questions in questions.items():
        bank = tmp_path / f"{workload}_bank.json"
        write_json(bank, workload_questions)
        banks[workload] = Shared.file_hash(bank)
        workloads[workload] = (bank, lambda values=workload_questions: values)
    monkeypatch.setattr(regrade, "WORKLOADS", workloads)
    monkeypatch.setattr(
        regrade.CodeBenchmark, "evaluate_question",
        staticmethod(lambda question, code: {
            "correct": "return 1" in code,
            "tests_passed": int("return 1" in code),
            "tests_total": 1,
            "error": None,
        }),
    )

    results_path = tmp_path / "results_host.json"
    results = {
        "version": "3.2",
        "engine": "llamacpp",
        "profile": {"hostname": "host"},
        "bank_versions": banks,
        "sample_ids": {},
        "llm": {"must": "remain"},
        **{workload: {"model": score_block()} for workload in questions},
    }
    write_json(results_path, results)
    raw_responses = {
        "mcq": "The answer is B.",
        "math": "The final answer is 5.",
        "code": "```python\ndef answer():\n    return 1\n```",
        "tool": "No offered tool applies.",
    }
    for workload, workload_questions in questions.items():
        question = workload_questions[0]
        sidecar = {
            "model": {
                "label": "Model",
                "answers": [{
                    "id": question["id"],
                    "category": "test",
                    "correct": False,
                    "raw_response": raw_responses[workload],
                }],
            },
        }
        write_json(regrade.answer_sidecar_path(results_path, workload), sidecar)
    return results_path, results


def test_build_regraded_outputs_preserves_non_grading_data_and_rebuilds_all_sidecars(
        tmp_path, monkeypatch):
    results_path, original = make_result_set(tmp_path, monkeypatch)

    outputs = regrade.build_regraded_outputs(results_path)
    regraded_results = outputs[regrade.regraded_path(results_path)]

    assert regraded_results["version"] == config.VERSION
    assert regraded_results["engine"] == original["engine"]
    assert regraded_results["profile"] == original["profile"]
    assert regraded_results["llm"] == original["llm"]
    assert regraded_results["bank_versions"] == original["bank_versions"]
    assert len(outputs) == 5
    assert all(regraded_results[workload]["model"]["correct"] == 1
               for workload in regrade.WORKLOADS)
    assert outputs[
        regrade.regraded_path(regrade.answer_sidecar_path(results_path, "math"))
    ]["model"]["answers"][0]["given"] == 5.0
    assert outputs[
        regrade.regraded_path(regrade.answer_sidecar_path(results_path, "code"))
    ]["model"]["answers"][0]["tests_passed"] == 1


def test_build_regraded_outputs_stops_before_scoring_on_bank_mismatch(
        tmp_path, monkeypatch):
    results_path, _ = make_result_set(tmp_path, monkeypatch)
    results = json.loads(results_path.read_text())
    results["bank_versions"]["math"] = "outdated"
    write_json(results_path, results)
    monkeypatch.setattr(
        regrade, "evaluate_raw_answer",
        lambda *args: pytest.fail("mismatched banks must not be scored"),
    )

    with pytest.raises(regrade.RegradeError, match="different question banks"):
        regrade.build_regraded_outputs(results_path)
    assert not list(tmp_path.glob("regraded_*"))


def test_build_regraded_outputs_requires_every_accuracy_block(tmp_path, monkeypatch):
    results_path, _ = make_result_set(tmp_path, monkeypatch)
    results = json.loads(results_path.read_text())
    results.pop("tool")
    write_json(results_path, results)
    with pytest.raises(regrade.RegradeError, match="has no tool grading block"):
        regrade.build_regraded_outputs(results_path)


def test_build_regraded_outputs_skips_unrun_empty_accuracy_blocks(tmp_path, monkeypatch):
    results_path, _ = make_result_set(tmp_path, monkeypatch)
    results = json.loads(results_path.read_text())
    for workload in ("math", "code", "tool"):
        results[workload] = {}
        regrade.answer_sidecar_path(results_path, workload).unlink()
    write_json(results_path, results)

    outputs = regrade.build_regraded_outputs(results_path)

    assert set(outputs) == {
        regrade.regraded_path(results_path),
        regrade.regraded_path(regrade.answer_sidecar_path(results_path, "mcq")),
    }


def test_build_regraded_outputs_rejects_non_object_empty_like_block(tmp_path, monkeypatch):
    results_path, _ = make_result_set(tmp_path, monkeypatch)
    results = json.loads(results_path.read_text())
    results["math"] = []
    write_json(results_path, results)
    with pytest.raises(regrade.RegradeError, match="math results must be a JSON object"):
        regrade.build_regraded_outputs(results_path)


def test_write_outputs_refuses_existing_output_without_changing_it(tmp_path):
    first = tmp_path / "regraded_first.json"
    second = tmp_path / "regraded_second.json"
    first.write_text("keep me")

    with pytest.raises(regrade.RegradeError, match="Refusing to overwrite"):
        regrade.write_outputs({first: {"new": True}, second: {"new": True}})

    assert first.read_text() == "keep me"
    assert not second.exists()


def test_write_outputs_writes_complete_json_files(tmp_path):
    outputs = {
        tmp_path / "regraded_one.json": {"one": [1, 2]},
        tmp_path / "regraded_two.json": {"two": {"complete": True}},
    }
    regrade.write_outputs(outputs)
    assert {path: json.loads(path.read_text()) for path in outputs} == outputs
    assert not list(tmp_path.glob(".*.tmp"))


def test_write_outputs_cleans_staged_files_after_serialization_failure(tmp_path):
    output = tmp_path / "regraded_invalid.json"
    with pytest.raises(TypeError):
        regrade.write_outputs({output: {"not_json": {1, 2}}})
    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_score_changes_reports_only_changed_model_totals():
    original = {workload: {"model": {"correct": 0, "answered": 0}}
                for workload in regrade.WORKLOADS}
    updated = {workload: {"model": {"correct": 0, "answered": 0}}
               for workload in regrade.WORKLOADS}
    updated["math"]["model"] = {"correct": 1, "answered": 1}
    assert regrade.score_changes(original, updated) == [
        "math/model: correct 0 -> 1, answered 0 -> 1"
    ]
