"""Regrade stored accuracy responses without modifying their source files."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import config
from code_benchmark import CodeBenchmark
from math_benchmark import MathBenchmark
from mcq_benchmark import MCQBenchmark
from reasoning_benchmark import ReasoningBenchmark
from shared import Shared
from tool_benchmark import ToolBenchmark


WORKLOADS = {
    "mcq": (MCQBenchmark.MCQ_DATA_PATH, MCQBenchmark.load_questions),
    "math": (MathBenchmark.MATH_DATA_PATH, MathBenchmark.load_questions),
    "reasoning": (ReasoningBenchmark.REASONING_DATA_PATH, ReasoningBenchmark.load_questions),
    "code": (CodeBenchmark.CODE_DATA_PATH, CodeBenchmark.load_questions),
    "tool": (ToolBenchmark.TOOL_DATA_PATH, ToolBenchmark.load_questions),
}


class RegradeError(Exception):
    """A stored result set cannot be regraded safely."""


def answer_sidecar_path(results_path: Path, workload: str) -> Path:
    name = results_path.name
    suffix = name[len("results_"):] if name.startswith("results_") else name
    return results_path.with_name(f"answers_{workload}_{suffix}")


def regraded_path(path: Path) -> Path:
    return path.with_name(f"regraded_{path.name}")


def load_json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegradeError(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RegradeError(f"Expected a JSON object in {path}")
    return value


def recorded_workloads(results: dict, results_path: Path) -> list[str]:
    recorded = results.get("bank_versions")
    if not isinstance(recorded, dict):
        raise RegradeError(f"{results_path} has no bank_versions object")
    workloads = [name for name in WORKLOADS if name in recorded]
    if not workloads:
        raise RegradeError(f"{results_path} has no recognized accuracy bank versions")
    return workloads


def current_bank_hashes(workloads=None) -> dict[str, str]:
    workloads = WORKLOADS if workloads is None else workloads
    return {name: Shared.file_hash(WORKLOADS[name][0]) for name in workloads}


def validate_bank_hashes(results: dict, results_path: Path) -> None:
    recorded = results.get("bank_versions")
    workloads = recorded_workloads(results, results_path)
    current = current_bank_hashes(workloads)
    mismatches = [
        f"{name}: stored {recorded.get(name, 'missing')}, current {current_hash}"
        for name, current_hash in current.items()
        if recorded.get(name) != current_hash
    ]
    if mismatches:
        raise RegradeError(
            f"{results_path} was generated with different question banks:\n  "
            + "\n  ".join(mismatches)
        )


def questions_for_result(results: dict, workload: str) -> list[dict]:
    _, loader = WORKLOADS[workload]
    questions = loader()
    all_sample_ids = results.get("sample_ids", {})
    if not isinstance(all_sample_ids, dict):
        raise RegradeError("sample_ids must be a JSON object")
    sample_ids = all_sample_ids.get(workload)
    if sample_ids is None:
        return questions
    if not isinstance(sample_ids, list) or len(sample_ids) != len(set(sample_ids)):
        raise RegradeError(f"sample_ids.{workload} must be a list of unique IDs")
    by_id = {question["id"]: question for question in questions}
    missing = [question_id for question_id in sample_ids if question_id not in by_id]
    if missing:
        raise RegradeError(f"sample_ids.{workload} contains unknown IDs: {missing}")
    return [by_id[question_id] for question_id in sample_ids]


def tool_calls_from_raw(raw_response: str) -> list:
    try:
        parsed = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list):
        return parsed["tool_calls"]
    return []


def evaluate_raw_answer(workload: str, question: dict, raw_response: str):
    if workload == "mcq":
        return MCQBenchmark.parse_answer(raw_response, question["choices"].keys())
    if workload == "math":
        return MathBenchmark.parse_answer(raw_response)
    if workload == "reasoning":
        return ReasoningBenchmark.parse_answer(raw_response, question["choices"].keys())
    if workload == "code":
        code = CodeBenchmark.extract_code(raw_response)
        return CodeBenchmark.evaluate_question(question, code)
    if workload == "tool":
        return ToolBenchmark.evaluate_question(question, tool_calls_from_raw(raw_response))
    raise RegradeError(f"Unknown workload: {workload}")


def score_answers(workload: str, questions: list[dict], answers: dict) -> dict:
    scorer = {
        "mcq": MCQBenchmark.score,
        "math": MathBenchmark.score,
        "reasoning": ReasoningBenchmark.score,
        "code": CodeBenchmark.score,
        "tool": ToolBenchmark.score,
    }[workload]
    return scorer(questions, answers)


def validate_answer_entries(workload: str, model: str, entries, questions: list[dict]) -> dict:
    if not isinstance(entries, list):
        raise RegradeError(f"{workload}/{model} answers must be a list")
    by_id = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise RegradeError(f"{workload}/{model} contains an answer without a valid ID")
        question_id = entry["id"]
        if question_id in by_id:
            raise RegradeError(f"{workload}/{model} repeats answer ID {question_id}")
        if not isinstance(entry.get("raw_response"), str):
            raise RegradeError(f"{workload}/{model}/{question_id} has no raw_response string")
        by_id[question_id] = entry

    expected_ids = [question["id"] for question in questions]
    if set(by_id) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(by_id))
        extra = sorted(set(by_id) - set(expected_ids))
        raise RegradeError(
            f"{workload}/{model} answer IDs do not match the bank "
            f"(missing={missing}, extra={extra})"
        )
    return by_id


def merge_diagnostics(original: dict, scored: dict) -> dict:
    merged = {**original, **{key: value for key, value in scored.items() if key != "all"}}
    if "likely_loop_ids" in merged:
        incorrect_ids = {entry["id"] for entry in scored["incorrect"]}
        loop_ids = [question_id for question_id in merged["likely_loop_ids"]
                    if question_id in incorrect_ids]
        if loop_ids:
            merged["likely_loop_ids"] = loop_ids
            merged["likely_loop_count"] = len(loop_ids)
        else:
            merged.pop("likely_loop_ids", None)
            merged.pop("likely_loop_count", None)
    return merged


def regrade_workload(workload: str, questions: list[dict], result_models: dict,
                     sidecar_models: dict) -> tuple[dict, dict]:
    if not isinstance(result_models, dict) or not isinstance(sidecar_models, dict):
        raise RegradeError(f"{workload} results and sidecar must be JSON objects")
    if set(result_models) != set(sidecar_models):
        raise RegradeError(
            f"{workload} model keys differ between results and sidecar "
            f"(results={sorted(result_models)}, sidecar={sorted(sidecar_models)})"
        )

    regraded_results = {}
    regraded_sidecar = {}
    for model, original_result in result_models.items():
        sidecar_model = sidecar_models[model]
        if not isinstance(original_result, dict) or not isinstance(sidecar_model, dict):
            raise RegradeError(f"{workload}/{model} must be a JSON object")
        entries = validate_answer_entries(
            workload, model, sidecar_model.get("answers"), questions,
        )
        parsed_answers = {}
        for question in questions:
            question_id = question["id"]
            try:
                parsed_answers[question_id] = evaluate_raw_answer(
                    workload, question, entries[question_id]["raw_response"],
                )
            except Exception as exc:
                raise RegradeError(
                    f"Failed to regrade {workload}/{model}/{question_id}: {exc}"
                ) from exc
        scored = score_answers(workload, questions, parsed_answers)
        regraded_results[model] = merge_diagnostics(original_result, scored)
        regraded_sidecar[model] = {
            "label": sidecar_model.get("label", original_result.get("label", model)),
            "answers": [
                {
                    **entry,
                    "raw_response": entries[entry["id"]]["raw_response"],
                }
                for entry in scored["all"]
            ],
        }
    return regraded_results, regraded_sidecar


def build_regraded_outputs(results_path: Path) -> dict[Path, dict]:
    results_path = Path(results_path)
    results = load_json_object(results_path)
    validate_bank_hashes(results, results_path)
    workloads = recorded_workloads(results, results_path)

    for workload in workloads:
        if workload not in results:
            raise RegradeError(f"{results_path} has no {workload} grading block")
        if not isinstance(results[workload], dict):
            raise RegradeError(f"{workload} results must be a JSON object")
    workloads = [workload for workload in workloads if results[workload]]

    output_results = dict(results)
    output_results["version"] = config.VERSION
    outputs = {}
    for workload in workloads:
        sidecar_path = answer_sidecar_path(results_path, workload)
        sidecar = load_json_object(sidecar_path)
        questions = questions_for_result(results, workload)
        regraded_results, regraded_sidecar = regrade_workload(
            workload, questions, results[workload], sidecar,
        )
        output_results[workload] = regraded_results
        outputs[regraded_path(sidecar_path)] = regraded_sidecar

    outputs[regraded_path(results_path)] = output_results
    return outputs


def write_outputs(outputs: dict[Path, dict]) -> None:
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise RegradeError("Refusing to overwrite existing files: " + ", ".join(map(str, existing)))

    staged = []
    try:
        for output_path, data in outputs.items():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp",
            )
            temporary_path = Path(temporary_name)
            staged.append((temporary_path, output_path))
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(data, stream, indent=2, allow_nan=False)
        for temporary_path, output_path in staged:
            os.replace(temporary_path, output_path)
    finally:
        for temporary_path, _ in staged:
            temporary_path.unlink(missing_ok=True)


def score_changes(original: dict, regraded: dict) -> list[str]:
    changes = []
    for workload in WORKLOADS:
        if workload not in original or workload not in regraded:
            continue
        for model, new_score in regraded[workload].items():
            old_score = original[workload][model]
            if (old_score.get("correct"), old_score.get("answered")) != (
                    new_score.get("correct"), new_score.get("answered")):
                changes.append(
                    f"{workload}/{model}: correct {old_score.get('correct')} -> "
                    f"{new_score.get('correct')}, answered {old_score.get('answered')} -> "
                    f"{new_score.get('answered')}"
                )
    return changes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Regrade an existing results file from its raw-answer sidecars.",
    )
    parser.add_argument("results_file", type=Path)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate and regrade in memory without writing output files",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:  # pragma: no cover - thin CLI wrapper
    args = parse_args(argv)
    try:
        original = load_json_object(args.results_file)
        outputs = build_regraded_outputs(args.results_file)
        regraded_results_path = regraded_path(args.results_file)
        changes = score_changes(original, outputs[regraded_results_path])
        if not args.dry_run:
            write_outputs(outputs)
        action = "Validated" if args.dry_run else "Wrote"
        Shared.ok(f"{action} {len(outputs)} regraded files for {args.results_file.name}")
        if changes:
            for change in changes:
                Shared.log(f"  {change}")
        else:
            Shared.log("  No model-level score totals changed")
        return 0
    except RegradeError as exc:
        Shared.err(str(exc))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
