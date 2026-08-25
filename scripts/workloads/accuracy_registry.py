"""Shared metadata for the five accuracy workloads."""

from dataclasses import dataclass
from pathlib import Path

from scripts.workloads.code_benchmark import CodeBenchmark
from scripts.workloads.math_benchmark import MathBenchmark
from scripts.workloads.mcq_benchmark import MCQBenchmark
from scripts.workloads.reasoning_benchmark import ReasoningBenchmark
from scripts.workloads.tool_benchmark import ToolBenchmark


@dataclass(frozen=True)
class AccuracySpec:
    benchmark: type
    data_path: Path


ACCURACY_SPECS = {
    "mcq": AccuracySpec(MCQBenchmark, MCQBenchmark.MCQ_DATA_PATH),
    "math": AccuracySpec(MathBenchmark, MathBenchmark.MATH_DATA_PATH),
    "reasoning": AccuracySpec(ReasoningBenchmark, ReasoningBenchmark.REASONING_DATA_PATH),
    "code": AccuracySpec(CodeBenchmark, CodeBenchmark.CODE_DATA_PATH),
    "tool": AccuracySpec(ToolBenchmark, ToolBenchmark.TOOL_DATA_PATH),
}


def accuracy_spec(stage_name: str) -> AccuracySpec:
    try:
        return ACCURACY_SPECS[stage_name]
    except KeyError as exc:
        raise ValueError(f"unknown accuracy stage: {stage_name}") from exc


def selected_questions(stage_name: str, sample_size: int | None = None) -> list[dict]:
    from scripts.runtime.shared import Shared

    questions = accuracy_spec(stage_name).benchmark.load_questions()
    return Shared.stratified_sample(questions, sample_size) if sample_size is not None else questions
