"""Static and runtime model compatibility preflight policy."""

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from scripts.setup.model_compatibility import (
    CompatibilityCheck, ModelCompatibility, chat_template_check,
    context_capacity_check, declared_context_length, preflight_verdict,
    tool_support_check, weight_completeness_check,
)


TOOL_TESTS = {"tool", "conc_tool"}


@dataclass(frozen=True)
class PreflightOutcome:
    reports: tuple[ModelCompatibility, ...]
    runnable_tags: frozenset[str]
    blocked_workloads: dict[str, frozenset[str]]
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return {
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "models": {report.tag: report.to_dict() for report in self.reports},
        }


def build_static_report(engine, model: dict, tests: list[str] | tuple[str, ...],
                        requested_context: int | None, force_all: bool) -> ModelCompatibility:
    tag = model["tag"]
    metadata, read_error = engine.compatibility_metadata(tag)
    architecture = metadata.get("general.architecture")
    if isinstance(architecture, bytes):
        architecture = architecture.decode("utf-8", errors="replace")
    if not isinstance(architecture, str):
        architecture = None
    if not engine.model_pulled(tag):
        weight_check = CompatibilityCheck(
            "weights", "incomplete", "hard_failure", "Model artifacts are not complete.",
        )
    elif engine.model_artifacts_are_local():
        weight_check = weight_completeness_check(engine.model_paths(tag))
    else:
        weight_check = CompatibilityCheck(
            "weights", "unavailable", "warning",
            "An external runtime does not expose local weight files for inspection.",
        )
    checks = (
        weight_check,
        chat_template_check(metadata, read_error),
        context_capacity_check(
            declared_context_length(metadata), requested_context,
            engine.max_context_length(tag, default=0),
        ),
        tool_support_check(engine.supports_tool_calls(tag), bool(set(tests) & TOOL_TESTS)),
    )
    status = preflight_verdict(checks, force_all)
    detail = next(
        (check.detail for check in checks if check.severity in {"hard_failure", "warning"}),
        "All static compatibility checks passed.",
    )
    return ModelCompatibility(engine.name, tag, architecture, status, detail, checks)


def run_static_preflight(engine, models: list[dict], tests: list[str] | tuple[str, ...],
                         requested_context: int | None, force_all: bool,
                         *, monotonic=time.monotonic) -> PreflightOutcome:
    started = monotonic()
    reports = tuple(
        build_static_report(engine, model, tests, requested_context, force_all)
        for model in models
    )
    runnable = frozenset(report.tag for report in reports if report.status != "excluded")
    blocked = {
        "tool": frozenset(
            report.tag for report in reports
            if any(check.scope == "tool" and check.severity == "workload_blocking"
                   for check in report.checks)
        )
    }
    return PreflightOutcome(reports, runnable, blocked, monotonic() - started)


def filter_models(models: list[dict], allowed_tags: frozenset[str]) -> list[dict]:
    return [model for model in models if model["tag"] in allowed_tags]


def maximum_requested_context(tests: list[str] | tuple[str, ...],
                              contexts_by_test: Mapping[str, Sequence[int]]) -> int | None:
    values = [
        value for test in tests for value in contexts_by_test.get(test, ())
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    ]
    return max(values, default=None)
