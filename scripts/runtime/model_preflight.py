"""Static and runtime model compatibility preflight policy."""

import time
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from scripts.setup.model_compatibility import (
    CompatibilityCheck, ModelCompatibility, chat_template_check,
    context_capacity_check, declared_context_length, preflight_verdict,
    tool_support_check, weight_completeness_check,
)


TOOL_TESTS = {"tool", "conc_tool"}
FORMAT_PROBE_MESSAGES = [
    {"role": "system", "content": "Reply with the single word ready."},
    {"role": "user", "content": "Confirm readiness."},
]
RAW_TEMPLATE_PATTERN = re.compile(r"(?:\{[{%].*?[}%]\}|<\|[^>]+\|>)", re.DOTALL)


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
    if read_error and engine.model_artifacts_are_local():
        weight_check = CompatibilityCheck(
            "weights", "unreadable", "hard_failure",
            f"Model metadata is unreadable: {read_error}",
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
        (check.detail for check in checks
         if check.severity in {"hard_failure", "workload_blocking", "warning"}),
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


def formatting_response_check(response_text: str, finish_reason: str | None) -> CompatibilityCheck:
    if not response_text.strip():
        return CompatibilityCheck(
            "formatting_probe", "empty", "hard_failure", "Formatting probe returned no text.",
        )
    if RAW_TEMPLATE_PATTERN.search(response_text):
        return CompatibilityCheck(
            "formatting_probe", "raw_markup", "hard_failure",
            "Formatting probe emitted raw chat-template markup.",
        )
    return CompatibilityCheck(
        "formatting_probe", "passed", "info", "Formatting round-trip passed.",
        evidence={"response_nonempty": True, "finish_reason": finish_reason or "not_reported"},
    )


def run_formatting_probe(engine, tag: str, num_ctx: int, timeout: int) -> CompatibilityCheck:
    if not getattr(engine, "can_reset_model_state", lambda: True)():
        return CompatibilityCheck(
            "formatting_probe", "clean_state_unavailable", "hard_failure",
            "The engine cannot reset externally managed model state after the probe.",
        )
    try:
        measurement = engine.chat(
            tag, FORMAT_PROBE_MESSAGES, timeout=timeout, num_ctx=num_ctx, num_predict=8,
        )
        check = formatting_response_check(measurement.response_text, measurement.finish_reason)
    except Exception as exc:
        check = CompatibilityCheck(
            "formatting_probe", "load_or_request_failed", "hard_failure", str(exc),
        )
    try:
        engine.unload(tag)
        clean = engine.wait_until_unloaded(tag)
    except Exception as exc:
        return CompatibilityCheck(
            "formatting_probe", "cleanup_failed", "hard_failure",
            f"Clean-state reset failed after formatting probe: {exc}",
        )
    if not clean:
        return CompatibilityCheck(
            "formatting_probe", "cleanup_failed", "hard_failure",
            "Model remained loaded after formatting probe.",
        )
    return check


def add_runtime_check(report: ModelCompatibility,
                      check: CompatibilityCheck) -> ModelCompatibility:
    checks = (*report.checks, check)
    status = "excluded" if check.severity == "hard_failure" else report.status
    detail = check.detail if check.severity == "hard_failure" else report.detail
    return ModelCompatibility(
        report.engine, report.tag, report.architecture, status, detail, checks,
    )


def run_runtime_preflight(outcome: PreflightOutcome, engine, num_ctx: int, timeout: int,
                          *, monotonic=time.monotonic) -> PreflightOutcome:
    started = monotonic()
    reports = tuple(
        add_runtime_check(report, run_formatting_probe(engine, report.tag, num_ctx, timeout))
        if report.tag in outcome.runnable_tags else report
        for report in outcome.reports
    )
    runnable = frozenset(report.tag for report in reports if report.status != "excluded")
    return PreflightOutcome(
        reports, runnable, outcome.blocked_workloads,
        outcome.elapsed_seconds + monotonic() - started,
    )
