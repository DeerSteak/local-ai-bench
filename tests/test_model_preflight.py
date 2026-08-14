from scripts.runtime.model_preflight import (
    build_static_report, filter_models, maximum_requested_context, run_static_preflight,
)


class FakeEngine:
    name = "fake"

    def __init__(self, metadata=None, pulled=True, paths=(), tools=True, max_context=8192):
        self.metadata = metadata or {
            "general.architecture": "muse", "tokenizer.chat_template": "{{ messages }}",
            "muse.context_length": max_context,
        }
        self.pulled = pulled
        self.paths = paths
        self.tools = tools
        self.max_context = max_context

    def compatibility_metadata(self, _tag): return self.metadata, None
    def model_pulled(self, _tag): return self.pulled
    def model_paths(self, _tag): return self.paths
    def model_artifacts_are_local(self) -> bool: return True
    def supports_tool_calls(self, _tag): return self.tools
    def max_context_length(self, _tag, default): return self.max_context or default


def test_static_report_records_all_passing_checks(tmp_path):
    weight = tmp_path / "model.gguf"
    weight.write_bytes(b"weights")
    report = build_static_report(FakeEngine(paths=(weight,)), {"tag": "muse"}, ["llm"], 4096, False)
    assert report.status == "passed"
    assert [check.name for check in report.checks] == [
        "weights", "chat_template", "context_capacity", "tool_calls",
    ]
    assert all(check.status in {"passed", "not_applicable"} for check in report.checks)


def test_hard_failure_excludes_only_affected_model_even_with_force_all(tmp_path):
    weight = tmp_path / "model.gguf"
    weight.write_bytes(b"weights")

    class PerModelEngine(FakeEngine):
        def model_pulled(self, tag): return tag == "good"
        def model_paths(self, tag): return (weight,) if tag == "good" else ()

    outcome = run_static_preflight(
        PerModelEngine(), [{"tag": "bad"}, {"tag": "good"}], ["llm"], 4096, True,
        monotonic=iter((2.0, 2.25)).__next__,
    )
    assert outcome.runnable_tags == {"good"}
    assert outcome.elapsed_seconds == 0.25
    assert filter_models([{"tag": "bad"}, {"tag": "good"}], outcome.runnable_tags) == [
        {"tag": "good"}
    ]


def test_unsupported_tools_limit_workload_without_excluding_model(tmp_path):
    weight = tmp_path / "model.gguf"
    weight.write_bytes(b"weights")
    outcome = run_static_preflight(
        FakeEngine(paths=(weight,), tools=False), [{"tag": "muse"}], ["tool"], 4096, False,
    )
    assert outcome.runnable_tags == {"muse"}
    assert outcome.blocked_workloads["tool"] == {"muse"}
    assert outcome.reports[0].status == "workload_limited"


def test_force_all_bypasses_context_warning(tmp_path):
    weight = tmp_path / "model.gguf"
    weight.write_bytes(b"weights")
    engine = FakeEngine(paths=(weight,), max_context=4096)
    assert build_static_report(engine, {"tag": "muse"}, ["llm"], 8192, False).status == "warning"
    assert build_static_report(engine, {"tag": "muse"}, ["llm"], 8192, True).status == "passed"


def test_external_runtime_records_unavailable_weights_without_hard_failure():
    class ExternalEngine(FakeEngine):
        def model_artifacts_are_local(self) -> bool: return False

    engine = ExternalEngine(paths=())
    report = build_static_report(engine, {"tag": "muse"}, ["llm"], 4096, False)
    assert report.status == "warning"
    assert report.checks[0].status == "unavailable"


def test_maximum_requested_context_uses_only_selected_workloads():
    contexts = {"llm": [512, 8192], "conv": [2048, 98304], "tool": [4096]}
    assert maximum_requested_context(["llm", "tool"], contexts) == 8192
    assert maximum_requested_context(["conv"], contexts) == 98304
    assert maximum_requested_context(["img"], contexts) is None
