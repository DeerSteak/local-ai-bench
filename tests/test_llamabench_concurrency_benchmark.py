"""Tests for LlamaBenchConcurrencyBenchmark — see docs/testing.md."""

import json
import subprocess
import threading
from pathlib import Path

import pytest

from scripts.runtime import config
from scripts.runtime.engines.llamacpp import LlamaCppEngine
from scripts.workloads.llamabench_concurrency_benchmark import LlamaBenchConcurrencyBenchmark as LBC
from scripts.runtime.shared import Shared


# ══════════════════════════════════════════════════════════════════════════
#  Binary resolution
# ══════════════════════════════════════════════════════════════════════════


def test_find_binary_via_llamacpp_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.workloads.llamabench_concurrency_benchmark.platform.system", lambda: "Linux")
    monkeypatch.setattr(config, "LLAMACPP_DIR", tmp_path)
    nested = tmp_path / "build" / "bin"
    nested.mkdir(parents=True)
    exe = nested / "llama-batched-bench"
    exe.write_text("")
    monkeypatch.setattr("scripts.workloads.llamabench_concurrency_benchmark.shutil.which", lambda name: None)
    assert LBC.find_binary() == str(exe)


def test_find_binary_skips_a_same_named_source_directory(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.workloads.llamabench_concurrency_benchmark.platform.system", lambda: "Linux")
    monkeypatch.setattr(config, "LLAMACPP_DIR", tmp_path)
    (tmp_path / "tools" / "llama-batched-bench").mkdir(parents=True)
    nested = tmp_path / "build" / "bin"
    nested.mkdir(parents=True)
    exe = nested / "llama-batched-bench"
    exe.write_text("")
    monkeypatch.setattr("scripts.workloads.llamabench_concurrency_benchmark.shutil.which", lambda name: None)
    assert LBC.find_binary() == str(exe)


def test_find_binary_falls_back_to_path(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.workloads.llamabench_concurrency_benchmark.platform.system", lambda: "Linux")
    monkeypatch.setattr(config, "LLAMACPP_DIR", tmp_path / "nonexistent")
    monkeypatch.setattr(
        "scripts.workloads.llamabench_concurrency_benchmark.shutil.which",
        lambda name: "/usr/local/bin/llama-batched-bench" if name == "llama-batched-bench" else None,
    )
    assert LBC.find_binary() == "/usr/local/bin/llama-batched-bench"


def test_find_binary_checks_macos_homebrew_prefixes(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.workloads.llamabench_concurrency_benchmark.platform.system", lambda: "Darwin")
    monkeypatch.setattr(config, "LLAMACPP_DIR", tmp_path / "nonexistent")
    monkeypatch.setattr("scripts.workloads.llamabench_concurrency_benchmark.shutil.which", lambda name: None)

    real_is_file = Path.is_file

    def fake_is_file(self):
        return str(self) == "/opt/homebrew/bin/llama-batched-bench" or real_is_file(self)

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    assert LBC.find_binary() == "/opt/homebrew/bin/llama-batched-bench"


def test_find_binary_returns_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.workloads.llamabench_concurrency_benchmark.platform.system", lambda: "Linux")
    monkeypatch.setattr(config, "LLAMACPP_DIR", tmp_path / "nonexistent")
    monkeypatch.setattr("scripts.workloads.llamabench_concurrency_benchmark.shutil.which", lambda name: None)
    assert LBC.find_binary() is None


# ══════════════════════════════════════════════════════════════════════════
#  fit_npl
# ══════════════════════════════════════════════════════════════════════════


def test_fit_npl_keeps_everything_when_context_is_ample():
    pp, npl = LBC.fit_npl(8192, [128, 512], [1, 2, 4, 8, 16], 1_000_000)
    assert pp == 8192
    assert npl == [1, 2, 4, 8, 16]


def test_fit_npl_drops_levels_that_would_not_fit():
    # 8192+512 = 8704 per sequence; 32768 // 8704 = 3, so only 1 and 2 fit.
    pp, npl = LBC.fit_npl(8192, [128, 512], [1, 2, 4, 8, 16], 32768)
    assert pp == 8192
    assert npl == [1, 2]


def test_fit_npl_clamps_prompt_depth_to_model_max():
    pp, npl = LBC.fit_npl(8192, [128, 512], [1, 2, 4], 4096)
    assert pp == 4096 - 512
    assert npl == [1]


def test_fit_npl_falls_back_to_single_sequence_when_nothing_fits():
    pp, npl = LBC.fit_npl(8192, [128, 512], [2, 4, 8], 4096)
    assert npl == [1]
    assert pp == 3584


def test_fit_npl_never_returns_a_nonpositive_prompt_depth():
    pp, npl = LBC.fit_npl(8192, [128, 512], [1, 2], 256)
    assert pp == 1
    assert npl == [1]


def test_fit_npl_uses_the_largest_tg_value_for_headroom():
    """Headroom must come from max(tg), not the first/smallest one."""
    pp, _ = LBC.fit_npl(8192, [512, 128], [1], 8192)
    assert pp == 8192 - 512


# ══════════════════════════════════════════════════════════════════════════
#  build_command
# ══════════════════════════════════════════════════════════════════════════


def test_build_command_shape():
    cmd = LBC.build_command(
        "llama-batched-bench", Path("/models/x.gguf"), 4096, 512, [128, 512], [1, 2, 4], 2048, 512, 999,
    )
    assert cmd == [
        "llama-batched-bench", "-m", "/models/x.gguf",
        "-c", "4096",
        "-npp", "512", "-ntg", "128,512", "-npl", "1,2,4",
        "-b", "2048", "-ub", "512",
        "-ngl", "999", "--split-mode", "layer",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--output-format", "jsonl",
    ]


def test_build_command_has_no_progress_or_reps_flags():
    cmd = LBC.build_command(
        "llama-batched-bench", Path("/models/x.gguf"), 4096, 512, [128], [1], 2048, 512, 999,
    )
    assert "--progress" not in cmd
    assert "-r" not in cmd


def test_build_command_cpu_only_ngl():
    cmd = LBC.build_command(
        "llama-batched-bench", Path("/models/x.gguf"), 4096, 512, [128], [1], 2048, 512, 0,
    )
    assert cmd[cmd.index("-ngl") + 1] == "0"
    assert cmd[cmd.index("--cache-type-k") + 1] == config.LLAMACPP_KV_CACHE_TYPE


def test_build_command_can_disable_repacking(monkeypatch):
    monkeypatch.setattr(config, "LLAMACPP_NO_REPACK", True)
    cmd = LBC.build_command(
        "llama-batched-bench", Path("/models/x.gguf"), 4096, 512,
        [128], [1], 2048, 512, 999,
    )
    assert "--no-repack" in cmd


def test_build_command_uses_f16_cache_for_tensor_split(monkeypatch):
    monkeypatch.setattr(config, "LLAMACPP_GPU_SPLIT_MODE", "tensor")
    cmd = LBC.build_command(
        "llama-batched-bench", Path("/models/x.gguf"), 4096, 512,
        [128], [1], 2048, 512, 999,
    )
    assert cmd[cmd.index("--cache-type-k") + 1] == "f16"
    assert cmd[cmd.index("--cache-type-v") + 1] == "f16"


# ══════════════════════════════════════════════════════════════════════════
#  run_one
# ══════════════════════════════════════════════════════════════════════════


class _FakePopen:
    """Mimics the slice of subprocess.Popen's interface run_one drains: line-iterable
    stdout/stderr pipes, poll(), wait(), and kill(). `hang=True` simulates a process that
    never finishes on its own, so run_one's idle watchdog is what has to kill it."""

    def __init__(self, returncode=0, stdout_lines=(), stderr_lines=(), hang=False):
        self.returncode = returncode
        self.stdout = iter(stdout_lines)
        self.stderr = iter(stderr_lines)
        self._hang = hang
        self.killed = False

    def poll(self):
        if self._hang and not self.killed:
            return None
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.killed = True


def _row(pl, speed_tg=10.0, tg=128):
    return {"pp": 8192, "tg": tg, "pl": pl, "speed_tg": speed_tg}


def _popen_factory(proc):
    return lambda cmd, stdout, stderr, text: proc


def test_run_one_parses_jsonl_rows(monkeypatch):
    lines = [json.dumps(_row(1)) + "\n", json.dumps(_row(2)) + "\n"]
    monkeypatch.setattr("scripts.workloads.llamabench_concurrency_benchmark.subprocess.Popen",
                        _popen_factory(_FakePopen(0, stdout_lines=lines)))
    entries = LBC.run_one("b", Path("/x.gguf"), 4096, 8192, [128], [1, 2], 2048, 512, 999, 60)
    assert [e["pl"] for e in entries] == [1, 2]


def test_run_one_reports_progress_per_row(monkeypatch):
    lines = [json.dumps(_row(1, 40.0)) + "\n", json.dumps(_row(4, 120.0)) + "\n"]
    monkeypatch.setattr("scripts.workloads.llamabench_concurrency_benchmark.subprocess.Popen",
                        _popen_factory(_FakePopen(0, stdout_lines=lines)))
    seen = []
    LBC.run_one("b", Path("/x.gguf"), 4096, 8192, [128], [1, 4], 2048, 512, 999, 60, on_progress=seen.append)
    assert seen == [
        "pp8192+tg128 @ 1-way: 40.0 tok/s aggregate",
        "pp8192+tg128 @ 4-way: 120.0 tok/s aggregate",
    ]


def test_run_one_succeeds_with_empty_stderr(monkeypatch):
    """This build is silent on stderr — an empty stderr must not be treated as a failure."""
    monkeypatch.setattr("scripts.workloads.llamabench_concurrency_benchmark.subprocess.Popen",
                        _popen_factory(_FakePopen(0, stdout_lines=[json.dumps(_row(1)) + "\n"], stderr_lines=[])))
    assert len(LBC.run_one("b", Path("/x.gguf"), 4096, 8192, [128], [1], 2048, 512, 999, 60)) == 1


def test_run_one_skips_blank_and_malformed_lines(monkeypatch):
    lines = ["\n", "warming up...\n", json.dumps(_row(1)) + "\n", "{broken\n", "   \n"]
    monkeypatch.setattr("scripts.workloads.llamabench_concurrency_benchmark.subprocess.Popen",
                        _popen_factory(_FakePopen(0, stdout_lines=lines)))
    entries = LBC.run_one("b", Path("/x.gguf"), 4096, 8192, [128], [1], 2048, 512, 999, 60)
    assert entries == [_row(1)]


def test_run_one_raises_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr("scripts.workloads.llamabench_concurrency_benchmark.subprocess.Popen",
                        _popen_factory(_FakePopen(1, stderr_lines=["error: out of memory\n"])))
    with pytest.raises(RuntimeError, match="out of memory"):
        LBC.run_one("b", Path("/x.gguf"), 4096, 8192, [128], [1], 2048, 512, 999, 60)


def test_run_one_raises_when_no_entries_parsed(monkeypatch):
    monkeypatch.setattr("scripts.workloads.llamabench_concurrency_benchmark.subprocess.Popen",
                        _popen_factory(_FakePopen(0, stdout_lines=["not json\n"])))
    with pytest.raises(RuntimeError, match="no parseable JSONL output"):
        LBC.run_one("b", Path("/x.gguf"), 4096, 8192, [128], [1], 2048, 512, 999, 60)


def test_run_one_propagates_timeout(monkeypatch):
    fake_proc = _FakePopen(stdout_lines=[json.dumps(_row(1)) + "\n"], hang=True)
    monkeypatch.setattr("scripts.workloads.llamabench_concurrency_benchmark.subprocess.Popen", _popen_factory(fake_proc))
    monkeypatch.setattr(LBC, "IDLE_POLL_INTERVAL", 0.001)
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        LBC.run_one("b", Path("/x.gguf"), 4096, 8192, [128], [1], 2048, 512, 999, 0.01)
    assert fake_proc.killed
    assert getattr(caught.value, "partial_entries") == [_row(1)]


def test_run_one_no_timeout_when_output_keeps_arriving(monkeypatch):
    fake_proc = _FakePopen(0, stdout_lines=[json.dumps(_row(1)) + "\n"])
    monkeypatch.setattr("scripts.workloads.llamabench_concurrency_benchmark.subprocess.Popen", _popen_factory(fake_proc))
    entries = LBC.run_one("b", Path("/x.gguf"), 4096, 8192, [128], [1], 2048, 512, 999, 60)
    assert len(entries) == 1
    assert not fake_proc.killed


def test_run_one_delivers_callbacks_on_the_calling_thread(monkeypatch):
    """Matches LlamaBenchBenchmark.run_one: callbacks persist results, and a journal's SQLite
    connection rejects use from any thread but its creator's."""
    lines = [json.dumps(_row(1)) + "\n", json.dumps(_row(2)) + "\n"]
    monkeypatch.setattr("scripts.workloads.llamabench_concurrency_benchmark.subprocess.Popen",
                        _popen_factory(_FakePopen(0, stdout_lines=lines)))
    caller = threading.current_thread().ident
    seen = []
    LBC.run_one("b", Path("/x.gguf"), 4096, 8192, [128], [1, 2], 2048, 512, 999, 60,
                on_entry=lambda _e: seen.append(("entry", threading.current_thread().ident)),
                on_progress=lambda _m: seen.append(("progress", threading.current_thread().ident)))
    assert seen, "callbacks must still fire"
    assert all(ident == caller for _kind, ident in seen)
    # on_progress still trails its own entry, as it did when both ran on the drain thread.
    assert [kind for kind, _ in seen] == ["entry", "progress", "entry", "progress"]


def test_run_one_propagates_entry_checkpoint_failure(monkeypatch):
    fake_proc = _FakePopen(0, stdout_lines=[json.dumps(_row(1)) + "\n"])
    monkeypatch.setattr("scripts.workloads.llamabench_concurrency_benchmark.subprocess.Popen", _popen_factory(fake_proc))
    with pytest.raises(OSError, match="disk full"):
        LBC.run_one(
            "b", Path("/x.gguf"), 4096, 8192, [128], [1], 2048, 512, 999, 60,
            on_entry=lambda entry: (_ for _ in ()).throw(OSError("disk full")),
        )


# ══════════════════════════════════════════════════════════════════════════
#  format_entry
# ══════════════════════════════════════════════════════════════════════════


def test_format_entry():
    assert LBC.format_entry({"pp": 8192, "tg": 512, "pl": 4, "speed_tg": 178.93}) == \
        "pp8192+tg512 @ 4-way: 178.9 tok/s aggregate"


def test_format_entry_tolerates_missing_fields():
    assert LBC.format_entry({}) == "pp0+tg0 @ 0-way: 0.0 tok/s aggregate"


# ══════════════════════════════════════════════════════════════════════════
#  run() dispatch
# ══════════════════════════════════════════════════════════════════════════


class _NotLlamaCppEngine:
    name = "other-engine"


@pytest.fixture
def fake_engine(monkeypatch):
    """A real LlamaCppEngine (for the isinstance check) with pulled/resolve/ctx mocked."""
    engine = LlamaCppEngine()
    monkeypatch.setattr(LlamaCppEngine, "model_pulled", lambda self, tag: True)
    monkeypatch.setattr(LlamaCppEngine, "max_context_length", lambda self, tag: 1_000_000)
    monkeypatch.setattr(
        LlamaCppEngine, "_resolve_model_files",
        classmethod(lambda cls, tag: [Path(f"/models/{tag}.gguf")]),
    )
    monkeypatch.setattr(LBC, "find_binary", staticmethod(lambda: "llama-batched-bench"))
    return engine


_MODELS = [{"tag": "m1", "label": "Model One", "short": "m1"}]


def test_run_skips_non_llamacpp_engine():
    assert LBC().run(_NotLlamaCppEngine(), _MODELS) == {}


def test_run_returns_empty_when_binary_missing(monkeypatch):
    monkeypatch.setattr(LBC, "find_binary", staticmethod(lambda: None))
    assert LBC().run(LlamaCppEngine(), _MODELS) == {}


def test_run_skips_unpulled_models(fake_engine, monkeypatch):
    monkeypatch.setattr(LlamaCppEngine, "model_pulled", lambda self, tag: False)
    assert LBC().run(fake_engine, _MODELS) == {}


def test_run_records_error_when_files_missing(fake_engine, monkeypatch):
    monkeypatch.setattr(LlamaCppEngine, "_resolve_model_files", classmethod(lambda cls, tag: None))
    assert LBC().run(fake_engine, _MODELS)["m1"] == {"error": "model files not found"}


def test_run_records_entries_and_sweep_shape_on_success(fake_engine, monkeypatch):
    fake_entries = [_row(1), _row(2)]
    monkeypatch.setattr(LBC, "run_one", classmethod(lambda cls, *a, **kw: fake_entries))
    result = LBC().run(fake_engine, _MODELS)
    assert result["m1"]["entries"] == fake_entries
    assert result["m1"]["pp"] == config.LLAMABENCH_CONC_PP
    expected_ctx = max(config.LLAMABENCH_CONC_NPL) * (
        config.LLAMABENCH_CONC_PP + max(config.LLAMABENCH_CONC_TG))
    assert result["m1"]["ctx_size"] == expected_ctx


def test_run_attaches_memory_to_each_delivered_native_case(fake_engine, monkeypatch):
    class Telemetry:
        def __init__(self): self.calls = []
        def begin_model_load(self): self.calls.append("load")
        def begin_measured(self, name): self.calls.append(name)
        def finish_case(self):
            self.calls.append("finish")
            return {"summary": {"process_rss_gb": {"peak_gb": len(self.calls)}}}

    rows = [_row(1), _row(2)]
    monkeypatch.setattr(LBC, "run_one", classmethod(lambda cls, *a, **kw: rows))
    telemetry = Telemetry()
    result = LBC().run(fake_engine, _MODELS, telemetry=telemetry)
    assert telemetry.calls[0] == "load"
    assert telemetry.calls.count("finish") == 2
    assert all("memory" in entry for entry in result["m1"]["entries"])


def test_run_sizes_ctx_and_npl_from_the_model_context(fake_engine, monkeypatch):
    monkeypatch.setattr(LlamaCppEngine, "max_context_length", lambda self, tag: 32768)
    captured = {}

    def fake_run_one(cls, binary, model_path, ctx_size, pp, tg, npl, *a, **kw):
        captured.update(ctx_size=ctx_size, pp=pp, npl=npl)
        return [_row(1)]

    monkeypatch.setattr(LBC, "run_one", classmethod(fake_run_one))
    result = LBC().run(fake_engine, _MODELS)
    assert captured["npl"] == [1, 2, 4]
    assert captured["pp"] == 4096
    assert captured["ctx_size"] == 4 * (4096 + 512)
    assert result["m1"]["ctx_size"] == 4 * (4096 + 512)


def test_run_clamps_prompt_depth_on_small_context_models(fake_engine, monkeypatch):
    monkeypatch.setattr(LlamaCppEngine, "max_context_length", lambda self, tag: 4096)
    captured = {}

    def fake_run_one(cls, binary, model_path, ctx_size, pp, tg, npl, *a, **kw):
        captured.update(ctx_size=ctx_size, pp=pp, npl=npl)
        return [_row(1)]

    monkeypatch.setattr(LBC, "run_one", classmethod(fake_run_one))
    result = LBC().run(fake_engine, _MODELS)
    assert captured["pp"] == 3584
    assert captured["npl"] == [1]
    assert captured["ctx_size"] == 4096
    assert result["m1"]["pp"] == 3584


def test_run_passes_on_progress_to_run_one(fake_engine, monkeypatch):
    captured = {}

    def fake_run_one(cls, *a, on_progress=None, **kw):
        captured["on_progress"] = on_progress
        return [_row(1)]

    monkeypatch.setattr(LBC, "run_one", classmethod(fake_run_one))
    LBC().run(fake_engine, _MODELS)
    assert captured["on_progress"] is Shared.log


def test_run_calls_save_fn_after_each_model(fake_engine, monkeypatch):
    monkeypatch.setattr(LBC, "run_one", classmethod(lambda cls, *a, **kw: [_row(1)]))
    saved = []
    LBC().run(fake_engine, _MODELS, save_fn=lambda partial: saved.append(dict(partial)))
    assert len(saved) >= 1
    assert saved[0]["m1"]["entries"] == [_row(1)]


def test_run_records_timeout_error(fake_engine, monkeypatch):
    def fake_run_one(cls, *a, **kw):
        raise subprocess.TimeoutExpired(cmd=["llama-batched-bench"], timeout=config.LLAMABENCH_TIMEOUT)

    monkeypatch.setattr(LBC, "run_one", classmethod(fake_run_one))
    assert "no output" in LBC().run(fake_engine, _MODELS)["m1"]["error"]


def test_run_records_generic_exception(fake_engine, monkeypatch):
    def fake_run_one(cls, *a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(LBC, "run_one", classmethod(fake_run_one))
    assert LBC().run(fake_engine, _MODELS)["m1"]["error"] == "boom"


def test_run_one_model_failure_does_not_stop_the_rest(fake_engine, monkeypatch):
    models = [{"tag": "bad", "label": "Bad", "short": "bad"}, {"tag": "good", "label": "Good", "short": "good"}]

    def fake_run_one(cls, binary, model_path, *a, **kw):
        if "bad" in str(model_path):
            raise RuntimeError("boom")
        return [_row(1)]

    monkeypatch.setattr(LBC, "run_one", classmethod(fake_run_one))
    result = LBC().run(fake_engine, models)
    assert result["bad"]["error"] == "boom"
    assert "entries" in result["good"]


def test_run_passes_full_offload_ngl_by_default(fake_engine, monkeypatch):
    captured = {}

    def fake_run_one(cls, binary, model_path, ctx_size, pp, tg, npl, batch_size, ubatch_size,
                     ngl, timeout, on_progress=None, on_entry=None):
        captured["ngl"] = ngl
        return [_row(1)]

    monkeypatch.setattr(LBC, "run_one", classmethod(fake_run_one))
    LBC().run(fake_engine, _MODELS, cpu_only=False)
    assert captured["ngl"] == config.LLAMABENCH_FULL_OFFLOAD_NGL


def test_run_passes_zero_ngl_when_cpu_only(fake_engine, monkeypatch):
    captured = {}

    def fake_run_one(cls, binary, model_path, ctx_size, pp, tg, npl, batch_size, ubatch_size,
                     ngl, timeout, on_progress=None, on_entry=None):
        captured["ngl"] = ngl
        return [_row(1)]

    monkeypatch.setattr(LBC, "run_one", classmethod(fake_run_one))
    LBC().run(fake_engine, _MODELS, cpu_only=True)
    assert captured["ngl"] == 0
