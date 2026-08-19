"""Tests for LlamaBenchBenchmark — see docs/testing.md."""

import json
import subprocess
import threading
from pathlib import Path

import pytest

from scripts.runtime import config
from scripts.runtime.engines.llamacpp import LlamaCppEngine
from scripts.workloads.llamabench_benchmark import LlamaBenchBenchmark
from scripts.runtime.shared import Shared


def test_find_binary_via_llamacpp_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.workloads.llamabench_benchmark.platform.system", lambda: "Linux")
    monkeypatch.setattr(config, "LLAMACPP_DIR", tmp_path)
    nested = tmp_path / "build" / "bin"
    nested.mkdir(parents=True)
    exe = nested / "llama-bench"
    exe.write_text("")
    monkeypatch.setattr("scripts.workloads.llamabench_benchmark.shutil.which", lambda name: None)
    assert LlamaBenchBenchmark.find_binary() == str(exe)


def test_find_binary_skips_a_same_named_source_directory(monkeypatch, tmp_path):
    """A CMake source tree has tools/llama-bench/ (source) alongside build/bin/llama-bench
    (compiled) — rglob must not return the directory just because it sorts first."""
    monkeypatch.setattr("scripts.workloads.llamabench_benchmark.platform.system", lambda: "Linux")
    monkeypatch.setattr(config, "LLAMACPP_DIR", tmp_path)
    (tmp_path / "tools" / "llama-bench").mkdir(parents=True)
    nested = tmp_path / "build" / "bin"
    nested.mkdir(parents=True)
    exe = nested / "llama-bench"
    exe.write_text("")
    monkeypatch.setattr("scripts.workloads.llamabench_benchmark.shutil.which", lambda name: None)
    assert LlamaBenchBenchmark.find_binary() == str(exe)


def test_find_binary_falls_back_to_path(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.workloads.llamabench_benchmark.platform.system", lambda: "Linux")
    monkeypatch.setattr(config, "LLAMACPP_DIR", tmp_path / "nonexistent")
    monkeypatch.setattr(
        "scripts.workloads.llamabench_benchmark.shutil.which",
        lambda name: "/usr/local/bin/llama-bench" if name == "llama-bench" else None,
    )
    assert LlamaBenchBenchmark.find_binary() == "/usr/local/bin/llama-bench"


def test_find_binary_checks_macos_homebrew_prefixes(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.workloads.llamabench_benchmark.platform.system", lambda: "Darwin")
    monkeypatch.setattr(config, "LLAMACPP_DIR", tmp_path / "nonexistent")
    monkeypatch.setattr("scripts.workloads.llamabench_benchmark.shutil.which", lambda name: None)

    real_is_file = Path.is_file

    def fake_is_file(self):
        return str(self) == "/opt/homebrew/bin/llama-bench" or real_is_file(self)

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    assert LlamaBenchBenchmark.find_binary() == "/opt/homebrew/bin/llama-bench"


def test_find_binary_returns_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.workloads.llamabench_benchmark.platform.system", lambda: "Linux")
    monkeypatch.setattr(config, "LLAMACPP_DIR", tmp_path / "nonexistent")
    monkeypatch.setattr("scripts.workloads.llamabench_benchmark.shutil.which", lambda name: None)
    assert LlamaBenchBenchmark.find_binary() is None


def test_build_prefill_command_shape():
    cmd = LlamaBenchBenchmark.build_prefill_command(
        "llama-bench", Path("/models/x.gguf"), [512, 2048], 2048, 512, 3, 999,
    )
    assert cmd == [
        "llama-bench", "-m", "/models/x.gguf",
        "-b", "2048", "-ub", "512",
        "-ngl", "999", "--split-mode", "layer",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "-r", "3", "-o", "jsonl",
        "--progress",
        "-p", "512,2048", "-n", "0", "-d", "0",
    ]


def test_build_prefill_command_never_passes_unsupported_repack_option(monkeypatch):
    monkeypatch.setattr(config, "LLAMACPP_NO_REPACK", True)
    cmd = LlamaBenchBenchmark.build_prefill_command(
        "llama-bench", Path("/models/x.gguf"), [512], 2048, 512, 3, 999,
    )
    assert "--no-repack" not in cmd


def test_build_decode_command_shape():
    cmd = LlamaBenchBenchmark.build_decode_command(
        "llama-bench", Path("/models/x.gguf"), [512, 2048], [128, 512], 2048, 512, 3, 999,
    )
    assert cmd == [
        "llama-bench", "-m", "/models/x.gguf",
        "-b", "2048", "-ub", "512",
        "-ngl", "999", "--split-mode", "layer",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "-r", "3", "-o", "jsonl",
        "--progress",
        "-p", "0", "-n", "128,512", "-d", "512,2048",
    ]


def test_native_commands_use_selected_tensor_split(monkeypatch):
    monkeypatch.setattr(config, "LLAMACPP_GPU_SPLIT_MODE", "tensor")
    command = LlamaBenchBenchmark.build_prefill_command(
        "llama-bench", Path("/models/x.gguf"), [512], 2048, 512, 1, 999,
    )
    assert command[command.index("--split-mode") + 1] == "tensor"
    assert command[command.index("--cache-type-k") + 1] == "f16"
    assert command[command.index("--cache-type-v") + 1] == "f16"


def test_commands_suppress_llama_bench_unwanted_defaults():
    prefill = LlamaBenchBenchmark.build_prefill_command(
        "llama-bench", Path("/models/x.gguf"), [2048], 2048, 512, 3, 999,
    )
    decode = LlamaBenchBenchmark.build_decode_command(
        "llama-bench", Path("/models/x.gguf"), [2048], [512], 2048, 512, 3, 999,
    )
    assert prefill[prefill.index("-n") + 1] == "0"
    assert prefill[prefill.index("-d") + 1] == "0"
    assert decode[decode.index("-p") + 1] == "0"
    assert decode[decode.index("-d") + 1] == "2048"
    assert "-pg" not in prefill
    assert "-pg" not in decode


def test_build_decode_command_uses_depth_and_tg_cross_product():
    cmd = LlamaBenchBenchmark.build_decode_command(
        "llama-bench", Path("/models/x.gguf"), [512, 2048], [128, 512], 2048, 512, 3, 999,
    )
    assert cmd[cmd.index("-p") + 1] == "0"
    assert cmd[cmd.index("-n") + 1] == "128,512"
    assert cmd[cmd.index("-d") + 1] == "512,2048"
    assert "-c" not in cmd


def test_build_decode_command_cpu_only_ngl():
    cmd = LlamaBenchBenchmark.build_decode_command(
        "llama-bench", Path("/models/x.gguf"), [512], [128], 2048, 512, 3, 0,
    )
    assert cmd[cmd.index("-ngl") + 1] == "0"
    assert cmd[cmd.index("--cache-type-k") + 1] == config.LLAMACPP_KV_CACHE_TYPE


class _FakePopen:
    """Mimics the process interface run_one drains.
    `hang=True` requires the idle watchdog to kill it."""

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


def test_run_one_parses_jsonl_on_success(monkeypatch):
    fake_entries = [{"n_prompt": 512, "n_gen": 0, "avg_ts": 123.4, "stddev_ts": 1.2, "n_gpu_layers": 999}]
    stdout_line = json.dumps(fake_entries[0]) + "\n"
    monkeypatch.setattr(
        "scripts.workloads.llamabench_benchmark.subprocess.Popen",
        lambda cmd, stdout, stderr, text: _FakePopen(0, stdout_lines=[stdout_line]),
    )
    entries = LlamaBenchBenchmark.run_one(["llama-bench"], 60)
    assert entries == fake_entries


def test_run_one_streams_each_jsonl_result(monkeypatch):
    rows = [
        {"n_prompt": 512, "n_gen": 0, "avg_ts": 1.0},
        {"n_prompt": 2048, "n_gen": 0, "avg_ts": 2.0},
    ]
    monkeypatch.setattr(
        "scripts.workloads.llamabench_benchmark.subprocess.Popen",
        lambda cmd, stdout, stderr, text: _FakePopen(
            0, stdout_lines=[json.dumps(row) + "\n" for row in rows],
        ),
    )
    seen = []
    assert LlamaBenchBenchmark.run_one(["llama-bench"], 60, on_result=seen.append) == rows
    assert seen == rows


def test_run_one_delivers_results_on_the_calling_thread(monkeypatch):
    """on_result writes to the SQLite journal, which rejects use from any thread but its
    creator's — delivering from the stdout drain thread failed every llama-bench model."""
    rows = [{"n_prompt": 512, "n_gen": 0, "avg_ts": 1.0},
            {"n_prompt": 0, "n_gen": 128, "avg_ts": 2.0}]
    monkeypatch.setattr(
        "scripts.workloads.llamabench_benchmark.subprocess.Popen",
        lambda cmd, stdout, stderr, text: _FakePopen(
            0, stdout_lines=[json.dumps(row) + "\n" for row in rows],
        ),
    )
    caller = threading.current_thread().ident
    threads = []
    LlamaBenchBenchmark.run_one(
        ["llama-bench"], 60,
        on_result=lambda _row: threads.append(threading.current_thread().ident),
    )
    assert threads and all(ident == caller for ident in threads)


def test_run_one_delivers_every_row_even_when_the_process_exits_immediately(monkeypatch):
    # A process that never polls as running must still flush its rows to on_result.
    rows = [{"n_prompt": 512, "n_gen": 0, "avg_ts": 1.0},
            {"n_prompt": 1024, "n_gen": 0, "avg_ts": 2.0},
            {"n_prompt": 2048, "n_gen": 0, "avg_ts": 3.0}]
    monkeypatch.setattr(
        "scripts.workloads.llamabench_benchmark.subprocess.Popen",
        lambda cmd, stdout, stderr, text: _FakePopen(
            0, stdout_lines=[json.dumps(row) + "\n" for row in rows],
        ),
    )
    seen = []
    assert LlamaBenchBenchmark.run_one(["llama-bench"], 60, on_result=seen.append) == rows
    assert seen == rows, "rows delivered after exit must preserve order and completeness"


def test_run_one_propagates_stream_callback_failure(monkeypatch):
    row = {"n_prompt": 512, "n_gen": 0, "avg_ts": 1.0}
    monkeypatch.setattr(
        "scripts.workloads.llamabench_benchmark.subprocess.Popen",
        lambda cmd, stdout, stderr, text: _FakePopen(
            0, stdout_lines=[json.dumps(row) + "\n"],
        ),
    )
    with pytest.raises(OSError, match="checkpoint"):
        LlamaBenchBenchmark.run_one(
            ["llama-bench"], 60,
            on_result=lambda _: (_ for _ in ()).throw(OSError("checkpoint")),
        )


def test_run_one_streams_stderr_progress_lines(monkeypatch):
    fake_entries = [{"n_prompt": 512, "n_gen": 0, "avg_ts": 123.4, "stddev_ts": 1.2, "n_gpu_layers": 999}]
    stdout_line = json.dumps(fake_entries[0]) + "\n"
    monkeypatch.setattr(
        "scripts.workloads.llamabench_benchmark.subprocess.Popen",
        lambda cmd, stdout, stderr, text: _FakePopen(
            0, stdout_lines=[stdout_line], stderr_lines=["1/6: pp512\n", "2/6: pp512\n"],
        ),
    )
    seen = []
    LlamaBenchBenchmark.run_one(["llama-bench"], 60, on_progress=seen.append)
    assert seen == ["1/6: pp512", "2/6: pp512"]


def test_run_one_raises_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(
        "scripts.workloads.llamabench_benchmark.subprocess.Popen",
        lambda cmd, stdout, stderr, text: _FakePopen(1, stderr_lines=["error: out of memory\n"]),
    )
    with pytest.raises(RuntimeError, match="out of memory"):
        LlamaBenchBenchmark.run_one(["llama-bench"], 60)


def test_run_one_raises_on_malformed_json(monkeypatch):
    monkeypatch.setattr(
        "scripts.workloads.llamabench_benchmark.subprocess.Popen",
        lambda cmd, stdout, stderr, text: _FakePopen(0, stdout_lines=["not json"]),
    )
    with pytest.raises(RuntimeError, match="unparseable JSON"):
        LlamaBenchBenchmark.run_one(["llama-bench"], 60)


def test_run_one_propagates_timeout(monkeypatch):
    fake_proc = _FakePopen(hang=True)
    monkeypatch.setattr("scripts.workloads.llamabench_benchmark.subprocess.Popen", lambda cmd, stdout, stderr, text: fake_proc)
    monkeypatch.setattr(LlamaBenchBenchmark, "IDLE_POLL_INTERVAL", 0.001)
    with pytest.raises(subprocess.TimeoutExpired):
        LlamaBenchBenchmark.run_one(["llama-bench"], 0.01)
    assert fake_proc.killed


def test_run_one_no_timeout_when_output_keeps_arriving(monkeypatch):
    """A process that never idles for longer than the timeout should complete normally,
    not get killed just because the whole sweep runs long."""
    fake_entries = [{"n_prompt": 512, "n_gen": 0, "avg_ts": 1.0, "stddev_ts": 0.1, "n_gpu_layers": 999}]
    fake_proc = _FakePopen(0, stdout_lines=[json.dumps(fake_entries[0]) + "\n"])
    monkeypatch.setattr("scripts.workloads.llamabench_benchmark.subprocess.Popen", lambda cmd, stdout, stderr, text: fake_proc)
    entries = LlamaBenchBenchmark.run_one(["llama-bench"], 60)
    assert entries == fake_entries
    assert not fake_proc.killed


def test_format_entry_prompt_processing_only():
    line = LlamaBenchBenchmark.format_entry(
        {"n_prompt": 512, "n_gen": 0, "avg_ts": 100.0, "stddev_ts": 2.0, "n_gpu_layers": 999},
    )
    assert line.startswith("pp512 @ ngl=999:")


def test_format_entry_generation_only():
    line = LlamaBenchBenchmark.format_entry(
        {"n_prompt": 0, "n_gen": 128, "n_depth": 2048,
         "avg_ts": 50.0, "stddev_ts": 1.0, "n_gpu_layers": 0},
    )
    assert line.startswith("tg128 @ pp2048 @ ngl=0:")


def test_format_entry_combined():
    line = LlamaBenchBenchmark.format_entry(
        {"n_prompt": 512, "n_gen": 128, "avg_ts": 75.0, "stddev_ts": 1.5, "n_gpu_layers": 999},
    )
    assert line.startswith("pp512+tg128 @ ngl=999:")


def test_normalize_streamed_entry_preserves_internal_repetition_samples():
    entry = LlamaBenchBenchmark.normalize_streamed_entry(
        {"avg_ts": 11.0, "stddev_ts": 1.0, "samples_ts": [10.0, 11.0, 12.0]}, 3,
    )
    assert entry["ts_runs"] == [10.0, 11.0, 12.0]
    assert entry["requested_reps"] == 3
    assert entry["completed_reps"] == 3


class _NotLlamaCppEngine:
    name = "other-engine"


@pytest.fixture
def fake_engine(monkeypatch):
    """A real LlamaCppEngine (for the isinstance check) with pulled/resolve mocked."""
    engine = LlamaCppEngine()
    monkeypatch.setattr(LlamaCppEngine, "model_pulled", lambda self, tag: True)
    monkeypatch.setattr(
        LlamaCppEngine, "_resolve_model_files",
        classmethod(lambda cls, tag: [Path(f"/models/{tag}.gguf")]),
    )
    monkeypatch.setattr(LlamaBenchBenchmark, "find_binary", staticmethod(lambda: "llama-bench"))
    return engine


_MODELS = [{"tag": "m1", "label": "Model One", "short": "m1"}]


@pytest.fixture
def small_matrix(monkeypatch):
    monkeypatch.setattr(config, "LLAMABENCH_PP", [512])
    monkeypatch.setattr(config, "LLAMABENCH_TG", [128])


def test_run_skips_non_llamacpp_engine():
    result = LlamaBenchBenchmark().run(_NotLlamaCppEngine(), _MODELS, reps=3)
    assert result == {}


def test_run_returns_empty_when_binary_missing(monkeypatch):
    monkeypatch.setattr(LlamaBenchBenchmark, "find_binary", staticmethod(lambda: None))
    result = LlamaBenchBenchmark().run(LlamaCppEngine(), _MODELS, reps=3)
    assert result == {}


def test_run_skips_unpulled_models(fake_engine, monkeypatch):
    monkeypatch.setattr(LlamaCppEngine, "model_pulled", lambda self, tag: False)
    result = LlamaBenchBenchmark().run(fake_engine, _MODELS, reps=3)
    assert result == {}


def test_run_records_error_when_files_missing(fake_engine, monkeypatch):
    monkeypatch.setattr(LlamaCppEngine, "_resolve_model_files", classmethod(lambda cls, tag: None))
    result = LlamaBenchBenchmark().run(fake_engine, _MODELS, reps=3)
    assert result["m1"] == {"error": "model files not found"}


def test_run_records_entries_on_success(fake_engine, monkeypatch, small_matrix):
    def run_one(cls, command, *args, **kwargs):
        if command[command.index("-n") + 1] == "0":
            rows = [{"n_prompt": 512, "n_gen": 0, "avg_ts": 100.0,
                     "samples_ts": [99.0, 101.0]}]
        else:
            rows = [{"n_prompt": 0, "n_gen": 128, "n_depth": 512, "avg_ts": 50.0,
                     "samples_ts": [49.0, 51.0]}]
        for row in rows:
            kwargs["on_result"](row)
        return rows
    monkeypatch.setattr(LlamaBenchBenchmark, "run_one", classmethod(run_one))
    result = LlamaBenchBenchmark().run(fake_engine, _MODELS, reps=2)["m1"]
    assert result["prefill_entries"][0]["ts_runs"] == [99.0, 101.0]
    assert result["decode_entries"][0]["completed_reps"] == 2
    assert result["completed_cases"] == 2


def test_run_passes_on_progress_to_run_one(fake_engine, monkeypatch, small_matrix):
    captured = []

    def fake_run_one(cls, *a, on_progress=None, **kw):
        captured.append(on_progress)
        return [{"n_prompt": 512, "n_gen": 0, "avg_ts": 1.0}]

    monkeypatch.setattr(LlamaBenchBenchmark, "run_one", classmethod(fake_run_one))
    LlamaBenchBenchmark().run(fake_engine, _MODELS, reps=1)
    assert captured == [Shared.log, Shared.log]


def test_run_calls_save_fn_after_each_streamed_case(fake_engine, monkeypatch, small_matrix):
    def run_one(cls, command, *args, **kwargs):
        row = ({"n_prompt": 512, "n_gen": 0, "avg_ts": 1.0, "samples_ts": [1.0, 1.0]}
               if command[command.index("-n") + 1] == "0" else
               {"n_prompt": 0, "n_depth": 512, "n_gen": 128, "avg_ts": 1.0,
                "samples_ts": [1.0, 1.0]})
        kwargs["on_result"](row)
        return [row]
    monkeypatch.setattr(LlamaBenchBenchmark, "run_one", classmethod(run_one))
    saved = []
    LlamaBenchBenchmark().run(fake_engine, _MODELS, reps=2, save_fn=lambda partial: saved.append(dict(partial)))
    assert len(saved) == 3


def test_run_records_timeout_error(fake_engine, monkeypatch, small_matrix):
    def fake_run_one(cls, *a, **kw):
        raise subprocess.TimeoutExpired(cmd=["llama-bench"], timeout=config.LLAMABENCH_TIMEOUT)

    monkeypatch.setattr(LlamaBenchBenchmark, "run_one", classmethod(fake_run_one))
    result = LlamaBenchBenchmark().run(fake_engine, _MODELS, reps=3)
    assert "no output" in result["m1"]["error"]


def test_run_preserves_successful_prefill_sweep_before_decode_timeout(
        fake_engine, monkeypatch, small_matrix):
    calls = 0

    def fake_run_one(cls, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise subprocess.TimeoutExpired(cmd=["llama-bench"], timeout=1)
        row = {"n_prompt": 512, "n_gen": 0, "avg_ts": 123.0,
               "samples_ts": [122.0, 123.0, 124.0]}
        kwargs["on_result"](row)
        return [row]

    monkeypatch.setattr(LlamaBenchBenchmark, "run_one", classmethod(fake_run_one))
    result = LlamaBenchBenchmark().run(fake_engine, _MODELS, reps=3)["m1"]
    assert result["prefill_entries"][0]["ts_runs"] == [122.0, 123.0, 124.0]
    assert result["prefill_entries"][0]["completed_reps"] == 3
    assert result["timed_out"] is True


def test_run_preserves_rows_streamed_before_same_sweep_times_out(
        fake_engine, monkeypatch):
    monkeypatch.setattr(config, "LLAMABENCH_PP", [512, 2048, 4096])
    monkeypatch.setattr(config, "LLAMABENCH_TG", [128])

    def fake_run_one(cls, command, *args, **kwargs):
        for prompt in (512, 2048):
            kwargs["on_result"]({
                "n_prompt": prompt, "n_gen": 0, "avg_ts": 123.0,
                "samples_ts": [122.0, 123.0, 124.0],
            })
        raise subprocess.TimeoutExpired(cmd=command, timeout=1)

    monkeypatch.setattr(LlamaBenchBenchmark, "run_one", classmethod(fake_run_one))
    result = LlamaBenchBenchmark().run(fake_engine, _MODELS, reps=3)["m1"]
    assert [entry["n_prompt"] for entry in result["prefill_entries"]] == [512, 2048]
    assert all(entry["ts_runs"] == [122.0, 123.0, 124.0]
               for entry in result["prefill_entries"])
    assert result["completed_repetitions"] == 6
    assert result["completed_cases"] == 2
    assert result["timed_out"] is True


def test_run_journal_commits_each_row_before_same_sweep_timeout(
        fake_engine, monkeypatch, small_matrix):
    class Journal:
        def __init__(self):
            self.entries = []
            self.states = []
            self.finished = False
            self.telemetry_calls = []

        def begin_measured(self, subwindow="measured"):
            self.telemetry_calls.append(subwindow)

        def record_model_plan(self, model, requested_cases, reps):
            assert (requested_cases, reps) == (2, 3)

        @staticmethod
        def pending_sweeps(model, pp, tg):
            return [("prefill", pp, []), ("decode", pp, tg)]

        def record_entry(self, model, entry):
            self.telemetry_calls.append("record")
            self.entries.append(entry)

        def record_model_state(self, model, state, result):
            self.states.append((state, result))

        def finish(self):
            self.finished = True

        @staticmethod
        def export():
            return {"projected": True}

    def fake_run_one(cls, command, *args, **kwargs):
        assert journal.telemetry_calls == ["measured:native-sweep-includes-load"]
        kwargs["on_result"]({
            "n_prompt": 512, "n_gen": 0, "avg_ts": 123.0,
            "samples_ts": [122.0, 123.0, 124.0],
        })
        raise subprocess.TimeoutExpired(cmd=command, timeout=1)

    monkeypatch.setattr(LlamaBenchBenchmark, "run_one", classmethod(fake_run_one))
    journal = Journal()
    result = LlamaBenchBenchmark().run(fake_engine, _MODELS, reps=3, journal=journal)
    assert result == {"projected": True}
    assert len(journal.entries) == 1
    assert journal.entries[0]["completed_reps"] == 3
    assert journal.telemetry_calls == ["measured:native-sweep-includes-load", "record"]
    assert journal.states[0][0] == "timed_out"
    assert journal.finished is True


def test_run_records_generic_exception(fake_engine, monkeypatch, small_matrix):
    def fake_run_one(cls, *a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(LlamaBenchBenchmark, "run_one", classmethod(fake_run_one))
    result = LlamaBenchBenchmark().run(fake_engine, _MODELS, reps=3)
    assert result["m1"]["error"] == "boom"


def test_run_one_model_failure_does_not_stop_the_rest(fake_engine, monkeypatch, small_matrix):
    models = [{"tag": "bad", "label": "Bad", "short": "bad"}, {"tag": "good", "label": "Good", "short": "good"}]

    def fake_run_one(cls, command, *a, **kw):
        if any("bad" in arg for arg in command):
            raise RuntimeError("boom")
        row = {"n_prompt": 512, "n_gen": 0, "avg_ts": 1.0, "stddev_ts": 0.0, "n_gpu_layers": 999}
        kw["on_result"](row)
        return [row]

    monkeypatch.setattr(LlamaBenchBenchmark, "run_one", classmethod(fake_run_one))
    result = LlamaBenchBenchmark().run(fake_engine, models, reps=3)
    assert result["bad"]["error"] == "boom"
    assert "prefill_entries" in result["good"]
    assert "decode_entries" in result["good"]


def test_run_passes_full_offload_ngl_by_default(fake_engine, monkeypatch, small_matrix):
    captured = {}

    def fake_run_one(cls, command, timeout, on_progress=None, on_result=None):
        captured.setdefault("commands", []).append(command)
        return []

    monkeypatch.setattr(LlamaBenchBenchmark, "run_one", classmethod(fake_run_one))
    LlamaBenchBenchmark().run(fake_engine, _MODELS, reps=3, cpu_only=False)
    assert len(captured["commands"]) == 2
    assert all(command[command.index("-ngl") + 1] == str(config.LLAMABENCH_FULL_OFFLOAD_NGL)
               for command in captured["commands"])
    assert all(command[command.index("-r") + 1] == "3" for command in captured["commands"])


def test_run_passes_zero_ngl_when_cpu_only(fake_engine, monkeypatch, small_matrix):
    captured = {}

    def fake_run_one(cls, command, timeout, on_progress=None, on_result=None):
        captured.setdefault("commands", []).append(command)
        return []

    monkeypatch.setattr(LlamaBenchBenchmark, "run_one", classmethod(fake_run_one))
    LlamaBenchBenchmark().run(fake_engine, _MODELS, reps=3, cpu_only=True)
    assert len(captured["commands"]) == 2
    assert all(command[command.index("-ngl") + 1] == "0" for command in captured["commands"])
