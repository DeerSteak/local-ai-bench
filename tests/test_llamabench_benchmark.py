"""Tests for LlamaBenchBenchmark — see docs/testing.md."""

import subprocess
from pathlib import Path

import pytest

import config
from engines.llamacpp import LlamaCppEngine
from llamabench_benchmark import LlamaBenchBenchmark


# ══════════════════════════════════════════════════════════════════════════
#  Binary resolution
# ══════════════════════════════════════════════════════════════════════════


def test_find_binary_via_llamacpp_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("llamabench_benchmark.platform.system", lambda: "Linux")
    monkeypatch.setattr(config, "LLAMACPP_DIR", tmp_path)
    nested = tmp_path / "build" / "bin"
    nested.mkdir(parents=True)
    exe = nested / "llama-bench"
    exe.write_text("")
    monkeypatch.setattr("llamabench_benchmark.shutil.which", lambda name: None)
    assert LlamaBenchBenchmark.find_binary() == str(exe)


def test_find_binary_falls_back_to_path(monkeypatch, tmp_path):
    monkeypatch.setattr("llamabench_benchmark.platform.system", lambda: "Linux")
    monkeypatch.setattr(config, "LLAMACPP_DIR", tmp_path / "nonexistent")
    monkeypatch.setattr(
        "llamabench_benchmark.shutil.which",
        lambda name: "/usr/local/bin/llama-bench" if name == "llama-bench" else None,
    )
    assert LlamaBenchBenchmark.find_binary() == "/usr/local/bin/llama-bench"


def test_find_binary_checks_macos_homebrew_prefixes(monkeypatch, tmp_path):
    monkeypatch.setattr("llamabench_benchmark.platform.system", lambda: "Darwin")
    monkeypatch.setattr(config, "LLAMACPP_DIR", tmp_path / "nonexistent")
    monkeypatch.setattr("llamabench_benchmark.shutil.which", lambda name: None)

    real_exists = Path.exists

    def fake_exists(self):
        return str(self) == "/opt/homebrew/bin/llama-bench" or real_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    assert LlamaBenchBenchmark.find_binary() == "/opt/homebrew/bin/llama-bench"


def test_find_binary_returns_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("llamabench_benchmark.platform.system", lambda: "Linux")
    monkeypatch.setattr(config, "LLAMACPP_DIR", tmp_path / "nonexistent")
    monkeypatch.setattr("llamabench_benchmark.shutil.which", lambda name: None)
    assert LlamaBenchBenchmark.find_binary() is None


# ══════════════════════════════════════════════════════════════════════════
#  build_command
# ══════════════════════════════════════════════════════════════════════════


def test_build_command_shape():
    cmd = LlamaBenchBenchmark.build_command(
        "llama-bench", Path("/models/x.gguf"), [512, 2048], [128], 2048, 512, 3, 999,
    )
    assert cmd == [
        "llama-bench", "-m", "/models/x.gguf",
        "-p", "512,2048", "-n", "128",
        "-b", "2048", "-ub", "512",
        "-ngl", "999", "-r", "3", "-o", "json",
    ]


def test_build_command_cpu_only_ngl():
    cmd = LlamaBenchBenchmark.build_command(
        "llama-bench", Path("/models/x.gguf"), [512], [128], 2048, 512, 3, 0,
    )
    assert cmd[cmd.index("-ngl") + 1] == "0"


# ══════════════════════════════════════════════════════════════════════════
#  run_one
# ══════════════════════════════════════════════════════════════════════════


class _FakeCompleted:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_one_parses_json_on_success(monkeypatch):
    fake_entries = [{"n_prompt": 512, "n_gen": 0, "avg_ts": 123.4, "stddev_ts": 1.2, "n_gpu_layers": 999}]
    monkeypatch.setattr(
        "llamabench_benchmark.subprocess.run",
        lambda cmd, capture_output, text, timeout: _FakeCompleted(0, stdout=__import__("json").dumps(fake_entries)),
    )
    entries = LlamaBenchBenchmark.run_one("llama-bench", Path("/x.gguf"), [512], [0], 2048, 512, 3, 999, 60)
    assert entries == fake_entries


def test_run_one_raises_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(
        "llamabench_benchmark.subprocess.run",
        lambda cmd, capture_output, text, timeout: _FakeCompleted(1, stderr="error: out of memory"),
    )
    with pytest.raises(RuntimeError, match="out of memory"):
        LlamaBenchBenchmark.run_one("llama-bench", Path("/x.gguf"), [512], [0], 2048, 512, 3, 999, 60)


def test_run_one_raises_on_malformed_json(monkeypatch):
    monkeypatch.setattr(
        "llamabench_benchmark.subprocess.run",
        lambda cmd, capture_output, text, timeout: _FakeCompleted(0, stdout="not json"),
    )
    with pytest.raises(RuntimeError, match="unparseable JSON"):
        LlamaBenchBenchmark.run_one("llama-bench", Path("/x.gguf"), [512], [0], 2048, 512, 3, 999, 60)


def test_run_one_propagates_timeout(monkeypatch):
    def fake_run(cmd, capture_output, text, timeout):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr("llamabench_benchmark.subprocess.run", fake_run)
    with pytest.raises(subprocess.TimeoutExpired):
        LlamaBenchBenchmark.run_one("llama-bench", Path("/x.gguf"), [512], [0], 2048, 512, 3, 999, 60)


# ══════════════════════════════════════════════════════════════════════════
#  format_entry
# ══════════════════════════════════════════════════════════════════════════


def test_format_entry_prompt_processing_only():
    line = LlamaBenchBenchmark.format_entry(
        {"n_prompt": 512, "n_gen": 0, "avg_ts": 100.0, "stddev_ts": 2.0, "n_gpu_layers": 999},
    )
    assert line.startswith("pp512 @ ngl=999:")


def test_format_entry_generation_only():
    line = LlamaBenchBenchmark.format_entry(
        {"n_prompt": 0, "n_gen": 128, "avg_ts": 50.0, "stddev_ts": 1.0, "n_gpu_layers": 0},
    )
    assert line.startswith("tg128 @ ngl=0:")


def test_format_entry_combined():
    line = LlamaBenchBenchmark.format_entry(
        {"n_prompt": 512, "n_gen": 128, "avg_ts": 75.0, "stddev_ts": 1.5, "n_gpu_layers": 999},
    )
    assert line.startswith("pp512+tg128 @ ngl=999:")


# ══════════════════════════════════════════════════════════════════════════
#  run() dispatch
# ══════════════════════════════════════════════════════════════════════════


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


def test_run_records_entries_on_success(fake_engine, monkeypatch):
    fake_entries = [{"n_prompt": 512, "n_gen": 0, "avg_ts": 100.0, "stddev_ts": 1.0, "n_gpu_layers": 999}]
    monkeypatch.setattr(LlamaBenchBenchmark, "run_one", classmethod(lambda cls, *a, **kw: fake_entries))
    result = LlamaBenchBenchmark().run(fake_engine, _MODELS, reps=3)
    assert result["m1"] == {"entries": fake_entries}


def test_run_calls_save_fn_after_each_model(fake_engine, monkeypatch):
    monkeypatch.setattr(LlamaBenchBenchmark, "run_one", classmethod(lambda cls, *a, **kw: []))
    saved = []
    LlamaBenchBenchmark().run(fake_engine, _MODELS, reps=3, save_fn=lambda partial: saved.append(dict(partial)))
    assert saved == [{"m1": {"entries": []}}]


def test_run_records_timeout_error(fake_engine, monkeypatch):
    def fake_run_one(cls, *a, **kw):
        raise subprocess.TimeoutExpired(cmd=["llama-bench"], timeout=config.LLAMABENCH_TIMEOUT)

    monkeypatch.setattr(LlamaBenchBenchmark, "run_one", classmethod(fake_run_one))
    result = LlamaBenchBenchmark().run(fake_engine, _MODELS, reps=3)
    assert "timed out" in result["m1"]["error"]


def test_run_records_generic_exception(fake_engine, monkeypatch):
    def fake_run_one(cls, *a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(LlamaBenchBenchmark, "run_one", classmethod(fake_run_one))
    result = LlamaBenchBenchmark().run(fake_engine, _MODELS, reps=3)
    assert result["m1"] == {"error": "boom"}


def test_run_one_model_failure_does_not_stop_the_rest(fake_engine, monkeypatch):
    models = [{"tag": "bad", "label": "Bad", "short": "bad"}, {"tag": "good", "label": "Good", "short": "good"}]

    def fake_run_one(cls, binary, model_path, *a, **kw):
        if "bad" in str(model_path):
            raise RuntimeError("boom")
        return [{"n_prompt": 512, "n_gen": 0, "avg_ts": 1.0, "stddev_ts": 0.0, "n_gpu_layers": 999}]

    monkeypatch.setattr(LlamaBenchBenchmark, "run_one", classmethod(fake_run_one))
    result = LlamaBenchBenchmark().run(fake_engine, models, reps=3)
    assert result["bad"] == {"error": "boom"}
    assert "entries" in result["good"]


def test_run_passes_full_offload_ngl_by_default(fake_engine, monkeypatch):
    captured = {}

    def fake_run_one(cls, binary, model_path, pp, tg, batch_size, ubatch_size, reps, ngl, timeout):
        captured["ngl"] = ngl
        return []

    monkeypatch.setattr(LlamaBenchBenchmark, "run_one", classmethod(fake_run_one))
    LlamaBenchBenchmark().run(fake_engine, _MODELS, reps=3, cpu_only=False)
    assert captured["ngl"] == config.LLAMABENCH_FULL_OFFLOAD_NGL


def test_run_passes_zero_ngl_when_cpu_only(fake_engine, monkeypatch):
    captured = {}

    def fake_run_one(cls, binary, model_path, pp, tg, batch_size, ubatch_size, reps, ngl, timeout):
        captured["ngl"] = ngl
        return []

    monkeypatch.setattr(LlamaBenchBenchmark, "run_one", classmethod(fake_run_one))
    LlamaBenchBenchmark().run(fake_engine, _MODELS, reps=3, cpu_only=True)
    assert captured["ngl"] == 0
