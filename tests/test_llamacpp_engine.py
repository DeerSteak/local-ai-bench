"""Tests for LlamaCppEngine — see docs/testing.md for the full coverage breakdown."""

import json
import subprocess
from pathlib import Path
from typing import cast

import gguf
import pytest
import requests

from scripts.runtime import config
from scripts.runtime.engines.base import aggregate_generation_measurements, is_valid_measurement
from scripts.runtime.engines.llamacpp import LlamaCppEngine, model_placement_error
import scripts.runtime.engines.llamacpp as llamacpp_module
from scripts.runtime.shared import EngineBudgetExceeded, EngineLoopDetected, EngineTimeout


# ══════════════════════════════════════════════════════════════════════════
#  Group 0 — binary resolution
# ══════════════════════════════════════════════════════════════════════════


def test_repack_args_follow_the_explicit_runtime_setting(monkeypatch):
    monkeypatch.setattr(config, "LLAMACPP_NO_REPACK", False)
    assert LlamaCppEngine.repack_args() == []
    monkeypatch.setattr(config, "LLAMACPP_NO_REPACK", True)
    assert LlamaCppEngine.repack_args() == ["--no-repack"]


def test_no_host_args_follow_runtime_setting_and_tool_syntax(monkeypatch):
    monkeypatch.setattr(config, "LLAMACPP_NO_HOST", False)
    assert LlamaCppEngine.no_host_args() == []
    monkeypatch.setattr(config, "LLAMACPP_NO_HOST", True)
    assert LlamaCppEngine.no_host_args() == ["--no-host"]
    assert LlamaCppEngine.no_host_args(value_required=True) == ["--no-host", "1"]
    assert LlamaCppEngine.no_host_args(cpu_only=True) == []


def test_parse_model_placement_reports_layers_and_cpu_side_model_buffers():
    log = """0.05.100.001 I load_tensors: offloaded 35/41 layers to GPU
0.05.101.002 I load_tensors:          CPU_Mapped model buffer size =   272.81 MiB
0.05.102.003 I load_tensors:        CUDA0 model buffer size =  4385.96 MiB
0.05.103.004 I load_tensors:    CUDA_Host model buffer size = 14105.96 MiB
"""
    assert LlamaCppEngine.parse_model_placement(log) == {
        "gpu_layers": 35,
        "total_layers": 41,
        "cpu_model_buffer_gb": 14.042,
    }


def test_parse_model_placement_uses_last_load_and_tolerates_missing_buffers():
    log = "load_tensors: offloaded 10/41 layers to GPU\nload_tensors: offloaded 41/41 layers to GPU"
    assert LlamaCppEngine.parse_model_placement(log) == {
        "gpu_layers": 41,
        "total_layers": 41,
    }


def test_full_log_keeps_placement_evidence_older_than_the_diagnostic_tail(tmp_path):
    engine = LlamaCppEngine()
    log_path = tmp_path / "server.log"
    engine._log_path = log_path
    log_path.write_text(
        "load_tensors: offloaded 41/41 layers to GPU\n" + "recent line\n" * 250,
        encoding="utf-8",
    )

    assert "offloaded 41/41" not in engine.tail_log(engine.SPAWN_LOG_LINES)
    assert LlamaCppEngine.parse_model_placement(engine._full_log())["gpu_layers"] == 41


def test_accelerated_model_placement_rejects_cpu_fallback_and_missing_evidence():
    assert model_placement_error("rocm", {"gpu_layers": 0, "total_layers": 49}) == (
        "rocm model load offloaded zero layers to the GPU"
    )
    assert model_placement_error("rocm", {}) == (
        "rocm model load did not report GPU layer placement"
    )
    assert model_placement_error("rocm", {"gpu_layers": 49, "total_layers": 49}) is None


def test_cpu_model_placement_does_not_require_gpu_layers():
    assert model_placement_error("cpu", {}) is None
    assert model_placement_error(None, {}) is None


def test_binary_path_via_llamacpp_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(llamacpp_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(config, "LLAMACPP_DIR", tmp_path)
    nested = tmp_path / "build" / "bin"
    nested.mkdir(parents=True)
    exe = nested / "llama-server"
    exe.write_text("")
    monkeypatch.setattr(llamacpp_module.shutil, "which", lambda name: None)
    assert LlamaCppEngine._binary_path() == str(exe)


def test_binary_path_skips_a_same_named_source_directory(monkeypatch, tmp_path):
    """A CMake source tree has tools/server/ (source) alongside build/bin/llama-server
    (compiled) — mirrors the real bug where a same-named source directory shadowed
    llama-bench's compiled binary; rglob must not return the directory."""
    monkeypatch.setattr(llamacpp_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(config, "LLAMACPP_DIR", tmp_path)
    (tmp_path / "tools" / "llama-server").mkdir(parents=True)
    nested = tmp_path / "build" / "bin"
    nested.mkdir(parents=True)
    exe = nested / "llama-server"
    exe.write_text("")
    monkeypatch.setattr(llamacpp_module.shutil, "which", lambda name: None)
    assert LlamaCppEngine._binary_path() == str(exe)


def test_binary_path_falls_back_to_path(monkeypatch, tmp_path):
    monkeypatch.setattr(llamacpp_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(config, "LLAMACPP_DIR", tmp_path / "nonexistent")
    monkeypatch.setattr(
        llamacpp_module.shutil, "which",
        lambda name: "/usr/local/bin/llama-server" if name == "llama-server" else None,
    )
    assert LlamaCppEngine._binary_path() == "/usr/local/bin/llama-server"


def test_binary_path_checks_macos_homebrew_prefixes(monkeypatch, tmp_path):
    monkeypatch.setattr(llamacpp_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(config, "LLAMACPP_DIR", tmp_path / "nonexistent")
    monkeypatch.setattr(llamacpp_module.shutil, "which", lambda name: None)

    real_is_file = llamacpp_module.Path.is_file

    def fake_is_file(self):
        return str(self) == "/opt/homebrew/bin/llama-server" or real_is_file(self)

    monkeypatch.setattr(llamacpp_module.Path, "is_file", fake_is_file)
    assert LlamaCppEngine._binary_path() == "/opt/homebrew/bin/llama-server"


def test_binary_path_returns_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(llamacpp_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(config, "LLAMACPP_DIR", tmp_path / "nonexistent")
    monkeypatch.setattr(llamacpp_module.shutil, "which", lambda name: None)
    assert LlamaCppEngine._binary_path() is None


# ══════════════════════════════════════════════════════════════════════════
#  Group 1 — local model-file resolution
# ══════════════════════════════════════════════════════════════════════════


_FAKE_CATALOG = [
    {"tag": "phi4-mini", "hf_repo": "org/phi4-mini-gguf", "hf_file": "phi4-mini.Q4_K_M.gguf"},
    {"tag": "llama3.2:3b-instruct-q4_K_M", "hf_repo": "org/llama32-gguf",
     "hf_file": "llama32-3b.Q4_K_M.gguf"},
    {"tag": "split:model", "hf_repo": "org/split-gguf",
     "hf_file": ["split-00001-of-00002.gguf", "split-00002-of-00002.gguf"]},
    {"tag": "mtp:embedded", "hf_repo": "org/mtp-embedded",
     "hf_file": "embedded.gguf",
     "native_mtp": {"llamacpp": {"num_speculative_tokens": 3}}},
    {"tag": "mtp:separate", "hf_repo": "org/mtp-separate",
     "hf_file": "model.gguf", "native_mtp": {"llamacpp": {
         "num_speculative_tokens": 2,
         "draft_repo": "org/mtp-separate", "draft_file": "MTP/draft.gguf",
     }}},
]


@pytest.fixture
def fake_catalog(monkeypatch, tmp_path):
    """Point config.MODELS_DIR at tmp_path and swap in a small fixture
    catalog (LLM_MODELS + EMBED_MODELS combined) instead of the real one."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(llamacpp_module, "LLM_MODELS", _FAKE_CATALOG)
    monkeypatch.setattr(llamacpp_module, "EMBED_MODELS", [])
    return tmp_path


def _write_model_file(models_dir, tag, filename, content: bytes):
    slug = LlamaCppEngine._slug(tag)
    model_dir = models_dir / "llamacpp" / slug
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / filename).write_bytes(content)


def test_models_dir_namespaced_under_engine_name(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    assert LlamaCppEngine._models_dir() == tmp_path / "llamacpp"


def test_slug_replaces_colons_and_slashes():
    assert LlamaCppEngine._slug("llama3.2:3b-instruct-q4_K_M") == "llama3.2_3b-instruct-q4_K_M"
    assert LlamaCppEngine._slug("someorg/some-model") == "someorg_some-model"


def test_resolve_model_files_finds_single_file(fake_catalog):
    _write_model_file(fake_catalog, "phi4-mini", "phi4-mini.Q4_K_M.gguf", b"fake-gguf-bytes")
    paths = LlamaCppEngine._resolve_model_files("phi4-mini")
    assert paths is not None
    assert paths[0].read_bytes() == b"fake-gguf-bytes"


def test_resolve_model_files_missing_tag_returns_none(fake_catalog):
    assert LlamaCppEngine._resolve_model_files("not-in-catalog") is None


def test_resolve_model_files_finds_custom_dropped_in_model(fake_catalog):
    from scripts.app.benchmark import resolve_custom_models

    custom_dir = fake_catalog / "llamacpp" / "my-custom-model"
    custom_dir.mkdir(parents=True)
    model_path = custom_dir / "weights.gguf"
    model_path.write_bytes(b"custom")
    assert LlamaCppEngine._resolve_model_files("my-custom-model") == [model_path]
    engine = LlamaCppEngine()
    installed_tags = [model["tag"] for model in engine.list_installed_models()]
    selected = resolve_custom_models(["my-custom-model"], [], installed_tags)
    assert selected[0]["tag"] == "my-custom-model"
    assert engine.model_pulled(selected[0]["tag"]) is True


def test_resolve_model_files_requires_complete_custom_multipart_set(fake_catalog):
    custom_dir = fake_catalog / "llamacpp" / "custom-split"
    custom_dir.mkdir(parents=True)
    first = custom_dir / "weights-00001-of-00002.gguf"
    second = custom_dir / "weights-00002-of-00002.gguf"
    first.write_bytes(b"a")
    assert LlamaCppEngine._resolve_model_files("custom-split") is None
    second.write_bytes(b"b")
    assert LlamaCppEngine._resolve_model_files("custom-split") == [first, second]


def test_resolve_model_files_missing_file_returns_none(fake_catalog):
    # Catalog entry exists but the file was never downloaded.
    assert LlamaCppEngine._resolve_model_files("phi4-mini") is None


def test_resolve_model_files_requires_every_split_part(fake_catalog):
    _write_model_file(fake_catalog, "split:model", "split-00001-of-00002.gguf", b"a")
    assert LlamaCppEngine._resolve_model_files("split:model") is None  # part 2 missing
    _write_model_file(fake_catalog, "split:model", "split-00002-of-00002.gguf", b"b")
    paths = LlamaCppEngine._resolve_model_files("split:model")
    assert paths is not None
    assert [p.name for p in paths] == ["split-00001-of-00002.gguf", "split-00002-of-00002.gguf"]


def test_model_pulled_true_when_resolvable(fake_catalog):
    _write_model_file(fake_catalog, "phi4-mini", "phi4-mini.Q4_K_M.gguf", b"x")
    assert LlamaCppEngine().model_pulled("phi4-mini") is True


def test_model_pulled_false_when_not_resolvable(fake_catalog):
    assert LlamaCppEngine().model_pulled("phi4-mini") is False


def test_resume_artifacts_include_separate_mtp_predictor_only_when_enabled(fake_catalog):
    _write_model_file(fake_catalog, "mtp:separate", "model.gguf", b"model")
    _write_model_file(fake_catalog, "mtp:separate", "draft.gguf", b"draft")
    engine = LlamaCppEngine()
    assert [path.name for path in engine.resume_artifact_paths("mtp:separate")] == [
        "model.gguf",
    ]
    engine.set_mtp_enabled(True)
    assert [path.name for path in engine.resume_artifact_paths("mtp:separate")] == [
        "model.gguf", "draft.gguf",
    ]


def test_enabled_separate_mtp_requires_downloaded_predictor(fake_catalog):
    _write_model_file(fake_catalog, "mtp:separate", "model.gguf", b"model")
    engine = LlamaCppEngine()
    engine.set_mtp_enabled(True)
    with pytest.raises(RuntimeError, match="MTP predictor is missing.*rerun setup"):
        engine.resume_artifact_paths("mtp:separate")


def test_enabled_mtp_rejects_unsupported_and_custom_models(fake_catalog):
    engine = LlamaCppEngine()
    engine.set_mtp_enabled(True)
    with pytest.raises(RuntimeError, match="does not support native MTP"):
        engine._native_mtp_config("phi4-mini")
    with pytest.raises(RuntimeError, match="no cataloged native MTP configuration"):
        engine._native_mtp_config("custom")


def test_list_installed_models_lists_every_downloaded_catalog_tag(fake_catalog):
    _write_model_file(fake_catalog, "phi4-mini", "phi4-mini.Q4_K_M.gguf", b"aaa")
    _write_model_file(fake_catalog, "llama3.2:3b-instruct-q4_K_M", "llama32-3b.Q4_K_M.gguf", b"bbbbb")
    installed = {m["tag"]: m["size"] for m in LlamaCppEngine().list_installed_models()}
    assert installed == {"phi4-mini": 3, "llama3.2:3b-instruct-q4_K_M": 5}


def test_list_installed_models_includes_custom_dropped_in_model(fake_catalog):
    # A model dropped in manually, not in the catalog at all.
    custom_dir = fake_catalog / "llamacpp" / "my-custom-model"
    custom_dir.mkdir(parents=True)
    (custom_dir / "weights.gguf").write_bytes(b"cc")
    installed = {m["tag"]: m["size"] for m in LlamaCppEngine().list_installed_models()}
    assert installed == {"my-custom-model": 2}


def test_registered_custom_model_uses_its_display_label(fake_catalog, monkeypatch):
    custom_dir = fake_catalog / "llamacpp" / "my-custom-model"
    custom_dir.mkdir(parents=True)
    (custom_dir / "weights.gguf").write_bytes(b"cc")
    monkeypatch.setattr(
        llamacpp_module, "custom_model",
        lambda engine, tag: {"label": "My Imported Model"}
        if (engine, tag) == ("llamacpp", "my-custom-model") else None,
    )

    installed = LlamaCppEngine().list_installed_models()

    assert installed == [{"tag": "my-custom-model", "label": "My Imported Model", "size": 2}]


def test_removed_catalog_model_remains_available_by_its_folder_name(
        fake_catalog, monkeypatch):
    old_tag = "llama3.2:3b-instruct-q4_K_M"
    _write_model_file(fake_catalog, old_tag, "llama32-3b.Q4_K_M.gguf", b"legacy")
    monkeypatch.setattr(llamacpp_module, "LLM_MODELS", [])

    installed = LlamaCppEngine().list_installed_models()

    assert installed == [{
        "tag": LlamaCppEngine._slug(old_tag),
        "size": len(b"legacy"),
    }]
    assert LlamaCppEngine().model_pulled(LlamaCppEngine._slug(old_tag)) is True


def test_list_installed_models_omits_ambiguous_custom_directory(fake_catalog):
    custom_dir = fake_catalog / "llamacpp" / "ambiguous"
    custom_dir.mkdir(parents=True)
    (custom_dir / "one.gguf").write_bytes(b"a")
    (custom_dir / "two.gguf").write_bytes(b"b")
    assert LlamaCppEngine().list_installed_models() == []


def test_list_installed_models_empty_when_dir_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "does-not-exist")
    monkeypatch.setattr(llamacpp_module, "LLM_MODELS", [])
    monkeypatch.setattr(llamacpp_module, "EMBED_MODELS", [])
    assert LlamaCppEngine().list_installed_models() == []


def _write_gguf(path, context_length: int):
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_context_length(context_length)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_max_context_length_reads_architecture_prefixed_key(tmp_path, monkeypatch):
    gguf_path = tmp_path / "model.gguf"
    _write_gguf(gguf_path, 40960)
    monkeypatch.setattr(LlamaCppEngine, "_resolve_model_files", classmethod(lambda cls, tag: [gguf_path]))
    assert LlamaCppEngine().max_context_length("qwen3.5:4b") == 40960


def test_max_context_length_ignores_rope_scaling_original_context_length(tmp_path, monkeypatch):
    """Regression test for the rope-scaling-key bug — see max_context_length's docstring."""
    gguf_path = tmp_path / "model.gguf"
    w = gguf.GGUFWriter(str(gguf_path), "qwen3next")
    w.add_rope_scaling_orig_ctx_len(512)
    w.add_context_length(262144)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    monkeypatch.setattr(LlamaCppEngine, "_resolve_model_files", classmethod(lambda cls, tag: [gguf_path]))
    assert LlamaCppEngine().max_context_length("qwen3-coder-next:80b-a3b-q4_K_M") == 262144


def test_max_context_length_falls_back_to_default_when_not_pulled(fake_catalog):
    assert LlamaCppEngine().max_context_length("phi4-mini", default=8192) == 8192


def test_max_context_length_falls_back_to_default_on_unparseable_file(tmp_path, monkeypatch):
    bad_path = tmp_path / "not-really-gguf.bin"
    bad_path.write_bytes(b"not a gguf file")
    monkeypatch.setattr(LlamaCppEngine, "_resolve_model_files", classmethod(lambda cls, tag: [bad_path]))
    assert LlamaCppEngine().max_context_length("some-tag", default=2048) == 2048


# ══════════════════════════════════════════════════════════════════════════
#  Group 2 — streaming / timing parsing
# ══════════════════════════════════════════════════════════════════════════


class _FakeResponse:
    """Mimics urllib.request.urlopen's return value: a context manager that
    iterates raw SSE lines as bytes."""

    def __init__(self, chunks):
        lines = []
        for c in chunks:
            lines.append(f"data: {json.dumps(c)}\n".encode())
        lines.append(b"data: [DONE]\n")
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._lines)


def _patch_urlopen(monkeypatch, chunks):
    monkeypatch.setattr(LlamaCppEngine, "_urlopen",
                        staticmethod(lambda req, timeout: _FakeResponse(chunks)))


# ── _sanitize_tps ──

def test_sanitize_tps_passes_through_plausible_value():
    assert LlamaCppEngine._sanitize_tps(120.0, tokens=50, ttft=0.5, total=1.0) == 120.0


def test_sanitize_tps_passes_through_at_exact_ceiling():
    ceiling = config.MAX_PLAUSIBLE_TPS
    assert LlamaCppEngine._sanitize_tps(ceiling, tokens=50, ttft=0.5, total=1.0) == ceiling


def test_sanitize_tps_falls_back_to_wall_clock_when_implausible():
    # Reproduces the real bug: llama-server reports a tiny predicted_ms under
    # heavy slot contention, producing a tps ratio with no physical basis.
    huge = config.MAX_PLAUSIBLE_TPS + 1
    result = LlamaCppEngine._sanitize_tps(huge, tokens=5, ttft=1.0, total=3.0)
    assert result == pytest.approx(2.5)  # 5 tokens / (3.0 - 1.0)s decode time


def test_sanitize_tps_returns_zero_when_decode_elapsed_not_positive():
    huge = config.MAX_PLAUSIBLE_TPS + 1
    assert LlamaCppEngine._sanitize_tps(huge, tokens=5, ttft=2.0, total=2.0) == 0


def _patch_ensure_model(monkeypatch):
    monkeypatch.setattr(LlamaCppEngine, "_ensure_model", lambda self, *a, **kw: None)


def _clock(*values):
    iterator = iter(values)
    last = values[-1]

    def now():
        nonlocal last
        try:
            last = next(iterator)
        except StopIteration:
            pass
        return last

    return now


# ── generate ──

def test_generate_requests_n_predict_from_config_constant(monkeypatch):
    # concurrency_benchmark.py's slot_ctx_for headroom relies on this constant.
    _patch_ensure_model(monkeypatch)
    captured = []

    def urlopen(req, timeout):
        captured.append(json.loads(req.data))
        return _FakeResponse([
            {"content": "hi", "tokens": [1]},
            {"content": "", "stop": True,
             "timings": {"predicted_n": 1, "predicted_ms": 100, "prompt_ms": 10}},
        ])

    monkeypatch.setattr(LlamaCppEngine, "_urlopen", staticmethod(urlopen))
    LlamaCppEngine().generate("some-tag", "prompt")
    assert captured[0]["n_predict"] == config.GENERATE_MAX_TOKENS
    assert captured[0]["cache_prompt"] is False
    from scripts.runtime.sampling import baseline_sampling_payload
    assert {
        key: captured[0][key] for key in baseline_sampling_payload("llamacpp")
    } == baseline_sampling_payload("llamacpp")


def test_generate_uses_server_reported_timings(monkeypatch):
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        *[{"content": "x", "tokens": [i]} for i in range(10)],
        {"content": "", "stop": True,
         "timings": {"predicted_n": 10, "predicted_ms": 2000, "prompt_ms": 500}},
    ])
    result = LlamaCppEngine().generate("some-tag", "prompt", num_ctx=2048)
    assert result.client_ttft_sec >= 0
    assert result.server_prompt_sec == pytest.approx(0.5)
    assert result.generated_tokens == 10
    assert result.tokens_per_sec == pytest.approx(5.0)


def test_generate_enables_prompt_cache_when_requested(monkeypatch):
    _patch_ensure_model(monkeypatch)
    captured = []

    def urlopen(req, timeout):
        captured.append(json.loads(req.data))
        return _FakeResponse([{"content": "x", "tokens": [1], "stop": True}])

    monkeypatch.setattr(LlamaCppEngine, "_urlopen", staticmethod(urlopen))
    LlamaCppEngine().generate("tag", "prompt", cache_prompt=True)
    assert captured[0]["cache_prompt"] is True


def test_generate_falls_back_to_wall_clock_ttft_when_server_omits_it(monkeypatch):
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(
        llamacpp_module.time, "perf_counter",
        _clock(90.0, 90.0, 100.0, 100.0, 101.5, 101.5, 101.7, 102.0),
    )
    _patch_urlopen(monkeypatch, [
        {"content": "Hi", "tokens": [1]},
        {"content": "", "stop": True, "timings": {"predicted_n": 1, "predicted_ms": 0}},
    ])
    result = LlamaCppEngine().generate("some-tag", "prompt")
    assert result.client_ttft_sec == pytest.approx(1.5)
    assert result.server_prompt_sec is None
    assert result.tokens_per_sec == pytest.approx(2.0)


def test_generate_marks_corrected_implausible_server_tps_invalid(monkeypatch):
    # Reproduces the real observed failure: predicted_n=1, predicted_ms=0.001 -> raw tps of 1000000.0.
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(
        llamacpp_module.time, "perf_counter",
        _clock(90.0, 90.0, 100.0, 100.0, 100.5, 100.5, 105.0, 110.5),
    )
    _patch_urlopen(monkeypatch, [
        {"content": "Hi", "tokens": [1]},
        {"content": "", "stop": True, "timings": {"predicted_n": 1, "predicted_ms": 0.001}},
    ])
    result = LlamaCppEngine().generate("some-tag", "prompt")
    assert result.client_ttft_sec == pytest.approx(0.5)
    assert result.tokens_per_sec == pytest.approx(0.1)
    assert result.decode_sec == pytest.approx(10.0)
    assert result.server_tps_implausible is True
    assert not is_valid_measurement(result)
    assert aggregate_generation_measurements([result], 1)["valid_runs"] == 0


def test_generate_counts_native_token_ids_not_sse_fragments(monkeypatch):
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"content": "several decoded pieces", "tokens": [11, 12]},
        {"content": "x", "tokens": []},
        {"content": "", "stop": True, "timings": {"predicted_n": 1, "predicted_ms": 1000}},
    ])
    result = LlamaCppEngine().generate("some-tag", "prompt")
    assert result.generated_tokens == 2


def test_generate_logs_raw_server_values_when_sanitizing(monkeypatch):
    # Assert the actual predicted_n/predicted_ms numbers appear, not just that some warning fired.
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", _clock(100.0, 100.0, 100.5, 100.5, 110.5))
    _patch_urlopen(monkeypatch, [
        {"content": "Hi", "tokens": [1]},
        {"content": "", "stop": True, "timings": {"predicted_n": 1, "predicted_ms": 0.001}},
    ])
    warnings = []
    monkeypatch.setattr(llamacpp_module.Shared, "warn", staticmethod(lambda msg: warnings.append(msg)))
    LlamaCppEngine().generate("some-tag", "prompt")
    assert len(warnings) == 1
    assert "some-tag" in warnings[0]
    assert "server predicted_n=1" in warnings[0]
    assert "response tokens=1" in warnings[0]
    assert "predicted_ms=0.001" in warnings[0]


def test_generate_does_not_warn_when_tps_is_plausible(monkeypatch):
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"content": "Hel", "tokens": [1]},
        {"content": "lo", "tokens": [2]},
        {"content": "", "stop": True,
         "timings": {"predicted_n": 10, "predicted_ms": 2000, "prompt_ms": 500}},
    ])
    warnings = []
    monkeypatch.setattr(llamacpp_module.Shared, "warn", staticmethod(lambda msg: warnings.append(msg)))
    LlamaCppEngine().generate("some-tag", "prompt", num_ctx=2048)
    assert warnings == []


def test_generate_preserves_request_to_first_output_ttft(monkeypatch):
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(
        llamacpp_module.time, "perf_counter",
        _clock(0.0, 0.0, 10.0, 10.0, 12.0, 12.0, 12.5, 13.0),
    )
    _patch_urlopen(monkeypatch, [
        {"content": "Hi", "tokens": [1]},
        {"content": "", "stop": True,
         "timings": {"predicted_n": 1, "predicted_ms": 500, "prompt_ms": 100}},
    ])
    result = LlamaCppEngine().generate("tag", "prompt")
    assert result.client_ttft_sec == pytest.approx(2.0)
    assert result.server_prompt_sec == pytest.approx(0.1)


def test_generate_enforces_total_deadline_and_keeps_partial_text(monkeypatch):
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(
        llamacpp_module.time, "perf_counter", _clock(0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 6.0),
    )
    _patch_urlopen(monkeypatch, [
        {"content": "partial", "tokens": [1]},
        {"content": " too late", "tokens": [2]},
    ])
    with pytest.raises(EngineTimeout) as exc_info:
        LlamaCppEngine().generate("tag", "prompt", timeout=5)
    assert exc_info.value.partial_text == "partial too late"


def test_generate_enforces_deadline_during_sse_keepalive_lines(monkeypatch):
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", _clock(0.0, 0.0, 6.0))
    response = type("Response", (), {
        "__enter__": lambda self: self,
        "__exit__": lambda self, *args: False,
        "__iter__": lambda self: iter([b": ping\n"]),
    })()
    monkeypatch.setattr(LlamaCppEngine, "_urlopen", staticmethod(lambda req, timeout: response))
    with pytest.raises(EngineTimeout):
        LlamaCppEngine().generate("tag", "prompt", timeout=5)


# ── chat ──

def test_chat_returns_content_and_server_timings(monkeypatch):
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"content": "H"}}]},
        {"choices": [{"delta": {"content": "e"}}]},
        {"choices": [{"delta": {"content": "l"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "timings": {"predicted_n": 4, "predicted_ms": 1000, "prompt_ms": 200, "prompt_n": 50}},
        {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 4, "total_tokens": 54}},
    ])
    result = LlamaCppEngine().chat(
        "tag", [{"role": "user", "content": "hi"}])
    assert result.response_text == "Hello"
    assert result.prompt_tokens == 50
    assert result.client_ttft_sec >= 0
    assert result.server_prompt_sec == pytest.approx(0.2)
    assert result.generated_tokens == 4
    assert result.tokens_per_sec == pytest.approx(4.0)


def test_chat_marks_corrected_implausible_server_tps_invalid(monkeypatch):
    # Same real bug as generate()'s equivalent test, via the chat() code path.
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(
        llamacpp_module.time, "perf_counter",
        _clock(100.0, 100.0, 100.0, 100.5, 105.0, 110.5),
    )
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"content": "Hi"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "timings": {"predicted_n": 1, "predicted_ms": 0.001}},
    ])
    result = LlamaCppEngine().chat("tag", [{"role": "user", "content": "hi"}])
    assert result.client_ttft_sec == pytest.approx(0.5)
    assert result.tokens_per_sec == pytest.approx(0.1)
    assert result.server_tps_implausible is True


def test_chat_prefers_usage_prompt_tokens_over_timings_prompt_n(monkeypatch):
    # usage.prompt_tokens (true total) must win over timings.prompt_n
    # (cache-miss-only count) — see chat()'s docstring.
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"content": "Hi"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "timings": {"predicted_n": 4, "predicted_ms": 1000, "prompt_n": 12}},
        {"choices": [], "usage": {"prompt_tokens": 2048, "completion_tokens": 4, "total_tokens": 2052}},
    ])
    result = LlamaCppEngine().chat(
        "tag", [{"role": "user", "content": "hi"}])
    assert result.prompt_tokens == 2048


def test_chat_falls_back_to_reasoning_text_when_content_empty(monkeypatch):
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"reasoning_content": "Let me consider... "}}]},
        {"choices": [{"delta": {"reasoning_content": "the answer is 42."}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "timings": {"predicted_n": 8, "predicted_ms": 1000, "prompt_n": 10}},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}},
    ])
    result = LlamaCppEngine().chat("tag", [{"role": "user", "content": "hi"}])
    assert result.response_text == "Let me consider... the answer is 42."
    assert result.generated_tokens == 8


def test_chat_prefers_content_over_reasoning_when_both_present(monkeypatch):
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"content": "answer", "reasoning_content": "reasoning"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "timings": {"predicted_n": 1, "predicted_ms": 1000, "prompt_n": 10}},
    ])
    result = LlamaCppEngine().chat("tag", [{"role": "user", "content": "hi"}])
    assert result.response_text == "answer"


def test_chat_check_loop_raises_early_on_repeated_hedging(monkeypatch):
    _patch_ensure_model(monkeypatch)
    import itertools

    counter = itertools.count(0, 1.0)
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", lambda: next(counter))
    monkeypatch.setattr(llamacpp_module.config, "LOOP_CHECK_INTERVAL", 0)
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"content": "wait, "}}]},
        {"choices": [{"delta": {"content": "wait, "}}]},
        {"choices": [{"delta": {"content": "wait, "}}]},
        {"choices": [{"delta": {"content": "wait, "}}]},
        {"choices": [{"delta": {"content": "wait, still stuck"}}]},
        {"choices": [{"delta": {"content": "this chunk should never be reached"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "timings": {"predicted_n": 6, "predicted_ms": 1000, "prompt_n": 10}},
    ])
    with pytest.raises(EngineLoopDetected) as exc_info:
        LlamaCppEngine().chat("tag", [{"role": "user", "content": "hi"}], check_loop=True)
    assert "loop" in str(exc_info.value).lower()
    assert "wait, wait, wait, wait, wait, still stuck" == exc_info.value.partial_text
    assert "never be reached" not in exc_info.value.partial_text


def test_chat_raises_engine_timeout_type_for_run_measured_calls_compat(monkeypatch):
    # run_measured_calls dispatches by isinstance on shared.py's types, not engine-specific subclasses.
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", _clock(0.0, 0.0, 1.0, 1.0, 100.0))
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"content": "hi"}}]},
        {"choices": [{"delta": {"content": "still going"}}]},
    ])
    with pytest.raises(EngineTimeout):
        LlamaCppEngine().chat("tag", [{"role": "user", "content": "hi"}], timeout=5)


def _chat_result(text, finish_reason, *, calls=None, tokens=2, prompt_tokens=10):
    return {
        "ttft": 0.2,
        "tokens": tokens,
        "tps": 2.0,
        "decode_seconds": 1.0,
        "wall_seconds": 1.2,
        "server_prompt_sec": 0.1,
        "prompt_eval_count": prompt_tokens,
        "response_text": text,
        "tool_calls": calls or [],
        "finish_reason": finish_reason,
    }


@pytest.mark.parametrize("finish_reason", ["stop", "tool_calls", None, "content_filter"])
def test_budgeted_chat_only_retries_literal_length(monkeypatch, finish_reason):
    _patch_ensure_model(monkeypatch)
    calls = []

    def fake_request(self, tag, messages, tools, deadline, num_predict,
                     check_loop, budget_nudged):
        calls.append((messages, num_predict, deadline, budget_nudged))
        return _chat_result("Answer: B", finish_reason)

    monkeypatch.setattr(LlamaCppEngine, "_chat_request", fake_request)
    result = LlamaCppEngine().chat(
        "tag", [{"role": "user", "content": "question"}],
        num_predict=-1, token_budget=10,
    )
    assert (result.response_text, result.budget_nudged) == ("Answer: B", False)
    assert [call[1] for call in calls] == [6]


def test_budgeted_chat_retries_with_original_history_and_grades_only_second(monkeypatch):
    _patch_ensure_model(monkeypatch)
    original = [{"role": "system", "content": "rules"}, {"role": "user", "content": "question"}]
    snapshot = [dict(message) for message in original]
    calls = []
    responses = iter([
        _chat_result("unfinished fragment", "length", tokens=6, prompt_tokens=20),
        _chat_result("Answer: C", "stop", tokens=2, prompt_tokens=30),
    ])

    def fake_request(self, tag, messages, tools, deadline, num_predict,
                     check_loop, budget_nudged):
        calls.append({
            "messages": [dict(message) for message in messages],
            "deadline": deadline,
            "num_predict": num_predict,
            "budget_nudged": budget_nudged,
        })
        return next(responses)

    monkeypatch.setattr(LlamaCppEngine, "_chat_request", fake_request)
    result = LlamaCppEngine().chat(
        "tag", original, timeout=60, num_predict=-1, token_budget=10,
    )
    assert original == snapshot
    assert [call["num_predict"] for call in calls] == [6, 4]
    assert calls[0]["deadline"] == calls[1]["deadline"]
    assert calls[1]["messages"] == snapshot + [
        {"role": "assistant", "content": "unfinished fragment"},
        {"role": "user", "content": config.ACC_FINALIZE_MESSAGE},
    ]
    assert calls[1]["budget_nudged"] is True
    assert result.client_ttft_sec == 0.2
    assert result.generated_tokens == 8
    assert result.tokens_per_sec == 4.0
    assert result.prompt_tokens == 30
    assert result.response_text == "Answer: C"
    assert result.budget_nudged is True


def test_budgeted_chat_two_streams_send_exact_split_and_return_second(monkeypatch):
    _patch_ensure_model(monkeypatch)
    captured = []
    streams = iter([
        [
            {"choices": [{"delta": {"content": "unfinished"}}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}],
             "timings": {"predicted_n": 6, "predicted_ms": 1000, "prompt_n": 10}},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 6}},
        ],
        [
            {"choices": [{"delta": {"content": "Answer: B"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}],
             "timings": {"predicted_n": 2, "predicted_ms": 500, "prompt_n": 20}},
            {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 2}},
        ],
    ])

    def urlopen(req, timeout):
        captured.append({"payload": json.loads(req.data), "timeout": timeout})
        return _FakeResponse(next(streams))

    monkeypatch.setattr(LlamaCppEngine, "_urlopen", staticmethod(urlopen))
    result = LlamaCppEngine().chat(
        "tag", [{"role": "user", "content": "question"}],
        timeout=60, num_predict=-1, token_budget=10,
    )
    assert [request["payload"]["n_predict"] for request in captured] == [6, 4]
    from scripts.runtime.sampling import baseline_sampling_payload
    for request in captured:
        assert {
            key: request["payload"][key] for key in baseline_sampling_payload("llamacpp")
        } == baseline_sampling_payload("llamacpp")
    assert captured[1]["payload"]["messages"][-2:] == [
        {"role": "assistant", "content": "unfinished"},
        {"role": "user", "content": config.ACC_FINALIZE_MESSAGE},
    ]
    assert result.generated_tokens == 8
    assert (result.response_text, result.budget_nudged) == ("Answer: B", True)


def test_budgeted_tool_chat_grades_replacement_call_without_merging(monkeypatch):
    _patch_ensure_model(monkeypatch)
    responses = iter([
        _chat_result("", "length", calls=[{"name": "partial", "arguments": {}, "incomplete": True}]),
        _chat_result("", "tool_calls", calls=[{"name": "final", "arguments": {"x": 1}}]),
    ])
    monkeypatch.setattr(
        LlamaCppEngine, "_chat_request",
        lambda self, *args, **kwargs: next(responses),
    )
    result = LlamaCppEngine().chat_tools(
        "tag", [{"role": "user", "content": "call"}], [{"type": "function"}],
        num_predict=-1, token_budget=10,
    )
    assert result.tool_calls == [{"name": "final", "arguments": {"x": 1}}]
    assert result.budget_nudged is True


def test_budgeted_chat_pass_two_length_raises_gradeable_exhaustion(monkeypatch):
    _patch_ensure_model(monkeypatch)
    responses = iter([
        _chat_result("first", "length"),
        _chat_result("Answer: D", "length"),
    ])
    monkeypatch.setattr(
        LlamaCppEngine, "_chat_request",
        lambda self, *args, **kwargs: next(responses),
    )
    with pytest.raises(EngineBudgetExceeded) as exc_info:
        LlamaCppEngine().chat(
            "tag", [{"role": "user", "content": "q"}],
            num_predict=-1, token_budget=10,
        )
    assert exc_info.value.partial_text == "Answer: D"
    assert exc_info.value.budget_nudged is True


def test_one_token_budget_scores_pass_one_as_exhausted_without_nudge(monkeypatch):
    _patch_ensure_model(monkeypatch)
    requests = []

    def fake_request(self, tag, messages, tools, deadline, num_predict,
                     check_loop, budget_nudged):
        requests.append(num_predict)
        return _chat_result("Answer: B", "length", tokens=1)

    monkeypatch.setattr(LlamaCppEngine, "_chat_request", fake_request)
    with pytest.raises(EngineBudgetExceeded) as exc_info:
        LlamaCppEngine().chat(
            "tag", [{"role": "user", "content": "q"}],
            num_predict=-1, token_budget=1,
        )
    assert requests == [1]
    assert exc_info.value.partial_text == "Answer: B"
    assert exc_info.value.budget_nudged is False


def test_budgeted_chat_pass_two_timeout_keeps_nudge_flag(monkeypatch):
    _patch_ensure_model(monkeypatch)
    calls = 0

    def fake_request(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _chat_result("first", "length")
        raise EngineTimeout("timed out", partial_text="Answer: A", budget_nudged=True)

    monkeypatch.setattr(LlamaCppEngine, "_chat_request", fake_request)
    with pytest.raises(EngineTimeout) as exc_info:
        LlamaCppEngine().chat(
            "tag", [{"role": "user", "content": "q"}],
            num_predict=-1, token_budget=10,
        )
    assert exc_info.value.partial_text == "Answer: A"
    assert exc_info.value.budget_nudged is True


def test_budgeted_chat_deadline_expiry_before_pass_two_scores_pass_one(monkeypatch):
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", _clock(0.0, 61.0))
    monkeypatch.setattr(
        LlamaCppEngine, "_chat_request",
        lambda self, *args, **kwargs: _chat_result("Answer: D", "length"),
    )
    with pytest.raises(EngineTimeout) as exc_info:
        LlamaCppEngine().chat(
            "tag", [{"role": "user", "content": "q"}],
            timeout=60, num_predict=-1, token_budget=10,
        )
    assert exc_info.value.partial_text == "Answer: D"
    assert exc_info.value.budget_nudged is False


def test_budgeted_chat_pass_two_loop_keeps_nudge_flag(monkeypatch):
    _patch_ensure_model(monkeypatch)
    calls = 0

    def fake_request(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _chat_result("first", "length")
        raise EngineLoopDetected(
            "loop", partial_text="wait, wait, wait", budget_nudged=True,
        )

    monkeypatch.setattr(LlamaCppEngine, "_chat_request", fake_request)
    with pytest.raises(EngineLoopDetected) as exc_info:
        LlamaCppEngine().chat(
            "tag", [{"role": "user", "content": "q"}],
            num_predict=-1, token_budget=10,
        )
    assert exc_info.value.partial_text == "wait, wait, wait"
    assert exc_info.value.budget_nudged is True


def test_budgeted_chat_rejects_finite_num_predict(monkeypatch):
    with pytest.raises(ValueError, match="finite num_predict"):
        LlamaCppEngine().chat("tag", [], num_predict=10, token_budget=10)


# ── chat_tools ──

def test_chat_tools_accumulates_fragmented_arguments(monkeypatch):
    # arguments streams as partial JSON text across chunks and must be
    # reassembled by index before parsing; name arrives once up front.
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "get_weather", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"location": "Par'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": 'is", "unit": "celsius"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
         "timings": {"predicted_n": 3, "predicted_ms": 1000, "prompt_n": 20}},
    ])
    result = LlamaCppEngine().chat_tools(
        "tag", [{"role": "user", "content": "weather?"}], tools=[{"type": "function"}])
    assert result.tool_calls == [{"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}]
    assert result.prompt_tokens == 20
    assert result.response_text == ""


def test_chat_tools_zero_tool_calls_returns_empty_list(monkeypatch):
    # A model that answers in prose instead of calling anything yields an
    # empty tool_calls list plus the response text.
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"content": "I can't help with that."}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "timings": {"predicted_n": 5, "predicted_ms": 1000, "prompt_n": 15}},
    ])
    result = LlamaCppEngine().chat_tools(
        "tag", [{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    assert result.tool_calls == []
    assert result.response_text == "I can't help with that."


def test_chat_tools_malformed_arguments_falls_back_to_empty_dict(monkeypatch):
    # Truncated arguments fall back to {} and are marked incomplete, distinct from a genuine empty-argument call.
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "set_timer", "arguments": '{"minutes":'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
         "timings": {"predicted_n": 1, "predicted_ms": 1000, "prompt_n": 10}},
    ])
    result = LlamaCppEngine().chat_tools(
        "tag", [{"role": "user", "content": "timer"}], tools=[{"type": "function"}])
    assert result.tool_calls == [{"name": "set_timer", "arguments": {}, "incomplete": True}]


def test_chat_tools_multiple_calls_ordered_by_index(monkeypatch):
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 1, "id": "c2", "function": {"name": "second", "arguments": "{}"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "first", "arguments": "{}"}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
         "timings": {"predicted_n": 2, "predicted_ms": 1000, "prompt_n": 10}},
    ])
    result = LlamaCppEngine().chat_tools(
        "tag", [{"role": "user", "content": "two"}], tools=[{"type": "function"}])
    assert [call["name"] for call in result.tool_calls] == ["first", "second"]


def test_chat_tools_timeout_serializes_completed_fragmented_call(monkeypatch):
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", _clock(0.0, 0.0, 1.0, 1.0, 2.0, 10.0))
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "get_weather", "arguments": '{"city":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '"Paris"}'}}]}}]},
        {"choices": [{"delta": {"content": "late"}}]},
    ])
    with pytest.raises(EngineTimeout) as exc_info:
        LlamaCppEngine().chat_tools("tag", [], [], timeout=5)
    assert json.loads(exc_info.value.partial_text) == [
        {"name": "get_weather", "arguments": {"city": "Paris"}},
    ]


def test_chat_tools_timeout_marks_incomplete_argument_evidence(monkeypatch):
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", _clock(0.0, 0.0, 1.0, 10.0))
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "set_timer", "arguments": '{"minutes":'}}]}}]},
    ])
    with pytest.raises(EngineTimeout) as exc_info:
        LlamaCppEngine().chat_tools("tag", [], [], timeout=5)
    assert json.loads(exc_info.value.partial_text) == [
        {"name": "set_timer", "arguments": {}, "incomplete": True},
    ]


def test_chat_tools_timeout_with_text_only_keeps_text(monkeypatch):
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", _clock(0.0, 0.0, 1.0, 10.0))
    _patch_urlopen(monkeypatch, [{"choices": [{"delta": {"content": "not calling"}}]}])
    with pytest.raises(EngineTimeout) as exc_info:
        LlamaCppEngine().chat_tools("tag", [], [], timeout=5)
    assert exc_info.value.partial_text == "not calling"


def test_chat_tools_falls_back_to_reasoning_text_when_content_empty(monkeypatch):
    # Mirrors chat()'s reasoning fallback — a declining model can stream its whole turn via reasoning_content.
    _patch_ensure_model(monkeypatch)
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"reasoning_content": "Let me consider... "}}]},
        {"choices": [{"delta": {"reasoning_content": "no tool fits."}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "timings": {"predicted_n": 8, "predicted_ms": 1000, "prompt_n": 10}},
    ])
    result = LlamaCppEngine().chat_tools(
        "tag", [{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    assert result.response_text == "Let me consider... no tool fits."
    assert result.tool_calls == []
    assert result.generated_tokens == 8


def test_chat_tools_check_loop_raises_during_reasoning_phase(monkeypatch):
    # Regression: check_loop must inspect reasoning_content too, not just content.
    _patch_ensure_model(monkeypatch)
    import itertools

    counter = itertools.count(0, 1.0)
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", lambda: next(counter))
    monkeypatch.setattr(llamacpp_module.config, "LOOP_CHECK_INTERVAL", 0)
    _patch_urlopen(monkeypatch, [
        {"choices": [{"delta": {"reasoning_content": "wait, "}}]},
        {"choices": [{"delta": {"reasoning_content": "wait, "}}]},
        {"choices": [{"delta": {"reasoning_content": "wait, "}}]},
        {"choices": [{"delta": {"reasoning_content": "wait, "}}]},
        {"choices": [{"delta": {"reasoning_content": "wait, still stuck"}}]},
        {"choices": [{"delta": {"reasoning_content": "this chunk should never be reached"}}]},
    ])
    with pytest.raises(EngineLoopDetected) as exc_info:
        LlamaCppEngine().chat_tools(
            "tag", [{"role": "user", "content": "hi"}], tools=[{"type": "function"}], check_loop=True)
    assert "wait, wait, wait, wait, wait, still stuck" == exc_info.value.partial_text
    assert "never be reached" not in exc_info.value.partial_text


def test_chat_tools_timeout_with_reasoning_only_keeps_reasoning_text(monkeypatch):
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", _clock(0.0, 0.0, 1.0, 10.0))
    _patch_urlopen(monkeypatch, [{"choices": [{"delta": {"reasoning_content": "still thinking"}}]}])
    with pytest.raises(EngineTimeout) as exc_info:
        LlamaCppEngine().chat_tools("tag", [], [], timeout=5)
    assert exc_info.value.partial_text == "still thinking"


# ── embed ──

def test_embed_returns_embeddings_in_index_order(monkeypatch):
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(
        "scripts.runtime.engines.llamacpp.requests.post",
        lambda url, json=None, timeout=None: type("R", (), {
            "ok": True,
            "json": lambda self: {"data": [
                {"index": 1, "embedding": [0.2]},
                {"index": 0, "embedding": [0.1]},
            ]},
        })(),
    )
    result = LlamaCppEngine().embed("nomic-embed-text", ["a", "b"])
    assert result.embeddings == [[0.1], [0.2]]
    assert result.client_wall_sec >= 0


def test_embed_raises_on_rejected_request(monkeypatch):
    _patch_ensure_model(monkeypatch)
    monkeypatch.setattr(
        "scripts.runtime.engines.llamacpp.requests.post",
        lambda url, json=None, timeout=None: type("R", (), {
            "ok": False, "status_code": 500, "json": lambda self: {"error": "oom"},
            "text": "oom",
        })(),
    )
    with pytest.raises(RuntimeError, match="oom"):
        LlamaCppEngine().embed("nomic-embed-text", ["a"])


# ══════════════════════════════════════════════════════════════════════════
#  Group 3 — maintenance seams
# ══════════════════════════════════════════════════════════════════════════


def test_is_connection_crash_true_for_connection_error():
    assert LlamaCppEngine().is_connection_crash(requests.exceptions.ConnectionError("down")) is True


def test_is_connection_crash_false_for_unrelated_error():
    assert LlamaCppEngine().is_connection_crash(ValueError("bad json")) is False


@pytest.mark.parametrize(("listing", "expected"), [
    ("CUDA0: NVIDIA RTX", "cuda"),
    ("HIP0: AMD Radeon", "rocm"),
    ("Metal: Apple M3", "metal"),
    ("Available devices:\n  MTL0: Apple M4 (18186 MiB, 18185 MiB free)", "metal"),
    ("SYCL0: Intel Arc", "xpu"),
    ("Vulkan0: AMD Radeon", "vulkan"),
    ("Available devices:\n", "cpu"),
])
def test_backend_from_device_listing(listing, expected):
    from scripts.runtime.llamacpp_tools import llamacpp_backend_from_device_listing
    assert llamacpp_backend_from_device_listing(listing) == expected


def test_runtime_backend_uses_binary_device_listing_and_cpu_override(monkeypatch):
    completed = type("Completed", (), {
        "stdout": "Available devices:\n  Vulkan0: AMD Radeon",
        "stderr": "",
        "returncode": 0,
    })()
    monkeypatch.setattr(LlamaCppEngine, "_binary_path", staticmethod(lambda: "llama-server"))
    monkeypatch.setattr(llamacpp_module.subprocess, "run", lambda *args, **kwargs: completed)
    engine = LlamaCppEngine()
    assert engine.runtime_backend("rocm") == "vulkan"
    assert engine._expected_backend == "vulkan"
    assert engine.runtime_backend("rocm", cpu_only=True) == "cpu"


def test_runtime_backend_preserves_hardware_accelerator_when_device_probe_falls_to_cpu(
        monkeypatch):
    monkeypatch.setattr(LlamaCppEngine, "_binary_path", staticmethod(lambda: "llama-server"))
    monkeypatch.setattr(llamacpp_module, "probe_llamacpp_backend", lambda *_args, **_kwargs: "cpu")
    engine = LlamaCppEngine()

    assert engine.runtime_backend("rocm") == "cpu"
    assert engine._expected_backend == "rocm"


def test_accelerated_preflight_rejects_runtime_device_loss(monkeypatch, tmp_path):
    monkeypatch.setattr(LlamaCppEngine, "_binary_path", staticmethod(lambda: "llama-server"))
    monkeypatch.setattr(LlamaCppEngine, "_models_dir", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(llamacpp_module, "probe_llamacpp_backend", lambda *_args, **_kwargs: "cpu")
    errors = []
    monkeypatch.setattr(llamacpp_module.Shared, "err", errors.append)
    engine = LlamaCppEngine()
    engine._expected_backend = "rocm"

    assert not engine.ensure_running()
    assert errors == ["llamacpp requires rocm, but its runtime exposes cpu"]


def test_runtime_backend_sources_oneapi_for_xpu_probe_and_processes(monkeypatch):
    environment = {"PATH": "/opt/intel/oneapi/bin", "LD_LIBRARY_PATH": "/opt/intel/lib"}
    completed = type("Completed", (), {
        "stdout": "Available devices:\n  SYCL0: Intel Arc Pro B65",
        "stderr": "",
        "returncode": 0,
    })()
    calls = []
    monkeypatch.setattr(LlamaCppEngine, "_binary_path", staticmethod(lambda: "llama-server"))
    monkeypatch.setattr(llamacpp_module, "oneapi_environment", lambda: environment)

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return completed

    monkeypatch.setattr(llamacpp_module.subprocess, "run", run)
    engine = LlamaCppEngine()

    assert engine.runtime_backend("xpu") == "xpu"
    assert engine.process_environment() == environment
    assert calls[0][1]["env"] == environment


def test_ensure_model_fast_path_does_not_probe_health(monkeypatch):
    engine = LlamaCppEngine()
    engine._loaded_tag = "tag"
    engine._loaded_num_ctx = 2048
    engine._loaded_embedding = False
    engine._loaded_n_parallel = 4
    engine._proc = cast(subprocess.Popen, type("Proc", (), {"poll": lambda self: None})())
    monkeypatch.setattr(engine, "available", lambda: pytest.fail("health probe should not run"))
    engine._ensure_model("tag", 2048, n_parallel=4)


def test_ensure_model_deadline_stops_in_progress_process(monkeypatch, tmp_path):
    class Proc:
        returncode = None

        def __init__(self):
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

        def kill(self):
            self.terminated = True

    proc = Proc()
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"x")
    monkeypatch.setattr(LlamaCppEngine, "_resolve_model_files", classmethod(lambda cls, tag: [model_path]))
    monkeypatch.setattr(LlamaCppEngine, "_binary_path", staticmethod(lambda: "llama-server"))
    monkeypatch.setattr(llamacpp_module.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(llamacpp_module.Shared, "_managed_procs", [])
    monkeypatch.setattr(llamacpp_module.time, "perf_counter", _clock(0.0, 0.0, 0.0, 2.0))
    engine = LlamaCppEngine()
    monkeypatch.setattr(engine, "available", lambda: False)

    with pytest.raises(EngineTimeout):
        engine._ensure_model("tag", 2048, deadline=1.0)

    assert proc.terminated is True
    assert engine._proc is None


@pytest.mark.parametrize(("gpu_visible", "expected_ngl"), [
    (True, "auto"),
    (False, "0"),
])
def test_ensure_model_ngl_lets_llama_server_fit_layers(monkeypatch, tmp_path, gpu_visible, expected_ngl):
    """-ngl must be "auto" (not a forced max) so --fit can offload only what fits in
    free VRAM, instead of OOM-ing when all layers are forced onto the GPU."""
    captured_args = {}

    class Proc:
        returncode = None

        def poll(self):
            return None

    def fake_popen(args, **kwargs):
        captured_args["args"] = args
        captured_args["env"] = kwargs.get("env")
        return Proc()

    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"x")
    monkeypatch.setattr(LlamaCppEngine, "_resolve_model_files", classmethod(lambda cls, tag: [model_path]))
    monkeypatch.setattr(LlamaCppEngine, "_binary_path", staticmethod(lambda: "llama-server"))
    monkeypatch.setattr(llamacpp_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(llamacpp_module.Shared, "_managed_procs", [])
    engine = LlamaCppEngine()
    engine._process_env = {"ONEAPI": "ready"}
    monkeypatch.setattr(engine, "available", lambda: True)
    monkeypatch.setattr(engine, "_fetch_props", lambda: {"model_path": model_path.name})
    engine._gpu_visible = gpu_visible

    engine._ensure_model("tag", 2048)

    args = captured_args["args"]
    assert args[args.index("-ngl") + 1] == expected_ngl
    assert args[args.index("-lv") + 1] == "4"
    assert captured_args["env"] == {"ONEAPI": "ready"}


@pytest.mark.parametrize(("tag", "embedding", "expected_tokens", "has_draft"), [
    ("mtp:embedded", False, "3", False),
    ("mtp:separate", False, "2", True),
    ("mtp:embedded", True, None, False),
])
def test_ensure_model_configures_cataloged_llamacpp_mtp(
        fake_catalog, monkeypatch, tag, embedding, expected_tokens, has_draft):
    captured_args = {}

    class Proc:
        returncode = None

        def poll(self):
            return None

    primary_name = "embedded.gguf" if tag == "mtp:embedded" else "model.gguf"
    _write_model_file(fake_catalog, tag, primary_name, b"model")
    if has_draft:
        _write_model_file(fake_catalog, tag, "draft.gguf", b"draft")
    monkeypatch.setattr(LlamaCppEngine, "_binary_path", staticmethod(lambda: "llama-server"))
    def popen(args, **_kwargs):
        captured_args["args"] = args
        return Proc()

    monkeypatch.setattr(llamacpp_module.subprocess, "Popen", popen)
    monkeypatch.setattr(llamacpp_module.Shared, "_managed_procs", [])
    engine = LlamaCppEngine()
    engine.set_mtp_enabled(True)
    monkeypatch.setattr(engine, "available", lambda: True)
    monkeypatch.setattr(engine, "_fetch_props", lambda: {"model_path": primary_name})

    engine._ensure_model(tag, 2048, embedding=embedding)

    args = captured_args["args"]
    if expected_tokens is None:
        assert "--spec-type" not in args
        assert engine._loaded_mtp_config is None
    else:
        assert args[args.index("--spec-type") + 1] == "draft-mtp"
        assert args[args.index("--spec-draft-n-max") + 1] == expected_tokens
        assert ("--spec-draft-model" in args) is has_draft
        if has_draft:
            assert Path(args[args.index("--spec-draft-model") + 1]).name == "draft.gguf"
        assert engine._loaded_mtp_config is not None


def test_ensure_model_restart_key_distinguishes_mtp_mode(monkeypatch):
    engine = LlamaCppEngine()
    engine._loaded_tag = "mtp:embedded"
    engine._loaded_num_ctx = 2048
    engine._loaded_embedding = False
    engine._loaded_n_parallel = 1
    engine._loaded_mtp_config = {"num_speculative_tokens": 3}
    engine._proc = cast(subprocess.Popen, type("Proc", (), {"poll": lambda self: None})())
    engine.set_mtp_enabled(True)
    monkeypatch.setattr(llamacpp_module, "LLM_MODELS", [_FAKE_CATALOG[-2]])
    monkeypatch.setattr(engine, "available", lambda: pytest.fail("health probe should not run"))
    engine._ensure_model("mtp:embedded", 2048)

    engine.set_mtp_enabled(False)
    monkeypatch.setattr(
        LlamaCppEngine, "_resolve_model_files",
        classmethod(lambda cls, tag: [Path("model.gguf")]),
    )
    monkeypatch.setattr(
        engine, "_stop_process",
        lambda: (_ for _ in ()).throw(RuntimeError("restart required")),
    )
    with pytest.raises(RuntimeError, match="restart required"):
        engine._ensure_model("mtp:embedded", 2048)


@pytest.mark.parametrize(("n_parallel", "num_ctx", "expected_ctx_arg"), [
    (1, 2048, "2048"),
    (4, 2048, "8192"),
])
def test_ensure_model_always_pins_parallel_flag(monkeypatch, tmp_path, n_parallel, num_ctx, expected_ctx_arg):
    """--parallel must be pinned even at 1 — see docs/engines.md's "--parallel is always pinned"."""
    captured_args = {}

    class Proc:
        returncode = None

        def poll(self):
            return None

    def fake_popen(args, **kwargs):
        captured_args["args"] = args
        return Proc()

    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"x")
    monkeypatch.setattr(LlamaCppEngine, "_resolve_model_files", classmethod(lambda cls, tag: [model_path]))
    monkeypatch.setattr(LlamaCppEngine, "_binary_path", staticmethod(lambda: "llama-server"))
    monkeypatch.setattr(llamacpp_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(llamacpp_module.Shared, "_managed_procs", [])
    monkeypatch.setattr(config, "LLAMACPP_NO_REPACK", True)
    engine = LlamaCppEngine()
    monkeypatch.setattr(engine, "available", lambda: True)
    monkeypatch.setattr(engine, "_fetch_props", lambda: {"model_path": model_path.name})

    engine._ensure_model("tag", num_ctx, n_parallel=n_parallel)

    args = captured_args["args"]
    assert "--parallel" in args
    assert args[args.index("--parallel") + 1] == str(n_parallel)
    assert args[args.index("-c") + 1] == expected_ctx_arg
    assert "--no-repack" in args
    assert engine._loaded_n_parallel == n_parallel


def test_reachable_or_abort_always_true_regardless_of_available(monkeypatch):
    # See reachable_or_abort's docstring — no shared daemon to check between models.
    monkeypatch.setattr(LlamaCppEngine, "available", lambda self: False)
    assert LlamaCppEngine().reachable_or_abort() is True
    monkeypatch.setattr(LlamaCppEngine, "available", lambda self: True)
    assert LlamaCppEngine().reachable_or_abort() is True


def test_wait_for_recovery_always_true_regardless_of_available(monkeypatch):
    # No passive self-heal to poll for (see wait_for_recovery's docstring) —
    # recovery happens synchronously on the next generate/chat/embed call.
    monkeypatch.setattr(LlamaCppEngine, "available", lambda self: False)
    assert LlamaCppEngine().wait_for_recovery() is True


def test_unload_stops_process_when_tag_matches(monkeypatch):
    engine = LlamaCppEngine()
    engine._loaded_tag = "phi4-mini"
    stopped = []
    monkeypatch.setattr(LlamaCppEngine, "_stop_process", lambda self: stopped.append(True))
    engine.unload("phi4-mini")
    assert stopped == [True]


def test_unload_noop_when_tag_does_not_match(monkeypatch):
    engine = LlamaCppEngine()
    engine._loaded_tag = "other-model"
    stopped = []
    monkeypatch.setattr(LlamaCppEngine, "_stop_process", lambda self: stopped.append(True))
    engine.unload("phi4-mini")
    assert stopped == []


def test_unload_all_unloads_the_loaded_model(monkeypatch):
    engine = LlamaCppEngine()
    engine._loaded_tag = "phi4-mini"
    unloaded = []
    monkeypatch.setattr(LlamaCppEngine, "unload", lambda self, tag: unloaded.append(tag))
    engine.unload_all()
    assert unloaded == ["phi4-mini"]


def test_unload_all_noop_when_nothing_loaded(monkeypatch):
    engine = LlamaCppEngine()
    unloaded = []
    monkeypatch.setattr(LlamaCppEngine, "unload", lambda self, tag: unloaded.append(tag))
    engine.unload_all()
    assert unloaded == []


def test_wait_until_unloaded_true_once_tag_no_longer_loaded():
    engine = LlamaCppEngine()
    engine._loaded_tag = None
    assert engine.wait_until_unloaded("phi4-mini") is True


def test_wait_until_unloaded_false_while_still_loaded():
    engine = LlamaCppEngine()
    engine._loaded_tag = "phi4-mini"
    assert engine.wait_until_unloaded("phi4-mini") is False


def test_tensor_split_uses_f16_cache_and_cpu_mode_disables_splitting(monkeypatch):
    monkeypatch.setattr(config, "LLAMACPP_GPU_SPLIT_MODE", "tensor")
    monkeypatch.setattr(llamacpp_module, "load_setup_config", lambda path: {})
    assert LlamaCppEngine.gpu_split_args(include_cache=True) == [
        "--split-mode", "tensor", "--cache-type-k", "f16", "--cache-type-v", "f16",
    ]
    assert LlamaCppEngine.gpu_split_args(include_cache=True, cpu_only=True) == [
        "--split-mode", "none", "--cache-type-k", config.LLAMACPP_KV_CACHE_TYPE,
        "--cache-type-v", config.LLAMACPP_KV_CACHE_TYPE,
    ]


def test_single_gpu_mode_disables_splitting_but_keeps_normal_cache(monkeypatch):
    monkeypatch.setattr(config, "LLAMACPP_GPU_SPLIT_MODE", "single")
    monkeypatch.setattr(llamacpp_module, "load_setup_config", lambda path: {})
    assert LlamaCppEngine.gpu_split_args(include_cache=True) == [
        "--split-mode", "none", "--cache-type-k", config.LLAMACPP_KV_CACHE_TYPE,
        "--cache-type-v", config.LLAMACPP_KV_CACHE_TYPE,
    ]


def test_single_gpu_mode_pins_first_prioritized_vulkan_device(monkeypatch):
    monkeypatch.setattr(config, "LLAMACPP_GPU_SPLIT_MODE", "single")
    monkeypatch.setattr(llamacpp_module, "load_setup_config", lambda path: {
        "gpu": {"devices": [
            {"backend": "vulkan", "type": "integrated", "vram_gb": 16},
            {"backend": "vulkan", "type": "discrete", "vram_gb": 32},
        ]},
    })
    assert LlamaCppEngine.gpu_split_args() == [
        "--split-mode", "none", "--device", "Vulkan1",
    ]


def test_layer_mode_applies_recorded_asymmetric_gpu_capacity(monkeypatch):
    monkeypatch.setattr(config, "LLAMACPP_GPU_SPLIT_MODE", "layer")
    monkeypatch.setattr(
        "scripts.runtime.engines.llamacpp.load_setup_config",
        lambda _path: {"gpu": {"devices": [
            {"backend": "vulkan", "vram_gb": 31.984375},
            {"backend": "vulkan", "vram_gb": 15.921875},
        ]}},
    )

    assert LlamaCppEngine.gpu_split_args() == [
        "--split-mode", "layer", "--device", "Vulkan0,Vulkan1",
        "--tensor-split", "2,1",
    ]


def test_explicit_gpu_ratio_is_omitted_for_single_and_cpu_modes(monkeypatch):
    monkeypatch.setattr(
        "scripts.runtime.engines.llamacpp.load_setup_config",
        lambda _path: {"gpu": {"devices": [
            {"backend": "vulkan", "vram_gb": 32},
            {"backend": "vulkan", "vram_gb": 16},
        ]}},
    )
    monkeypatch.setattr(config, "LLAMACPP_GPU_SPLIT_MODE", "single")
    assert LlamaCppEngine.gpu_split_args() == [
        "--split-mode", "none", "--device", "Vulkan0",
    ]
    monkeypatch.setattr(config, "LLAMACPP_GPU_SPLIT_MODE", "layer")
    assert LlamaCppEngine.gpu_split_args(cpu_only=True) == ["--split-mode", "none"]


# ── server identity on /props ──

def test_serving_model_file_reads_the_modern_model_path_key():
    from scripts.runtime.engines.llamacpp import LlamaCppEngine
    props = {"model_path": "/models/gemma3-1b-it-q4_K_M.gguf"}
    assert LlamaCppEngine.serving_model_file(props) == "gemma3-1b-it-q4_K_M.gguf"


def test_serving_model_file_falls_back_through_older_llama_server_shapes():
    from scripts.runtime.engines.llamacpp import LlamaCppEngine
    nested = {"default_generation_settings": {"model": "/models/mxbai-embed-large.gguf"}}
    assert LlamaCppEngine.serving_model_file(nested) == "mxbai-embed-large.gguf"
    assert LlamaCppEngine.serving_model_file({"model": "/m/a.gguf"}) == "a.gguf"


def test_serving_model_file_handles_windows_paths():
    from scripts.runtime.engines.llamacpp import LlamaCppEngine
    props = {"model_path": r"C:\\models\\granite4.1-3b-q4_K_M.gguf"}
    assert LlamaCppEngine.serving_model_file(props) == "granite4.1-3b-q4_K_M.gguf"


@pytest.mark.parametrize("props", [
    None, {}, "not-a-dict", {"model_path": ""}, {"model_path": "   "},
    {"model_path": None}, {"default_generation_settings": None},
    {"default_generation_settings": {}},
])
def test_serving_model_file_is_none_when_the_server_cannot_be_identified(props):
    """An unidentifiable server is not treated as a mismatch — the load proceeds
    rather than failing on a llama-server build that reports nothing useful."""
    from scripts.runtime.engines.llamacpp import LlamaCppEngine
    assert LlamaCppEngine.serving_model_file(props) is None


def test_serving_model_file_detects_a_foreign_server_on_the_port():
    """The embeddings server answering a concurrency runner's requests: /props names
    the embedding model, so the requested generation model is a mismatch."""
    from scripts.runtime.engines.llamacpp import LlamaCppEngine
    serving = LlamaCppEngine.serving_model_file({"model_path": "/m/mxbai-embed-large.gguf"})
    assert serving is not None and serving != "gemma3-1b-it-q4_K_M.gguf"
