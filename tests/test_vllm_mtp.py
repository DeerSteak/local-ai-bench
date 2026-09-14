import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.setup.vllm_mtp import MTP_RUNTIME_PROBE, probe_qwen_mtp_runtime, snapshot_qwen_mtp


@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("prefix", ["", "model.", "model.language_model.", "language_model."])
@pytest.mark.parametrize("indexed", [True, False])
def test_predictor_detection_accepts_supported_checkpoint_layouts(tmp_path, qwen_mtp_snapshot,
                                                                 packed, prefix, indexed):
    snapshot = qwen_mtp_snapshot(tmp_path, packed=packed, prefix=prefix, indexed=indexed)
    assert snapshot_qwen_mtp(snapshot) == {"method": "mtp", "num_speculative_tokens": 2}


@pytest.mark.parametrize("field,value", [
    ("mtp_num_hidden_layers", 0), ("mtp_num_hidden_layers", -1),
    ("mtp_num_hidden_layers", True), ("mtp_num_hidden_layers", "1"),
    ("mtp_num_hidden_layers", None), ("num_experts", 0), ("num_experts", True),
    ("model_type", "qwen3_5"), ("mtp", {"num_hidden_layers": 2}),
])
def test_predictor_detection_rejects_invalid_metadata(tmp_path, qwen_mtp_snapshot, field, value):
    snapshot = qwen_mtp_snapshot(tmp_path)
    path = snapshot / "config.json"
    data = json.loads(path.read_text())
    data["text_config"][field] = value
    path.write_text(json.dumps(data))
    assert snapshot_qwen_mtp(snapshot) is None


def test_text_only_and_legacy_layer_count_config(tmp_path, qwen_mtp_snapshot):
    snapshot = qwen_mtp_snapshot(tmp_path, layers=2)
    path = snapshot / "config.json"
    text = json.loads(path.read_text())["text_config"]
    text["num_nextn_predict_layers"] = text.pop("mtp_num_hidden_layers")
    text["architectures"] = ["Qwen4ExpForCausalLM"]
    path.write_text(json.dumps(text))
    assert snapshot_qwen_mtp(snapshot) is not None


@pytest.mark.parametrize("failure", ["missing_shard", "missing_head", "missing_expert", "index_lie",
                                      "broken_json", "truncated", "bad_offsets", "path", "unknown_arch"])
def test_predictor_detection_rejects_incomplete_or_untrusted_artifacts(tmp_path, qwen_mtp_snapshot, failure):
    snapshot = qwen_mtp_snapshot(tmp_path, packed=False)
    weights = snapshot / "model.safetensors"
    index = snapshot / "model.safetensors.index.json"
    data = json.loads(index.read_text())
    if failure == "missing_shard":
        weights.unlink()
    elif failure in {"missing_head", "missing_expert"}:
        key = "mtp.fc_hidden.weight" if failure == "missing_head" else "mtp.layers.0.mlp.experts.1.up_proj.weight"
        del data["weight_map"][key]
        index.write_text(json.dumps(data))
    elif failure == "index_lie":
        data["weight_map"]["mtp.fake.weight"] = "model.safetensors"
        index.write_text(json.dumps(data))
    elif failure == "broken_json":
        index.write_text("{")
    elif failure == "truncated":
        weights.write_bytes(weights.read_bytes()[:-1])
    elif failure == "bad_offsets":
        raw = weights.read_bytes()
        size = int.from_bytes(raw[:8], "little")
        header = json.loads(raw[8:8+size])
        header["mtp.fc_hidden.weight"]["data_offsets"] = [0, 9999999]
        encoded = json.dumps(header).encode()
        weights.write_bytes(len(encoded).to_bytes(8, "little") + encoded + raw[8+size:])
    elif failure == "path":
        data["weight_map"]["mtp.fc_hidden.weight"] = "../outside.safetensors"
        index.write_text(json.dumps(data))
    else:
        path = snapshot / "config.json"
        config = json.loads(path.read_text())
        config["architectures"] = ["SomeOtherModel"]
        path.write_text(json.dumps(config))
    assert snapshot_qwen_mtp(snapshot) is None


@pytest.mark.parametrize("verdict,code,expected", [("LAB_MTP_SUPPORTED", 0, True),
    ("vLLM startup log\nLAB_MTP_SUPPORTED", 0, True), ("unsupported", 0, False),
    ("LAB_MTP_SUPPORTED", 1, False), ("noise", 0, False), ("", 0, False)])
def test_runtime_probe_requires_explicit_supported_result(tmp_path, verdict, code, expected):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=verdict, returncode=code)
    assert probe_qwen_mtp_runtime(tmp_path / "python", "Qwen4ExpForCausalLM", run=run) is expected
    assert calls[0][-1] == "Qwen4ExpForCausalLM"
    assert "Qwen4ExpMTP" in calls[0][2]


def test_runtime_probe_failure_is_not_support(tmp_path):
    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired("probe", 30)
    assert not probe_qwen_mtp_runtime(tmp_path / "python", "Qwen4ExpForCausalLM", run=run)


@pytest.mark.parametrize("target,predictor,backend,expected", [
    (True, True, "cuda", True), (True, True, "rocm", True),
    (True, True, "cpu", False), (False, True, "cuda", False),
    (True, False, "cuda", False),
])
def test_runtime_probe_checks_both_architectures_and_backend(
        monkeypatch, capsys, target, predictor, backend, expected):
    architectures = (["Qwen4ExpForCausalLM"] if target else []) + (["Qwen4ExpMTP"] if predictor else [])
    registry = SimpleNamespace(get_supported_archs=lambda: architectures)
    platform = SimpleNamespace(is_cuda=lambda: backend == "cuda", is_rocm=lambda: backend == "rocm")
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(ModelRegistry=registry))
    monkeypatch.setitem(sys.modules, "vllm.platforms", SimpleNamespace(current_platform=platform))
    monkeypatch.setattr(sys, "argv", ["probe", "Qwen4ExpForCausalLM"])
    exec(MTP_RUNTIME_PROBE, {})
    assert capsys.readouterr().out.strip() == ("LAB_MTP_SUPPORTED" if expected else "unsupported")
