"""Read-only Qwen4 experimental predictor and runtime capability checks."""

import json
from pathlib import Path
import subprocess


QWEN_MTP_ARCHITECTURES = {"Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM"}
MAX_HEADER_BYTES = 64 * 1024 * 1024
MTP_RUNTIME_PROBE = (
    "import sys; from vllm import ModelRegistry; from vllm.platforms import current_platform; "
    "a = set(ModelRegistry.get_supported_archs()); "
    "print('LAB_MTP_SUPPORTED' if 'Qwen4ExpMTP' in a and "
    "sys.argv[1] in a and "
    "(current_platform.is_cuda() or current_platform.is_rocm()) else 'unsupported')"
)


def probe_qwen_mtp_runtime(python: Path, architecture: str, *, run=subprocess.run) -> bool:
    try:
        result = run([str(python), "-c", MTP_RUNTIME_PROBE, architecture], capture_output=True,
                     text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    lines = result.stdout.strip().splitlines()
    return result.returncode == 0 and bool(lines) and lines[-1] == "LAB_MTP_SUPPORTED"


def _tensor_names(path: Path) -> set[str]:
    with path.open("rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
        file_size = path.stat().st_size
        if not 0 < header_size <= min(MAX_HEADER_BYTES, file_size - 8):
            return set()
        header = json.loads(stream.read(header_size))
    if not isinstance(header, dict):
        return set()
    names = set()
    for name, value in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(value, dict):
            return set()
        shape, offsets = value.get("shape"), value.get("data_offsets")
        if (not isinstance(shape, list)
                or not all(type(dim) is int and dim > 0 for dim in shape)
                or not isinstance(value.get("dtype"), str)
                or not isinstance(offsets, list) or len(offsets) != 2
                or not all(type(offset) is int for offset in offsets)
                or not 0 <= offsets[0] < offsets[1] <= file_size - 8 - header_size):
            return set()
        names.add(name)
    return names


def _mtp_name(name: str) -> str:
    for prefix in ("model.language_model.", "language_model."):
        if name.startswith(prefix):
            name = name.removeprefix(prefix)
            break
    return name.removeprefix("model.")


def _predictor_present(names: set[str], layers: int, experts: int) -> bool:
    def has(stem):
        return stem in names or stem + ".weight" in names

    required = ["mtp.fc_embedding", "mtp.fc_hidden", "mtp.pre_fc_norm_embedding",
                "mtp.pre_fc_norm_hidden"]
    required += [f"mtp.hyper_connection_mixer.{part}" for part in (
        "hc_norm", "input_mix_weight_down", "input_mix_weight_up",
    )]
    for layer in range(layers):
        prefix = f"mtp.layers.{layer}"
        required += [f"{prefix}.self_attn.{part}" for part in (
            "q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm",
            "indexer.index_qk_proj", "indexer.q_layernorm", "indexer.k_layernorm",
        )]
        required += [f"{prefix}.{branch}_hyper_connection.{part}"
                     for branch in ("attn", "mlp") for part in (
                         "hc_norm", "input_mix_weight_down", "input_mix_weight_up",
                     )]
        required += [f"{prefix}.mlp.{part}" for part in (
            "gate", "shared_expert_gate", "shared_expert.gate_proj",
            "shared_expert.up_proj", "shared_expert.down_proj",
        )]
        packed = all(has(f"{prefix}.mlp.experts.{part}")
                     for part in ("gate_up_proj", "down_proj"))
        if not packed:
            required += [f"{prefix}.mlp.experts.{expert}.{part}"
                         for expert in range(experts)
                         for part in ("gate_proj", "up_proj", "down_proj")]
    return all(has(name) for name in required)


def snapshot_qwen_mtp(snapshot: Path) -> dict | None:
    """Require declared predictor layers and real local tensors, not names alone."""
    try:
        config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            return None
        architectures = config.get("architectures")
        if (not isinstance(architectures, list) or len(architectures) != 1
                or architectures[0] not in QWEN_MTP_ARCHITECTURES):
            return None
        text = config.get("text_config", config)
        if (config.get("model_type") not in {"qwen4_exp", "qwen4_exp_text"}
                or not isinstance(text, dict)
                or text.get("model_type") != "qwen4_exp_text"):
            return None
        layers = text.get("mtp_num_hidden_layers", text.get("num_nextn_predict_layers"))
        experts = text.get("num_experts")
        if (type(layers) is not int or not 0 < layers <= 128
                or type(experts) is not int or not 0 < experts <= 4096):
            return None
        nested = text.get("mtp")
        if nested is not None and (not isinstance(nested, dict)
                                   or nested.get("num_hidden_layers") != layers):
            return None
        index = snapshot / "model.safetensors.index.json"
        if index.exists() or index.is_symlink():
            data = json.loads(index.read_text(encoding="utf-8"))
            weights = data.get("weight_map") if isinstance(data, dict) else None
            if not isinstance(weights, dict) or not weights:
                return None
            if any(not isinstance(file, str) or Path(file).name != file
                   or not file.endswith(".safetensors") or not (snapshot / file).is_file()
                   for file in weights.values()):
                return None
            by_file: dict[str, set[str]] = {}
            for name, file in weights.items():
                if _mtp_name(name).startswith("mtp."):
                    by_file.setdefault(file, set()).add(name)
            names = set()
            for file, expected in by_file.items():
                if not expected <= _tensor_names(snapshot / file):
                    return None
                names.update(_mtp_name(name) for name in expected)
        else:
            files = list(snapshot.glob("*.safetensors"))
            if len(files) != 1:
                return None
            names = {_mtp_name(name) for name in _tensor_names(files[0])}
        if _predictor_present(names, layers, experts):
            return {"method": "mtp", "num_speculative_tokens": 2}
    except (OSError, ValueError, TypeError):
        return None
    return None
