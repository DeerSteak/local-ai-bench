"""Shared pytest configuration for package-based script imports."""

import ctypes
import sys

import pytest


def tk_display_unavailable_reason(platform, load_library=None):
    if platform != "darwin":
        return None
    if load_library is None:
        load_library = ctypes.CDLL
    graphics = load_library("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
    foundation = load_library("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    graphics.CGSessionCopyCurrentDictionary.restype = ctypes.c_void_p
    foundation.CFRelease.argtypes = [ctypes.c_void_p]
    session = graphics.CGSessionCopyCurrentDictionary()
    if not session:
        return "macOS WindowServer session unavailable (headless or sandboxed)"
    foundation.CFRelease(session)
    return None


def guard_tk_initialization(tk, reason, patcher):
    if reason is None:
        return

    def unavailable(*_args, **_kwargs):
        pytest.skip(reason)

    patcher.setattr(tk.Tk, "__init__", unavailable)


@pytest.fixture(scope="session", autouse=True)
def tk_display_guard():
    # macOS Tk can abort in native code before Python can catch TclError.
    reason = tk_display_unavailable_reason(sys.platform)
    if reason is None:
        yield
        return
    try:
        import tkinter
    except ImportError:
        yield
        return
    with pytest.MonkeyPatch.context() as patcher:
        guard_tk_initialization(tkinter, reason, patcher)
        yield


@pytest.fixture
def symlink_or_skip():
    """Create a symlink, skipping where the platform forbids it — Windows needs Developer
    Mode or admin. These tests cover POSIX symlink semantics, which junctions do not match."""
    def _make(link, target, *, directory=False):
        try:
            link.symlink_to(target, target_is_directory=directory)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are unavailable on this platform")
        return link
    return _make


@pytest.fixture
def qwen_mtp_snapshot():
    import json

    def create(snapshot, *, packed=True, indexed=True, prefix="", layers=1):
        snapshot.mkdir(parents=True, exist_ok=True)
        text = {"model_type": "qwen4_exp_text", "mtp_num_hidden_layers": layers,
                "mtp": {"num_hidden_layers": layers}, "num_experts": 2}
        (snapshot / "config.json").write_text(json.dumps({
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp", "text_config": text,
        }))
        names = ["mtp.fc_embedding.weight", "mtp.fc_hidden.weight",
                 "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight"]
        names += [f"mtp.hyper_connection_mixer.{part}.weight" for part in (
            "hc_norm", "input_mix_weight_down", "input_mix_weight_up",
        )]
        for layer in range(layers):
            base = f"mtp.layers.{layer}"
            names += [f"{base}.self_attn.{part}.weight" for part in (
                "q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm",
                "indexer.index_qk_proj", "indexer.q_layernorm", "indexer.k_layernorm",
            )]
            names += [f"{base}.{branch}_hyper_connection.{part}.weight"
                      for branch in ("attn", "mlp") for part in (
                          "hc_norm", "input_mix_weight_down", "input_mix_weight_up",
                      )]
            names += [f"{base}.mlp.{part}.weight" for part in (
                "gate", "shared_expert_gate", "shared_expert.gate_proj",
                "shared_expert.up_proj", "shared_expert.down_proj",
            )]
            names += ([f"{base}.mlp.experts.{part}" for part in ("gate_up_proj", "down_proj")]
                      if packed else [f"{base}.mlp.experts.{expert}.{part}.weight"
                                      for expert in range(2)
                                      for part in ("gate_proj", "up_proj", "down_proj")])
        names = [prefix + name for name in names]
        header = {name: {"dtype": "U8", "shape": [1], "data_offsets": [i, i + 1]}
                  for i, name in enumerate(names)}
        encoded = json.dumps(header).encode()
        (snapshot / "model.safetensors").write_bytes(
            len(encoded).to_bytes(8, "little") + encoded + b"x" * len(names),
        )
        if indexed:
            (snapshot / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": dict.fromkeys(names, "model.safetensors"),
            }))
        return snapshot

    return create
