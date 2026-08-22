"""Versioned text-generation sampling profiles and engine payload mappings."""

from copy import deepcopy


BASELINE_SAMPLING_PROFILE = "deterministic-baseline-v1"
BASELINE_CONTROLS = {
    "temperature": 0.0,
    "top_k": 0,
    "top_p": 1.0,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": 0,
    "logit_bias": {},
}
LLAMACPP_NEUTRAL_CONTROLS = {
    "typical_p": 1.0,
    "dry_multiplier": 0.0,
    "xtc_probability": 0.0,
    "top_n_sigma": -1.0,
    "dynatemp_range": 0.0,
    "mirostat": 0,
}


def baseline_sampling_payload(engine_name: str) -> dict:
    if engine_name == "vllm":
        return deepcopy(BASELINE_CONTROLS)
    if engine_name == "llamacpp":
        payload = {
            key: value for key, value in BASELINE_CONTROLS.items()
            if key != "repetition_penalty"
        }
        payload["repeat_penalty"] = BASELINE_CONTROLS["repetition_penalty"]
        payload["logit_bias"] = []
        payload.update(LLAMACPP_NEUTRAL_CONTROLS)
        return payload
    raise ValueError(f"unsupported sampling engine: {engine_name}")


def baseline_sampling_profile(engine_name: str) -> dict:
    return {
        "profile": BASELINE_SAMPLING_PROFILE,
        "semantic_controls": deepcopy(BASELINE_CONTROLS),
        "engine_controls": baseline_sampling_payload(engine_name),
    }
