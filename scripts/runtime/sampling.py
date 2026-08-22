"""Versioned text-generation sampling profiles and engine payload mappings."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path


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
PUBLISHER_SAMPLING_PROFILE = "publisher-recommended-v1"
PUBLISHER_CONTROL_KEYS = {
    "temperature", "top_k", "top_p", "min_p", "repetition_penalty",
    "presence_penalty", "frequency_penalty", "do_sample",
}
PUBLISHER_PROFILE_SCHEMA_VERSION = 1


def sampling_payload(engine_name: str, controls: dict) -> dict:
    if engine_name == "vllm":
        return {key: deepcopy(value) for key, value in controls.items() if key != "do_sample"}
    if engine_name == "llamacpp":
        payload = {
            key: deepcopy(value) for key, value in controls.items()
            if key not in {"repetition_penalty", "do_sample"}
        }
        payload["repeat_penalty"] = controls["repetition_penalty"]
        payload["logit_bias"] = []
        payload.update(LLAMACPP_NEUTRAL_CONTROLS)
        return payload
    raise ValueError(f"unsupported sampling engine: {engine_name}")


def baseline_sampling_payload(engine_name: str) -> dict:
    return sampling_payload(engine_name, BASELINE_CONTROLS)


def baseline_sampling_profile(engine_name: str) -> dict:
    return {
        "profile": BASELINE_SAMPLING_PROFILE,
        "semantic_controls": deepcopy(BASELINE_CONTROLS),
        "engine_controls": baseline_sampling_payload(engine_name),
    }


def publisher_sampling_profile(engine_name: str, *, name: str, repo: str,
                               revision: str, controls: dict) -> dict:
    if not all(isinstance(value, str) and value for value in (name, repo, revision)):
        raise ValueError("publisher sampling requires name, repository, and revision")
    unknown = set(controls) - PUBLISHER_CONTROL_KEYS
    if unknown:
        raise ValueError(f"unsupported publisher sampling controls: {', '.join(sorted(unknown))}")
    if not controls:
        raise ValueError("publisher sampling controls are empty")
    if controls.get("do_sample") is False:
        raise ValueError("publisher profile disables sampling; use the deterministic baseline")
    resolved = deepcopy(BASELINE_CONTROLS)
    resolved.update({key: value for key, value in controls.items() if key != "do_sample"})
    source_payload = json.dumps(controls, sort_keys=True, separators=(",", ":")).encode()
    return {
        "profile": f"{PUBLISHER_SAMPLING_PROFILE}:{name}",
        "source": {
            "repo": repo,
            "revision": revision,
            "path": "generation_config.json",
            "controls_sha256": hashlib.sha256(source_payload).hexdigest(),
        },
        "publisher_controls": deepcopy(controls),
        "semantic_controls": resolved,
        "engine_controls": sampling_payload(engine_name, resolved),
    }


def load_publisher_sampling_profile(path: Path, engine_name: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != PUBLISHER_PROFILE_SCHEMA_VERSION:
        raise ValueError("unsupported publisher sampling profile")
    source = value.get("source")
    if not isinstance(source, dict) or set(source) != {"repo", "revision"}:
        raise ValueError("publisher sampling profile requires an exact source")
    name, repo, revision = value.get("name"), source.get("repo"), source.get("revision")
    if not isinstance(name, str) or not name or not isinstance(repo, str) or not repo \
            or not isinstance(revision, str) or not revision:
        raise ValueError("publisher sampling profile requires an exact source")
    controls = value.get("controls")
    if not isinstance(controls, dict):
        raise ValueError("publisher sampling profile controls must be an object")
    return publisher_sampling_profile(
        engine_name, name=name, repo=repo, revision=revision, controls=controls,
    )


def sampling_profile_payload(engine_name: str, profile: dict | None) -> dict:
    expected = baseline_sampling_profile(engine_name) if profile is None else profile
    profile_name = expected.get("profile") if isinstance(expected, dict) else None
    if isinstance(profile_name, str) and profile_name.startswith(
            f"{PUBLISHER_SAMPLING_PROFILE}:"):
        source = expected.get("source")
        controls = expected.get("publisher_controls")
        if not isinstance(source, dict) or not isinstance(controls, dict):
            raise ValueError("publisher sampling profile is missing source evidence")
        repo, revision = source.get("repo"), source.get("revision")
        if not isinstance(repo, str) or not isinstance(revision, str):
            raise ValueError("publisher sampling profile is missing source evidence")
        rebuilt = publisher_sampling_profile(
            engine_name, name=profile_name.split(":", 1)[1],
            repo=repo, revision=revision, controls=controls,
        )
        if expected != rebuilt:
            raise ValueError("publisher sampling profile does not match its source controls")
    controls = expected.get("engine_controls") if isinstance(expected, dict) else None
    if not isinstance(controls, dict):
        raise ValueError("sampling profile has no resolved engine controls")
    return deepcopy(controls)
