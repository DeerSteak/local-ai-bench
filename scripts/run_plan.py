"""Immutable, serializable benchmark execution plan."""

import hashlib
import json
from dataclasses import dataclass


PLAN_SCHEMA_VERSION = 1
SAFE_CONFIG_KEYS = {
    "runs", "warmup_runs", "run_timeout_seconds", "accuracy_timeout_seconds",
    "accuracy_token_budget", "cpu_only", "force_all", "max_prompt_tokens",
    "context_lengths", "llamabench_pp", "llamabench_tg", "sample_size",
}
REQUIRED_CONFIG_KEYS = {"warmup_runs", "cpu_only", "force_all"}
MODEL_FAMILIES = {"llm", "concurrency", "embeddings", "images"}
SAFE_MODEL_KEYS = {"tag", "short", "size_gb", "params_b"}


def _canonical_json(value) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class RunPlan:
    application_version: str
    engine_name: str
    tests: tuple[str, ...]
    stage_order: tuple[str, ...]
    _models_json: str
    _config_json: str

    @classmethod
    def create(cls, *, application_version: str, engine_name: str, tests,
               stage_order, models: dict, effective_config: dict):
        tests = tuple(tests)
        stage_order = tuple(stage_order)
        if not engine_name:
            raise ValueError("run plan requires an engine")
        if len(set(tests)) != len(tests) or len(set(stage_order)) != len(stage_order):
            raise ValueError("run plan tests and stage order must not contain duplicates")
        if set(tests) != set(stage_order):
            raise ValueError("run plan tests and stage order must contain the same stages")
        unknown_config = set(effective_config) - SAFE_CONFIG_KEYS
        if unknown_config:
            raise ValueError(f"unsafe or unknown run-plan settings: {sorted(unknown_config)}")
        missing_config = REQUIRED_CONFIG_KEYS - set(effective_config)
        if missing_config:
            raise ValueError(f"missing required run-plan settings: {sorted(missing_config)}")
        unknown_families = set(models) - MODEL_FAMILIES
        if unknown_families:
            raise ValueError(f"unknown run-plan model families: {sorted(unknown_families)}")
        for family, entries in models.items():
            if not isinstance(entries, list):
                raise ValueError(f"run-plan model family must be a list: {family}")
            for entry in entries:
                unknown_model_keys = set(entry) - SAFE_MODEL_KEYS
                if unknown_model_keys:
                    raise ValueError(
                        f"unsafe or unknown model identity fields: {sorted(unknown_model_keys)}"
                    )
        return cls(
            application_version=str(application_version),
            engine_name=str(engine_name),
            tests=tests,
            stage_order=stage_order,
            _models_json=_canonical_json(models),
            _config_json=_canonical_json(effective_config),
        )

    @classmethod
    def from_dict(cls, value: dict):
        if value.get("schema_version") != PLAN_SCHEMA_VERSION:
            raise ValueError(f"unsupported run-plan schema: {value.get('schema_version')}")
        return cls.create(
            application_version=value["application_version"], engine_name=value["engine"],
            tests=value["requested_tests"], stage_order=value["stage_order"],
            models=value["models"], effective_config=value["effective_config"],
        )

    @property
    def models(self) -> dict:
        return json.loads(self._models_json)

    @property
    def effective_config(self) -> dict:
        return json.loads(self._config_json)

    @property
    def warmup_runs(self) -> int:
        return self.effective_config["warmup_runs"]

    @property
    def cpu_only(self) -> bool:
        return self.effective_config["cpu_only"]

    @property
    def force_all(self) -> bool:
        return self.effective_config["force_all"]

    def to_dict(self) -> dict:
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "application_version": self.application_version,
            "engine": self.engine_name,
            "requested_tests": list(self.tests),
            "stage_order": list(self.stage_order),
            "models": self.models,
            "effective_config": self.effective_config,
        }

    @property
    def plan_id(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()
