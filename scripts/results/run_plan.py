"""Immutable, serializable benchmark execution plan."""

import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path

from scripts.results.canonical_json import canonical_json, sha256_json


PLAN_SCHEMA_VERSION = 4
SUPPORTED_PLAN_SCHEMAS = {1, 2, 3, PLAN_SCHEMA_VERSION}
IDENTITY_SCHEME = "sha256-v1"
SAFE_CONFIG_KEYS = {
    "runs", "warmup_runs", "run_timeout_seconds", "accuracy_timeout_seconds",
    "accuracy_token_budget", "cpu_only", "force_all", "retry_crashed_models",
    "max_prompt_tokens",
    "context_lengths", "llamabench_pp", "llamabench_tg", "sample_size",
    "concurrency_tool_levels", "concurrency_chat_levels",
    "concurrency_tool_context", "concurrency_chat_context",
    "concurrency_chat_soft_exit_floor",
    "methodology_profile", "effective_optimizations", "offline",
    "gpu_split_mode",
    "llamacpp_no_repack",
    "memory_telemetry", "memory_telemetry_interval_sec",
    "power_telemetry", "power_telemetry_interval_sec", "power_source", "power_scope",
}
REQUIRED_CONFIG_KEYS = {"warmup_runs", "cpu_only", "force_all"}
MODEL_FAMILIES = {"llm", "concurrency", "embeddings", "images"}
SAFE_MODEL_KEYS = {"tag", "short", "size_gb", "params_b"}
EXECUTION_CONFIG_KEYS = set(SAFE_CONFIG_KEYS) - {
    "methodology_profile", "effective_optimizations", "offline", "gpu_split_mode",
    "retry_crashed_models", "llamacpp_no_repack",
    "memory_telemetry", "memory_telemetry_interval_sec",
    "power_telemetry", "power_telemetry_interval_sec", "power_source", "power_scope",
}


def _stable_id(kind: str, *parts) -> str:
    return f"{kind}_{sha256_json([kind, *parts])}"


@dataclass(frozen=True)
class RunPlan:
    schema_version: int
    _job_id: str
    application_version: str
    engine_name: str
    tests: tuple[str, ...]
    stage_order: tuple[str, ...]
    _models_json: str
    _config_json: str

    @classmethod
    def create(cls, *, application_version: str, engine_name: str, tests,
               stage_order, models: dict, effective_config: dict,
               schema_version: int = PLAN_SCHEMA_VERSION, job_id: str | None = None):
        tests = tuple(tests)
        stage_order = tuple(stage_order)
        if not engine_name:
            raise ValueError("run plan requires an engine")
        if schema_version not in SUPPORTED_PLAN_SCHEMAS:
            raise ValueError(f"unsupported run-plan schema: {schema_version}")
        if schema_version >= 2:
            job_id = job_id or f"job_{uuid.uuid4().hex}"
            if not isinstance(job_id, str) or not job_id.startswith("job_"):
                raise ValueError("run plan requires a valid job identity")
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
            schema_version=schema_version,
            _job_id=job_id or "",
            application_version=str(application_version),
            engine_name=str(engine_name),
            tests=tests,
            stage_order=stage_order,
            _models_json=canonical_json(models),
            _config_json=canonical_json(effective_config),
        )

    @classmethod
    def from_dict(cls, value: dict):
        schema_version = value.get("schema_version")
        if schema_version not in SUPPORTED_PLAN_SCHEMAS:
            raise ValueError(f"unsupported run-plan schema: {schema_version}")
        if schema_version >= 2 and value.get("identity_scheme") != IDENTITY_SCHEME:
            raise ValueError("unsupported or missing run-plan identity scheme")
        if schema_version >= 2 and not value.get("job_id"):
            raise ValueError("run-plan schema 2 requires a job identity")
        plan = cls.create(
            application_version=value["application_version"], engine_name=value["engine"],
            tests=value["requested_tests"], stage_order=value["stage_order"],
            models=value["models"], effective_config=value["effective_config"],
            schema_version=schema_version,
            job_id=value.get("job_id"),
        )
        if schema_version >= 3 and value.get("execution_identity") != plan.execution_identity:
            raise ValueError("run-plan execution identity is missing or inconsistent")
        return plan

    @property
    def models(self) -> dict:
        return json.loads(self._models_json)

    @property
    def effective_config(self) -> dict:
        return json.loads(self._config_json)

    @property
    def execution_identity(self) -> dict:
        settings = self.effective_config
        identity = {
            "workloads": {stage: self.application_version for stage in self.stage_order},
            "runtime": {"engine": self.engine_name, "adapter_contract": 1},
            "privacy": {
                "prompts": "not_in_result",
                "responses": "workload_dependent_sidecar",
            },
            "retry": {
                "implausible_tps_retries": 1,
                "engine_recovery": "bounded_same_sample",
            },
            "timeouts": {
                "run_seconds": settings.get("run_timeout_seconds"),
                "accuracy_seconds": settings.get("accuracy_timeout_seconds"),
            },
            "output": {"result_schema": 3, "event_schema": 1},
        }
        if "offline" in settings:
            identity["privacy"]["offline"] = settings["offline"]
        has_profile = "methodology_profile" in settings
        has_optimizations = "effective_optimizations" in settings
        if has_profile != has_optimizations:
            raise ValueError("methodology profile and effective optimizations must be recorded together")
        if has_profile:
            identity["methodology"] = {
                "profile": settings["methodology_profile"],
                "effective_optimizations": settings.get("effective_optimizations", []),
            }
        if settings.get("power_telemetry"):
            identity["telemetry"] = {
                "power": {
                    "interval_sec": settings["power_telemetry_interval_sec"],
                    "source": settings["power_source"],
                    "scope": settings["power_scope"],
                },
            }
        return identity

    @property
    def warmup_runs(self) -> int:
        return self.effective_config["warmup_runs"]

    @property
    def cpu_only(self) -> bool:
        return self.effective_config["cpu_only"]

    @property
    def force_all(self) -> bool:
        return self.effective_config["force_all"]

    @property
    def retry_crashed_models(self) -> bool:
        return self.effective_config.get("retry_crashed_models", False)

    def to_dict(self) -> dict:
        value = {
            "schema_version": self.schema_version,
            "application_version": self.application_version,
            "engine": self.engine_name,
            "requested_tests": list(self.tests),
            "stage_order": list(self.stage_order),
            "models": self.models,
            "effective_config": self.effective_config,
        }
        if self.schema_version >= 2:
            value["identity_scheme"] = IDENTITY_SCHEME
            value["job_id"] = self.job_id
        if self.schema_version >= 3:
            value["execution_identity"] = self.execution_identity
        return value

    def validate_for_execution(self) -> None:
        missing_families = MODEL_FAMILIES - set(self.models)
        missing_config = EXECUTION_CONFIG_KEYS - set(self.effective_config)
        if not self.tests or missing_families or missing_config:
            raise ValueError(
                f"incomplete run plan: model families={sorted(missing_families)}, "
                f"settings={sorted(missing_config)}"
            )
        settings = self.effective_config
        integer_ranges = {
            "runs": (1, 10), "warmup_runs": (0, None),
            "run_timeout_seconds": (1, None), "accuracy_timeout_seconds": (1, None),
            "accuracy_token_budget": (1, None),
        }
        for key, (minimum, maximum) in integer_ranges.items():
            value = settings[key]
            if (isinstance(value, bool) or not isinstance(value, int) or value < minimum
                    or (maximum is not None and value > maximum)):
                raise ValueError(f"invalid execution setting: {key}")
        for key in ("cpu_only", "force_all", "retry_crashed_models", "offline",
                    "llamacpp_no_repack", "memory_telemetry", "power_telemetry"):
            if key == "retry_crashed_models" and key not in settings:
                continue
            if key not in settings and key in {
                    "offline", "llamacpp_no_repack", "memory_telemetry", "power_telemetry"}:
                continue
            if not isinstance(settings[key], bool):
                raise ValueError(f"invalid execution setting: {key}")
        interval = settings.get("memory_telemetry_interval_sec")
        if settings.get("memory_telemetry"):
            if (isinstance(interval, bool) or not isinstance(interval, (int, float))
                    or not math.isfinite(interval) or interval <= 0):
                raise ValueError("invalid execution setting: memory_telemetry_interval_sec")
        elif interval is not None:
            raise ValueError("memory telemetry interval requires memory telemetry")
        power_interval = settings.get("power_telemetry_interval_sec")
        if settings.get("power_telemetry"):
            if not settings.get("memory_telemetry"):
                raise ValueError("power telemetry requires memory telemetry")
            if (isinstance(power_interval, bool)
                    or not isinstance(power_interval, (int, float))
                    or not math.isfinite(power_interval) or power_interval <= 0):
                raise ValueError("invalid execution setting: power_telemetry_interval_sec")
            for key in ("power_source", "power_scope"):
                if not isinstance(settings.get(key), str) or not settings[key]:
                    raise ValueError(f"invalid execution setting: {key}")
        elif any(settings.get(key) is not None for key in (
                "power_telemetry_interval_sec", "power_source", "power_scope")):
            raise ValueError("power telemetry settings require power telemetry")
        if settings.get("gpu_split_mode", "layer") not in ("single", "layer", "tensor"):
            raise ValueError("invalid execution setting: gpu_split_mode")
        if "methodology_profile" in settings:
            if settings["methodology_profile"] != "neutral-v1":
                raise ValueError("invalid execution setting: methodology_profile")
            optimizations = settings.get("effective_optimizations")
            if (not isinstance(optimizations, list)
                    or any(not isinstance(value, str) or not value for value in optimizations)
                    or len(optimizations) != len(set(optimizations))):
                raise ValueError("invalid execution setting: effective_optimizations")
        for key in (
            "max_prompt_tokens", "sample_size", "concurrency_tool_context",
            "concurrency_chat_context", "concurrency_chat_soft_exit_floor",
        ):
            value = settings[key]
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)
                                      or value < 1):
                raise ValueError(f"invalid execution setting: {key}")
        for key in (
            "context_lengths", "llamabench_pp", "llamabench_tg",
            "concurrency_tool_levels", "concurrency_chat_levels",
        ):
            values = settings[key]
            invalid_values = not isinstance(values, list) or not values or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in values
            )
            if invalid_values or len(values) != len(set(values)):
                raise ValueError(f"invalid execution setting: {key}")
        for family, models in self.models.items():
            required_keys = ("short",) if family == "images" else ("tag", "short")
            for model in models:
                if not all(isinstance(model.get(key), str) and model[key]
                           for key in required_keys):
                    raise ValueError(f"invalid model identity in family: {family}")

    @property
    def plan_id(self) -> str:
        value = self.to_dict()
        value.pop("job_id", None)
        return sha256_json(value)

    @property
    def job_id(self) -> str:
        return self._job_id or f"job_{self.plan_id}"

    def stage_id(self, stage: str) -> str:
        if stage not in self.stage_order:
            raise ValueError(f"stage is not present in run plan: {stage}")
        return _stable_id("stage", self.job_id, stage)

    def model_id(self, family: str, identity: dict) -> str:
        if family not in MODEL_FAMILIES or identity not in self.models.get(family, []):
            raise ValueError("model identity is not present in run plan")
        return _stable_id("model", self.job_id, family, identity)

    def case_id(self, stage: str, model_id: str, case_key) -> str:
        return _stable_id("case", self.stage_id(stage), model_id, case_key)

    @staticmethod
    def attempt_id(case_id: str, attempt: int) -> str:
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
            raise ValueError("attempt number must be a positive integer")
        return _stable_id("attempt", case_id, attempt)

    @staticmethod
    def sample_id(attempt_id: str, sample: int) -> str:
        if not isinstance(sample, int) or isinstance(sample, bool) or sample < 1:
            raise ValueError("sample number must be a positive integer")
        return _stable_id("sample", attempt_id, sample)


def load_run_plan(path: Path) -> RunPlan:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict) and "run" in value:
        run = value.get("run")
        value = run.get("plan") if isinstance(run, dict) else None
    if not isinstance(value, dict):
        raise ValueError("File does not contain a benchmark run plan.")
    return RunPlan.from_dict(value)
