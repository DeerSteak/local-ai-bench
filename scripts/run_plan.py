"""Immutable, serializable benchmark execution plan."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


PLAN_SCHEMA_VERSION = 2
SUPPORTED_PLAN_SCHEMAS = {1, PLAN_SCHEMA_VERSION}
IDENTITY_SCHEME = "sha256-v1"
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


def _stable_id(kind: str, *parts) -> str:
    payload = _canonical_json([kind, *parts]).encode("utf-8")
    return f"{kind}_{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True)
class RunPlan:
    schema_version: int
    application_version: str
    engine_name: str
    tests: tuple[str, ...]
    stage_order: tuple[str, ...]
    _models_json: str
    _config_json: str

    @classmethod
    def create(cls, *, application_version: str, engine_name: str, tests,
               stage_order, models: dict, effective_config: dict,
               schema_version: int = PLAN_SCHEMA_VERSION):
        tests = tuple(tests)
        stage_order = tuple(stage_order)
        if not engine_name:
            raise ValueError("run plan requires an engine")
        if schema_version not in SUPPORTED_PLAN_SCHEMAS:
            raise ValueError(f"unsupported run-plan schema: {schema_version}")
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
            application_version=str(application_version),
            engine_name=str(engine_name),
            tests=tests,
            stage_order=stage_order,
            _models_json=_canonical_json(models),
            _config_json=_canonical_json(effective_config),
        )

    @classmethod
    def from_dict(cls, value: dict):
        schema_version = value.get("schema_version")
        if schema_version not in SUPPORTED_PLAN_SCHEMAS:
            raise ValueError(f"unsupported run-plan schema: {schema_version}")
        if schema_version == PLAN_SCHEMA_VERSION and value.get("identity_scheme") != IDENTITY_SCHEME:
            raise ValueError("unsupported or missing run-plan identity scheme")
        return cls.create(
            application_version=value["application_version"], engine_name=value["engine"],
            tests=value["requested_tests"], stage_order=value["stage_order"],
            models=value["models"], effective_config=value["effective_config"],
            schema_version=schema_version,
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
        return value

    @property
    def plan_id(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    @property
    def job_id(self) -> str:
        return f"job_{self.plan_id}"

    def stage_id(self, stage: str) -> str:
        if stage not in self.stage_order:
            raise ValueError(f"stage is not present in run plan: {stage}")
        return _stable_id("stage", self.plan_id, stage)

    def model_id(self, family: str, identity: dict) -> str:
        if family not in MODEL_FAMILIES or identity not in self.models.get(family, []):
            raise ValueError("model identity is not present in run plan")
        return _stable_id("model", self.plan_id, family, identity)

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
