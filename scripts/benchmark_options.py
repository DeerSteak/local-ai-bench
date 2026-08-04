"""Typed public benchmark option metadata shared by CLI and frontends."""

from dataclasses import dataclass

import config


@dataclass(frozen=True)
class OptionSpec:
    value_type: str
    classification: str
    ui_status: str
    ui_location: str
    default: object = None
    minimum: int | None = None
    maximum: int | None = None
    choices: tuple[object, ...] = ()


TEST_CHOICES = (
    "llm", "conv", "llamabench", "llamabenchconc", "emb", "mcq", "math",
    "reasoning", "code", "tool", "acc", "conc_tool", "conc_chat", "conc", "img",
)
TG_TOKEN_CHOICES = (128, 512, 1024)
TIER_CHOICES = ("xsmall", "small", "medium", "large")


def _spec(value_type, classification, ui_status, ui_location, **kwargs):
    return OptionSpec(value_type, classification, ui_status, ui_location, **kwargs)


PUBLIC_OPTION_SCHEMA = {
    "--tests": _spec("string-list", "guided", "exposed", "Test selection screen", choices=TEST_CHOICES),
    "--engine": _spec("choice", "guided", "exposed", "Engine selection screen"),
    "--llm-models": _spec("string-list", "guided", "exposed", "LLM model selection screen"),
    "--models": _spec("string-list", "guided", "equivalent", "Alias of --llm-models"),
    "--embedding-models": _spec("string-list", "guided", "exposed", "Embedding model selection screen"),
    "--image-models": _spec("string-list", "guided", "exposed", "Image model selection screen"),
    "--max-prompt-tokens": _spec("integer", "guided", "exposed", "Prompt-processing cap screen", minimum=1),
    "--tg-tokens": _spec("integer-list", "guided", "exposed", "Generation-size screen", choices=TG_TOKEN_CHOICES),
    "--maxtier": _spec("choice", "guided", "equivalent", "Explicit model selection is more precise", choices=TIER_CHOICES),
    "--list-models": _spec("boolean", "contextual", "equivalent", "Installed models appear in selection screens", default=False),
    "--sample": _spec("integer", "developer-only", "excluded", "Developer-only non-comparable accuracy sampling", minimum=1),
    "--fork-plan": _spec("path", "developer-only", "excluded", "Internal reviewed-fork provenance guard"),
    "--warmup": _spec("integer", "advanced", "exposed", "Graphical execution settings", default=config.WARMUP_RUNS, minimum=0),
    "--runs": _spec("integer", "advanced", "exposed", "Graphical execution settings", default=config.N_RUNS, minimum=1, maximum=10),
    "--timeout": _spec("integer", "advanced", "exposed", "Graphical execution settings", default=300, minimum=1),
    "--acc-timeout": _spec("integer", "advanced", "exposed", "Graphical execution settings", default=config.ACC_TIMEOUT, minimum=1),
    "--acc-token-budget": _spec("integer", "advanced", "exposed", "Graphical execution settings", default=config.ACC_TOKEN_BUDGET, minimum=1),
    "--cpu-only": _spec("boolean", "advanced", "exposed", "Graphical execution settings", default=False),
    "--force-all": _spec("boolean", "advanced", "exposed", "Graphical execution settings", default=False),
    "--offline": _spec("boolean", "advanced", "exposed", "Graphical execution settings", default=False),
    "--out": _spec("path", "advanced", "exposed", "Graphical path settings", default=""),
    "--comfyui": _spec("path", "advanced", "exposed", "Graphical path settings", default=""),
}


GUI_OPTION_FLAGS = {
    "warmup": "--warmup", "runs": "--runs", "timeout": "--timeout",
    "acc_timeout": "--acc-timeout", "acc_token_budget": "--acc-token-budget",
    "cpu_only": "--cpu-only", "force_all": "--force-all", "offline": "--offline", "out": "--out",
    "comfyui": "--comfyui",
}


def gui_option_defaults() -> dict:
    return {key: PUBLIC_OPTION_SCHEMA[flag].default for key, flag in GUI_OPTION_FLAGS.items()}


def option_value_errors(values: dict[str, object]) -> list[str]:
    errors = []
    for flag, value in values.items():
        spec = PUBLIC_OPTION_SCHEMA[flag]
        if value is None:
            continue
        if spec.value_type == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"{flag} must be a whole number.")
                continue
            if spec.minimum is not None and value < spec.minimum:
                errors.append(f"{flag} must be at least {spec.minimum}.")
            if spec.maximum is not None and value > spec.maximum:
                errors.append(f"{flag} must be at most {spec.maximum}.")
    return errors
