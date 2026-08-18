"""Authoritative metadata for benchmark stages and their result sections."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StageSpec:
    key: str
    label: str
    section: str
    model_family: str
    ui_family: str
    category: str
    menu_label: str | None = None
    default_enabled: bool = False
    menu_visible: bool = True
    native_engine: str | None = None
    journal_owned: bool = False
    selected_retry: bool = False


STAGE_SPECS = (
    StageSpec("llm", "Single-shot LLM", "llm", "llm", "llm", "llm", default_enabled=True,
              journal_owned=True, selected_retry=True),
    StageSpec("conv", "Conversation", "llm_conversation", "llm", "llm", "llm",
              default_enabled=True, journal_owned=True, selected_retry=True),
    StageSpec("llamabench", "llama-bench throughput", "llamabench", "llm", "llm", "llm", menu_label="llama-bench (throughput + concurrency)", native_engine="llamacpp", journal_owned=True),
    StageSpec("llamabenchconc", "llama-bench concurrency", "llamabenchconc", "llm", "llm", "llm", menu_visible=False, native_engine="llamacpp"),
    StageSpec("vllmbench", "vllm bench (latency + throughput)", "vllmbench", "llm", "llm", "llm", native_engine="vllm"),
    StageSpec("sustained", "Sustained load", "sustained", "llm", "llm", "llm",
              journal_owned=True, selected_retry=True),
    StageSpec("emb", "Embeddings", "embeddings", "embeddings", "embedding", "embedding",
              default_enabled=True, journal_owned=True, selected_retry=True),
    StageSpec("mcq", "MCQ accuracy", "mcq", "llm", "llm", "accuracy",
              journal_owned=True, selected_retry=True),
    StageSpec("math", "Math accuracy", "math", "llm", "llm", "accuracy",
              journal_owned=True, selected_retry=True),
    StageSpec("reasoning", "Reasoning accuracy", "reasoning", "llm", "llm", "accuracy",
              journal_owned=True, selected_retry=True),
    StageSpec("code", "Code accuracy", "code", "llm", "llm", "accuracy",
              journal_owned=True, selected_retry=True),
    StageSpec("tool", "Tool accuracy", "tool", "llm", "llm", "accuracy",
              journal_owned=True, selected_retry=True),
    StageSpec("conc_tool", "Tool concurrency", "concurrency_tool", "concurrency", "llm", "concurrency",
              journal_owned=True, selected_retry=True),
    StageSpec("conc_chat", "Chat concurrency", "concurrency_chat", "concurrency", "llm", "concurrency",
              journal_owned=True, selected_retry=True),
    StageSpec("img", "Image generation", "images", "images", "image", "image", default_enabled=True),
)
STAGE_BY_KEY = {spec.key: spec for spec in STAGE_SPECS}
STAGE_ORDER = tuple(spec.key for spec in STAGE_SPECS)
ACCURACY_TESTS = [spec.key for spec in STAGE_SPECS if spec.category == "accuracy"]
CONCURRENCY_TESTS = [spec.key for spec in STAGE_SPECS if spec.category == "concurrency"]
LLM_TESTS = [spec.key for spec in STAGE_SPECS if spec.category in {"llm", "accuracy"}]
JOURNAL_STAGES = frozenset(spec.key for spec in STAGE_SPECS if spec.journal_owned)
SELECTED_RETRY_STAGES = frozenset(spec.key for spec in STAGE_SPECS if spec.selected_retry)


def stage_spec(key: str) -> StageSpec:
    try:
        return STAGE_BY_KEY[key]
    except KeyError as exc:
        raise ValueError(f"unknown benchmark stage: {key}") from exc


def engine_incompatible_tests(tests: list[str], engine_name: str) -> list[str]:
    return [
        key for key in tests
        if (spec := STAGE_BY_KEY.get(key)) is not None
        and spec.native_engine is not None and spec.native_engine != engine_name
    ]
