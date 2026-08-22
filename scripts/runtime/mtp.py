"""Native multi-token prediction capability and benchmark pass planning."""

from collections.abc import Sequence


MTP_MODES = ("off", "on", "both")
MTP_SERVER_TESTS = frozenset({
    "llm", "conv", "mcq", "math", "reasoning", "code", "tool",
    "conc_tool", "conc_chat", "sustained",
})


def native_mtp_config(model: dict, engine_name: str) -> dict | None:
    value = model.get("native_mtp")
    if not isinstance(value, dict):
        return None
    config = value.get(engine_name)
    if not isinstance(config, dict):
        return None
    tokens = config.get("num_speculative_tokens")
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
        return None
    return {"num_speculative_tokens": tokens}


def native_mtp_models(models: Sequence[dict], engine_name: str) -> list[dict]:
    return [model for model in models if native_mtp_config(model, engine_name)]


def mtp_mode_states(mode: str) -> tuple[bool, ...]:
    if mode == "off":
        return (False,)
    if mode == "on":
        return (True,)
    if mode == "both":
        return (False, True)
    raise ValueError(f"unknown MTP mode: {mode}")


def mtp_tests(tests: Sequence[str], enabled: bool) -> list[str]:
    return list(tests) if not enabled else [test for test in tests if test in MTP_SERVER_TESTS]


def mtp_pass_label(engine_name: str, enabled: bool) -> str:
    return f"{engine_name} · MTP {'on' if enabled else 'off'}"
