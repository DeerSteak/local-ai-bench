"""Native multi-token prediction capability and benchmark pass planning."""

from collections.abc import Sequence


MTP_MODES = ("off", "on", "both")
MTP_SERVER_TESTS = frozenset({
    "llm", "conv", "mcq", "math", "reasoning", "code", "tool",
    "conc_tool", "conc_chat", "sustained",
})
MTP_CONCURRENCY_TESTS = frozenset({"conc_tool", "conc_chat"})


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
    resolved: dict = {"num_speculative_tokens": tokens}
    draft_repo = config.get("draft_repo")
    draft_file = config.get("draft_file")
    if draft_repo is not None or draft_file is not None:
        if not isinstance(draft_repo, str) or not draft_repo.strip():
            return None
        if not isinstance(draft_file, str) or not draft_file.strip():
            return None
        resolved.update({"draft_repo": draft_repo, "draft_file": draft_file})
    return resolved


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


def expand_mtp_passes(scopes: Sequence[dict], mode: str) -> list[dict]:
    passes = []
    states = mtp_mode_states(mode)
    for scope in scopes:
        for enabled in states:
            tests = mtp_tests(scope["tests"], enabled)
            llm_models = scope["llm_models"]
            concurrency_models = scope["concurrency_models"]
            if enabled:
                llm_models = native_mtp_models(llm_models, scope["name"])
                concurrency_models = native_mtp_models(
                    concurrency_models, scope["name"],
                )
                if not llm_models:
                    tests = [test for test in tests if test in MTP_CONCURRENCY_TESTS]
                if not concurrency_models:
                    tests = [test for test in tests if test not in MTP_CONCURRENCY_TESTS]
            if not tests:
                continue
            passes.append({
                **scope,
                "tests": tests,
                "llm_models": llm_models,
                "concurrency_models": concurrency_models,
                "mtp_enabled": enabled,
                "progress_name": (
                    mtp_pass_label(scope["name"], enabled)
                    if mode != "off" else scope["name"]
                ),
            })
    return passes


def mtp_progress_names(engine_names: Sequence[str], mode: str,
                       catalog_models: Sequence[dict]) -> list[str]:
    names = []
    for engine_name in engine_names:
        for enabled in mtp_mode_states(mode):
            if enabled and not native_mtp_models(catalog_models, engine_name):
                continue
            names.append(
                mtp_pass_label(engine_name, enabled) if mode != "off" else engine_name
            )
    return names
