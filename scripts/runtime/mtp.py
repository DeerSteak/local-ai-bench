"""Native multi-token prediction capability and benchmark pass planning."""

from collections.abc import Mapping, Sequence
from scripts.runtime.engine_identity import engine_family


MTP_MODES = ("off", "on", "both")
MTP_SERVER_TESTS = frozenset({
    "llm", "llm_cached", "conv", "mcq", "math", "reasoning", "code", "tool",
    "conc_tool", "conc_chat", "sustained",
})
MTP_CONCURRENCY_TESTS = frozenset({"conc_tool", "conc_chat"})


def native_mtp_config(model: dict, engine_name: str) -> dict | None:
    engine_name = engine_family(engine_name)
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
    method = config.get("method")
    if engine_name == "vllm" and method is not None:
        if not isinstance(method, str) or not method.strip():
            return None
        resolved["method"] = method
    draft_repo = config.get("draft_repo")
    draft_file = config.get("draft_file")
    if engine_name == "llamacpp" and (draft_repo is not None or draft_file is not None):
        if not isinstance(draft_repo, str) or not draft_repo.strip():
            return None
        if not isinstance(draft_file, str) or not draft_file.strip():
            return None
        resolved.update({"draft_repo": draft_repo, "draft_file": draft_file})
    return resolved


def native_mtp_models(models: Sequence[dict], engine_name: str) -> list[dict]:
    return [model for model in models if native_mtp_config(model, engine_name)]


def active_mtp_configurations(models: Sequence[dict], engine_name: str,
                              enabled: bool) -> dict[str, dict]:
    if not enabled:
        return {}
    configurations = {}
    for model in models:
        config = native_mtp_config(model, engine_name)
        if config is None:
            continue
        tag = model.get("tag")
        if not isinstance(tag, str) or not tag:
            raise ValueError("MTP-capable model requires a tag")
        recorded = {
            "num_speculative_tokens": config["num_speculative_tokens"],
            "predictor": "separate" if "draft_file" in config else "embedded",
        }
        if "method" in config:
            recorded["method"] = config["method"]
        if tag in configurations and configurations[tag] != recorded:
            raise ValueError(f"conflicting MTP configuration for model: {tag}")
        configurations[tag] = recorded
    return configurations


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


def mtp_selection_error(engine_models: Mapping[str, Sequence[dict]], mode: str,
                        tests: Sequence[str]) -> str | None:
    if mode == "off":
        return None
    incompatible = [test for test in tests if test not in MTP_SERVER_TESTS]
    if mode == "on" and incompatible:
        return (
            "--mtp on cannot run non-MTP workloads: " + ", ".join(incompatible)
            + "; use --mtp both to run them once in the baseline pass"
        )
    capable_engines = {
        engine_name for engine_name, models in engine_models.items()
        if native_mtp_models(models, engine_name)
    }
    if mode == "on":
        missing = [
            engine_name for engine_name in engine_models if engine_name not in capable_engines
        ]
        if missing:
            return (
                "--mtp on requires a selected model with cataloged native MTP support "
                "for every selected engine; missing: " + ", ".join(missing)
            )
    elif mode == "both":
        if not capable_engines:
            return (
                "--mtp both requires a selected model with cataloged native MTP support; "
                "use --mtp off for a baseline-only run"
            )
        if not set(tests) & MTP_SERVER_TESTS:
            return (
                "--mtp both requires at least one server-backed text workload; "
                "use --mtp off for a baseline-only run"
            )
    return None


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


def mtp_progress_names(engine_models: Mapping[str, Sequence[dict]], mode: str) -> list[str]:
    names = []
    for engine_name, models in engine_models.items():
        for enabled in mtp_mode_states(mode):
            if enabled and not native_mtp_models(models, engine_name):
                continue
            names.append(
                mtp_pass_label(engine_name, enabled) if mode != "off" else engine_name
            )
    return names
