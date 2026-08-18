"""Shared effective execution-profile construction for runs and recovery."""

from scripts.runtime.shared import Shared


ENGINE_BACKED_STAGES = {
    "llm", "conv", "llamabench", "llamabenchconc", "emb", "mcq", "math", "reasoning",
    "code", "tool", "conc_tool", "conc_chat", "sustained",
}


def build_execution_profile(engine, tests, *, cpu_only: bool,
                            hardware_profile: dict | None = None) -> dict:
    hardware = dict(hardware_profile or Shared.build_profile())
    hardware_backend = hardware["backend"]
    backend = (engine.runtime_backend(hardware_backend, cpu_only=cpu_only)
               if set(tests) & ENGINE_BACKED_STAGES else hardware_backend)
    return {**hardware, "hardware_backend": hardware_backend, "backend": backend}
