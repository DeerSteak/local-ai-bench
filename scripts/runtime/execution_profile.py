"""Shared effective execution-profile construction for runs and recovery."""

from scripts.runtime.shared import Shared
from scripts.release.qualification import engine_support_profile
from scripts.setup.runtime_identity import engine_runtime_version
from scripts.runtime import config


ENGINE_BACKED_STAGES = {
    "llm", "conv", "llamabench", "llamabenchconc", "emb", "mcq", "math", "reasoning",
    "code", "tool", "conc_tool", "conc_chat", "sustained",
}


def build_execution_profile(engine, tests, *, cpu_only: bool,
                            engine_name: str = "llamacpp",
                            hardware_profile: dict | None = None,
                            runtime_version: str | None = None) -> dict:
    hardware = dict(hardware_profile or Shared.build_profile())
    hardware_backend = hardware["backend"]
    backend = (engine.runtime_backend(hardware_backend, cpu_only=cpu_only)
               if set(tests) & ENGINE_BACKED_STAGES else hardware_backend)
    runtime_version = runtime_version or engine_runtime_version(engine_name, engine)
    os_name = str(hardware.get("os", "")).split(" ", 1)[0]
    support = engine_support_profile(
        system=os_name, architecture=str(hardware.get("arch", "")),
        wsl=hardware.get("wsl") is True, runtime=engine_name,
        runtime_version=runtime_version, backend=backend, current_version=config.VERSION,
        accelerator=str(hardware.get("hostname", "")),
    )
    return {
        **hardware, "hardware_backend": hardware_backend, "backend": backend,
        "engine_support": support,
    }
