from scripts.runtime.execution_profile import build_execution_profile


class Engine:
    def __init__(self):
        self.calls = []

    def runtime_backend(self, hardware_backend, *, cpu_only=False):
        self.calls.append((hardware_backend, cpu_only))
        return "cpu" if cpu_only else "cuda"


def hardware():
    return {"hostname": "host", "os": "Darwin 25.0", "arch": "arm64",
            "backend": "metal", "timestamp": "now"}


def test_engine_backed_profile_records_hardware_and_effective_backend():
    engine = Engine()
    profile = build_execution_profile(
        engine, ["llm", "img"], cpu_only=False, hardware_profile=hardware(),
        runtime_version="b6000",
    )
    assert profile == {
        "hostname": "host", "os": "Darwin 25.0", "arch": "arm64",
        "timestamp": "now", "hardware_backend": "metal",
        "backend": "cuda", "engine_support": {
            "support_level": "unverified",
            "caveat": "No full lifecycle qualification matches this exact runtime.",
            "qualification_id": None, "qualified_at": None, "suite_version": None,
            "runtime_version": "b6000", "platform": "macos", "architecture": "arm64",
            "backend": "cuda", "stale": False,
        },
    }
    assert engine.calls == [("metal", False)]


def test_cpu_only_profile_uses_the_same_runtime_policy_as_execution():
    engine = Engine()
    assert build_execution_profile(
        engine, ["emb"], cpu_only=True, hardware_profile=hardware(), runtime_version="b6000",
    )["backend"] == "cpu"
    assert engine.calls == [("metal", True)]


def test_image_only_profile_does_not_consult_the_inference_engine():
    engine = Engine()
    profile = build_execution_profile(
        engine, ["img"], cpu_only=False, hardware_profile=hardware(), runtime_version="b6000",
    )
    assert profile["backend"] == profile["hardware_backend"] == "metal"
    assert engine.calls == []
