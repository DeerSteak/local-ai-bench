from dataclasses import dataclass

import pytest

from scripts.release.qualification_install import (
    qualification_install_plan, qualification_model, qualification_vllm_handoff,
    qualification_vllm_index,
    validate_vllm_runtime,
)


CATALOG = [{"tag": "tiny", "label": "Tiny", "vllm_repo": "example/tiny"}]


@dataclass
class Support:
    installable: bool
    detail: str = "test support"


def test_model_selection_requires_one_exact_catalog_entry():
    assert qualification_model("tiny", CATALOG)["label"] == "Tiny"
    with pytest.raises(ValueError, match="one catalog entry"):
        qualification_model("missing", CATALOG)


def test_llamacpp_plan_keeps_runtime_models_and_cache_under_isolated_root(tmp_path,
                                                                         monkeypatch):
    monkeypatch.setattr("scripts.release.qualification_install.LLM_MODELS", CATALOG)
    plan = qualification_install_plan(
        root=tmp_path, engine="llamacpp", model_tag="tiny", system="Darwin",
        machine="arm64", nvidia=False, rocm=False, runtime_version="b7000",
    )
    assert plan["runtime_dir"] == str(tmp_path / "llama.cpp")
    assert plan["models_dir"] == str(tmp_path / "models")
    assert plan["cache_dir"] == str(tmp_path / "qualification-cache")
    assert plan["runtime_version"] == "b7000"
    assert plan["coverage_models"] == {
        "llm": "tiny", "embeddings": "nomic-embed-text", "images": "sd15",
    }
    assert not (tmp_path / "llama.cpp").exists()


def test_vllm_plan_rejects_an_uninstallable_platform(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.release.qualification_install.LLM_MODELS", CATALOG)
    with pytest.raises(ValueError, match="cannot be installed"):
        qualification_install_plan(
            root=tmp_path, engine="vllm", model_tag="tiny", system="Darwin",
            machine="arm64", nvidia=False, rocm=False, runtime_version="0.27.1+rocm723",
            vllm_support=Support(False),
        )


def test_vllm_plan_accepts_installable_support_and_records_exact_model(tmp_path,
                                                                       monkeypatch):
    monkeypatch.setattr("scripts.release.qualification_install.LLM_MODELS", CATALOG)
    plan = qualification_install_plan(
        root=tmp_path, engine="vllm", model_tag="tiny", system="Linux",
        machine="x86_64", nvidia=True, rocm=False,
        runtime_version="0.27.1+rocm723", vllm_support=Support(True),
    )
    assert plan["runtime_dir"] == str(tmp_path / "vllm-env")
    assert plan["model"] == {"tag": "tiny", "label": "Tiny"}
    assert plan["runtime_version"] == "0.27.1+rocm723"
    assert plan["coverage_models"]["images"] is None


def test_plan_never_leaves_the_qualification_version_floating(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.release.qualification_install.LLM_MODELS", CATALOG)
    with pytest.raises(ValueError, match="exact runtime version"):
        qualification_install_plan(
            root=tmp_path, engine="vllm", model_tag="tiny", system="Linux",
            machine="x86_64", nvidia=True, rocm=False, vllm_support=Support(True),
        )


def test_dgx_qualification_uses_the_reviewed_cuda_13_release_index():
    index = qualification_vllm_index("cu130_wheel")
    assert index is not None
    assert index.endswith("/0.27.1/cu130")
    assert qualification_vllm_index("cuda_wheel") is None


def test_vllm_runtime_validation_imports_the_installed_environment(tmp_path):
    result = type("Result", (), {"returncode": 0, "stdout": "0.27.1+rocm723\n", "stderr": ""})()
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return result

    assert validate_vllm_runtime(tmp_path / "vllm-env", run) == (
        True, "0.27.1+rocm723",
    )
    assert calls[0][0][0].endswith("vllm-env/bin/python")
    assert calls[0][1]["timeout"] == 300


def test_vllm_runtime_validation_preserves_import_failure_detail(tmp_path):
    result = type("Result", (), {
        "returncode": 1, "stdout": "", "stderr": "libmpi_cxx.so.40: not found\n",
    })()
    assert validate_vllm_runtime(tmp_path / "vllm-env", lambda *_args, **_kwargs: result) == (
        False, "libmpi_cxx.so.40: not found",
    )


def test_qualification_handoff_selects_the_managed_runtime_and_cache(tmp_path):
    runtime = tmp_path / "vllm-env"
    executable = runtime / "bin" / "vllm"
    executable.parent.mkdir(parents=True)
    executable.touch()
    cache = tmp_path / "qualification-vllm-cache"
    assert qualification_vllm_handoff(runtime, cache, system="Linux") == {
        "executable": str(executable), "launcher": None, "server_url": None,
        "launcher_extra_args": [], "hf_home": str(cache),
    }
