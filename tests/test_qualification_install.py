from dataclasses import dataclass

import pytest

from scripts.release.qualification_install import (
    qualification_install_plan, qualification_model,
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


def test_plan_never_leaves_the_qualification_version_floating(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.release.qualification_install.LLM_MODELS", CATALOG)
    with pytest.raises(ValueError, match="exact runtime version"):
        qualification_install_plan(
            root=tmp_path, engine="vllm", model_tag="tiny", system="Linux",
            machine="x86_64", nvidia=True, rocm=False, vllm_support=Support(True),
        )
