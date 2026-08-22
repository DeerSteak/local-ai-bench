import pytest

from scripts.runtime.engines import engine_display_name, engine_names, get_engine
from scripts.runtime.engines.llamacpp import LlamaCppEngine
from scripts.runtime.engines.vllm import VllmEngine


def test_engine_names_lists_every_registered_engine():
    assert engine_names() == ["llamacpp", "vllm"]


def test_engine_display_name_only_expands_llamacpp_branding():
    assert engine_display_name("llamacpp") == "llama.cpp"
    assert engine_display_name("vllm") == "vllm"


def test_get_engine_returns_registered_type():
    assert isinstance(get_engine("llamacpp"), LlamaCppEngine)
    assert isinstance(get_engine("vllm"), VllmEngine)


def test_every_registered_engine_implements_the_interface():
    from scripts.runtime.engines.base import InferenceEngine
    for name in engine_names():
        engine = get_engine(name)
        assert isinstance(engine, InferenceEngine)
        assert engine.name == name


def test_get_engine_raises_on_unknown_name():
    with pytest.raises(ValueError, match="Unknown inference engine"):
        get_engine("nope")
