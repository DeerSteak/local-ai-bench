import pytest

from engines import engine_names, get_engine
from engines.llamacpp import LlamaCppEngine


def test_engine_names_lists_every_registered_engine():
    assert engine_names() == ["llamacpp"]


def test_get_engine_returns_registered_type():
    assert isinstance(get_engine("llamacpp"), LlamaCppEngine)


def test_get_engine_raises_on_unknown_name():
    with pytest.raises(ValueError, match="Unknown inference engine"):
        get_engine("nope")
