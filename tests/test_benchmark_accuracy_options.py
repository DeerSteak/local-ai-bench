import argparse

import pytest

from benchmark import positive_int


@pytest.mark.parametrize("value", ["1", "8192"])
def test_positive_int_accepts_positive_values(value):
    assert positive_int(value) == int(value)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_positive_int_rejects_non_positive_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(value)
