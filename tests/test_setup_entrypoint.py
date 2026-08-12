import importlib
import sys


def test_setup_entrypoint_import_does_not_load_workflow():
    sys.modules.pop("scripts.setup.setup_workflow", None)
    module = importlib.import_module("scripts.setup.setup_check")

    assert callable(module.main)
    assert "scripts.setup.setup_workflow" not in sys.modules
