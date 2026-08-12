from pathlib import Path
import importlib
import sys


SETUP_CHECK = Path(__file__).parents[1] / "scripts" / "setup" / "setup_check.py"


def test_setup_coordinator_retains_install_and_summary_stages():
    source = SETUP_CHECK.read_text(encoding="utf-8")

    assert "ensure_comfyui(" in source
    assert "provision_comfyui_assets(" in source
    assert "write_setup_config(" in source
    assert 'section("Summary")' in source


def test_setup_coordinator_import_is_safe():
    sys.modules.pop("scripts.setup.setup_check", None)
    module = importlib.import_module("scripts.setup.setup_check")

    assert callable(module.main)
