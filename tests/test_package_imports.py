import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def test_internal_modules_use_package_imports():
    internal_names = {
        path.stem for path in SCRIPTS.rglob("*.py") if path.stem != "__init__"
    }
    violations = []
    for path in SCRIPTS.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".", 1)[0]
                if root in internal_names and root != "scripts":
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno} from {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root in internal_names and root != "scripts":
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno} import {alias.name}")
    assert violations == []
