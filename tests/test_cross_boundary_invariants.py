import re
from pathlib import Path

from scripts.workloads.llm_conversation_benchmark import LLMConversationBenchmark
from scripts.workloads.models import EMBED_MODELS, IMAGE_MODELS, LLM_MODELS


DASHBOARD_CONSTANTS = Path(__file__).parents[1] / "dashboard" / "src" / "constants.ts"


def _typescript_string_array(source: str, name: str) -> list[str]:
    match = re.search(rf"export const {name} = (\[[^;]+\]);", source, re.DOTALL)
    assert match is not None, f"dashboard constant {name} is missing or no longer a literal array"
    # This intentionally parses catalog-controlled literal arrays, not general TypeScript.
    body = re.sub(r"//.*?$|/\*.*?\*/", "", match.group(1), flags=re.MULTILINE | re.DOTALL)
    return re.findall(r'"([^"]+)"', body)


def _context_label(tokens: int) -> str:
    return f"{tokens // 1024}K" if tokens % 1024 == 0 else f"{tokens / 1024:g}K"


def test_typescript_array_scraper_ignores_quoted_comments():
    source = 'export const VALUES = ["real", /* "block" */ // "line"\n];'
    assert _typescript_string_array(source, "VALUES") == ["real"]


def test_conversation_checkpoints_exist_in_dashboard_context_order():
    source = DASHBOARD_CONSTANTS.read_text(encoding="utf-8")
    dashboard_contexts = set(_typescript_string_array(source, "CTX_ORDER"))
    checkpoint_labels = {
        _context_label(tokens) for tokens in LLMConversationBenchmark.CONV_CHECKPOINTS
    }
    assert checkpoint_labels <= dashboard_contexts


def test_dashboard_model_registries_match_python_catalog():
    source = DASHBOARD_CONSTANTS.read_text(encoding="utf-8")
    assert _typescript_string_array(source, "LLM_MODEL_ORDER") == [
        model["short"] for model in LLM_MODELS
    ]
    assert _typescript_string_array(source, "IMAGE_MODEL_ORDER") == [
        model["short"] for model in IMAGE_MODELS
    ]
    assert _typescript_string_array(source, "EMBED_MODEL_ORDER") == [
        model["short"] for model in EMBED_MODELS
    ]
