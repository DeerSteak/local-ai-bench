from types import SimpleNamespace

from scripts.app.benchmark import apply_quick_preset
from scripts.workloads.models import LLM_MODELS_XSMALL


def test_quick_preset_resolves_a_single_short_llm_smoke_test():
    args = SimpleNamespace(
        quick=True, tests=["img"], warmup=2, runs=3, max_prompt_tokens=None,
        maxtier=None, llm_models=None, engine="vllm", out="chosen.json",
    )
    apply_quick_preset(args)
    assert args.tests == ["llm"]
    assert args.warmup == 0
    assert args.runs == 1
    assert args.max_prompt_tokens == 2048
    assert args.maxtier == "xsmall"
    assert args.llm_models == [LLM_MODELS_XSMALL[0]["tag"]]
    assert (args.engine, args.out) == ("vllm", "chosen.json")


def test_quick_preset_leaves_arguments_unchanged_when_disabled():
    args = SimpleNamespace(quick=False, tests=["img"], runs=3)
    apply_quick_preset(args)
    assert vars(args) == {"quick": False, "tests": ["img"], "runs": 3}
