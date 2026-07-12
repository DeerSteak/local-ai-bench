from llm_conversation_benchmark import LLMConversationBenchmark as Conv


def test_followup_prompt_cycles_through_sections():
    seen = [Conv._conv_followup_prompt(n) for n in range(1, Conv.CONV_NUM_SECTIONS * 2 + 1)]
    for i, prompt in enumerate(seen, start=1):
        section = ((i - 1) % Conv.CONV_NUM_SECTIONS) + 1
        assert f"Section {section}" in prompt


def test_followup_prompt_wraps_around_after_last_section():
    last = Conv._conv_followup_prompt(Conv.CONV_NUM_SECTIONS)
    wrapped = Conv._conv_followup_prompt(Conv.CONV_NUM_SECTIONS + 1)
    assert f"Section {Conv.CONV_NUM_SECTIONS}" in last
    assert "Section 1" in wrapped


def test_checkpoints_ascending_and_within_target_ctx():
    assert Conv.CONV_CHECKPOINTS == sorted(Conv.CONV_CHECKPOINTS)
    assert Conv.CONV_CHECKPOINTS[-1] < Conv.CONV_TARGET_CTX
