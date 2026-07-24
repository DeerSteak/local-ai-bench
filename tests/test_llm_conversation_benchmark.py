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


# ── compute_growth_step ──

def test_growth_step_takes_the_full_gap_when_it_fits_within_step_max():
    # Small gap: the whole remaining distance becomes the step, no need to
    # divide it up.
    step, out_of_room = Conv.compute_growth_step(
        cumulative_tokens=1900, target=2048, num_ctx=100_000, is_last_checkpoint=False)
    assert out_of_room is False
    assert step == 2048 - 1900


def test_growth_step_divides_a_large_gap_by_the_divisor():
    step, out_of_room = Conv.compute_growth_step(
        cumulative_tokens=0, target=65536, num_ctx=200_000, is_last_checkpoint=False)
    assert out_of_room is False
    remaining = 65536
    expected = max(Conv.CONV_STEP_MIN, remaining // Conv.CONV_STEP_DIVISOR)
    expected = min(expected, Conv.CONV_STEP_MAX_FAR)
    assert step == expected


def test_growth_step_never_exceeds_step_max_far():
    step, out_of_room = Conv.compute_growth_step(
        cumulative_tokens=0, target=1_000_000, num_ctx=2_000_000, is_last_checkpoint=False)
    assert out_of_room is False
    assert step <= Conv.CONV_STEP_MAX_FAR


def test_growth_step_uses_smaller_step_max_when_close_to_target():
    # When remaining <= 8192, step_max should be CONV_STEP_MAX (1024), not CONV_STEP_MAX_FAR (4096)
    step, out_of_room = Conv.compute_growth_step(
        cumulative_tokens=0, target=8192, num_ctx=100_000, is_last_checkpoint=False)
    assert out_of_room is False
    # remaining // divisor = 8192 // 4 = 2048, which is larger than CONV_STEP_MAX (1024),
    # so it should be clamped to CONV_STEP_MAX (1024)
    assert step == Conv.CONV_STEP_MAX


def test_growth_step_never_below_step_min_when_room_allows():
    # A tiny gap still produces at least CONV_STEP_MIN when there's room for it.
    step, out_of_room = Conv.compute_growth_step(
        cumulative_tokens=2040, target=2048, num_ctx=100_000, is_last_checkpoint=False)
    assert out_of_room is False
    assert step >= Conv.CONV_STEP_MIN


def test_growth_step_holds_back_safety_margin_for_non_final_checkpoint():
    # num_ctx leaves just enough room for the step itself plus the safety
    # margin — step must be clamped so the margin survives.
    cumulative = 100
    num_ctx = cumulative + Conv.CONV_STEP_MIN + Conv.CONV_SAFETY_MARGIN
    step, out_of_room = Conv.compute_growth_step(
        cumulative_tokens=cumulative, target=cumulative + 10_000,
        num_ctx=num_ctx, is_last_checkpoint=False)
    assert out_of_room is False
    assert step == Conv.CONV_STEP_MIN
    assert cumulative + step + Conv.CONV_SAFETY_MARGIN <= num_ctx


def test_growth_step_uses_full_room_on_final_step_of_last_checkpoint():
    # No next turn to protect on the very last step of the very last
    # checkpoint — it can use every token of room up to num_ctx, no reserve.
    cumulative = 100
    remaining = 500  # <= CONV_STEP_MAX, so this is a final step
    num_ctx = cumulative + remaining  # exactly enough room, no margin to spare
    step, out_of_room = Conv.compute_growth_step(
        cumulative_tokens=cumulative, target=cumulative + remaining,
        num_ctx=num_ctx, is_last_checkpoint=True)
    assert out_of_room is False
    assert step == remaining


def test_growth_step_reports_out_of_room_when_margin_cannot_be_kept():
    cumulative = 100
    num_ctx = cumulative + Conv.CONV_STEP_MIN - 1 + Conv.CONV_SAFETY_MARGIN
    step, out_of_room = Conv.compute_growth_step(
        cumulative_tokens=cumulative, target=cumulative + 10_000,
        num_ctx=num_ctx, is_last_checkpoint=False)
    assert out_of_room is True
    assert step is None


def test_growth_step_out_of_room_even_on_last_checkpoint_if_room_below_min():
    cumulative = 100
    num_ctx = cumulative + Conv.CONV_STEP_MIN - 1
    step, out_of_room = Conv.compute_growth_step(
        cumulative_tokens=cumulative, target=cumulative + 200,
        num_ctx=num_ctx, is_last_checkpoint=True)
    assert out_of_room is True
    assert step is None


def test_growth_sequence_at_the_ceiling_never_exceeds_num_ctx():
    """Caller must pass compute_growth_step an effective depth including the last
    turn's pending response, or requests can overshoot num_ctx — see docs/workloads.md."""
    num_ctx = 32768
    target_threshold = int(num_ctx * 0.995)
    followup_tokens = 20   # a short synthetic followup prompt, e.g. _conv_followup_prompt's length
    cumulative_tokens = 16769   # depth entering the final checkpoint's growth phase
    pending_response_tokens = 0

    for _ in range(1000):
        if cumulative_tokens >= target_threshold:
            break
        effective = cumulative_tokens + pending_response_tokens
        step, out_of_room = Conv.compute_growth_step(
            effective, target_threshold, num_ctx, is_last_checkpoint=True)
        assert out_of_room is False, "should never run out of room in this scenario"

        this_request_tokens = cumulative_tokens + pending_response_tokens + followup_tokens
        assert this_request_tokens <= num_ctx, (
            f"request of {this_request_tokens} tokens would exceed num_ctx={num_ctx}"
        )
        # Model uses its full requested budget — the worst realistic case.
        cumulative_tokens = this_request_tokens
        pending_response_tokens = step
    else:
        raise AssertionError("growth loop did not converge")


# ── conv_ctx_plan ──

def test_ctx_plan_uses_full_target_and_checkpoints_regardless_of_tier():
    target_ctx, checkpoints, num_ctx = Conv.conv_ctx_plan(131072)
    assert target_ctx == Conv.CONV_TARGET_CTX
    assert checkpoints == Conv.CONV_CHECKPOINTS
    assert num_ctx == min(target_ctx + Conv.CONV_CTX_HEADROOM, 131072)


def test_ctx_plan_a_lower_model_max_cuts_off_target_and_checkpoints_early():
    # A model whose real ceiling is below 128K gets a correspondingly
    # shorter plan — no tier-based cap involved.
    target_ctx, checkpoints, num_ctx = Conv.conv_ctx_plan(32768)
    assert target_ctx == 32768
    assert checkpoints[-1] == 32768
    # target_ctx + headroom would exceed model_max, so num_ctx clamps to it.
    assert num_ctx == 32768


def test_ctx_plan_num_ctx_never_exceeds_model_max():
    _, _, num_ctx = Conv.conv_ctx_plan(100000)
    assert num_ctx == 100000
