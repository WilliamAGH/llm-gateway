"""Tests for the upstream HTTP timeout policy."""
import asyncio

import pytest

from app.common.http_timeout import (
    CONNECT_TIMEOUT_S,
    POOL_TIMEOUT_S,
    WRITE_TIMEOUT_S,
    StreamDeadlinePolicy,
    StreamStalled,
    build_http_timeout,
    header_value,
    is_batch_tier,
    iter_with_stall_guard,
)

POLICY = StreamDeadlinePolicy(
    saas_first_byte=60.0,
    onprem_first_byte=75.0,
    batch_first_byte=900.0,
    saas_idle=20.0,
    onprem_idle=60.0,
)


def test_is_batch_tier_single_source_of_truth():
    # The one shared tier check used by both the deadline policy and the Anthropic cache gate.
    assert is_batch_tier("batch") is True
    assert is_batch_tier("  BATCH  ") is True  # trimmed + case-insensitive
    assert is_batch_tier("production-a") is False
    assert is_batch_tier("batch-job") is False  # only the canonical token counts
    assert is_batch_tier(None) is False
    assert is_batch_tier("") is False


def test_policy_saas_production_stays_tight():
    assert POLICY.for_request(tier="production-a", target_model="gpt-5.4") == (60.0, 20.0)


def test_policy_onprem_production_gets_warm_budget():
    # :onprem alias, non-batch tier -> 75s first byte, 60s idle.
    assert POLICY.for_request(tier="production-b", target_model="qwen3.6:onprem") == (75.0, 60.0)


def test_policy_batch_tier_gets_long_budget_model_agnostic():
    # batch tier wins even when the model alias was mapped away from :onprem,
    # so a suffix-stripped on-prem batch request still gets the cold-start budget.
    assert POLICY.for_request(tier="batch", target_model="qwen3.6") == (900.0, 60.0)


def test_policy_batch_signal_is_case_insensitive():
    assert POLICY.for_request(tier="BATCH", target_model="gpt-5.4") == (900.0, 60.0)


def test_policy_missing_tier_defaults_to_class_by_model():
    assert POLICY.for_request(tier=None, target_model="claude-3") == (60.0, 20.0)
    assert POLICY.for_request(tier=None, target_model="qwen3.6:onprem") == (75.0, 60.0)


def test_policy_onprem_published_alias_drives_warm_budget_with_neutral_source():
    # Real deployment: the :onprem suffix is a published-alias signal on the
    # requested model, while the provider-side source name is neutral. The
    # requested model must drive the warm budget even when target_model is bare.
    assert POLICY.for_request(
        tier="production-b", requested_model="qwen3.6:onprem", target_model="qwen3.6"
    ) == (75.0, 60.0)


def test_policy_neutral_models_without_any_onprem_signal_stay_tight():
    # Nothing marks this on-prem (no batch tier, no :onprem on either model),
    # so a slow upstream still fails fast on the SaaS budget.
    assert POLICY.for_request(
        tier="production-a", requested_model="qwen3.6", target_model="qwen3.6"
    ) == (60.0, 20.0)


def test_header_value_is_case_insensitive():
    assert header_value({"X-Tier": "batch"}, "x-tier") == "batch"
    assert header_value({"x-tier": "production-a"}, "X-TIER") == "production-a"
    assert header_value({}, "x-tier") is None


def test_build_http_timeout_keeps_read_budget_and_caps_phases():
    timeout = build_http_timeout(1800)
    # The generation read budget is preserved...
    assert timeout.read == 1800.0
    # ...while connect/write/pool are capped so a dead host cannot wedge a slot
    # for the full read budget (the bare-int httpx default would do exactly that).
    assert timeout.connect == CONNECT_TIMEOUT_S
    assert timeout.write == WRITE_TIMEOUT_S
    assert timeout.pool == POOL_TIMEOUT_S


async def test_stall_guard_passes_clean_stream_through():
    async def source():
        yield b"a"
        yield b"b"
        yield b"c"

    out = [
        chunk
        async for chunk in iter_with_stall_guard(
            source(), first_byte_timeout=1.0, idle_timeout=1.0
        )
    ]
    assert out == [b"a", b"b", b"c"]


async def test_stall_guard_trips_on_first_byte():
    async def never_first_byte():
        await asyncio.sleep(1.0)
        yield b"too late"

    with pytest.raises(StreamStalled) as excinfo:
        async for _ in iter_with_stall_guard(
            never_first_byte(), first_byte_timeout=0.05, idle_timeout=0.05
        ):
            pass
    assert excinfo.value.phase == "first byte"


async def test_stall_guard_trips_on_inter_chunk_gap_after_first_byte():
    async def stall_after_one():
        yield b"hi"
        await asyncio.sleep(1.0)
        yield b"never"

    seen: list[bytes] = []
    with pytest.raises(StreamStalled) as excinfo:
        async for chunk in iter_with_stall_guard(
            stall_after_one(), first_byte_timeout=1.0, idle_timeout=0.05
        ):
            seen.append(chunk)
    # The first byte arrived under its (larger) budget; the gap to the second
    # chunk is what trips the tighter idle deadline.
    assert seen == [b"hi"]
    assert excinfo.value.phase == "inter-chunk"
