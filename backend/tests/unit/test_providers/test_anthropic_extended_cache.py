"""Batch-tier 1h prompt-cache: cache_control ttl="1h" rewrite (GA, no beta header)."""

from app.providers.anthropic_client import (
    EXTENDED_CACHE_TTL,
    AnthropicClient,
    _apply_extended_cache,
    _is_batch_tier,
    _set_cache_control_ttl,
)


def _body_with_cache_blocks():
    return {
        "model": "claude-opus-4-7",
        "system": [{"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}}],
        "tools": [{"name": "x", "cache_control": {"type": "ephemeral", "ttl": "5m"}}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]},
        ],
    }


def test_is_batch_tier_only_for_batch():
    assert _is_batch_tier({"x-tier": "batch"}) is True
    assert _is_batch_tier({"X-Tier": "BATCH"}) is True
    assert _is_batch_tier({"x-tier": "production-z"}) is False
    assert _is_batch_tier({}) is False


def test_set_cache_control_ttl_upgrades_every_ephemeral_block():
    body = _body_with_cache_blocks()
    _set_cache_control_ttl(body, EXTENDED_CACHE_TTL)
    assert body["system"][0]["cache_control"]["ttl"] == "1h"
    assert body["tools"][0]["cache_control"]["ttl"] == "1h"
    assert body["messages"][0]["content"][0]["cache_control"]["ttl"] == "1h"


def test_apply_extended_cache_batch_only():
    batch = _apply_extended_cache(_body_with_cache_blocks(), {"x-tier": "batch"})
    assert batch["system"][0]["cache_control"]["ttl"] == "1h"

    live = _apply_extended_cache(_body_with_cache_blocks(), {"x-tier": "production-z"})
    assert "ttl" not in live["system"][0]["cache_control"]  # untouched on the live tier


def test_prepare_headers_adds_no_extended_cache_beta():
    # The 1h TTL is GA — no extended-cache beta header is sent on any tier.
    client = AnthropicClient()
    batch = client._prepare_headers({"x-tier": "batch"}, "sk-test")
    assert "extended-cache-ttl" not in batch.get("anthropic-beta", "")

    live = client._prepare_headers({"x-tier": "production-z"}, "sk-test")
    assert "extended-cache-ttl" not in live.get("anthropic-beta", "")
