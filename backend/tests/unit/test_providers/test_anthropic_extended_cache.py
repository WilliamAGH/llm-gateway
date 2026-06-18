"""Batch-tier 1h prompt-cache: cache_control ttl="1h" rewrite (GA, no beta header)."""

from app.providers.anthropic_client import (
    EXTENDED_CACHE_TTL,
    AnthropicClient,
    _apply_extended_cache,
    _ensure_cache_breakpoints,
    _has_cache_control,
    _is_batch_tier,
    _set_cache_control_ttl,
)


def _body_with_cache_blocks():
    return {
        "model": "claude-opus-4-7",
        "system": [
            {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}}
        ],
        "tools": [{"name": "x", "cache_control": {"type": "ephemeral", "ttl": "5m"}}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hi",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
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


def test_apply_extended_cache_does_not_mutate_input():
    # Root cause: the rewrite must deep-copy, so a retry/failover re-forwarding the
    # caller's original body is never poisoned with a stale ttl="1h".
    original = _body_with_cache_blocks()
    result = _apply_extended_cache(original, {"x-tier": "batch"})
    assert result is not original
    assert result["system"][0]["cache_control"]["ttl"] == "1h"
    assert "ttl" not in original["system"][0]["cache_control"]  # input untouched
    assert (
        original["tools"][0]["cache_control"]["ttl"] == "5m"
    )  # input's explicit 5m preserved


def test_apply_extended_cache_skips_minimax():
    # Framework-first: MiniMax's Anthropic-compat layer doesn't honor the extended TTL,
    # so the batch-tier rewrite is skipped and the body passes through unchanged.
    body = _body_with_cache_blocks()
    result = _apply_extended_cache(body, {"x-tier": "batch"}, is_minimax=True)
    assert result is body
    assert "ttl" not in result["system"][0]["cache_control"]


def _body_without_breakpoints(*, with_tools: bool, system="You are a sub-agent."):
    body = {
        "model": "claude-sonnet-4-6",
        "system": system,
        "messages": [{"role": "user", "content": "do the task"}],
    }
    if with_tools:
        body["tools"] = [{"name": "a"}, {"name": "b"}]
    return body


def test_has_cache_control_detects_existing_breakpoints():
    assert _has_cache_control(_body_with_cache_blocks()) is True
    assert _has_cache_control(_body_without_breakpoints(with_tools=True)) is False


def test_ensure_breakpoints_tags_static_prefix_when_client_set_none():
    # Sub-agent turns ship no cache_control; tag the stable prefix (last tool + last system block)
    # so Anthropic caches it. System (a string) is wrapped into a cache-tagged text block.
    body = _body_without_breakpoints(with_tools=True)
    _ensure_cache_breakpoints(body)
    assert body["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert body["tools"][0].get("cache_control") is None  # only the LAST tool
    assert isinstance(body["system"], list)
    assert body["system"][-1]["cache_control"] == {"type": "ephemeral"}
    # never tag the volatile trailing message
    assert _has_cache_control(body["messages"]) is False


def test_ensure_breakpoints_tags_last_system_block_in_list():
    body = {
        "model": "claude-sonnet-4-6",
        "system": [
            {"type": "text", "text": "x-anthropic-billing-header: cch=abc;"},
            {"type": "text", "text": "real sub-agent prompt"},
        ],
        "messages": [{"role": "user", "content": "hi"}],
    }
    _ensure_cache_breakpoints(body)
    assert "cache_control" not in body["system"][0]  # billing marker block untouched
    assert body["system"][1]["cache_control"] == {"type": "ephemeral"}


def test_ensure_breakpoints_respects_client_breakpoints():
    # Anthropic allows max 4 breakpoints and the client's own placement wins — never add on top.
    body = _body_with_cache_blocks()
    before = {
        "system": [b.get("cache_control") for b in body["system"]],
        "tools": [t.get("cache_control") for t in body["tools"]],
    }
    _ensure_cache_breakpoints(body)
    assert [b.get("cache_control") for b in body["system"]] == before["system"]
    assert [t.get("cache_control") for t in body["tools"]] == before["tools"]


def test_ensure_breakpoints_uses_distinct_dicts_so_ttl_upgrade_is_independent():
    # Each injected cache_control must be its own dict; _set_cache_control_ttl mutates in place.
    body = _body_without_breakpoints(with_tools=True)
    _ensure_cache_breakpoints(body)
    assert body["tools"][-1]["cache_control"] is not body["system"][-1]["cache_control"]
    _set_cache_control_ttl(body, EXTENDED_CACHE_TTL)
    assert body["tools"][-1]["cache_control"]["ttl"] == "1h"
    assert body["system"][-1]["cache_control"]["ttl"] == "1h"


def test_apply_extended_cache_injects_breakpoints_for_untagged_batch_request():
    # End-to-end: a batch sub-agent request with no breakpoints comes out tagged AND at 1h ttl.
    out = _apply_extended_cache(
        _body_without_breakpoints(with_tools=True), {"x-tier": "batch"}
    )
    assert out["system"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert out["tools"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_apply_extended_cache_does_not_inject_on_live_tier():
    out = _apply_extended_cache(
        _body_without_breakpoints(with_tools=True), {"x-tier": "production-z"}
    )
    assert (
        _has_cache_control(out) is False
    )  # live tier untouched; client owns breakpoints there


def test_prepare_headers_strips_internal_routing_headers():
    # Encapsulation: x-tier (internal routing) and the x-lgw-* observability namespace
    # must never leak to the upstream provider; the GA 1h TTL needs no beta header either.
    client = AnthropicClient()
    out = client._prepare_headers(
        {
            "x-tier": "batch",
            "X-Lgw-Trace-Id": "abc123",
            "anthropic-version": "2023-06-01",
        },
        "sk-test",
    )
    assert "x-tier" not in {k.lower() for k in out}
    assert not any(k.lower().startswith("x-lgw-") for k in out)
    assert "extended-cache-ttl" not in out.get("anthropic-beta", "")
