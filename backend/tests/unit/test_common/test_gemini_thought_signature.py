"""Gemini thought_signature persistence for non-OpenAI clients.

A non-OpenAI client (e.g. Anthropic /v1/messages from Claude Code) routed to a
Gemini provider pivots through OpenAI *inside* the converter, so the OpenAI-gated
protocol hooks never see the intermediate. Without the plumbing exercised here,
Gemini's required ``thought_signature`` is dropped and the next tool turn 400s.
"""

from types import SimpleNamespace

import pytest

from app.common.protocol.converters import (
    Protocol,
    SDKRequestConverter,
    SDKResponseConverter,
    SDKStreamConverter,
    _apply_tool_call_extra_content,
    _collect_tool_call_extra_content,
)
from app.services.protocol_hooks import ProtocolConversionHooks

SIG = "THOUGHT_SIG_ABC123"
EXTRA = {"google": {"thought_signature": SIG}}


def test_apply_injects_extra_content_by_id():
    body = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_x", "type": "function",
                     "function": {"name": "f", "arguments": "{}"}}
                ],
            }
        ]
    }
    _apply_tool_call_extra_content(body, {"call_x": EXTRA})
    assert body["messages"][0]["tool_calls"][0]["extra_content"] == EXTRA


def test_apply_preserves_existing_and_skips_unknown():
    body = {
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"id": "call_keep", "extra_content": {"already": True},
                 "function": {"name": "f", "arguments": "{}"}},
                {"id": "call_unknown", "function": {"name": "g", "arguments": "{}"}},
            ]}
        ]
    }
    _apply_tool_call_extra_content(body, {"call_keep": EXTRA})
    tcs = body["messages"][0]["tool_calls"]
    assert tcs[0]["extra_content"] == {"already": True}  # not overwritten
    assert "extra_content" not in tcs[1]  # no cache entry -> untouched


def test_collect_harvests_extra_content_by_id():
    body = {"choices": [{"message": {"tool_calls": [
        {"id": "call_y", "extra_content": EXTRA, "function": {"name": "f", "arguments": "{}"}},
        {"id": "call_z", "function": {"name": "g", "arguments": "{}"}},
    ]}}]}
    assert _collect_tool_call_extra_content(body) == {"call_y": EXTRA}


def test_request_converter_restores_signature_onto_gemini_functioncall():
    """ANTHROPIC -> GEMINI: an injected signature lands on the Gemini functionCall part."""
    body = {
        "model": "claude",
        "max_tokens": 256,
        "tools": [{"name": "get_weather", "description": "w",
                   "input_schema": {"type": "object",
                                    "properties": {"city": {"type": "string"}}}}],
        "messages": [
            {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                 "input": {"city": "Paris"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "15C"}]},
        ],
    }
    options = {"tool_call_extra_content_inject": {"toolu_1": EXTRA}}
    conv = SDKRequestConverter(Protocol.ANTHROPIC, Protocol.GEMINI)
    result = conv.convert("/v1/messages", body, "gemini-3.5-flash", options=options)

    # private carrier popped so it never reaches the SDK
    assert "tool_call_extra_content_inject" not in options
    sigs = [
        part.get("thoughtSignature")
        for content in result.body.get("contents", [])
        for part in content.get("parts", [])
        if "functionCall" in part
    ]
    assert SIG in sigs


def test_response_converter_collects_signature_into_sink():
    """GEMINI -> ANTHROPIC: the signature is harvested into the proxy sink."""
    gemini_body = {
        "candidates": [{"content": {"role": "model", "parts": [
            {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}},
             "thoughtSignature": SIG}
        ]}, "finishReason": "STOP"}],
    }
    sink: dict = {}
    options = {"tool_call_extra_content_sink": sink}
    conv = SDKResponseConverter(Protocol.GEMINI, Protocol.ANTHROPIC)
    conv.convert(gemini_body, "gemini-3.5-flash", options=options)

    assert "tool_call_extra_content_sink" not in options  # popped
    assert any(v == EXTRA for v in sink.values())
    assert sink  # at least one tool_call captured


@pytest.mark.asyncio
async def test_stream_tap_persists_signature():
    async def openai_stream():
        chunk = (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_s",'
            '"type":"function","function":{"name":"f","arguments":"{}"},'
            '"extra_content":{"google":{"thought_signature":"' + SIG + '"}}}]}}]}\n\n'
        )
        yield chunk.encode()
        yield b"data: [DONE]\n\n"

    captured: list[tuple] = []

    async def store_cb(tool_call_id, extra):
        captured.append((tool_call_id, extra))

    conv = SDKStreamConverter(Protocol.GEMINI, Protocol.ANTHROPIC)
    out = [c async for c in conv._tap_tool_call_extra_content(openai_stream(), store_cb)]

    assert captured == [("call_s", EXTRA)]
    assert b"".join(out)  # chunks passed through unchanged


@pytest.mark.asyncio
async def test_hooks_prefetch_and_store_roundtrip():
    store: dict[str, str] = {}

    class FakeKV:
        async def get(self, key):
            return SimpleNamespace(value=store[key]) if key in store else None

        async def set(self, key, value, ttl_seconds=None):
            store[key] = value
            return SimpleNamespace(value=value)

    hooks = ProtocolConversionHooks(kv_repo=FakeKV())
    await hooks.cache_tool_call_extra_content("toolu_1", EXTRA)

    anthropic_body = {"messages": [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}]},
    ]}
    fetched = await hooks.prefetch_tool_call_extra_content(anthropic_body)
    assert fetched == {"toolu_1": EXTRA}


@pytest.mark.asyncio
async def test_hooks_prefetch_without_kv_is_empty():
    hooks = ProtocolConversionHooks(kv_repo=None)
    out = await hooks.prefetch_tool_call_extra_content(
        {"messages": [{"role": "assistant", "content": [
            {"type": "tool_use", "id": "t", "name": "f", "input": {}}]}]}
    )
    assert out == {}
