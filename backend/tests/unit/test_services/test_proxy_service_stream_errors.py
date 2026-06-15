"""Stream failure surfacing for ProxyService.

Covers the two silent-exit paths:
- An empty upstream stream (200, zero chunks) must surface an explicit 502 to the
  retry/failover layer instead of an empty stream.
- The in-band error frame format used when a stream fails after headers are sent
  (single owner, both client protocols).
"""
from unittest.mock import AsyncMock, patch

import pytest

from app.common.time import utc_now
from app.domain.model import ModelMapping
from app.providers.base import ProviderResponse
from app.rules.models import CandidateProvider
from app.services.proxy_service import ProxyService, _stream_error_frames


def test_stream_error_frames_openai():
    frames = _stream_error_frames("openai", "boom")
    assert frames[0] == b'data: {"error": {"message": "boom"}}\n\n'
    assert frames[-1] == b"data: [DONE]\n\n"


def test_stream_error_frames_defaults_to_openai_when_unknown():
    assert _stream_error_frames(None, "x")[-1] == b"data: [DONE]\n\n"


def test_stream_error_frames_anthropic():
    frames = _stream_error_frames("anthropic", "boom")
    assert frames[0].startswith(b"event: error\n")
    assert b'"message": "boom"' in frames[0]
    assert frames[1].startswith(b"event: message_stop\n")


def _model_mapping() -> ModelMapping:
    now = utc_now()
    return ModelMapping(
        requested_model="test-model",
        strategy="round_robin",
        matching_rules=None,
        capabilities=None,
        is_active=True,
        created_at=now,
        updated_at=now,
    )


def _candidate() -> CandidateProvider:
    return CandidateProvider(
        provider_id=1,
        provider_name="p-openai",
        base_url="https://example.com",
        protocol="openai",
        api_key="sk-test",
        target_model="gpt-4o-mini",
        priority=0,
        weight=1,
    )


@pytest.mark.asyncio
async def test_empty_upstream_stream_yields_502_not_silent():
    service = ProxyService(
        model_repo=AsyncMock(),
        provider_repo=AsyncMock(),
        log_repo=AsyncMock(),
    )
    service._resolve_candidates = AsyncMock(  # type: ignore[method-assign]
        return_value=(_model_mapping(), [_candidate()], 0, "openai", {})
    )

    async def empty_forward_stream(**kwargs):
        # Upstream returns 200 then zero chunks.
        return
        yield  # pragma: no cover - makes this an async generator

    fake_client = AsyncMock()
    fake_client.forward_stream = empty_forward_stream

    with patch("app.services.proxy_service.get_provider_client", return_value=fake_client):
        with patch(
            "app.services.proxy_service.convert_request_for_supplier",
            return_value=("/v1/chat/completions", {"model": "gpt-4o-mini", "messages": []}),
        ):
            initial_response, _gen, _conv = await service.process_request_stream(
                api_key_id=1,
                api_key_name="k",
                request_protocol="openai",
                path="/v1/chat/completions",
                request_url="/v1/chat/completions",
                method="POST",
                headers={},
                body={"model": "test-model", "messages": []},
            )

    assert initial_response.status_code == 502
    assert "empty" in (initial_response.error or "").lower()


@pytest.mark.asyncio
async def test_mid_stream_provider_error_surfaces_in_band_frame():
    """A stall (or any failure) after the first good chunk arrives as an
    unsuccessful tuple. The stream already returned 200, so the client must get
    an explicit in-band error frame, not a silently truncated stream."""
    service = ProxyService(
        model_repo=AsyncMock(),
        provider_repo=AsyncMock(),
        log_repo=AsyncMock(),
    )
    service._resolve_candidates = AsyncMock(  # type: ignore[method-assign]
        return_value=(_model_mapping(), [_candidate()], 0, "openai", {})
    )

    async def stalling_forward_stream(**kwargs):
        ok = ProviderResponse(
            status_code=200, headers={"content-type": "text/event-stream"}
        )
        yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', ok
        # Mid-stream stall surfaced by the provider stall-guard as a 504 tuple.
        yield b"", ProviderResponse(
            status_code=504, error="upstream stream stalled (inter-chunk > 20s)"
        )

    fake_client = AsyncMock()
    fake_client.forward_stream = stalling_forward_stream

    with patch("app.services.proxy_service.get_provider_client", return_value=fake_client):
        with patch(
            "app.services.proxy_service.convert_request_for_supplier",
            return_value=("/v1/chat/completions", {"model": "gpt-4o-mini", "messages": []}),
        ):
            initial_response, gen, _conv = await service.process_request_stream(
                api_key_id=1,
                api_key_name="k",
                request_protocol="openai",
                path="/v1/chat/completions",
                request_url="/v1/chat/completions",
                method="POST",
                headers={},
                body={"model": "test-model", "messages": []},
            )
            body = b"".join([chunk async for chunk in gen])

    # The stream started (first chunk was a success) ...
    assert initial_response.status_code == 200
    assert b"hi" in body
    # ... and the mid-stream failure is surfaced explicitly, not truncated.
    assert b'"error"' in body
    assert b"stalled" in body
    assert b"[DONE]" in body
