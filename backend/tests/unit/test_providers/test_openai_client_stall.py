"""forward_stream stall + unexpected-error handling for OpenAIClient.

A provider that returns 200 headers then never sends a token must be surfaced
as a 504 (so retry/failover moves on) instead of occupying the connection for
the full HTTP_TIMEOUT budget. An unexpected mid-stream error must surface as a
500 tuple instead of silently killing the generator.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.common.http_timeout import StreamDeadlinePolicy
from app.providers.openai_client import OpenAIClient


def _uniform_policy(first_byte: float, idle: float) -> StreamDeadlinePolicy:
    """A policy that returns the same (first_byte, idle) for every class, so the
    stall tests can force tiny deadlines without depending on tier/model routing."""
    return StreamDeadlinePolicy(
        saas_first_byte=first_byte,
        onprem_first_byte=first_byte,
        batch_first_byte=first_byte,
        saas_idle=idle,
        onprem_idle=idle,
    )


def _patch_stream(mock_client_cls, *, aiter_bytes, status_code=200):
    mock_client = AsyncMock()
    mock_client.stream = MagicMock()

    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.headers = {"content-type": "text/event-stream"}
    mock_response.aiter_bytes.return_value = aiter_bytes

    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__aenter__.return_value = mock_response
    mock_stream_ctx.__aexit__.return_value = None
    mock_client.stream.return_value = mock_stream_ctx

    mock_client_cls.return_value.__aenter__.return_value = mock_client


async def _run(client):
    chunks = []
    async for chunk, resp in client.forward_stream(
        base_url="https://api.openai.com",
        api_key="sk-test",
        path="/v1/chat/completions",
        method="POST",
        headers={},
        body={"model": "gpt-4", "stream": True},
        target_model="gpt-4",
    ):
        chunks.append((chunk, resp))
    return chunks


@pytest.mark.asyncio
async def test_first_byte_stall_yields_504():
    client = OpenAIClient()
    client.stream_policy = _uniform_policy(0.05, 0.05)

    async def never_first_byte():
        await asyncio.sleep(1.0)
        yield b"data: too late\n\n"

    with patch("httpx.AsyncClient") as mock_client_cls:
        _patch_stream(mock_client_cls, aiter_bytes=never_first_byte())
        chunks = await _run(client)

    assert len(chunks) == 1
    assert chunks[0][1].status_code == 504


@pytest.mark.asyncio
async def test_inter_chunk_stall_after_first_byte_yields_504_tail():
    client = OpenAIClient()
    client.stream_policy = _uniform_policy(1.0, 0.05)

    async def stall_after_first():
        yield b"data: chunk1\n\n"
        await asyncio.sleep(1.0)
        yield b"data: chunk2\n\n"

    with patch("httpx.AsyncClient") as mock_client_cls:
        _patch_stream(mock_client_cls, aiter_bytes=stall_after_first())
        chunks = await _run(client)

    # First good chunk streamed, then a 504 marker tuple (empty body) instead of
    # a 30-minute hang. The success chunk carries the live provider response.
    assert chunks[0][0] == b"data: chunk1\n\n"
    assert chunks[0][1].status_code == 200
    assert chunks[-1][0] == b""
    assert chunks[-1][1].status_code == 504


@pytest.mark.asyncio
async def test_onprem_batch_request_does_not_trip_short_budget():
    """A :onprem model on the batch tier must NOT trip the SaaS first-byte budget.

    The upstream sends its first token after a gap that would kill a SaaS request
    but is well within the on-prem/batch budget, so the stream succeeds.
    """
    client = OpenAIClient()
    # SaaS budget is tiny; on-prem/batch budget is generous.
    client.stream_policy = StreamDeadlinePolicy(
        saas_first_byte=0.05,
        onprem_first_byte=5.0,
        batch_first_byte=5.0,
        saas_idle=0.05,
        onprem_idle=5.0,
    )

    async def slow_cold_start():
        await asyncio.sleep(0.2)  # would trip the 0.05s SaaS budget
        yield b"data: warmed up\n\n"
        yield b"data: [DONE]\n\n"

    chunks = []
    with patch("httpx.AsyncClient") as mock_client_cls:
        _patch_stream(mock_client_cls, aiter_bytes=slow_cold_start())
        async for chunk, resp in client.forward_stream(
            base_url="http://10.0.0.5:8000",
            api_key=None,
            path="/v1/chat/completions",
            method="POST",
            headers={"x-tier": "batch"},
            body={"model": "qwen3.6:onprem", "stream": True},
            target_model="qwen3.6:onprem",
        ):
            chunks.append((chunk, resp))

    # No 504: the cold start was tolerated and both data chunks streamed.
    assert [c[0] for c in chunks] == [b"data: warmed up\n\n", b"data: [DONE]\n\n"]
    assert all(c[1].status_code == 200 for c in chunks)


@pytest.mark.asyncio
async def test_unexpected_error_yields_500_not_raise():
    client = OpenAIClient()

    async def boom():
        raise RuntimeError("kaboom")
        yield b""  # unreachable; makes this an async generator

    with patch("httpx.AsyncClient") as mock_client_cls:
        _patch_stream(mock_client_cls, aiter_bytes=boom())
        chunks = await _run(client)

    assert len(chunks) == 1
    assert chunks[0][1].status_code == 500
    assert "kaboom" in (chunks[0][1].error or "")
