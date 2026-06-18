"""HTTP timeout policy for upstream provider calls.

Single owner for two related concerns that were previously implicit:

1. ``build_http_timeout`` turns the flat ``HTTP_TIMEOUT`` budget into a granular
   ``httpx.Timeout``. A bare ``httpx.AsyncClient(timeout=1800)`` applies 1800s to
   *every* phase, so a stalled TCP connect or a hung pool checkout would wait the
   full generation budget. We keep the generous ``read`` budget (so long
   generations stream) but cap connect/write/pool.

2. ``iter_with_stall_guard`` enforces a short first-byte deadline and a short
   inter-chunk idle deadline on a streaming body. In httpx the ``read`` timeout
   *is* the per-chunk idle timeout, but it is tied to the whole-generation budget
   (1800s), so an upstream that returns 200 headers then never sends a token is
   not detected for the full budget. This guard catches that in seconds and lets
   the caller surface a 504 so the retry/failover layer moves on.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Mapping, Optional

import httpx

# Tier header set by the upstream queue (x-tier: batch | production-a | ...).
TIER_HEADER = "x-tier"
BATCH_TIER = "batch"
# Public on-prem model alias suffix (e.g. "qwen3.6:onprem").
ONPREM_MODEL_SUFFIX = ":onprem"

# Phase caps applied alongside the generation ``read`` budget. Short enough that a
# dead host never wedges a slot; long enough to tolerate a busy but live upstream.
CONNECT_TIMEOUT_S = 10.0
WRITE_TIMEOUT_S = 30.0
POOL_TIMEOUT_S = 5.0


class StreamStalled(Exception):
    """An upstream stream produced no bytes within its first-byte/idle deadline.

    Distinct from ``httpx.TimeoutException`` so callers can label the phase and
    map it to a 504 without conflating it with a connect/total read timeout.
    """

    def __init__(self, phase: str, timeout_s: float) -> None:
        self.phase = phase
        self.timeout_s = timeout_s
        super().__init__(f"upstream stream stalled ({phase} > {timeout_s:g}s)")


def build_http_timeout(total_timeout_s: float) -> httpx.Timeout:
    """Granular timeout: capped connect/write/pool, generous read = generation budget."""
    return httpx.Timeout(
        connect=CONNECT_TIMEOUT_S,
        read=float(total_timeout_s),
        write=WRITE_TIMEOUT_S,
        pool=POOL_TIMEOUT_S,
    )


def header_value(headers: Mapping[str, str], name: str) -> Optional[str]:
    """Case-insensitive header lookup (ASGI lowercases, but upstreams vary)."""
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return None


def is_batch_tier(tier: Optional[str]) -> bool:
    """Single source of truth for the batch-tier check: True when a resolved tier string is the
    batch lane (the long-running harness lane). Shared by the stream-deadline policy and the
    Anthropic prompt-cache gate so the comparison lives in exactly one place."""
    return (tier or "").strip().lower() == BATCH_TIER


@dataclass(frozen=True)
class StreamDeadlinePolicy:
    """Chooses (first_byte, idle) stream deadlines by request class.

    On-prem local models can spend minutes on prompt processing before the first
    token, so they need a much larger first-byte budget than SaaS providers. The
    long latency is time-to-first-token, not inter-chunk, so idle stays tight and
    SaaS/production traffic still fails fast on the same path.

    The long class is recognized by EITHER signal: an ``x-tier: batch`` header
    (set by the queue) OR an ``:onprem`` model alias. This composes with the
    queue's own per-(lane, tier) guard — the effective deadline is the min of the
    two, so the queue's finer split (e.g. 75s on-prem production) stays authoritative
    while Squirrel never trips before it for legitimate on-prem traffic.
    """

    saas_first_byte: float
    onprem_first_byte: float
    batch_first_byte: float
    saas_idle: float
    onprem_idle: float

    @classmethod
    def from_settings(cls, settings) -> "StreamDeadlinePolicy":
        return cls(
            saas_first_byte=float(settings.STREAM_FIRST_BYTE_TIMEOUT),
            onprem_first_byte=float(settings.STREAM_ONPREM_FIRST_BYTE_TIMEOUT),
            batch_first_byte=float(settings.STREAM_BATCH_FIRST_BYTE_TIMEOUT),
            saas_idle=float(settings.STREAM_IDLE_TIMEOUT),
            onprem_idle=float(settings.STREAM_ONPREM_IDLE_TIMEOUT),
        )

    def for_request(
        self,
        *,
        tier: Optional[str],
        requested_model: str = "",
        target_model: str = "",
    ) -> tuple[float, float]:
        if is_batch_tier(tier):
            return self.batch_first_byte, self.onprem_idle
        # The :onprem signal is a published-alias suffix carried on the
        # requested model (e.g. "qwen3.6:onprem"). target_model is the
        # provider-side source name and is usually provider-neutral, so prefer
        # requested_model and keep target_model only as a fallback for providers
        # that do carry the suffix.
        candidates = (requested_model, target_model)
        if any((m or "").strip().lower().endswith(ONPREM_MODEL_SUFFIX) for m in candidates):
            return self.onprem_first_byte, self.onprem_idle
        return self.saas_first_byte, self.saas_idle


async def iter_with_stall_guard(
    source: AsyncIterator[bytes],
    *,
    first_byte_timeout: float,
    idle_timeout: float,
) -> AsyncIterator[bytes]:
    """Yield chunks from ``source``, raising :class:`StreamStalled` when the first
    chunk exceeds ``first_byte_timeout`` or any later gap exceeds ``idle_timeout``.

    The first ``__anext__`` (time-to-first-token after headers) gets the larger
    first-byte budget; every subsequent read gets the tighter idle budget.
    ``asyncio.CancelledError`` (client disconnect) propagates untouched.
    """
    iterator = source.__aiter__()
    first = True
    while True:
        deadline = first_byte_timeout if first else idle_timeout
        try:
            async with asyncio.timeout(deadline):
                chunk = await iterator.__anext__()
        except StopAsyncIteration:
            return
        except TimeoutError as exc:
            raise StreamStalled("first byte" if first else "inter-chunk", deadline) from exc
        first = False
        yield chunk
