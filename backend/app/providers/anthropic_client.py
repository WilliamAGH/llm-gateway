"""
Anthropic Protocol Client

Implements Anthropic-compatible request forwarding.
"""

import copy
import json
import logging
from urllib.parse import urlparse
from typing import Any, AsyncGenerator, Optional

import httpx

from app.common.upstream_url import build_upstream_url
from app.common.timer import Timer
from app.common.http_timeout import (
    BATCH_TIER,
    TIER_HEADER,
    StreamDeadlinePolicy,
    StreamStalled,
    build_http_timeout,
    header_value,
    iter_with_stall_guard,
)
from app.config import get_settings
from app.providers.base import ProviderClient, ProviderResponse

logger = logging.getLogger(__name__)

# Extended (1-hour) prompt-cache opt-in. Anthropic keeps an ephemeral cache_control block warm for 5m
# by default; the 1h tier is now GA and needs only ttl="1h" on the block — the old
# extended-cache-ttl-2025-04-11 beta header is no longer required. Applied ONLY to the batch tier
# (long-running harness work that re-reads a stable prefix across slow iterations / resume-after-backoff),
# never the live tier — a 1h cache WRITE costs 2x base vs 1.25x for 5m, so short one-off chats keep 5m.
EXTENDED_CACHE_TTL = "1h"


def _is_batch_tier(headers: dict[str, str]) -> bool:
    """True for the batch tier (the long-running harness lane); mirrors http_timeout's tier check."""
    return (header_value(headers, TIER_HEADER) or "").strip().lower() == BATCH_TIER


def _set_cache_control_ttl(node: Any, ttl: str) -> None:
    """Recursively upgrade every ephemeral cache_control block in `node` to `ttl`. Mutates in place;
    callers pass a copy they own (see `_apply_extended_cache`), never a shared request body."""
    if isinstance(node, dict):
        cache_control = node.get("cache_control")
        if isinstance(cache_control, dict) and cache_control.get("type") == "ephemeral":
            cache_control["ttl"] = ttl
        for value in node.values():
            _set_cache_control_ttl(value, ttl)
    elif isinstance(node, list):
        for value in node:
            _set_cache_control_ttl(value, ttl)


def _apply_extended_cache(
    body: dict[str, Any],
    headers: dict[str, str],
    *,
    is_minimax: bool = False,
) -> dict[str, Any]:
    """Batch tier, real-Anthropic only: return a DEEP COPY of `body` with ephemeral cache_control
    blocks upgraded to the 1h TTL so a stable prefix stays cache-readable across slow iterations /
    resume-after-backoff. Returns `body` unchanged (no copy) when:
      - not the batch tier (live traffic keeps the cheaper 5m default), or
      - the upstream is MiniMax, whose Anthropic-compat layer does not honor the extended TTL
        (sending it would silently degrade caching with no error).
    Deep-copies before mutating so a retry/failover that re-forwards the caller's original body is
    never poisoned with a stale ttl="1h"."""
    if is_minimax or not _is_batch_tier(headers):
        return body
    body = copy.deepcopy(body)
    _set_cache_control_ttl(body, EXTENDED_CACHE_TTL)
    return body


class AnthropicClient(ProviderClient):
    """
    Anthropic Protocol Client
    
    Supports Anthropic-style API request forwarding, including:
    - /v1/messages
    """
    
    # Anthropic API Version
    ANTHROPIC_VERSION = "2023-06-01"
    
    def __init__(self):
        """Initialize client"""
        settings = get_settings()
        self.timeout = build_http_timeout(settings.HTTP_TIMEOUT)
        self.stream_policy = StreamDeadlinePolicy.from_settings(settings)

    @staticmethod
    def _is_minimax_base_url(base_url: str) -> bool:
        """Detect MiniMax Anthropic-compatible endpoints by hostname."""
        try:
            hostname = (urlparse(base_url).hostname or "").lower()
        except Exception:
            return False
        return "minimax" in hostname

    def _sanitize_minimax_headers(
        self,
        headers: dict[str, str],
    ) -> dict[str, str]:
        """
        Strip Anthropic-specific experimental headers that MiniMax does not
        document as supported on its compatibility endpoint.
        """
        sanitized = dict(headers)
        for key in list(sanitized.keys()):
            lowered = key.lower()
            if lowered == "anthropic-beta":
                del sanitized[key]
            elif lowered == "anthropic-dangerous-direct-browser-access":
                del sanitized[key]
            elif lowered.startswith("x-stainless-"):
                del sanitized[key]
            elif lowered == "x-app":
                del sanitized[key]
        return sanitized

    def _sanitize_minimax_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """
        Strip high-risk Anthropic-only fields that are ignored or unstable on
        MiniMax's Anthropic compatibility layer to reduce long-context failures.
        """
        sanitized = dict(body)
        for key in (
            "context_management",
            "mcp_servers",
            "container",
            "service_tier",
            "thinking",
        ):
            sanitized.pop(key, None)
        return sanitized
    
    def _prepare_headers(
        self,
        headers: dict[str, str],
        api_key: Optional[str],
        extra_headers: Optional[dict[str, str]] = None,
    ) -> dict[str, str]:
        """
        Prepare Anthropic request headers
        
        Anthropic uses x-api-key header for authentication.
        
        Args:
            headers: Original request headers
            api_key: Provider API Key
            extra_headers: Extra headers
        
        Returns:
            dict: Processed request headers
        """
        new_headers = dict(headers)
        
        # Remove original authentication headers and auto-generated headers
        keys_to_remove = [
            "authorization",
            "x-api-key",
            "api-key",
            "x-user-id",
            "content-length",
            "host",
            "content-type",
            "accept-encoding",
            TIER_HEADER,  # internal routing tier (x-tier); routing-only, never forward upstream
        ]
        # Strip exact internal headers plus the gateway's own x-lgw-* observability namespace,
        # so no internal header leaks to the upstream provider.
        for key in list(new_headers.keys()):
            lowered = key.lower()
            if lowered in keys_to_remove or lowered.startswith("x-lgw-"):
                del new_headers[key]
        
        # Add Anthropic specific header
        if api_key:
            new_headers["x-api-key"] = api_key
        
        # Ensure Anthropic version is set
        if "anthropic-version" not in [k.lower() for k in new_headers.keys()]:
            new_headers["anthropic-version"] = self.ANTHROPIC_VERSION
            
        # Merge extra headers (overwrite existing)
        if extra_headers:
            new_headers.update(extra_headers)

        for key in list(new_headers.keys()):
            if key.lower() == "x-user-id":
                del new_headers[key]

        return new_headers
    
    async def forward(
        self,
        base_url: str,
        api_key: Optional[str],
        path: str,
        method: str,
        headers: dict[str, str],
        body: dict[str, Any],
        target_model: str,
        response_mode: str = "parsed",
        extra_headers: Optional[dict[str, str]] = None,
        proxy_config: Optional[dict[str, str]] = None,
    ) -> ProviderResponse:
        """
        Forward request to Anthropic-compatible provider
        
        Args:
            base_url: Provider base URL
            api_key: Provider API Key
            path: Request path
            method: HTTP method
            headers: Request headers
            body: Request body
            target_model: Target model name
            response_mode: Response mode, "parsed" (parse JSON) or "raw" (return raw bytes)
            extra_headers: Extra headers
        
        Returns:
            ProviderResponse: Provider response
        """
        url = build_upstream_url(base_url, path)
        is_minimax = self._is_minimax_base_url(base_url)
        prepared_body = self._prepare_body(body, target_model)
        prepared_body = _apply_extended_cache(prepared_body, headers, is_minimax=is_minimax)
        prepared_headers = self._prepare_headers(headers, api_key, extra_headers)
        if is_minimax:
            prepared_body = self._sanitize_minimax_body(prepared_body)
            prepared_headers = self._sanitize_minimax_headers(prepared_headers)
        prepared_headers["Content-Type"] = "application/json"

        logger.debug(
            "Anthropic Request: method=%s url=%s headers=%s body=%s",
            method,
            url,
            prepared_headers,
            json.dumps(prepared_body, ensure_ascii=False),
        )
        
        timer = Timer().start()
        
        try:
            proxy_url = proxy_config.get("all://") if proxy_config else None
            async with httpx.AsyncClient(timeout=self.timeout, proxy=proxy_url) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=prepared_headers,
                    json=prepared_body,
                )
                
                timer.mark_first_byte()
                
                if response_mode == "raw":
                    response_body: Any = response.content
                else:
                    response_body = response.text
                    try:
                        response_body = response.json()
                    except json.JSONDecodeError:
                        pass
                
                timer.stop()
                
                return ProviderResponse(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    body=response_body,
                    first_byte_delay_ms=timer.first_byte_delay_ms,
                    total_time_ms=timer.total_time_ms,
                )
        
        except httpx.TimeoutException as e:
            timer.stop()
            return ProviderResponse(
                status_code=504,
                error=f"Request timeout: {str(e)}",
                first_byte_delay_ms=timer.first_byte_delay_ms,
                total_time_ms=timer.total_time_ms,
            )
        
        except httpx.RequestError as e:
            timer.stop()
            return ProviderResponse(
                status_code=502,
                error=f"Request error: {str(e)}",
                first_byte_delay_ms=timer.first_byte_delay_ms,
                total_time_ms=timer.total_time_ms,
            )
        
        except Exception as e:
            timer.stop()
            return ProviderResponse(
                status_code=500,
                error=f"Unexpected error: {str(e)}",
                first_byte_delay_ms=timer.first_byte_delay_ms,
                total_time_ms=timer.total_time_ms,
            )
    
    async def list_models(
        self,
        base_url: str,
        api_key: Optional[str],
        extra_headers: Optional[dict[str, str]] = None,
        proxy_config: Optional[dict[str, str]] = None,
    ) -> ProviderResponse:
        """
        List available models from Anthropic-compatible provider
        """
        url = build_upstream_url(base_url, "/v1/models")
        prepared_headers = self._prepare_headers({}, api_key, extra_headers)

        logger.debug(
            "Anthropic List Models: url=%s headers=%s",
            url,
            prepared_headers,
        )

        timer = Timer().start()

        try:
            proxy_url = proxy_config.get("all://") if proxy_config else None
            async with httpx.AsyncClient(timeout=self.timeout, proxy=proxy_url) as client:
                response = await client.request(
                    method="GET",
                    url=url,
                    headers=prepared_headers,
                )

                timer.mark_first_byte()

                response_body: Any = response.text
                try:
                    response_body = response.json()
                except json.JSONDecodeError:
                    pass

                timer.stop()

                return ProviderResponse(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    body=response_body,
                    first_byte_delay_ms=timer.first_byte_delay_ms,
                    total_time_ms=timer.total_time_ms,
                )

        except httpx.TimeoutException as e:
            timer.stop()
            return ProviderResponse(
                status_code=504,
                error=f"Request timeout: {str(e)}",
                first_byte_delay_ms=timer.first_byte_delay_ms,
                total_time_ms=timer.total_time_ms,
            )

        except httpx.RequestError as e:
            timer.stop()
            return ProviderResponse(
                status_code=502,
                error=f"Request error: {str(e)}",
                first_byte_delay_ms=timer.first_byte_delay_ms,
                total_time_ms=timer.total_time_ms,
            )

        except Exception as e:
            timer.stop()
            return ProviderResponse(
                status_code=500,
                error=f"Unexpected error: {str(e)}",
                first_byte_delay_ms=timer.first_byte_delay_ms,
                total_time_ms=timer.total_time_ms,
            )

    async def forward_stream(
        self,
        base_url: str,
        api_key: Optional[str],
        path: str,
        method: str,
        headers: dict[str, str],
        body: dict[str, Any],
        target_model: str,
        requested_model: str = "",
        extra_headers: Optional[dict[str, str]] = None,
        proxy_config: Optional[dict[str, str]] = None,
    ) -> AsyncGenerator[tuple[bytes, ProviderResponse], None]:
        """
        Forward streaming request to Anthropic-compatible provider
        
        Args:
            base_url: Provider base URL
            api_key: Provider API Key
            path: Request path
            method: HTTP method
            headers: Request headers
            body: Request body
            target_model: Target model name
            extra_headers: Extra headers
        
        Yields:
            tuple[bytes, ProviderResponse]: (Data chunk, Response info)
        """
        url = build_upstream_url(base_url, path)
        is_minimax = self._is_minimax_base_url(base_url)
        prepared_body = self._prepare_body(body, target_model)
        prepared_body = _apply_extended_cache(prepared_body, headers, is_minimax=is_minimax)
        prepared_headers = self._prepare_headers(headers, api_key, extra_headers)
        if is_minimax:
            prepared_body = self._sanitize_minimax_body(prepared_body)
            prepared_headers = self._sanitize_minimax_headers(prepared_headers)
        prepared_headers["Content-Type"] = "application/json"

        logger.debug(
            "Anthropic Stream Request: method=%s url=%s headers=%s body=%s",
            method,
            url,
            prepared_headers,
            json.dumps(prepared_body, ensure_ascii=False),
        )
        
        timer = Timer().start()
        first_chunk = True
        
        try:
            proxy_url = proxy_config.get("all://") if proxy_config else None
            async with httpx.AsyncClient(timeout=self.timeout, proxy=proxy_url) as client:
                async with client.stream(
                    method=method,
                    url=url,
                    headers=prepared_headers,
                    json=prepared_body,
                ) as response:
                    provider_response = ProviderResponse(
                        status_code=response.status_code,
                        headers=dict(response.headers),
                    )

                    if response.status_code >= 400:
                        body_bytes = await response.aread()
                        timer.mark_first_byte()
                        timer.stop()
                        provider_response.first_byte_delay_ms = timer.first_byte_delay_ms
                        provider_response.total_time_ms = timer.total_time_ms
                        provider_response.body = body_bytes
                        reason = response.reason_phrase or "Upstream error"
                        provider_response.error = f"{response.status_code} {reason}"
                        yield body_bytes or b"", provider_response
                        return
                    
                    first_byte_timeout, idle_timeout = self.stream_policy.for_request(
                        tier=header_value(headers, TIER_HEADER),
                        requested_model=requested_model,
                        target_model=target_model,
                    )
                    async for chunk in iter_with_stall_guard(
                        response.aiter_bytes(),
                        first_byte_timeout=first_byte_timeout,
                        idle_timeout=idle_timeout,
                    ):
                        if first_chunk:
                            timer.mark_first_byte()
                            provider_response.first_byte_delay_ms = (
                                timer.first_byte_delay_ms
                            )
                            first_chunk = False

                        yield chunk, provider_response

                    timer.stop()
                    provider_response.total_time_ms = timer.total_time_ms

        except StreamStalled as e:
            timer.stop()
            yield b"", ProviderResponse(
                status_code=504,
                error=f"Request timeout: {str(e)}",
                first_byte_delay_ms=timer.first_byte_delay_ms,
                total_time_ms=timer.total_time_ms,
            )

        except httpx.TimeoutException as e:
            timer.stop()
            yield b"", ProviderResponse(
                status_code=504,
                error=f"Request timeout: {str(e)}",
                first_byte_delay_ms=timer.first_byte_delay_ms,
                total_time_ms=timer.total_time_ms,
            )

        except httpx.RequestError as e:
            timer.stop()
            yield b"", ProviderResponse(
                status_code=502,
                error=f"Request error: {str(e)}",
                first_byte_delay_ms=timer.first_byte_delay_ms,
                total_time_ms=timer.total_time_ms,
            )

        except Exception as e:
            # Catch-all so an unexpected error never silently kills the generator
            # mid-stream. CancelledError is a BaseException and is intentionally
            # NOT caught (client disconnect).
            timer.stop()
            yield b"", ProviderResponse(
                status_code=500,
                error=f"Unexpected stream error: {str(e)}",
                first_byte_delay_ms=timer.first_byte_delay_ms,
                total_time_ms=timer.total_time_ms,
            )
