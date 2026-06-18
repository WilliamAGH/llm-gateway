"""
Protocol Conversion (OpenAI <-> Anthropic <-> OpenAI Responses)

Convert request/response between different LLM API protocols
when provider protocol differs from user request protocol.

This module provides backward-compatible functions that delegate
to the new modular protocol conversion architecture.

Main entry points:
    - convert_request_for_supplier(): Convert user request to supplier protocol
    - convert_response_for_user(): Convert supplier response to user protocol
    - convert_stream_for_user(): Convert supplier stream to user protocol
    - normalize_protocol(): Normalize protocol string to canonical form
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, AsyncGenerator, Optional

from app.common.errors import ServiceError
from app.common.reasoning import (
    normalize_reasoning_for_dashscope,
    normalize_reasoning_for_deepseek,
)

# Import from new modular architecture
from app.common.protocol import (
    ConversionResult,
    Protocol,
    ProtocolConversionError,
    UnsupportedConversionError,
)
from app.common.protocol import (
    convert_request as _convert_request,
)
from app.common.protocol import (
    convert_response as _convert_response,
)
from app.common.protocol import (
    convert_stream as _convert_stream,
)
from app.common.protocol import (
    normalize_protocol as _normalize_protocol,
)
from app.common.provider_protocols import (
    ANTHROPIC_PROTOCOL,
    IMPLEMENTATION_PROTOCOLS,
    OPENAI_PROTOCOL,
    OPENAI_RESPONSES_PROTOCOL,
    normalize_frontend_protocol,
    resolve_implementation_protocol,
    uses_dashscope_thinking,
    uses_deepseek_compatible_thinking,
)

logger = logging.getLogger(__name__)


def normalize_protocol(protocol: str) -> str:
    """
    Normalize a protocol string to its canonical form.

    Args:
        protocol: Protocol string (e.g., "openai", "openai_chat", "anthropic")

    Returns:
        Canonical protocol name (openai, openai_responses, or anthropic)

    Raises:
        ServiceError: If protocol is not supported
    """
    try:
        implementation = resolve_implementation_protocol(protocol)
        implementation = (implementation or OPENAI_PROTOCOL).lower().strip()
        if implementation not in IMPLEMENTATION_PROTOCOLS:
            raise ServiceError(
                message=f"Unsupported protocol '{protocol}'",
                code="unsupported_protocol",
            )
        return implementation
    except Exception as e:
        if isinstance(e, ServiceError):
            raise
        raise ServiceError(
            message=f"Unsupported protocol '{protocol}'",
            code="unsupported_protocol",
        ) from e


_IMAGE_PATHS = {"/v1/images/generations", "/v1/images/edits", "/v1/images/variations"}
_LEGACY_IMAGE_RESPONSE_FORMAT_MODELS = {"dall-e-2", "dall-e-3"}
_OPENAI_USER_IDENTIFIER_MAX_LENGTH = 64
_ANTHROPIC_BILLING_SYSTEM_PREFIX = "x-anthropic-billing-header:"
# OpenAI's default ("in-memory") prompt cache lives only minutes and routes best-effort, so identical
# repeats intermittently return cached_tokens=0. "24h" opts into extended caching (Responses API only).
_OPENAI_RESPONSES_CACHE_RETENTION = "24h"


def _apply_image_defaults(path: str, body: dict[str, Any], target_model: str) -> None:
    """Apply default parameters for image API requests."""
    model = str(body.get("model") or target_model).strip().lower()
    if path in _IMAGE_PATHS and model in _LEGACY_IMAGE_RESPONSE_FORMAT_MODELS:
        body.setdefault("response_format", "b64_json")


def _normalize_openai_user_identifier(body: dict[str, Any]) -> None:
    """Keep OpenAI-bound user identifiers within the provider's 64-character limit.

    The id is carried on whichever field the supplier body uses: OpenAI/Responses put it in ``user``,
    while an ``anthropic``-protocol supplier keeps it in ``metadata.user_id``. The latter matters because
    an Anthropic-protocol upstream can itself front an OpenAI backend, which rejects a >64-char user
    identifier ("Invalid 'user': string too long"). A sha256 hex digest is exactly 64 chars, so it
    stays a stable per-user identifier while satisfying the limit.
    """
    user = body.get("user")
    if isinstance(user, str) and len(user) > _OPENAI_USER_IDENTIFIER_MAX_LENGTH:
        body["user"] = hashlib.sha256(user.encode("utf-8")).hexdigest()
    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        user_id = metadata.get("user_id")
        if (
            isinstance(user_id, str)
            and len(user_id) > _OPENAI_USER_IDENTIFIER_MAX_LENGTH
        ):
            metadata["user_id"] = hashlib.sha256(user_id.encode("utf-8")).hexdigest()


def strip_anthropic_billing_system_blocks(body: dict[str, Any]) -> dict[str, Any]:
    """Drop the Claude Agent SDK's ``x-anthropic-billing-header`` system block for OpenAI-bound traffic.

    The SDK prepends an Anthropic-internal billing/telemetry marker as the first ``system`` text block,
    carrying a ``cch=`` token that rotates every request. Anthropic consumes it server-side, but a
    conversion to an OpenAI(/Responses) supplier merges every system block into the ``instructions``
    prefix, so the rotating token lands at byte 0 of OpenAI's prompt-cache prefix and defeats caching on
    every call (``cached_tokens`` stays 0 even on a 300k-token prompt). The marker carries no instruction
    value for an OpenAI model, so strip it from the source ``system`` list before conversion. Returns a
    shallow copy when a block is removed; the original body otherwise (so the Anthropic path is untouched).
    """
    system = body.get("system")
    if not isinstance(system, list):
        return body
    kept = [
        block
        for block in system
        if not (
            isinstance(block, dict)
            and isinstance(block.get("text"), str)
            and block["text"].lstrip().startswith(_ANTHROPIC_BILLING_SYSTEM_PREFIX)
        )
    ]
    if len(kept) == len(system):
        return body
    return {**body, "system": kept}


def convert_request_for_supplier(
    *,
    request_protocol: str,
    supplier_protocol: str,
    path: str,
    body: dict[str, Any],
    target_model: str,
    options: Optional[dict[str, Any]] = None,
) -> tuple[str, dict[str, Any]]:
    """
    Convert user request protocol to supplier protocol request body/path.

    Supports conversion between:
    - OpenAI: /v1/chat/completions
    - Anthropic: /v1/messages
    - OpenAI Responses: /v1/responses

    Args:
        request_protocol: Protocol of the incoming user request
        supplier_protocol: Protocol expected by the supplier/provider
        path: Original request path
        body: Request body in user protocol format
        target_model: Target model name for the supplier

    Returns:
        tuple[str, dict]: (target_path, converted_body)

    Raises:
        ServiceError: If conversion fails or is not supported
    """
    try:
        supplier_frontend_protocol = normalize_frontend_protocol(supplier_protocol)

        # Normalize protocols
        request_protocol = normalize_protocol(request_protocol)
        supplier_protocol = normalize_protocol(supplier_protocol)

        # Strip the Claude Agent SDK's rotating `x-anthropic-billing-header` system block before an
        # OpenAI-bound conversion folds it into the (otherwise stable) instructions prefix and poisons
        # OpenAI prompt caching. Anthropic-bound traffic keeps it (the upstream consumes it).
        source_body = body
        if request_protocol == ANTHROPIC_PROTOCOL and supplier_protocol in (
            OPENAI_PROTOCOL,
            OPENAI_RESPONSES_PROTOCOL,
        ):
            source_body = strip_anthropic_billing_system_blocks(body)

        # Use new conversion module
        result = _convert_request(
            source_protocol=request_protocol,
            target_protocol=supplier_protocol,
            path=path,
            body=source_body,
            target_model=target_model,
            options=options,
        )

        converted_body = result.body
        if uses_deepseek_compatible_thinking(supplier_frontend_protocol):
            converted_body = normalize_reasoning_for_deepseek(
                converted_body,
                source_body=body,
            )
        elif uses_dashscope_thinking(supplier_frontend_protocol):
            converted_body = normalize_reasoning_for_dashscope(
                converted_body,
                source_body=body,
            )

        if supplier_protocol in (
            OPENAI_PROTOCOL,
            OPENAI_RESPONSES_PROTOCOL,
            ANTHROPIC_PROTOCOL,
        ):
            _normalize_openai_user_identifier(converted_body)

        if (
            supplier_protocol == OPENAI_RESPONSES_PROTOCOL
            and "prompt_cache_retention" not in converted_body
        ):
            # Honor an explicit client choice (the conversion can drop it), else default to 24h.
            client_retention = (
                body.get("prompt_cache_retention") if isinstance(body, dict) else None
            )
            converted_body["prompt_cache_retention"] = (
                client_retention or _OPENAI_RESPONSES_CACHE_RETENTION
            )

        _apply_image_defaults(result.path, converted_body, target_model)

        return result.path, converted_body

    except UnsupportedConversionError as e:
        raise ServiceError(
            message=e.message,
            code=e.code,
        ) from e
    except ProtocolConversionError as e:
        raise ServiceError(
            message=e.message,
            code=e.code,
        ) from e
    except ServiceError:
        raise
    except Exception as e:
        logger.exception(
            "Unexpected error during request conversion: %s -> %s",
            request_protocol,
            supplier_protocol,
        )
        raise ServiceError(
            message=f"Request conversion failed: {str(e)}",
            code="conversion_error",
        ) from e


def convert_response_for_user(
    *,
    request_protocol: str,
    supplier_protocol: str,
    body: Any,
    target_model: str,
) -> Any:
    """
    Convert supplier response to user request protocol response body.

    Args:
        request_protocol: Protocol the user expects (original request protocol)
        supplier_protocol: Protocol of the supplier response
        body: Response body from supplier
        target_model: Target model name

    Returns:
        Converted response body in user's expected protocol format

    Raises:
        ServiceError: If conversion fails or is not supported
    """
    try:
        # Normalize protocols
        request_protocol = normalize_protocol(request_protocol)
        supplier_protocol = normalize_protocol(supplier_protocol)

        # No conversion needed for same protocol
        if request_protocol == supplier_protocol:
            return body

        # Skip non-dict bodies
        if not isinstance(body, dict):
            return body

        # Use new conversion module
        # Note: For response conversion, we convert FROM supplier TO user request protocol
        return _convert_response(
            source_protocol=supplier_protocol,
            target_protocol=request_protocol,
            body=body,
            target_model=target_model,
        )

    except UnsupportedConversionError as e:
        raise ServiceError(
            message=e.message,
            code=e.code,
        ) from e
    except ProtocolConversionError as e:
        raise ServiceError(
            message=e.message,
            code=e.code,
        ) from e
    except ServiceError:
        raise
    except Exception as e:
        logger.exception(
            "Unexpected error during response conversion: %s -> %s",
            supplier_protocol,
            request_protocol,
        )
        raise ServiceError(
            message=f"Response conversion failed: {str(e)}",
            code="conversion_error",
        ) from e


async def convert_stream_for_user(
    *,
    request_protocol: str,
    supplier_protocol: str,
    upstream: AsyncGenerator[bytes, None],
    model: str,
    input_tokens: Optional[int] = None,
) -> AsyncGenerator[bytes, None]:
    """
    Convert supplier SSE bytes stream to user request protocol SSE bytes stream.

    SSE Formats:
    - OpenAI: data: {chat.completion.chunk}\n\n + data: [DONE]\n\n
    - Anthropic: data: {type: ...}\n\n (ends with message_stop event)
    - OpenAI Responses: data: {type: ...}\n\n

    Args:
        request_protocol: Protocol the user expects
        supplier_protocol: Protocol of the supplier stream
        upstream: Async generator yielding bytes from upstream provider
        model: Model name for the response

    Yields:
        Converted SSE bytes in user's expected protocol format

    Raises:
        ServiceError: If conversion fails or is not supported
    """
    try:
        # Normalize protocols
        request_protocol = normalize_protocol(request_protocol)
        supplier_protocol = normalize_protocol(supplier_protocol)

        # No conversion needed for same protocol
        if request_protocol == supplier_protocol:
            async for chunk in upstream:
                yield chunk
            return

        # Use new conversion module
        # Note: For stream conversion, we convert FROM supplier TO user request protocol
        async for chunk in _convert_stream(
            source_protocol=supplier_protocol,
            target_protocol=request_protocol,
            upstream=upstream,
            model=model,
            options={"input_tokens": input_tokens}
            if input_tokens is not None
            else None,
        ):
            yield chunk

    except UnsupportedConversionError as e:
        raise ServiceError(
            message=e.message,
            code=e.code,
        ) from e
    except ProtocolConversionError as e:
        raise ServiceError(
            message=e.message,
            code=e.code,
        ) from e
    except ServiceError:
        raise
    except Exception as e:
        logger.exception(
            "Unexpected error during stream conversion: %s -> %s",
            supplier_protocol,
            request_protocol,
        )
        raise ServiceError(
            message=f"Stream conversion failed: {str(e)}",
            code="conversion_error",
        ) from e


# Export for backward compatibility
__all__ = [
    "normalize_protocol",
    "convert_request_for_supplier",
    "convert_response_for_user",
    "convert_stream_for_user",
]
