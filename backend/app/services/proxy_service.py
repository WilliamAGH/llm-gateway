"""Proxy Core Service Module

Implements core business logic for request proxying."""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Optional

import anyio

from app.common.costs import CostBreakdown, calculate_cost_from_billing, resolve_billing
from app.common.errors import NotFoundError, ServiceError
from app.common.protocol_conversion import (
    convert_request_for_supplier,
    convert_response_for_user,
    convert_stream_for_user,
    normalize_protocol,
    strip_anthropic_billing_system_blocks,
)
from app.common.provider_protocols import resolve_implementation_protocol
from app.common.proxy import build_proxy_config
from app.common.sanitizer import sanitize_headers
from app.common.stream_usage import StreamUsageAccumulator
from app.common.time import utc_now
from app.common.upstream_url import build_upstream_url
from app.common.token_counter import get_token_counter
from app.common.usage_extractor import ensure_openai_usage_details, extract_usage_details
from app.common.utils import generate_trace_id
from app.domain.log import RequestLogCreate
from app.domain.model import ModelMapping, ModelMappingProviderResponse
from app.domain.provider import Provider
from app.providers import ProviderResponse, get_provider_client
from app.repositories.log_repo import LogRepository
from app.repositories.model_repo import ModelRepository
from app.repositories.provider_repo import ProviderRepository
from app.repositories.kv_store_repo import KVStoreRepository
from app.rules import CandidateProvider, RuleContext, RuleEngine, TokenUsage
from app.services.retry_handler import AttemptRecord, RetryHandler
from app.services.protocol_hooks import OPENAI_IMAGE_PATHS, ProtocolConversionHooks
from app.services.strategy import (
    CostFirstStrategy,
    PrefixAffinityStrategy,
    PriorityStrategy,
    RoundRobinStrategy,
    SelectionStrategy,
)

logger = logging.getLogger(__name__)

MAX_LOG_TEXT_LENGTH = 10000
MAX_USER_ID_LENGTH = 255
CandidateKey = tuple[str, int] | tuple[str, int, str]

# Prompt-cache observability. A request that carried a cache signal but came back with zero
# cached tokens is only suspicious on a *repeat* of the same prefix within the cache TTL window —
# a first request legitimately misses (cold). We track recently-seen prompt_cache_keys so the
# WARN fires on a genuine repeat-miss (the signature of a broken/invalidated prefix), not on cold
# starts. Process-local and best-effort; the authoritative signal is the per-request structured log.
_CACHE_KEY_TTL_SECONDS = 600.0
_PROMPT_CACHE_PREFIX_CHARS = 8192
_EXACT_RESPONSE_CACHE_TTL_SECONDS = 600
_EXACT_RESPONSE_CACHE_MIN_INPUT_TOKENS = 1024
_EXACT_RESPONSE_CACHE_PREFIX = "exact_response:v1:"
_EXACT_RESPONSE_CACHE_HIT_HEADER = "x-llm-gateway-response-cache"
_PROMPT_CACHE_KEY_HINT_HEADER = "x-lgw-cache-key"
_recent_prompt_cache_keys: dict[str, float] = {}


def _non_empty_prompt_cache_key(body: Any) -> Optional[str]:
    if not isinstance(body, dict):
        return None
    key = body.get("prompt_cache_key")
    return key if isinstance(key, str) and key.strip() else None


def _prompt_cache_namespace(*values: Any) -> Optional[str]:
    """Sole owner of prompt-cache target detection: maps request values (models, base_url)
    to the namespace that seeds the cache key, or None when no target matches."""
    lowered = [str(v).lower() for v in values if v is not None]
    kimi_markers = ("kimi", "moonshot", "api.kimi.com", "api.moonshot.ai")
    if any(marker in value for value in lowered for marker in kimi_markers):
        return "kimi"
    if any(
        value.startswith("gpt-") and not value.startswith("gpt-oss")
        for value in lowered
    ):
        return "openai"
    return None


def _prompt_cache_key_hint(headers: dict[str, str]) -> Optional[str]:
    for key, value in headers.items():
        if key.lower() == _PROMPT_CACHE_KEY_HINT_HEADER:
            trimmed = str(value).strip()
            return trimmed[:MAX_USER_ID_LENGTH] if trimmed else None
    return None


def _body_prompt_cache_session(body: dict[str, Any]) -> Optional[str]:
    metadata = body.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("user_id"), str):
        user_id = metadata["user_id"].strip()
        if user_id:
            return user_id
    user = body.get("user")
    if isinstance(user, str):
        user = user.strip()
        if user:
            return user
    return None


def _prompt_cache_prefix(body: dict[str, Any]) -> str:
    body = strip_anthropic_billing_system_blocks(body)
    prefix_owner = {
        key: body[key]
        for key in ("messages", "system", "instructions", "input", "tools", "response_format")
        if key in body
    }
    return json.dumps(
        prefix_owner or body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )[:_PROMPT_CACHE_PREFIX_CHARS]


def _first_conversation_turn(body: dict[str, Any]) -> Optional[dict[str, Any]]:
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "system":
                return {
                    "role": message.get("role"),
                    "content": message.get("content"),
                }
        for message in messages:
            if isinstance(message, dict):
                return {
                    "role": message.get("role"),
                    "content": message.get("content"),
                }

    input_value = body.get("input")
    if isinstance(input_value, list) and input_value:
        return {"input": input_value[0]}
    if isinstance(input_value, str) and input_value.strip():
        return {"input": input_value}

    return None


def _kimi_session_seed_owner(body: dict[str, Any], requested_model: str, session_hint: str) -> dict[str, Any]:
    first_turn = _first_conversation_turn(strip_anthropic_billing_system_blocks(body))
    if first_turn:
        return {
            "conversation": first_turn,
            "model": requested_model,
            "session": session_hint,
        }
    return {
        "model": requested_model,
        "prefix": _prompt_cache_prefix(body),
        "session": session_hint,
    }


def _prompt_cache_key_for_request(
    body: Any,
    requested_model: str,
    target_model: list[str] | None = None,
    cache_key_hint: Optional[str] = None,
) -> Optional[str]:
    existing = _non_empty_prompt_cache_key(body)
    if existing:
        return existing
    if not isinstance(body, dict):
        return None
    targets = [requested_model, *(target_model or [])]
    namespace = _prompt_cache_namespace(*targets)
    if namespace is None:
        return None
    if namespace == "kimi":
        session_hint = cache_key_hint or _body_prompt_cache_session(body)
        if session_hint:
            seed_owner = _kimi_session_seed_owner(body, requested_model, session_hint)
        else:
            seed_owner = {"model": requested_model, "prefix": _prompt_cache_prefix(body)}
    else:
        seed_owner = {"model": requested_model, "prefix": _prompt_cache_prefix(body)}
    seed = json.dumps(
        seed_owner,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"llmgw:{namespace}:{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:24]}"


def _body_with_prompt_cache_key(body: Any, prompt_cache_key: Optional[str]) -> Any:
    if not prompt_cache_key or not isinstance(body, dict) or _non_empty_prompt_cache_key(body):
        return body
    return {**body, "prompt_cache_key": prompt_cache_key}


def _supplier_body_with_prompt_cache_key(
    body: Any,
    prompt_cache_key: Optional[str],
    *,
    requested_model: str,
    target_model: str,
    base_url: str,
    supplier_protocol: Optional[str],
) -> Any:
    if (
        not prompt_cache_key
        or not isinstance(body, dict)
        or _non_empty_prompt_cache_key(body)
        or supplier_protocol not in {"openai", "openai_responses"}
        or _prompt_cache_namespace(requested_model, target_model, base_url) is None
    ):
        return body
    return {**body, "prompt_cache_key": prompt_cache_key}


def _request_has_cache_signal(body: Any) -> bool:
    """True if the request asked for prompt caching (OpenAI prompt_cache_key or Anthropic
    cache_control on a system/message block)."""
    if not isinstance(body, dict):
        return False
    pck = body.get("prompt_cache_key")
    if isinstance(pck, str) and pck:
        return True

    def _has_cache_control(blocks: Any) -> bool:
        return isinstance(blocks, list) and any(
            isinstance(b, dict) and b.get("cache_control") for b in blocks
        )

    if _has_cache_control(body.get("system")):
        return True
    return any(
        isinstance(m, dict) and _has_cache_control(m.get("content"))
        for m in (body.get("messages") or [])
    )


def _exact_response_cache_key(
    *,
    api_key_id: Optional[int],
    method: str,
    supplier_path: str,
    requested_model: str,
    provider_id: int,
    provider_mapping_id: Optional[int],
    target_model: str,
    supplier_body: Any,
) -> str:
    seed = json.dumps(
        {
            "api_key_id": api_key_id,
            "method": method.upper(),
            "supplier_path": supplier_path,
            "requested_model": requested_model,
            "provider_id": provider_id,
            "provider_mapping_id": provider_mapping_id,
            "target_model": target_model,
            "supplier_body": supplier_body,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"{_EXACT_RESPONSE_CACHE_PREFIX}{digest}"


def _exact_response_cache_allowed(
    *,
    request_protocol: str,
    supplier_protocol: Optional[str],
    supplier_body: Any,
    method: str,
    prompt_cache_key: Optional[str],
    requested_model: str,
    target_model: str,
    base_url: str,
    input_tokens: Optional[int],
) -> bool:
    if method.upper() != "POST" or not prompt_cache_key:
        return False
    if normalize_protocol(request_protocol) == normalize_protocol(supplier_protocol):
        return False
    namespace = _prompt_cache_namespace(requested_model, target_model, base_url)
    if namespace == "openai":
        if supplier_protocol != "openai_responses":
            return False
    elif namespace == "kimi":
        if supplier_protocol != "openai":
            return False
    else:
        return False
    if int(input_tokens or 0) < _EXACT_RESPONSE_CACHE_MIN_INPUT_TOKENS:
        return False
    if not isinstance(supplier_body, dict):
        return False
    if supplier_body.get("stream"):
        return False
    if supplier_body.get("background"):
        return False
    if supplier_body.get("store") is True:
        return False
    uncacheable_keys = {
        "tools",
        "tool_choice",
        "previous_response_id",
        "conversation",
        "include",
        "prompt",
        "_files",
    }
    return not any(key in supplier_body for key in uncacheable_keys)


def _exact_response_cache_payload(response: ProviderResponse) -> Optional[str]:
    if not response.is_success or not isinstance(response.body, (dict, list)):
        return None
    return json.dumps(
        {
            "status_code": response.status_code,
            "headers": response.headers or {},
            "body": response.body,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _provider_response_from_exact_cache(value: str, elapsed_ms: int) -> Optional[ProviderResponse]:
    try:
        payload = json.loads(value)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    headers = payload.get("headers") if isinstance(payload.get("headers"), dict) else {}
    return ProviderResponse(
        status_code=int(payload.get("status_code") or 200),
        headers={**headers, _EXACT_RESPONSE_CACHE_HIT_HEADER: "hit"},
        body=payload.get("body"),
        first_byte_delay_ms=elapsed_ms,
        total_time_ms=elapsed_ms,
    )


def _body_with_gateway_response_cache_usage(
    body: Any, input_tokens: Optional[int]
) -> Any:
    estimated_cached_tokens = int(input_tokens or 0)
    if estimated_cached_tokens <= 0 or not isinstance(body, dict):
        return body
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return body

    updated_body = dict(body)
    updated_usage = dict(usage)
    reported_input_tokens = (
        updated_usage.get("prompt_tokens")
        or updated_usage.get("input_tokens")
        or estimated_cached_tokens
    )
    cached_tokens = min(estimated_cached_tokens, int(reported_input_tokens or 0))
    if cached_tokens <= 0:
        return body
    if "prompt_tokens" in updated_usage:
        details = updated_usage.get("prompt_tokens_details")
        updated_usage["prompt_tokens_details"] = {
            **(details if isinstance(details, dict) else {}),
            "cached_tokens": cached_tokens,
        }
    elif "input_tokens_details" in updated_usage:
        details = updated_usage.get("input_tokens_details")
        updated_usage["input_tokens_details"] = {
            **(details if isinstance(details, dict) else {}),
            "cached_tokens": cached_tokens,
        }
    elif "input_tokens" in updated_usage:
        updated_usage["cache_read_input_tokens"] = cached_tokens
    else:
        return body
    updated_body["usage"] = updated_usage
    return updated_body


def _note_and_check_repeat_miss(
    prompt_cache_key: Optional[str], cache_hit: bool, now: float
) -> bool:
    """Record a prompt_cache_key sighting and report whether this is a repeat-miss.

    Returns True only when the key was already seen within the TTL window AND this request did
    not hit cache — i.e. a prefix that should be warm but isn't. First sight (cold) returns False.
    Prunes expired keys on each call to stay bounded.
    """
    if not prompt_cache_key:
        return False
    expired = [k for k, t in _recent_prompt_cache_keys.items() if now - t > _CACHE_KEY_TTL_SECONDS]
    for k in expired:
        _recent_prompt_cache_keys.pop(k, None)
    seen_recently = prompt_cache_key in _recent_prompt_cache_keys
    _recent_prompt_cache_keys[prompt_cache_key] = now
    return seen_recently and not cache_hit


def _truncate_log_text(text: str) -> str:
    if len(text) <= MAX_LOG_TEXT_LENGTH:
        return text
    return f"{text[:MAX_LOG_TEXT_LENGTH]}...[truncated]"


def _smart_truncate(data: Any, max_list: int = 20, max_str: int = 1000) -> Any:
    """
    Recursively truncate data structures for logging.
    """
    if isinstance(data, dict):
        return {k: _smart_truncate(v, max_list, max_str) for k, v in data.items()}

    if isinstance(data, list):
        if len(data) > max_list:
            # Check if it's a list of numbers (likely embedding vector)
            if data and isinstance(data[0], (int, float)):
                return data[:5] + [f"...({len(data) - 5} items)..."]

            truncated = [_smart_truncate(x, max_list, max_str) for x in data[:max_list]]
            truncated.append(f"...({len(data) - max_list} more items)...")
            return truncated
        return [_smart_truncate(x, max_list, max_str) for x in data]

    if isinstance(data, str) and len(data) > max_str:
        return data[:max_str] + "...[truncated]"

    return data


def _extract_user_id(headers: dict[str, str]) -> str | None:
    for key, value in headers.items():
        if key.lower() == "x-user-id":
            user_id = str(value).strip()
            if not user_id:
                return None
            return user_id[:MAX_USER_ID_LENGTH]
    return None


class StreamInterrupted(Exception):
    """A provider stream failed after its first successful chunk.

    Raised when a mid-stream chunk carries an unsuccessful ProviderResponse (for
    example the provider stall-guard's 504). It is a plain ``Exception`` so the
    stream handler's ``except Exception`` surfaces it as an in-band error frame,
    while ``asyncio.CancelledError`` (client disconnect) still propagates.
    """


def _stream_error_frames(request_protocol: Optional[str], message: str) -> list[bytes]:
    """In-band error frames for a stream that fails after headers are sent.

    Once the 200 + headers have flushed the HTTP status cannot change, so the
    client is told explicitly (in its own protocol) instead of receiving a
    silently truncated stream. Single owner for the OpenAI vs Anthropic SSE
    shapes used wherever a mid-stream failure is surfaced to the client.
    """
    if (request_protocol or "openai").lower() == "anthropic":
        return [
            f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': message}}, ensure_ascii=False)}\n\n".encode(
                "utf-8"
            ),
            f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'}, ensure_ascii=False)}\n\n".encode(
                "utf-8"
            ),
        ]
    return [
        f"data: {json.dumps({'error': {'message': message}}, ensure_ascii=False)}\n\n".encode(
            "utf-8"
        ),
        b"data: [DONE]\n\n",
    ]


class ProxyService:
    """
    Proxy Core Service

    Handles the complete flow of proxy requests:
    1. Parse request, extract requested_model
    2. Calculate input Token
    3. Rule engine match, get candidate providers
    4. Selection strategy selects provider
    5. Replace model field, forward request
    6. Handle retry and failover
    7. Calculate output Token
    8. Record log
    9. Return response
    """

    def __init__(
        self,
        model_repo: ModelRepository,
        provider_repo: ProviderRepository,
        log_repo: LogRepository,
        round_robin_strategy: Optional[SelectionStrategy] = None,
        cost_first_strategy: Optional[SelectionStrategy] = None,
        priority_strategy: Optional[SelectionStrategy] = None,
        prefix_affinity_strategy: Optional[SelectionStrategy] = None,
        protocol_hooks: Optional[ProtocolConversionHooks] = None,
        kv_repo: Optional[KVStoreRepository] = None,
    ):
        """
        Initialize Service

        Args:
            model_repo: Model Repository
            provider_repo: Provider Repository
            log_repo: Log Repository
            round_robin_strategy: Optional Round Robin Strategy instance
            cost_first_strategy: Optional Cost First Strategy instance
            priority_strategy: Optional Priority Strategy instance
        """
        self.model_repo = model_repo
        self.provider_repo = provider_repo
        self.log_repo = log_repo
        self.rule_engine = RuleEngine()
        # Strategy selection instances (reused for performance)
        self._round_robin_strategy = round_robin_strategy or RoundRobinStrategy()
        self._cost_first_strategy = cost_first_strategy or CostFirstStrategy()
        self._priority_strategy = priority_strategy or PriorityStrategy()
        self._prefix_affinity_strategy = (
            prefix_affinity_strategy or PrefixAffinityStrategy()
        )
        self._protocol_hooks = protocol_hooks or ProtocolConversionHooks()
        self._kv_repo = kv_repo

    async def _write_log(self, log_data: RequestLogCreate) -> None:
        await self.log_repo.create(log_data)

    def _get_strategy(self, strategy_name: str) -> SelectionStrategy:
        """
        Get strategy instance based on strategy name

        Args:
            strategy_name: Strategy name ("round_robin", "cost_first", or "priority")

        Returns:
            SelectionStrategy: Strategy instance
        """
        if strategy_name == "cost_first":
            return self._cost_first_strategy
        if strategy_name == "priority":
            return self._priority_strategy
        if strategy_name == "prefix_affinity":
            return self._prefix_affinity_strategy
        else:
            # Default to round_robin for unknown strategies
            return self._round_robin_strategy

    @staticmethod
    def _provider_mapping_key(
        provider_id: int,
        target_model_name: str,
        provider_mapping_id: Optional[int] = None,
    ) -> CandidateKey:
        if provider_mapping_id is not None:
            return ("mapping", provider_mapping_id)
        return ("provider_target", provider_id, target_model_name)

    @classmethod
    def _candidate_key(cls, candidate: CandidateProvider) -> CandidateKey:
        return cls._provider_mapping_key(
            provider_id=candidate.provider_id,
            target_model_name=candidate.target_model,
            provider_mapping_id=candidate.provider_mapping_id,
        )

    @staticmethod
    def _serialize_response_body(body: Any) -> str | None:
        if body is None:
            return None

        data = body
        if isinstance(body, (bytes, bytearray)):
            if b"\x00" in body:
                return f"[binary data: {len(body)} bytes]"
            try:
                decoded = body.decode("utf-8")
                # Try to parse as JSON first
                try:
                    data = json.loads(decoded)
                except json.JSONDecodeError:
                    return _truncate_log_text(decoded)
            except UnicodeDecodeError:
                return f"[binary data: {len(body)} bytes]"

        # If it's already a dict/list or successfully parsed
        if isinstance(data, (dict, list)):
            try:
                truncated_data = _smart_truncate(data)
                return json.dumps(truncated_data, ensure_ascii=False)
            except Exception:
                # Fallback
                return _truncate_log_text(str(data))

        return _truncate_log_text(str(data))

    @staticmethod
    def _sanitize_request_body_for_log(body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body, dict) or "_files" not in body:
            return body

        safe_files = []
        for item in body.get("_files", []):
            if not isinstance(item, dict):
                continue
            data = item.get("data")
            safe_files.append(
                {
                    "field": item.get("field"),
                    "filename": item.get("filename"),
                    "content_type": item.get("content_type"),
                    "size": len(data) if isinstance(data, (bytes, bytearray)) else None,
                }
            )
        sanitized = dict(body)
        sanitized["_files"] = safe_files
        return sanitized

    @staticmethod
    def _build_conversion_options(
        provider_options: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if not provider_options:
            return None
        if not isinstance(provider_options, dict):
            return None
        default_params = provider_options.get("default_parameters")
        if not isinstance(default_params, dict) or not default_params:
            return None
        return {"default_parameters": default_params}

    @staticmethod
    def _use_no_suffix(provider_options: Optional[dict[str, Any]]) -> bool:
        if not isinstance(provider_options, dict):
            return False
        return bool(provider_options.get("no_suffix"))

    async def _resolve_candidates(
        self,
        requested_model: str,
        request_protocol: str,
        headers: dict[str, str],
        body: dict[str, Any],
    ) -> tuple[
        ModelMapping,
        list[CandidateProvider],
        int,
        str,
        dict[CandidateKey, ModelMappingProviderResponse],
    ]:
        """
        Resolve model and provider candidate list

        Returns:
            tuple: (model_mapping, candidates, input_tokens, protocol, provider_mapping_by_id)
        """
        request_protocol = (request_protocol or "openai").lower()
        model_mapping = await self.model_repo.get_mapping(requested_model)
        if not model_mapping:
            raise NotFoundError(
                message=f"Model '{requested_model}' is not configured",
                code="model_not_found",
            )

        if not model_mapping.is_active:
            raise ServiceError(
                message=f"Model '{requested_model}' is disabled",
                code="model_disabled",
            )

        provider_mappings = await self.model_repo.get_provider_mappings(
            requested_model=requested_model,
            is_active=True,
        )

        if not provider_mappings:
            raise ServiceError(
                message=f"No providers configured for model '{requested_model}'",
                code="no_available_provider",
            )

        provider_ids = [pm.provider_id for pm in provider_mappings]
        providers: dict[int, Provider] = {}
        for pid in provider_ids:
            provider = await self.provider_repo.get_by_id(pid)
            if provider:
                providers[pid] = provider

        eligible_provider_mappings = [
            pm
            for pm in provider_mappings
            if (provider := providers.get(pm.provider_id)) is not None and provider.is_active
        ]
        eligible_providers = {pid: p for pid, p in providers.items() if p.is_active}

        if not eligible_provider_mappings:
            raise ServiceError(
                message="No available providers", code="no_available_provider"
            )

        provider_mapping_by_id = {
            self._provider_mapping_key(
                provider_id=pm.provider_id,
                target_model_name=pm.target_model_name,
                provider_mapping_id=pm.id,
            ): pm
            for pm in eligible_provider_mappings
        }

        token_counter = get_token_counter(request_protocol)
        input_tokens = token_counter.count_request(body, requested_model)

        context = RuleContext(
            current_model=requested_model,
            headers=headers,
            request_body=body,
            token_usage=TokenUsage(input_tokens=input_tokens),
        )

        candidates = await self.rule_engine.evaluate(
            context=context,
            model_mapping=model_mapping,
            provider_mappings=eligible_provider_mappings,
            providers=eligible_providers,
        )

        if not candidates:
            raise ServiceError(
                message="No providers matched the rules",
                code="no_available_provider",
            )

        return (
            model_mapping,
            candidates,
            input_tokens,
            request_protocol,
            provider_mapping_by_id,
        )

    async def process_request(
        self,
        api_key_id: Optional[int],
        api_key_name: Optional[str],
        request_protocol: str,
        path: str,
        request_url: Optional[str],
        method: str,
        headers: dict[str, str],
        body: dict[str, Any],
        *,
        force_parse_response: bool = False,
    ) -> tuple[ProviderResponse, dict[str, Any]]:
        """
        Process Proxy Request

        Args:
            api_key_id: API Key ID
            api_key_name: API Key Name
            path: Request path
            method: HTTP method
            headers: Request headers
            body: Request body

        Returns:
            tuple[ProviderResponse, dict]: (Provider response, Log info)

        Raises:
            NotFoundError: Model not configured
            ServiceError: No available provider
        """
        trace_id = generate_trace_id()
        request_time = utc_now()
        sanitized_body = self._sanitize_request_body_for_log(body)
        user_id = _extract_user_id(headers)

        # 1. Extract requested_model
        requested_model = body.get("model")
        if not requested_model:
            raise ServiceError(
                message="Model is required in request body",
                code="missing_model",
            )

        # 2. Get model mapping
        (
            model_mapping,
            candidates,
            input_tokens,
            protocol,
            provider_mapping_by_id,
        ) = await self._resolve_candidates(
            requested_model=requested_model,
            request_protocol=request_protocol,
            headers=headers,
            body=body,
        )
        prompt_cache_key = _prompt_cache_key_for_request(
            body,
            requested_model,
            [candidate.target_model for candidate in candidates],
            _prompt_cache_key_hint(headers),
        )
        cache_signal_body = _body_with_prompt_cache_key(body, prompt_cache_key)
        token_counter = get_token_counter(protocol)

        # Extract image count for per-image billing
        image_count: Optional[int] = None
        if path in OPENAI_IMAGE_PATHS:
            try:
                image_count = int(body.get("n") or 1)
            except (ValueError, TypeError):
                image_count = 1

        # DEBUG: Log matched providers
        candidates_info = [
            {
                "id": c.provider_id,
                "name": c.provider_name,
                "priority": c.priority,
                "weight": c.weight,
            }
            for c in candidates
        ]
        logger.debug(
            f"Matched Providers: {json.dumps(candidates_info, ensure_ascii=False)}"
        )

        # Select strategy based on model configuration
        strategy = self._get_strategy(model_mapping.strategy)
        retry_handler = RetryHandler(strategy)

        failed_attempt_logged = False
        # Track protocol conversion data for logging
        conversion_data: dict[str, Any] = {
            "request_protocol": request_protocol,
            "supplier_protocol": None,
            "converted_request_body": None,
            "upstream_response_body": None,
        }
        gateway_response_cache_hit = False

        async def log_failed_attempt(attempt: AttemptRecord) -> None:
            nonlocal failed_attempt_logged
            provider_mapping = provider_mapping_by_id.get(
                self._candidate_key(attempt.provider)
            )
            billing = resolve_billing(
                input_tokens=input_tokens,
                model_input_price=model_mapping.input_price,
                model_output_price=model_mapping.output_price,
                model_billing_mode=model_mapping.billing_mode,
                model_per_request_price=model_mapping.per_request_price,
                model_per_image_price=model_mapping.per_image_price,
                model_tiered_pricing=model_mapping.tiered_pricing,
                model_cache_billing_enabled=getattr(model_mapping, "cache_billing_enabled", None),
                model_cached_input_price=getattr(model_mapping, "cached_input_price", None),
                model_cached_output_price=getattr(model_mapping, "cached_output_price", None),
                provider_billing_mode=provider_mapping.billing_mode
                if provider_mapping
                else None,
                provider_per_request_price=provider_mapping.per_request_price
                if provider_mapping
                else None,
                provider_per_image_price=provider_mapping.per_image_price
                if provider_mapping
                else None,
                provider_tiered_pricing=provider_mapping.tiered_pricing
                if provider_mapping
                else None,
                provider_input_price=provider_mapping.input_price
                if provider_mapping
                else None,
                provider_output_price=provider_mapping.output_price
                if provider_mapping
                else None,
                provider_cache_billing_enabled=getattr(provider_mapping, "cache_billing_enabled", None)
                if provider_mapping
                else None,
                provider_cached_input_price=getattr(provider_mapping, "cached_input_price", None)
                if provider_mapping
                else None,
                provider_cached_output_price=getattr(provider_mapping, "cached_output_price", None)
                if provider_mapping
                else None,
            )
            attempt_log = RequestLogCreate(
                request_time=attempt.request_time,
                api_key_id=api_key_id,
                api_key_name=api_key_name,
                user_id=user_id,
                requested_model=requested_model,
                target_model=attempt.provider.target_model,
                provider_id=attempt.provider.provider_id,
                provider_name=attempt.provider.provider_name,
                retry_count=attempt.attempt_index + 1,
                matched_provider_count=len(candidates),
                first_byte_delay_ms=attempt.response.first_byte_delay_ms,
                total_time_ms=attempt.response.total_time_ms,
                input_tokens=input_tokens,
                output_tokens=None,
                total_cost=None,
                input_cost=None,
                output_cost=None,
                price_source=billing.price_source,
                request_headers=sanitize_headers(headers),
                response_headers=sanitize_headers(attempt.response.headers),
                request_body=sanitized_body,
                response_status=attempt.response.status_code,
                response_body=self._serialize_response_body(attempt.response.body),
                error_info=attempt.response.error,
                trace_id=trace_id,
                is_stream=False,
                request_path=path,
                request_url=request_url,
                request_method=method,
                upstream_url=conversion_data.get("upstream_url"),
                # Protocol conversion fields
                request_protocol=request_protocol,
                supplier_protocol=resolve_implementation_protocol(
                    attempt.provider.protocol
                ),
                converted_request_body=_smart_truncate(
                    conversion_data.get("converted_request_body")
                ),
                upstream_response_body=self._serialize_response_body(
                    attempt.response.body
                ),
            )
            try:
                await self._write_log(attempt_log)
                failed_attempt_logged = True
            except Exception:
                logger.exception(
                    "Failed to write attempt log: trace_id=%s provider_id=%s attempt_index=%s",
                    trace_id,
                    attempt.provider.provider_id,
                    attempt.attempt_index,
                )

        # 8. Execute request (with retry)
        async def forward_fn(candidate: CandidateProvider) -> ProviderResponse:
            nonlocal gateway_response_cache_hit
            supplier_protocol: Optional[str] = None
            try:
                is_image_path = path in OPENAI_IMAGE_PATHS
                supplier_protocol = resolve_implementation_protocol(candidate.protocol)
                client = get_provider_client(supplier_protocol)
                conversion_options = self._build_conversion_options(
                    candidate.provider_options
                )
                hooked_body = await self._protocol_hooks.before_request_conversion(
                    cache_signal_body,
                    request_protocol,
                    supplier_protocol,
                )
                if hooked_body is None:
                    hooked_body = cache_signal_body
                if is_image_path:
                    hooked_image_body = (
                        await self._protocol_hooks.before_image_request_conversion(
                            hooked_body,
                            request_protocol,
                            supplier_protocol,
                            path,
                        )
                    )
                    if hooked_image_body is not None:
                        hooked_body = hooked_image_body
                # Non-OpenAI client -> Gemini provider pivots through OpenAI inside
                # the converter, so the OpenAI-gated thought_signature hooks miss it.
                # Pre-fetch the cached signatures by tool_use id and hand them to the
                # converter to restore onto the OpenAI-intermediate before Gemini.
                if (
                    normalize_protocol(supplier_protocol) == "gemini"
                    and normalize_protocol(request_protocol) != "openai"
                ):
                    inject_extra = (
                        await self._protocol_hooks.prefetch_tool_call_extra_content(
                            hooked_body
                        )
                    )
                    if inject_extra:
                        conversion_options = dict(conversion_options or {})
                        conversion_options["tool_call_extra_content_inject"] = inject_extra
                supplier_path, supplier_body = convert_request_for_supplier(
                    request_protocol=request_protocol,
                    supplier_protocol=candidate.protocol,
                    path=path,
                    body=hooked_body,
                    target_model=candidate.target_model,
                    options=conversion_options,
                )
                if self._use_no_suffix(candidate.provider_options):
                    supplier_path = ""
                hooked_supplier_body = await self._protocol_hooks.after_request_conversion(
                    supplier_body,
                    request_protocol,
                    supplier_protocol,
                )
                if hooked_supplier_body is not None:
                    supplier_body = hooked_supplier_body
                supplier_body = _supplier_body_with_prompt_cache_key(
                    supplier_body,
                    prompt_cache_key,
                    requested_model=requested_model,
                    target_model=candidate.target_model,
                    base_url=candidate.base_url,
                    supplier_protocol=supplier_protocol,
                )
                if is_image_path:
                    hooked_image_supplier_body = (
                        await self._protocol_hooks.after_image_request_conversion(
                            supplier_body,
                            request_protocol,
                            supplier_protocol,
                            path,
                        )
                    )
                    if hooked_image_supplier_body is not None:
                        supplier_body = hooked_image_supplier_body
                # Track conversion data for logging
                conversion_data["supplier_protocol"] = supplier_protocol
                conversion_data["converted_request_body"] = supplier_body
                conversion_data["upstream_url"] = build_upstream_url(
                    candidate.base_url, supplier_path
                )
                exact_cache_key = None
                if _exact_response_cache_allowed(
                    request_protocol=request_protocol,
                    supplier_protocol=supplier_protocol,
                    supplier_body=supplier_body,
                    method=method,
                    prompt_cache_key=prompt_cache_key,
                    requested_model=requested_model,
                    target_model=candidate.target_model,
                    base_url=candidate.base_url,
                    input_tokens=input_tokens,
                ):
                    exact_cache_key = _exact_response_cache_key(
                        api_key_id=api_key_id,
                        method=method,
                        supplier_path=supplier_path,
                        requested_model=requested_model,
                        provider_id=candidate.provider_id,
                        provider_mapping_id=candidate.provider_mapping_id,
                        target_model=candidate.target_model,
                        supplier_body=supplier_body,
                    )
                    if self._kv_repo is not None:
                        cache_start = time.monotonic()
                        try:
                            cached = await self._kv_repo.get(exact_cache_key)
                            if cached is not None:
                                elapsed_ms = int((time.monotonic() - cache_start) * 1000)
                                cached_response = _provider_response_from_exact_cache(
                                    cached.value, elapsed_ms
                                )
                                if cached_response is not None:
                                    gateway_response_cache_hit = True
                                    logger.info(
                                        "gateway exact response cache hit model=%s provider_id=%s key=%s",
                                        requested_model,
                                        candidate.provider_id,
                                        exact_cache_key,
                                    )
                                    return cached_response
                        except Exception:
                            logger.exception(
                                "gateway exact response cache read failed model=%s provider_id=%s",
                                requested_model,
                                candidate.provider_id,
                            )
                same_protocol = normalize_protocol(
                    request_protocol
                ) == normalize_protocol(supplier_protocol)
                proxy_config = build_proxy_config(
                    candidate.proxy_enabled,
                    candidate.proxy_url,
                )
                response = await client.forward(
                    base_url=candidate.base_url,
                    api_key=candidate.api_key,
                    path=supplier_path,
                    method=method,
                    headers=headers,
                    body=supplier_body,
                    target_model=candidate.target_model,
                    response_mode="parsed"
                    if force_parse_response
                    else ("raw" if same_protocol else "parsed"),
                    extra_headers=candidate.extra_headers,
                    proxy_config=proxy_config,
                )
                if exact_cache_key and self._kv_repo is not None:
                    payload = _exact_response_cache_payload(response)
                    if payload is not None:
                        try:
                            await self._kv_repo.set(
                                exact_cache_key,
                                payload,
                                ttl_seconds=_EXACT_RESPONSE_CACHE_TTL_SECONDS,
                            )
                            logger.info(
                                "gateway exact response cache stored model=%s provider_id=%s key=%s ttl_seconds=%s",
                                requested_model,
                                candidate.provider_id,
                                exact_cache_key,
                                _EXACT_RESPONSE_CACHE_TTL_SECONDS,
                            )
                        except Exception:
                            logger.exception(
                                "gateway exact response cache write failed model=%s provider_id=%s",
                                requested_model,
                                candidate.provider_id,
                            )
                return response
            except Exception as e:
                error_msg = str(e)
                logger.error(
                    "Error during request forwarding: provider_id=%s, provider_name=%s, "
                    "request_protocol=%s, supplier_protocol=%s, error=%s",
                    candidate.provider_id,
                    candidate.provider_name,
                    request_protocol,
                    supplier_protocol or candidate.protocol,
                    error_msg,
                )
                return ProviderResponse(status_code=400, error=error_msg)

        result = await retry_handler.execute_with_retry(
            candidates=candidates,
            requested_model=requested_model,
            forward_fn=forward_fn,
            input_tokens=input_tokens,
            image_count=image_count,
            # Pin a stable prefix to one backend so a repeated prefix reuses that backend's warm
            # cache; only the prefix_affinity strategy reads it.
            affinity_key=prompt_cache_key,
            on_failure_attempt=log_failed_attempt,
        )

        if result.response.body is not None and result.final_provider is not None:
            try:
                is_image_path = path in OPENAI_IMAGE_PATHS
                supplier_protocol = resolve_implementation_protocol(
                    result.final_provider.protocol
                )
                same_protocol = normalize_protocol(
                    request_protocol
                ) == normalize_protocol(supplier_protocol)
                hooked_upstream_body = await self._protocol_hooks.before_response_conversion(
                    result.response.body,
                    request_protocol,
                    supplier_protocol,
                )
                if hooked_upstream_body is None:
                    hooked_upstream_body = result.response.body
                if is_image_path:
                    hooked_image_upstream_body = (
                        await self._protocol_hooks.before_image_response_conversion(
                            hooked_upstream_body,
                            request_protocol,
                            supplier_protocol,
                            path,
                        )
                    )
                    if hooked_image_upstream_body is not None:
                        hooked_upstream_body = hooked_image_upstream_body
                # Capture upstream response before protocol conversion
                conversion_data["upstream_response_body"] = hooked_upstream_body
                response_body = hooked_upstream_body
                if not same_protocol:
                    # Harvest Gemini thought_signatures from the converter's
                    # OpenAI-intermediate for the non-OpenAI client path the
                    # OpenAI-gated hooks miss, then persist them for later turns.
                    response_extra_sink: dict[str, Any] = {}
                    response_body = convert_response_for_user(
                        request_protocol=request_protocol,
                        supplier_protocol=supplier_protocol,
                        body=hooked_upstream_body,
                        target_model=result.final_provider.target_model,
                        options=(
                            {"tool_call_extra_content_sink": response_extra_sink}
                            if (
                                normalize_protocol(supplier_protocol) == "gemini"
                                and normalize_protocol(request_protocol) != "openai"
                            )
                            else None
                        ),
                    )
                    if response_extra_sink:
                        await self._protocol_hooks.cache_tool_call_extra_content_map(
                            response_extra_sink
                        )
                hooked_response_body = await self._protocol_hooks.after_response_conversion(
                    response_body,
                    request_protocol,
                    supplier_protocol,
                )
                if hooked_response_body is not None:
                    response_body = hooked_response_body
                if is_image_path:
                    hooked_image_response_body = (
                        await self._protocol_hooks.after_image_response_conversion(
                            response_body,
                            request_protocol,
                            supplier_protocol,
                            path,
                        )
                    )
                    if hooked_image_response_body is not None:
                        response_body = hooked_image_response_body
                if normalize_protocol(request_protocol) == "openai" and not is_image_path:
                    ensure_openai_usage_details(
                        response_body, conversion_data["upstream_response_body"]
                    )
                if gateway_response_cache_hit:
                    response_body = _body_with_gateway_response_cache_usage(
                        response_body, input_tokens
                    )
                result.response.body = response_body
            except Exception as e:
                error_msg = str(e)
                logger.error(
                    "Error during response conversion: provider_id=%s, provider_name=%s, "
                    "request_protocol=%s, supplier_protocol=%s, error=%s",
                    result.final_provider.provider_id,
                    result.final_provider.provider_name,
                    request_protocol,
                    supplier_protocol,
                    error_msg,
                )
                result.response = ProviderResponse(
                    status_code=502,
                    headers=result.response.headers,
                    error=error_msg,
                    first_byte_delay_ms=result.response.first_byte_delay_ms,
                    total_time_ms=result.response.total_time_ms,
                )

        # 9. Calculate Output Token and usage details
        output_tokens = 0
        usage_details: Optional[dict[str, Any]] = None
        if result.success and result.response.body:
            upstream_body = conversion_data.get("upstream_response_body")
            details = None
            try:
                details = extract_usage_details(upstream_body) or extract_usage_details(
                    result.response.body
                )
            except Exception:
                details = None

            if details:
                usage_details = dict(details.__dict__)
                if details.input_tokens:
                    input_tokens = details.input_tokens
                if details.output_tokens:
                    output_tokens = details.output_tokens
                else:
                    output_tokens = token_counter.count_output_body(
                        result.response.body, requested_model
                    )
                    usage_details["output_tokens"] = output_tokens
                    usage_details["source"] = "mixed"
                if not usage_details.get("input_tokens"):
                    usage_details["input_tokens"] = input_tokens
                    usage_details["source"] = "mixed"
                if not usage_details.get("total_tokens") and usage_details.get(
                    "input_tokens"
                ):
                    usage_details["total_tokens"] = usage_details["input_tokens"] + (
                        usage_details.get("output_tokens") or 0
                    )
            else:
                output_tokens = token_counter.count_output_body(
                    result.response.body, requested_model
                )
                usage_details = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": (input_tokens or 0) + (output_tokens or 0),
                    "source": "estimated",
                }
            if gateway_response_cache_hit:
                if usage_details is None:
                    usage_details = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": (input_tokens or 0) + (output_tokens or 0),
                    }
                usage_details["source"] = "gateway_response_cache"
                usage_details["gateway_response_cache_hit"] = True
                usage_details["cache_read_input_tokens"] = (
                    usage_details.get("input_tokens") or input_tokens or 0
                )

        # 10. Record log
        provider_mapping = (
            provider_mapping_by_id.get(self._candidate_key(result.final_provider))
            if result.final_provider is not None
            else None
        )
        billing = resolve_billing(
            input_tokens=input_tokens,
            model_input_price=model_mapping.input_price,
            model_output_price=model_mapping.output_price,
            model_billing_mode=model_mapping.billing_mode,
            model_per_request_price=model_mapping.per_request_price,
            model_per_image_price=model_mapping.per_image_price,
            model_tiered_pricing=model_mapping.tiered_pricing,
            model_cache_billing_enabled=getattr(model_mapping, "cache_billing_enabled", None),
            model_cached_input_price=getattr(model_mapping, "cached_input_price", None),
            model_cached_output_price=getattr(model_mapping, "cached_output_price", None),
            provider_billing_mode=provider_mapping.billing_mode
            if provider_mapping
            else None,
            provider_per_request_price=provider_mapping.per_request_price
            if provider_mapping
            else None,
            provider_per_image_price=provider_mapping.per_image_price
            if provider_mapping
            else None,
            provider_tiered_pricing=provider_mapping.tiered_pricing
            if provider_mapping
            else None,
            provider_input_price=provider_mapping.input_price
            if provider_mapping
            else None,
            provider_output_price=provider_mapping.output_price
            if provider_mapping
            else None,
            provider_cache_billing_enabled=getattr(provider_mapping, "cache_billing_enabled", None)
            if provider_mapping
            else None,
            provider_cached_input_price=getattr(provider_mapping, "cached_input_price", None)
            if provider_mapping
            else None,
            provider_cached_output_price=getattr(provider_mapping, "cached_output_price", None)
            if provider_mapping
            else None,
        )
        # Extract cached tokens from usage details
        cached_input_tokens = None
        if usage_details:
            cached_input_tokens = (
                usage_details.get("cached_tokens")
                or usage_details.get("cache_read_input_tokens")
            )

        # Cache-effectiveness observability: a structured outcome line on every request (queryable
        # for hit-rate metrics) plus a loud WARN on a repeat-miss — a prompt_cache_key seen again
        # within the TTL that still returns zero cached tokens, i.e. a prefix that should be warm.
        # Never breaks the response path.
        try:
            cache_hit = bool(cached_input_tokens)
            logger.info(
                "cache outcome model=%s cached_tokens=%s input_tokens=%s hit=%s had_signal=%s",
                requested_model,
                cached_input_tokens or 0,
                input_tokens or 0,
                cache_hit,
                _request_has_cache_signal(cache_signal_body),
            )
            if _note_and_check_repeat_miss(
                prompt_cache_key, cache_hit, time.monotonic()
            ):
                logger.warning(
                    "prompt cache repeat-miss model=%s prompt_cache_key=%s input_tokens=%s — "
                    "prefix may be unstable, below the cacheable token floor, or routed to a cold "
                    "backend (check prefix stability and routing affinity)",
                    requested_model,
                    prompt_cache_key,
                    input_tokens or 0,
                )
        except Exception:
            pass
        if gateway_response_cache_hit:
            cost = CostBreakdown(total_cost=0.0, input_cost=0.0, output_cost=0.0)
        else:
            cost = calculate_cost_from_billing(
                billing=billing,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                image_count=image_count,
                cached_input_tokens=cached_input_tokens,
            )
        log_data = RequestLogCreate(
            request_time=request_time,
            api_key_id=api_key_id,
            api_key_name=api_key_name,
            user_id=user_id,
            requested_model=requested_model,
            target_model=result.final_provider.target_model
            if result.final_provider
            else None,
            provider_id=result.final_provider.provider_id
            if result.final_provider
            else None,
            provider_name=result.final_provider.provider_name
            if result.final_provider
            else None,
            retry_count=result.retry_count,
            matched_provider_count=len(candidates),
            first_byte_delay_ms=result.response.first_byte_delay_ms,
            total_time_ms=result.response.total_time_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_cost=cost.total_cost,
            input_cost=cost.input_cost,
            output_cost=cost.output_cost,
            cached_input_cost=cost.cached_input_cost,
            cached_output_cost=cost.cached_output_cost,
            price_source=billing.price_source,
            request_headers=sanitize_headers(headers),
            response_headers=sanitize_headers(result.response.headers),
            request_body=sanitized_body,
            response_status=result.response.status_code,
            response_body=self._serialize_response_body(result.response.body),
            usage_details=usage_details,
            error_info=result.response.error,
            trace_id=trace_id,
            is_stream=False,
            request_path=path,
            request_url=request_url,
            request_method=method,
            upstream_url=conversion_data.get("upstream_url"),
            # Protocol conversion fields
            request_protocol=conversion_data.get("request_protocol"),
            supplier_protocol=conversion_data.get("supplier_protocol"),
            converted_request_body=_smart_truncate(
                conversion_data.get("converted_request_body")
            ),
            upstream_response_body=self._serialize_response_body(
                conversion_data.get("upstream_response_body")
            ),
        )

        # DEBUG: Log request details
        try:
            logger.debug(f"Request Log: {log_data.model_dump_json()}")
        except AttributeError:
            # Fallback for Pydantic v1
            logger.debug(f"Request Log: {log_data.json()}")

        if result.success or not failed_attempt_logged:
            await self._write_log(log_data)

        return result.response, {
            "trace_id": trace_id,
            "retry_count": result.retry_count,
            "target_model": result.final_provider.target_model
            if result.final_provider
            else None,
            "provider_name": result.final_provider.provider_name
            if result.final_provider
            else None,
        }

    async def process_request_stream(
        self,
        api_key_id: Optional[int],
        api_key_name: Optional[str],
        request_protocol: str,
        path: str,
        request_url: Optional[str],
        method: str,
        headers: dict[str, str],
        body: dict[str, Any],
    ) -> tuple[ProviderResponse, AsyncGenerator[bytes, None], dict[str, Any]]:
        """
        Process Streaming Proxy Request

        Args:
            api_key_id: API Key ID
            api_key_name: API Key Name
            path: Request path
            method: HTTP method
            headers: Request headers
            body: Request body

        Returns:
            tuple: (Initial response, Stream generator, Log info)
        """
        trace_id = generate_trace_id()
        request_time = utc_now()
        start_monotonic = time.monotonic()
        sanitized_body = self._sanitize_request_body_for_log(body)
        user_id = _extract_user_id(headers)

        # 1-7. Same model resolution and rule matching logic
        requested_model = body.get("model")
        if not requested_model:
            raise ServiceError(message="Model is required", code="missing_model")

        (
            model_mapping,
            candidates,
            input_tokens,
            protocol,
            provider_mapping_by_id,
        ) = await self._resolve_candidates(
            requested_model=requested_model,
            request_protocol=request_protocol,
            headers=headers,
            body=body,
        )
        prompt_cache_key = _prompt_cache_key_for_request(
            body,
            requested_model,
            [candidate.target_model for candidate in candidates],
            _prompt_cache_key_hint(headers),
        )
        cache_signal_body = _body_with_prompt_cache_key(body, prompt_cache_key)

        # Extract image count for per-image billing
        image_count: Optional[int] = None
        if path in OPENAI_IMAGE_PATHS:
            try:
                image_count = int(body.get("n") or 1)
            except (ValueError, TypeError):
                image_count = 1

        # DEBUG: Log matched providers
        candidates_info = [
            {
                "id": c.provider_id,
                "name": c.provider_name,
                "priority": c.priority,
                "weight": c.weight,
            }
            for c in candidates
        ]
        logger.debug(
            f"Matched Providers: {json.dumps(candidates_info, ensure_ascii=False)}"
        )

        # Select strategy based on model configuration
        strategy = self._get_strategy(model_mapping.strategy)
        retry_handler = RetryHandler(strategy)

        # Track protocol conversion data for logging
        stream_conversion_data: dict[str, Any] = {
            "request_protocol": request_protocol,
            "supplier_protocol": None,
            "converted_request_body": None,
            "upstream_chunks": [],
        }

        # 8. Execute streaming request
        async def forward_stream_fn(candidate: CandidateProvider):
            async def error_gen(msg: str):
                yield b"", ProviderResponse(status_code=400, error=msg)

            supplier_protocol: Optional[str] = None
            try:
                is_image_path = path in OPENAI_IMAGE_PATHS
                supplier_protocol = resolve_implementation_protocol(candidate.protocol)
                client = get_provider_client(supplier_protocol)
                conversion_options = self._build_conversion_options(
                    candidate.provider_options
                )
                hooked_body = await self._protocol_hooks.before_request_conversion(
                    cache_signal_body,
                    request_protocol,
                    supplier_protocol,
                )
                if hooked_body is None:
                    hooked_body = cache_signal_body
                if is_image_path:
                    hooked_image_body = (
                        await self._protocol_hooks.before_image_request_conversion(
                            hooked_body,
                            request_protocol,
                            supplier_protocol,
                            path,
                        )
                    )
                    if hooked_image_body is not None:
                        hooked_body = hooked_image_body
                # Non-OpenAI client -> Gemini provider pivots through OpenAI inside
                # the converter, so the OpenAI-gated thought_signature hooks miss it.
                # Pre-fetch the cached signatures by tool_use id and hand them to the
                # converter to restore onto the OpenAI-intermediate before Gemini.
                if (
                    normalize_protocol(supplier_protocol) == "gemini"
                    and normalize_protocol(request_protocol) != "openai"
                ):
                    inject_extra = (
                        await self._protocol_hooks.prefetch_tool_call_extra_content(
                            hooked_body
                        )
                    )
                    if inject_extra:
                        conversion_options = dict(conversion_options or {})
                        conversion_options["tool_call_extra_content_inject"] = inject_extra
                supplier_path, supplier_body = convert_request_for_supplier(
                    request_protocol=request_protocol,
                    supplier_protocol=candidate.protocol,
                    path=path,
                    body=hooked_body,
                    target_model=candidate.target_model,
                    options=conversion_options,
                )
                hooked_supplier_body = await self._protocol_hooks.after_request_conversion(
                    supplier_body,
                    request_protocol,
                    supplier_protocol,
                )
                if hooked_supplier_body is not None:
                    supplier_body = hooked_supplier_body
                supplier_body = _supplier_body_with_prompt_cache_key(
                    supplier_body,
                    prompt_cache_key,
                    requested_model=requested_model,
                    target_model=candidate.target_model,
                    base_url=candidate.base_url,
                    supplier_protocol=supplier_protocol,
                )
                if is_image_path:
                    hooked_image_supplier_body = (
                        await self._protocol_hooks.after_image_request_conversion(
                            supplier_body,
                            request_protocol,
                            supplier_protocol,
                            path,
                        )
                    )
                    if hooked_image_supplier_body is not None:
                        supplier_body = hooked_image_supplier_body
                # Track conversion data for logging
                stream_conversion_data["supplier_protocol"] = supplier_protocol
                stream_conversion_data["converted_request_body"] = supplier_body
                stream_conversion_data["upstream_url"] = build_upstream_url(
                    candidate.base_url, supplier_path
                )
            except Exception as e:
                error_msg = str(e)
                logger.error(
                    "Error during stream request conversion: provider_id=%s, provider_name=%s, "
                    "request_protocol=%s, supplier_protocol=%s, error=%s",
                    candidate.provider_id,
                    candidate.provider_name,
                    request_protocol,
                    supplier_protocol or candidate.protocol,
                    error_msg,
                )
                return error_gen(error_msg)

            proxy_config = build_proxy_config(
                candidate.proxy_enabled,
                candidate.proxy_url,
            )
            upstream_gen = client.forward_stream(
                base_url=candidate.base_url,
                api_key=candidate.api_key,
                path=supplier_path,
                method=method,
                headers=headers,
                body=supplier_body,
                target_model=candidate.target_model,
                requested_model=requested_model,
                extra_headers=candidate.extra_headers,
                proxy_config=proxy_config,
            )

            async def wrapped() -> AsyncGenerator[tuple[bytes, ProviderResponse], None]:
                try:
                    first_chunk, first_resp = await anext(upstream_gen)
                except StopAsyncIteration:
                    # Upstream produced zero chunks (e.g. 200 with an empty body).
                    # Surface an explicit failure so retry/failover and the request
                    # log get a clear reason instead of an empty stream.
                    logger.warning(
                        "Empty upstream stream: provider_id=%s, provider_name=%s",
                        candidate.provider_id,
                        candidate.provider_name,
                    )
                    yield b"", ProviderResponse(
                        status_code=502,
                        error="Upstream returned an empty stream (no chunks)",
                    )
                    return

                if not first_resp.is_success:
                    yield first_chunk, first_resp
                    async for chunk, resp in upstream_gen:
                        yield chunk, resp
                    return

                async def upstream_bytes() -> AsyncGenerator[bytes, None]:
                    # Reset upstream chunks for the current attempt
                    stream_conversion_data["upstream_chunks"] = []

                    # Buffer for complete SSE events (events end with \n\n)
                    event_buffer = b""

                    async def process_chunk(chunk: bytes) -> AsyncGenerator[bytes, None]:
                        nonlocal event_buffer
                        stream_conversion_data["upstream_chunks"].append(chunk)
                        event_buffer += chunk

                        # Process complete SSE events (each event ends with \n\n)
                        while b"\n\n" in event_buffer:
                            # Find the position of the event delimiter
                            delimiter_pos = event_buffer.index(b"\n\n")
                            # Extract the complete event including the delimiter
                            complete_event = event_buffer[: delimiter_pos + 2]
                            event_buffer = event_buffer[delimiter_pos + 2 :]

                            # Call hook with complete SSE event
                            hooked_event = (
                                await self._protocol_hooks.before_stream_chunk_conversion(
                                    complete_event,
                                    request_protocol,
                                    supplier_protocol,
                                )
                            )
                            if hooked_event is None:
                                hooked_event = complete_event
                            yield hooked_event

                    async for event in process_chunk(first_chunk):
                        yield event
                    async for chunk, chunk_resp in upstream_gen:
                        if not chunk_resp.is_success:
                            # A mid-stream failure (e.g. the provider stall-guard
                            # 504) arrives as an error tuple after the first
                            # chunk. The 200 + headers are already flushed, so
                            # raise and let wrapped()'s handler emit an in-band
                            # error frame instead of silently truncating.
                            raise StreamInterrupted(
                                chunk_resp.error
                                or f"upstream returned {chunk_resp.status_code} mid-stream"
                            )
                        async for event in process_chunk(chunk):
                            yield event

                    # Flush any remaining data in buffer (incomplete event)
                    if event_buffer:
                        hooked_remaining = (
                            await self._protocol_hooks.before_stream_chunk_conversion(
                                event_buffer,
                                request_protocol,
                                supplier_protocol,
                            )
                        )
                        if hooked_remaining is None:
                            hooked_remaining = event_buffer
                        yield hooked_remaining

                try:
                    same_protocol = normalize_protocol(
                        request_protocol
                    ) == normalize_protocol(supplier_protocol)
                    if same_protocol:
                        async for chunk in upstream_bytes():
                            hooked_chunk = (
                                await self._protocol_hooks.after_stream_chunk_conversion(
                                    chunk,
                                    request_protocol,
                                    supplier_protocol,
                                )
                            )
                            if hooked_chunk is None:
                                hooked_chunk = chunk
                            yield hooked_chunk, first_resp
                    else:
                        async for out_chunk in convert_stream_for_user(
                            request_protocol=request_protocol,
                            supplier_protocol=supplier_protocol,
                            upstream=upstream_bytes(),
                            model=candidate.target_model,
                            input_tokens=input_tokens,
                            extra_content_store_cb=(
                                self._protocol_hooks.cache_tool_call_extra_content
                                if (
                                    normalize_protocol(supplier_protocol) == "gemini"
                                    and normalize_protocol(request_protocol) != "openai"
                                )
                                else None
                            ),
                        ):
                            hooked_out_chunk = (
                                await self._protocol_hooks.after_stream_chunk_conversion(
                                    out_chunk,
                                    request_protocol,
                                    supplier_protocol,
                                )
                            )
                            if hooked_out_chunk is None:
                                hooked_out_chunk = out_chunk
                            yield hooked_out_chunk, first_resp
                except Exception as e:
                    err = str(e)
                    logger.error(
                        "Error during stream response conversion: provider_id=%s, provider_name=%s, "
                        "request_protocol=%s, supplier_protocol=%s, error=%s",
                        candidate.provider_id,
                        candidate.provider_name,
                        request_protocol,
                        supplier_protocol or candidate.protocol,
                        err,
                    )
                    for frame in _stream_error_frames(request_protocol, err):
                        yield frame, first_resp
                    return

            return wrapped()

        async def log_failed_attempt(attempt: AttemptRecord) -> None:
            provider_mapping = provider_mapping_by_id.get(
                self._candidate_key(attempt.provider)
            )
            billing = resolve_billing(
                input_tokens=input_tokens,
                model_input_price=model_mapping.input_price,
                model_output_price=model_mapping.output_price,
                model_billing_mode=model_mapping.billing_mode,
                model_per_request_price=model_mapping.per_request_price,
                model_per_image_price=model_mapping.per_image_price,
                model_tiered_pricing=model_mapping.tiered_pricing,
                model_cache_billing_enabled=getattr(model_mapping, "cache_billing_enabled", None),
                model_cached_input_price=getattr(model_mapping, "cached_input_price", None),
                model_cached_output_price=getattr(model_mapping, "cached_output_price", None),
                provider_billing_mode=provider_mapping.billing_mode
                if provider_mapping
                else None,
                provider_per_request_price=provider_mapping.per_request_price
                if provider_mapping
                else None,
                provider_per_image_price=provider_mapping.per_image_price
                if provider_mapping
                else None,
                provider_tiered_pricing=provider_mapping.tiered_pricing
                if provider_mapping
                else None,
                provider_input_price=provider_mapping.input_price
                if provider_mapping
                else None,
                provider_output_price=provider_mapping.output_price
                if provider_mapping
                else None,
                provider_cache_billing_enabled=getattr(provider_mapping, "cache_billing_enabled", None)
                if provider_mapping
                else None,
                provider_cached_input_price=getattr(provider_mapping, "cached_input_price", None)
                if provider_mapping
                else None,
                provider_cached_output_price=getattr(provider_mapping, "cached_output_price", None)
                if provider_mapping
                else None,
            )
            attempt_log = RequestLogCreate(
                request_time=attempt.request_time,
                api_key_id=api_key_id,
                api_key_name=api_key_name,
                user_id=user_id,
                requested_model=requested_model,
                target_model=attempt.provider.target_model,
                provider_id=attempt.provider.provider_id,
                provider_name=attempt.provider.provider_name,
                retry_count=attempt.attempt_index + 1,
                matched_provider_count=len(candidates),
                first_byte_delay_ms=attempt.response.first_byte_delay_ms,
                total_time_ms=attempt.response.total_time_ms,
                input_tokens=input_tokens,
                output_tokens=None,
                total_cost=None,
                input_cost=None,
                output_cost=None,
                price_source=billing.price_source,
                request_headers=sanitize_headers(headers),
                response_headers=sanitize_headers(attempt.response.headers),
                request_body=sanitized_body,
                response_status=attempt.response.status_code,
                response_body=self._serialize_response_body(attempt.response.body),
                error_info=attempt.response.error,
                trace_id=trace_id,
                is_stream=True,
                request_path=path,
                request_url=request_url,
                request_method=method,
                upstream_url=stream_conversion_data.get("upstream_url"),
                # Protocol conversion fields
                request_protocol=request_protocol,
                supplier_protocol=resolve_implementation_protocol(
                    attempt.provider.protocol
                ),
                converted_request_body=_smart_truncate(
                    stream_conversion_data.get("converted_request_body")
                ),
                upstream_response_body=self._serialize_response_body(
                    attempt.response.body
                ),
            )
            try:
                with anyio.CancelScope(shield=True):
                    await self._write_log(attempt_log)
            except Exception:
                pass

        stream_gen = retry_handler.execute_with_retry_stream(
            candidates,
            requested_model,
            forward_stream_fn,
            input_tokens=input_tokens,
            image_count=image_count,
            # See non-streaming path: pins a stable prefix to one backend for cache reuse.
            affinity_key=prompt_cache_key,
            on_failure_attempt=log_failed_attempt,
        )

        # Get first chunk to determine status
        try:
            first_chunk, initial_response, final_provider, retry_count = await anext(
                stream_gen
            )
        except StopAsyncIteration:
            raise ServiceError(message="Stream ended unexpectedly", code="stream_error")
        except Exception as e:
            raise ServiceError(
                message=f"Stream connection error: {str(e)}", code="stream_error"
            )

        # Wrap generator to handle logging
        async def wrapped_generator():
            nonlocal input_tokens
            usage_acc = StreamUsageAccumulator(
                protocol=protocol,
                model=requested_model,
            )
            raw_stream_chunks: list[bytes] = []
            stream_error: Optional[str] = None

            def record_stream_chunk(chunk: Any) -> None:
                if not chunk:
                    return
                if isinstance(chunk, (bytes, bytearray)):
                    raw_stream_chunks.append(bytes(chunk))
                    return
                raw_stream_chunks.append(str(chunk).encode("utf-8"))

            try:
                usage_acc.feed(first_chunk)
                record_stream_chunk(first_chunk)
                yield first_chunk
                async for chunk, _, _, _ in stream_gen:
                    usage_acc.feed(chunk)
                    record_stream_chunk(chunk)
                    yield chunk
            except asyncio.CancelledError:
                stream_error = "client_disconnected"
                raise
            except Exception as e:
                # 200 + headers are already flushed, so the HTTP status can't
                # change; surface an explicit in-band error in the client's
                # protocol instead of a silently truncated stream, then end.
                stream_error = str(e)
                for frame in _stream_error_frames(request_protocol, str(e)):
                    record_stream_chunk(frame)
                    yield frame
                return
            finally:
                usage_result = usage_acc.finalize()
                usage_details = usage_result.usage_details
                if usage_result.input_tokens:
                    input_tokens = usage_result.input_tokens
                if usage_details is None:
                    usage_details = {
                        "input_tokens": input_tokens,
                        "output_tokens": usage_result.output_tokens,
                        "total_tokens": (input_tokens or 0)
                        + (usage_result.output_tokens or 0),
                        "source": "estimated",
                    }
                elif not usage_details.get("input_tokens"):
                    usage_details["input_tokens"] = input_tokens
                    usage_details["source"] = "mixed"
                if not usage_details.get("output_tokens"):
                    usage_details["output_tokens"] = usage_result.output_tokens
                    usage_details["source"] = "mixed"
                if not usage_details.get("total_tokens") and usage_details.get(
                    "input_tokens"
                ):
                    usage_details["total_tokens"] = usage_details["input_tokens"] + (
                        usage_details.get("output_tokens") or 0
                    )
                total_time_ms = initial_response.total_time_ms
                if total_time_ms is None:
                    total_time_ms = int((time.monotonic() - start_monotonic) * 1000)

                # 10. Record log (after stream ends)
                # Record the raw stream response (SSE) plus a reconstructed summary in one field.
                provider_mapping = (
                    provider_mapping_by_id.get(self._candidate_key(final_provider))
                    if final_provider is not None
                    else None
                )
                billing = resolve_billing(
                    input_tokens=input_tokens,
                    model_input_price=model_mapping.input_price,
                    model_output_price=model_mapping.output_price,
                    model_billing_mode=model_mapping.billing_mode,
                    model_per_request_price=model_mapping.per_request_price,
                    model_per_image_price=model_mapping.per_image_price,
                    model_tiered_pricing=model_mapping.tiered_pricing,
                    model_cache_billing_enabled=getattr(model_mapping, "cache_billing_enabled", None),
                    model_cached_input_price=getattr(model_mapping, "cached_input_price", None),
                    model_cached_output_price=getattr(model_mapping, "cached_output_price", None),
                    provider_billing_mode=provider_mapping.billing_mode
                    if provider_mapping
                    else None,
                    provider_per_request_price=provider_mapping.per_request_price
                    if provider_mapping
                    else None,
                    provider_per_image_price=provider_mapping.per_image_price
                    if provider_mapping
                    else None,
                    provider_tiered_pricing=provider_mapping.tiered_pricing
                    if provider_mapping
                    else None,
                    provider_input_price=provider_mapping.input_price
                    if provider_mapping
                    else None,
                    provider_output_price=provider_mapping.output_price
                    if provider_mapping
                    else None,
                    provider_cache_billing_enabled=getattr(provider_mapping, "cache_billing_enabled", None)
                    if provider_mapping
                    else None,
                    provider_cached_input_price=getattr(provider_mapping, "cached_input_price", None)
                    if provider_mapping
                    else None,
                    provider_cached_output_price=getattr(provider_mapping, "cached_output_price", None)
                    if provider_mapping
                    else None,
                )
                # Extract cached tokens from stream usage details
                stream_cached_input_tokens = None
                if usage_details:
                    stream_cached_input_tokens = (
                        usage_details.get("cached_tokens")
                        or usage_details.get("cache_read_input_tokens")
                    )
                cost = calculate_cost_from_billing(
                    billing=billing,
                    input_tokens=input_tokens,
                    output_tokens=usage_result.output_tokens,
                    image_count=image_count,
                    cached_input_tokens=stream_cached_input_tokens,
                )
                raw_stream_text = (
                    b"".join(raw_stream_chunks).decode("utf-8", errors="replace")
                    if raw_stream_chunks
                    else ""
                )
                reconstructed_body = json.dumps(
                    {
                        "type": "stream_reconstruction",
                        "protocol": protocol,
                        "output_text": usage_result.output_text,
                        "upstream_reported_output_tokens": usage_result.upstream_reported_output_tokens,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                combined_body = raw_stream_text
                log_data = RequestLogCreate(
                    request_time=request_time,
                    api_key_id=api_key_id,
                    api_key_name=api_key_name,
                    user_id=user_id,
                    requested_model=requested_model,
                    target_model=final_provider.target_model
                    if final_provider
                    else None,
                    provider_id=final_provider.provider_id if final_provider else None,
                    provider_name=final_provider.provider_name
                    if final_provider
                    else None,
                    retry_count=retry_count,
                    matched_provider_count=len(candidates),
                    first_byte_delay_ms=initial_response.first_byte_delay_ms,
                    total_time_ms=total_time_ms,
                    input_tokens=input_tokens,
                    output_tokens=usage_result.output_tokens,
                    total_cost=cost.total_cost,
                    input_cost=cost.input_cost,
                    output_cost=cost.output_cost,
                    cached_input_cost=cost.cached_input_cost,
                    cached_output_cost=cost.cached_output_cost,
                    price_source=billing.price_source,
                    request_headers=sanitize_headers(headers),
                    response_headers=sanitize_headers(initial_response.headers),
                    request_body=sanitized_body,
                    response_body=combined_body
                    if raw_stream_text or reconstructed_body
                    else None,
                    response_status=initial_response.status_code,
                    usage_details=usage_details,
                    error_info=initial_response.error or stream_error,
                    trace_id=trace_id,
                    is_stream=True,
                    request_path=path,
                    request_url=request_url,
                    request_method=method,
                    upstream_url=stream_conversion_data.get("upstream_url"),
                    # Protocol conversion fields
                    request_protocol=stream_conversion_data.get("request_protocol"),
                    supplier_protocol=stream_conversion_data.get("supplier_protocol"),
                    converted_request_body=_smart_truncate(
                        stream_conversion_data.get("converted_request_body")
                    ),
                    # For stream, upstream_response_body is the raw stream captured from upstream
                    upstream_response_body=(
                        b"".join(stream_conversion_data["upstream_chunks"]).decode(
                            "utf-8", errors="replace"
                        )
                        if stream_conversion_data.get("upstream_chunks")
                        else (raw_stream_text if raw_stream_text else None)
                    ),
                )

                # DEBUG: Log request details
                try:
                    logger.debug(f"Request Log: {log_data.model_dump_json()}")
                except AttributeError:
                    # Fallback for Pydantic v1
                    logger.debug(f"Request Log: {log_data.json()}")

                # client disconnect triggers cancellation, use shield to ensure logs are written to DB
                try:
                    with anyio.CancelScope(shield=True):
                        await self._write_log(log_data)
                except Exception:
                    # Log writing failure does not affect main flow
                    pass

        return (
            initial_response,
            wrapped_generator(),
            {
                "trace_id": trace_id,
                "retry_count": retry_count,
                "target_model": final_provider.target_model if final_provider else None,
                "provider_name": final_provider.provider_name
                if final_provider
                else None,
            },
        )
