# Guide: Adding a New Protocol Conversion Path

This document explains how to add a new protocol or extend protocol conversion paths in `backend/app/common/protocol_conversion.py`.

## Goals

- Use registry-style converters so adding a protocol/path does not require changing core logic.
- Keep request, response, and streaming conversion extensions consistent.

## Key Locations

- Protocol constants and normalization: `backend/app/common/provider_protocols.py`
- Conversion core: `backend/app/common/protocol_conversion.py`
- OpenAI Responses conversion helpers: `backend/app/common/openai_responses.py`

## Minimum Steps to Add a Protocol

1. **Declare protocol constants**
   - Add protocol constants in `backend/app/common/provider_protocols.py`.
   - Add the protocol to `IMPLEMENTATION_PROTOCOLS` and extend `resolve_implementation_protocol` mapping rules.

2. **Implement conversion functions**
   - Request conversion: add `_convert_request_<from>_to_<to>`.
   - Response conversion: add `_convert_response_<from>_to_<to>`.
   - Stream conversion: add `_convert_stream_<from>_to_<to>`.

3. **Register conversion routes**
   - Add request conversion in the `converters` map inside `convert_request_for_supplier`.
   - Add response conversion in the `converters` map inside `convert_response_for_user`.
   - Add stream conversion in the `converters` map inside `convert_stream_for_user`.

4. **Complete protocol-specific behavior**
   - If the new protocol has special fields (tool calls, system messages, etc.), reuse or add normalization helpers.
   - Prefer official SDK converters when available; otherwise implement a minimal fallback.

5. **Test and validate**
   - Add unit tests for each new path under `backend/tests/unit/`.
   - Manually validate stream event ordering and termination semantics for the target protocol.

## Development Conventions

- Request/response/stream converters should be idempotent, composable, and low side-effect.
- Reuse existing helpers when adding protocols (for example `_normalize_openai_tooling_fields`).
- Do not stack protocol branching logic in shared entry points; route through the converter registry.

## Example: Adding a `Foo` Protocol

The following pseudocode shows a minimal OpenAI -> Foo request conversion:

```python
# 1) Add request conversion function

def _convert_request_openai_to_foo(*, path: str, body: dict[str, Any], target_model: str) -> tuple[str, dict[str, Any]]:
    if path != "/v1/chat/completions":
        raise ServiceError(message=f"Unsupported OpenAI endpoint for conversion: {path}", code="unsupported_protocol_conversion")
    foo_body = {...}
    foo_body["model"] = target_model
    return "/v1/foo/messages", foo_body

# 2) Register request conversion
converters = {
    (OPENAI_PROTOCOL, FOO_PROTOCOL): _convert_request_openai_to_foo,
}
```

## Common Gotchas

- **Path checks**: Every converter should validate whether the `path` is supported.
- **Model field**: Ensure the converted payload sets `model` explicitly (unless not required downstream).
- **Stream termination semantics**: End-of-stream behavior differs between protocols and must be translated correctly.
- **Tool calls**: Ensure two-way compatibility for `tools` / `tool_choice` / `tool_calls`.
