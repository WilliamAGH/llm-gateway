# Follow-up: Anthropic batch-tier 1h prompt cache

Tracking doc (GitHub issues are disabled on this fork). Follow-up from `1eb8f42`
— batch-tier 1h prompt cache via a `cache_control.ttl="1h"` rewrite, which also
removed the now-GA `extended-cache-ttl-2025-04-11` beta-header plumbing (the 1h
TTL no longer requires a beta header; verified against the live Anthropic
prompt-caching docs).

All three items below are **pre-existing** characteristics of the
`_apply_extended_cache` feature, **non-blocking**, in
`backend/app/providers/anthropic_client.py`.

## 1. Shallow-copy body mutation

`base.py::_prepare_body` does `body.copy()` (shallow), so `_set_cache_control_ttl`
mutates the **original** caller's nested `cache_control` dicts in place when it
rewrites `ttl="1h"`. Harmless for request-scoped bodies (the docstring
acknowledges it), but a latent footgun if any retry/failover path re-forwards
the same `body` object — the mutated `ttl` would persist into the next attempt,
even one that routes to a non-batch tier or a different provider.

- **Fix:** deep-copy the relevant subtree before mutating (scoped to batch tier),
  e.g. `copy.deepcopy` inside `_apply_extended_cache`.

## 2. Internal `x-tier` routing header leaks upstream

`x-tier` is a client-supplied inbound header (`/v1/messages` forwards
`dict(request.headers)`). `_is_batch_tier` reads it, but it is **not** in
`_prepare_headers`'s `keys_to_remove`, so it is forwarded to Anthropic.
Anthropic ignores unknown `x-*` headers (functionally harmless), but this leaks
internal routing taxonomy upstream.

- **Fix:** add `TIER_HEADER` (and any `x-lgw-*` internal headers) to
  `keys_to_remove` in `_prepare_headers`.

## 3. MiniMax + batch tier gets `ttl="1h"` injected

`_apply_extended_cache` runs **before** `_sanitize_minimax_body`, and the
sanitizer strips `context_management` / `mcp_servers` / etc. but **not**
`cache_control`. So a MiniMax batch request would carry `ttl="1h"`. MiniMax's
Anthropic-compat layer may not honor the extended-TTL field.

- **Fix:** skip `_apply_extended_cache` for MiniMax base URLs, or strip
  `cache_control.ttl` in `_sanitize_minimax_body`.

## Testing note

The header / `ttl` forwarding behavior is asserted by
`backend/tests/unit/test_providers/test_anthropic_extended_cache.py` — the
correct layer, since these are upstream-forwarding details that are **not**
black-box-observable from a gateway client. Live testing against
`api.llm-gateway.iocloudhost.net` can only confirm end-to-end that batch-tier
caching still works (`/v1/messages` with `x-tier: batch` + a `cache_control`
block succeeds and `usage.cache_read_input_tokens > 0` on a repeat).
