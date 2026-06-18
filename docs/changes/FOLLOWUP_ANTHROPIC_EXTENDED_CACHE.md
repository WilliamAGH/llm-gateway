# Anthropic batch-tier 1h prompt cache — fixes + open issue

Context: `1eb8f42` shipped the batch-tier 1h prompt cache (a `cache_control.ttl="1h"`
rewrite) and removed the now-GA `extended-cache-ttl-2025-04-11` beta header. The 1h
TTL is GA — `ttl="1h"` on the block is all that's required (verified against the live
Anthropic prompt-caching docs).

## Resolved (were wrongly classified non-blocking; fixed)

All three in `backend/app/providers/anthropic_client.py`, covered by
`backend/tests/unit/test_providers/test_anthropic_extended_cache.py`.

1. **Root cause — masked error via shallow-copy mutation.** `_set_cache_control_ttl`
   mutated nested `cache_control` dicts in place, and `_prepare_body` only shallow-copies,
   so the caller's original body was corrupted — a retry/failover re-forwarding it (to a
   non-batch tier or a different provider) would carry a stale `ttl="1h"`.
   **Fix:** `_apply_extended_cache` now `copy.deepcopy`s before mutating and returns the
   copy; the input is never touched. Test: `test_apply_extended_cache_does_not_mutate_input`.

2. **Encapsulation — internal routing header leaked upstream.** `x-tier` (the routing
   tier) and the gateway's `x-lgw-*` observability namespace were forwarded to Anthropic.
   **Fix:** `_prepare_headers` now strips `TIER_HEADER` and any `x-lgw-*` header.
   Test: `test_prepare_headers_strips_internal_routing_headers`.

3. **Framework-first — unsupported vendor field on MiniMax.** `_apply_extended_cache` ran
   before `_sanitize_minimax_body`, injecting `ttl="1h"` into MiniMax requests whose
   Anthropic-compat layer doesn't honor it (silent cache degradation).
   **Fix:** the rewrite is skipped for MiniMax (`is_minimax` gate at the call site;
   `_apply_extended_cache(..., is_minimax=True)` is a no-op). Test:
   `test_apply_extended_cache_skips_minimax`.

## Open — the feature isn't activating in production (consistent 5m)

The 1h rewrite only fires when `_is_batch_tier(headers)` is true, i.e. the inbound header
is literally `x-tier: batch`. Per `app/common/http_timeout.py`, **that header is set by the
upstream queue** (`x-tier: batch | production-a | ...`) — *not* by the client and *not*
derived by the gateway. (An earlier draft of this doc wrongly called it "client-supplied.")

Production shows consistent 5m caching, which means real Anthropic batch traffic reaching
the gateway is **not carrying `x-tier: batch`** — it's on a different tier value, or it
isn't transiting the lane that stamps `batch`.

Note the asymmetry with the stream-deadline policy: `StreamDeadlinePolicy.for_request`
recognizes the long class by `x-tier: batch` **OR** an `:onprem` model suffix, whereas
`_is_batch_tier` checks only the header. (`:onprem` models are self-hosted and don't do
Anthropic caching, so that specific signal isn't the fix — but it shows the cache gate is
narrower than the rest of the codebase's "long class.")

**Decision needed before fixing:** is the queue supposed to stamp `x-tier: batch` on
Anthropic batch work (fix in the `llm-inference-network` queue/edge), or should the gateway
classify batch from a signal the harness already sends (fix `_is_batch_tier`)? Confirm what
real traffic carries — via the gateway admin logs (`/api/admin/logs`) or the queue config —
before changing either side.
