# Token Accounting Research and Unified Specification

This report summarizes usage payload structures and token accounting approaches for OpenAI, Anthropic, Google Gemini, and OpenRouter, then defines this project's unified structured recording and estimation strategy. Key point: token counting before forwarding is for routing reference only; if the upstream response includes usage data, upstream values are authoritative and override local estimates.

## 1. Vendor Usage Structures and Token Accounting

### 1.1 OpenAI
- Chat Completions:
  - `usage.prompt_tokens`, `usage.completion_tokens`, `usage.total_tokens`
- Responses API:
  - `usage.input_tokens`, `usage.output_tokens`, `usage.total_tokens`
  - Optional detailed fields: `usage.input_tokens_details` / `usage.output_tokens_details`
    - Common fields: `cached_tokens`, `audio_tokens`, `image_tokens`, `reasoning_tokens`, `tool_tokens`
- Multimodal:
  - Text: counted by model tokenizer (for example `tiktoken`).
  - Image: low detail is usually a fixed token count; high detail is tile-based (512px tiles).
  - Audio: often reflected in detailed usage fields such as `audio_tokens`.
  - Video: some models/interfaces provide detailed fields such as `video_tokens`.

### 1.2 Anthropic
- Messages API:
  - `usage.input_tokens`, `usage.output_tokens`
  - Cache-related: `cache_creation_input_tokens`, `cache_read_input_tokens`
- Multimodal:
  - Image content is included in input/output token accounting; usage may mainly expose totals and cache deltas.

### 1.3 Google Gemini
- `usageMetadata`:
  - `promptTokenCount`, `candidatesTokenCount`, `totalTokenCount`
  - `cachedContentTokenCount`
- Multimodal tokens are included in prompt/candidate totals.

### 1.4 OpenRouter (OpenAI-compatible)
- Usually reuses OpenAI usage fields:
  - `prompt_tokens`, `completion_tokens`, `total_tokens`
- Potential extensions:
  - `cached_tokens`, or details similar to OpenAI Responses.

## 2. Unified Structured Usage Schema (Log Storage)

The log table adds `usage_details` (JSON) with this unified schema (fields may be omitted if unavailable):

```json
{
  "input_tokens": 123,
  "output_tokens": 45,
  "total_tokens": 168,
  "cached_tokens": 12,
  "cache_creation_input_tokens": 0,
  "cache_read_input_tokens": 12,
  "input_audio_tokens": 0,
  "output_audio_tokens": 0,
  "input_image_tokens": 85,
  "output_image_tokens": 0,
  "input_video_tokens": 0,
  "output_video_tokens": 0,
  "reasoning_tokens": 0,
  "tool_tokens": 0,
  "source": "upstream|estimated|mixed",
  "raw_usage": { "prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168 },
  "extra_usage": { "vendor_field": "..." }
}
```

Notes:
- `source=upstream`: response includes usage and upstream values are used.
- `source=estimated`: response omits usage and local estimates are used.
- `source=mixed`: some fields are missing and values combine upstream + local estimates.
- `raw_usage`: original usage payload (preserved as completely as possible).
- `extra_usage`: non-standard usage fields retained to avoid data loss.

## 3. Multimodal Token Estimation Strategy (Local Fallback)

Used only when upstream usage is unavailable.

### 3.1 Text
- OpenAI: use `tiktoken` (for example `cl100k_base`).
- Anthropic: currently estimated from character length (average 4 chars/token), including multimodal content blocks.

### 3.2 Image
- OpenAI estimation rule (aligned with published billing guidance):
  - `detail=low`: about 85 tokens
  - `detail=high` or unknown: tile-based, `tokens = tiles * 170`
  - Tiles: `ceil(width/512) * ceil(height/512)`
- Anthropic/Google/OpenRouter: currently uses the same tile heuristic until official tokenizers are integrated.
- If only Base64 is available, image dimensions are inferred from PNG/JPEG headers before estimating.

### 3.3 Audio
- Prefer upstream usage values (for example `audio_tokens`).
- When usage is missing:
  - If duration is known: `tokens ~= duration_seconds * 50`
  - If only byte size is known: `tokens ~= bytes / 1000`

### 3.4 Video
- Prefer upstream usage values (for example `video_tokens`).
- When usage is missing:
  - If duration is known: `tokens ~= duration_seconds * 200`
  - If only byte size is known: `tokens ~= bytes / 2000`

## 4. Processing and Update Logic

1. **Before forwarding**: estimate input tokens from request payload (text/image/audio/video) for routing decisions.
2. **After response**:
   - If usage exists: replace input/output counts with upstream usage values.
   - If usage is absent: estimate output tokens locally and keep input token estimate.
3. **Logging**: always store `usage_details`, and preserve raw usage for later analysis.

## 5. Risks and Improvement Directions

- Multimodal token estimation depends on vendor-specific rules and should be treated as approximate routing guidance.
- Future improvements: integrate official tokenizers (for example Anthropic tokenizer) or more accurate multimodal cost models.
