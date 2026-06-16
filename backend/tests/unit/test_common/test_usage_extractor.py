from app.common.usage_extractor import (
    ensure_openai_usage_details,
    extract_output_tokens,
    extract_usage_details,
)


def test_extract_usage_details_openai_prompt_completion():
    body = {"usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}}
    details = extract_usage_details(body)
    assert details is not None
    assert details.input_tokens == 12
    assert details.output_tokens == 7
    assert details.total_tokens == 19


def test_extract_usage_details_openai_details_fields():
    body = {
        "usage": {
            "input_tokens": 10,
            "output_tokens": 4,
            "input_tokens_details": {"cached_tokens": 2, "audio_tokens": 3},
            "output_tokens_details": {"image_tokens": 5, "reasoning_tokens": 1},
        }
    }
    details = extract_usage_details(body)
    assert details is not None
    assert details.input_tokens == 10
    assert details.output_tokens == 4
    assert details.cached_tokens == 2
    assert details.input_audio_tokens == 3
    assert details.output_image_tokens == 5
    assert details.reasoning_tokens == 1


def test_extract_usage_details_anthropic_cache_fields():
    body = {
        "usage": {
            "input_tokens": 20,
            "output_tokens": 5,
            "cache_creation_input_tokens": 3,
            "cache_read_input_tokens": 2,
        }
    }
    details = extract_usage_details(body)
    assert details is not None
    assert details.cache_creation_input_tokens == 3
    assert details.cache_read_input_tokens == 2


def test_extract_usage_details_gemini_metadata():
    body = {
        "usageMetadata": {
            "promptTokenCount": 8,
            "candidatesTokenCount": 6,
            "totalTokenCount": 14,
            "cachedContentTokenCount": 4,
        }
    }
    details = extract_usage_details(body)
    assert details is not None
    assert details.input_tokens == 8
    assert details.output_tokens == 6
    assert details.total_tokens == 14
    assert details.cached_tokens == 4


def test_extract_usage_details_gemini_modality_details():
    """Gemini usageMetadata with promptTokensDetails and candidatesTokensDetails."""
    body = {
        "usageMetadata": {
            "promptTokenCount": 6,
            "candidatesTokenCount": 1220,
            "totalTokenCount": 1377,
            "promptTokensDetails": [
                {"modality": "TEXT", "tokenCount": 6},
            ],
            "candidatesTokensDetails": [
                {"modality": "IMAGE", "tokenCount": 1120},
            ],
            "thoughtsTokenCount": 151,
        }
    }
    details = extract_usage_details(body)
    assert details is not None
    assert details.input_tokens == 6
    assert details.output_tokens == 1220
    assert details.total_tokens == 1377
    assert details.output_image_tokens == 1120
    assert details.reasoning_tokens == 151
    # TEXT modality in promptTokensDetails doesn't map to image/audio/video
    assert details.input_image_tokens is None
    assert details.input_audio_tokens is None
    # Parsed fields should not appear in extra_usage
    assert details.extra_usage is None or "promptTokensDetails" not in details.extra_usage
    assert details.extra_usage is None or "candidatesTokensDetails" not in details.extra_usage
    assert details.extra_usage is None or "thoughtsTokenCount" not in details.extra_usage


def test_extract_usage_details_gemini_multimodal_input():
    """Gemini usageMetadata with image and audio in prompt."""
    body = {
        "usageMetadata": {
            "promptTokenCount": 500,
            "candidatesTokenCount": 100,
            "totalTokenCount": 600,
            "promptTokensDetails": [
                {"modality": "TEXT", "tokenCount": 50},
                {"modality": "IMAGE", "tokenCount": 300},
                {"modality": "AUDIO", "tokenCount": 150},
            ],
            "candidatesTokensDetails": [
                {"modality": "TEXT", "tokenCount": 100},
            ],
        }
    }
    details = extract_usage_details(body)
    assert details is not None
    assert details.input_tokens == 500
    assert details.output_tokens == 100
    assert details.input_image_tokens == 300
    assert details.input_audio_tokens == 150
    assert details.output_image_tokens is None
    assert details.reasoning_tokens is None


def test_extract_output_tokens_fallback_total_minus_input():
    body = {"usage": {"total_tokens": 20, "prompt_tokens": 12}}
    assert extract_output_tokens(body) == 8


def test_ensure_openai_usage_details_reattaches_cached_tokens_from_responses_upstream():
    # Responses-API upstream carries cache info under input_tokens_details; the converted
    # chat response lost it down to the flat trio. The detail sub-object must be restored.
    upstream = {
        "usage": {
            "input_tokens": 2707,
            "output_tokens": 5,
            "input_tokens_details": {"cached_tokens": 2304},
            "output_tokens_details": {"reasoning_tokens": 3},
        }
    }
    response = {"usage": {"prompt_tokens": 2707, "completion_tokens": 5, "total_tokens": 2712}}

    ensure_openai_usage_details(response, upstream)

    assert response["usage"]["prompt_tokens_details"] == {"cached_tokens": 2304}
    assert response["usage"]["completion_tokens_details"] == {"reasoning_tokens": 3}


def test_ensure_openai_usage_details_preserves_existing_details():
    # Same-protocol verbatim responses already carry richer details — never downgrade them.
    rich = {"cached_tokens": 2304, "audio_tokens": 0}
    response = {
        "usage": {
            "prompt_tokens": 2707,
            "completion_tokens": 5,
            "total_tokens": 2712,
            "prompt_tokens_details": rich,
        }
    }

    ensure_openai_usage_details(response, {"usage": {"input_tokens_details": {"cached_tokens": 1}}})

    assert response["usage"]["prompt_tokens_details"] is rich


def test_ensure_openai_usage_details_noop_without_usage():
    response = {"choices": []}
    ensure_openai_usage_details(response, {"usage": {"prompt_tokens_details": {"cached_tokens": 9}}})
    assert "usage" not in response
