"""Unit tests for prompt-cache observability helpers in proxy_service."""

import app.services.proxy_service as ps
from app.services.proxy_service import (
    _body_with_prompt_cache_key,
    _body_with_gateway_response_cache_usage,
    _exact_response_cache_allowed,
    _exact_response_cache_key,
    _note_and_check_repeat_miss,
    _prompt_cache_key_for_request,
    _request_has_cache_signal,
    _supplier_body_with_prompt_cache_key,
)


class TestRequestHasCacheSignal:
    def test_prompt_cache_key(self):
        assert _request_has_cache_signal({"prompt_cache_key": "k"}) is True

    def test_empty_prompt_cache_key_is_not_a_signal(self):
        assert _request_has_cache_signal({"prompt_cache_key": ""}) is False

    def test_cache_control_on_system_block(self):
        body = {"system": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]}
        assert _request_has_cache_signal(body) is True

    def test_cache_control_on_message_content(self):
        body = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "x", "cache_control": {}}]}
            ]
        }
        # An empty dict is falsy, so this is NOT a signal — only a populated cache_control counts.
        assert _request_has_cache_signal(body) is False
        body["messages"][0]["content"][0]["cache_control"] = {"type": "ephemeral"}
        assert _request_has_cache_signal(body) is True

    def test_no_signal(self):
        assert _request_has_cache_signal({"messages": [{"role": "user", "content": "hi"}]}) is False

    def test_non_dict(self):
        assert _request_has_cache_signal("not a dict") is False


class TestPromptCacheKeyForRequest:
    def test_preserves_existing_prompt_cache_key(self):
        key = _prompt_cache_key_for_request(
            {"prompt_cache_key": "client-key"},
            "kimi-k2.7-code",
        )

        assert key == "client-key"

    def test_derives_stable_kimi_key_from_stable_prefix(self):
        stable_prefix = "stable-prefix " * 900
        body_a = {
            "messages": [
                {"role": "system", "content": stable_prefix},
                {"role": "user", "content": "first delta after stable prefix"},
            ]
        }
        body_b = {
            "messages": [
                {"role": "system", "content": stable_prefix},
                {"role": "user", "content": "second delta after stable prefix"},
            ]
        }

        key_a = _prompt_cache_key_for_request(body_a, "kimi-k2.7-code")
        key_b = _prompt_cache_key_for_request(body_b, "kimi-k2.7-code")

        assert key_a is not None
        assert key_a.startswith("llmgw:kimi:")
        assert key_a == key_b

    def test_derives_from_kimi_target_model_when_requested_alias_is_neutral(self):
        key = _prompt_cache_key_for_request(
            {"messages": [{"role": "user", "content": "hello"}]},
            "researchly-code",
            ["kimi-for-coding"],
        )

        assert key is not None
        assert key.startswith("llmgw:kimi:")

    def test_kimi_session_hint_wins_over_changing_prefix(self):
        body_a = {
            "messages": [
                {"role": "system", "content": "stable-prefix " * 900},
                {"role": "user", "content": "first turn"},
            ]
        }
        body_b = {
            "messages": [
                {"role": "system", "content": "different-prefix " * 900},
                {"role": "user", "content": "second turn"},
            ]
        }

        key_a = _prompt_cache_key_for_request(body_a, "kimi-k2.7-code", cache_key_hint="harness-run:123")
        key_b = _prompt_cache_key_for_request(body_b, "kimi-k2.7-code", cache_key_hint="harness-run:123")

        assert key_a is not None
        assert key_a.startswith("llmgw:kimi:")
        assert key_a == key_b

    def test_derives_from_gpt_prefixed_target_model(self):
        key = _prompt_cache_key_for_request(
            {"messages": [{"role": "user", "content": "hello"}]},
            "researchly-gpt",
            ["gpt-5.4"],
        )

        assert key is not None
        assert key.startswith("llmgw:openai:")

    def test_derives_from_generic_gpt_prefixed_target_model(self):
        key = _prompt_cache_key_for_request(
            {"messages": [{"role": "user", "content": "hello"}]},
            "researchly-gpt",
            ["gpt-example"],
        )

        assert key is not None
        assert key.startswith("llmgw:openai:")

    def test_gpt_key_ignores_rotating_claude_code_billing_system_block(self):
        stable_system = {"type": "text", "text": "You are Claude Code."}
        body_a = {
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: cch=first;"},
                stable_system,
            ],
            "messages": [{"role": "user", "content": "hello"}],
        }
        body_b = {
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: cch=second;"},
                stable_system,
            ],
            "messages": [{"role": "user", "content": "hello"}],
        }

        key_a = _prompt_cache_key_for_request(body_a, "gpt-5.4")
        key_b = _prompt_cache_key_for_request(body_b, "gpt-5.4")

        assert key_a is not None
        assert key_a.startswith("llmgw:openai:")
        assert key_a == key_b
        assert "prompt_cache_key" not in body_a
        assert "prompt_cache_key" not in body_b

    def test_does_not_derive_key_for_local_gpt_oss_model(self):
        key = _prompt_cache_key_for_request(
            {"messages": [{"role": "user", "content": "hello"}]},
            "gpt-oss-120b",
            ["gpt-oss-120b"],
        )

        assert key is None


class TestPromptCacheKeyBodies:
    def test_body_with_prompt_cache_key_returns_copy_and_preserves_input(self):
        body = {"messages": [{"role": "user", "content": "hello"}]}

        with_key = _body_with_prompt_cache_key(body, "derived-key")

        assert with_key == {**body, "prompt_cache_key": "derived-key"}
        assert body == {"messages": [{"role": "user", "content": "hello"}]}

    def test_body_with_prompt_cache_key_preserves_existing_key(self):
        body = {"prompt_cache_key": "client-key"}

        assert _body_with_prompt_cache_key(body, "derived-key") is body

    def test_supplier_body_injects_for_openai_kimi_target(self):
        body = {"messages": [{"role": "user", "content": "hello"}]}

        supplier_body = _supplier_body_with_prompt_cache_key(
            body,
            "derived-key",
            requested_model="researchly-code",
            target_model="kimi-for-coding",
            base_url="https://api.kimi.com/coding/v1",
            supplier_protocol="openai",
        )

        assert supplier_body == {**body, "prompt_cache_key": "derived-key"}
        assert body == {"messages": [{"role": "user", "content": "hello"}]}

    def test_supplier_body_injects_for_openai_responses_gpt_prefixed_target(self):
        body = {"input": "hello"}

        supplier_body = _supplier_body_with_prompt_cache_key(
            body,
            "derived-key",
            requested_model="researchly-gpt",
            target_model="gpt-5.4",
            base_url="https://api.openai.com/v1",
            supplier_protocol="openai_responses",
        )

        assert supplier_body == {**body, "prompt_cache_key": "derived-key"}
        assert body == {"input": "hello"}

    def test_supplier_body_does_not_inject_for_non_cache_target_or_non_openai(self):
        body = {"messages": [{"role": "user", "content": "hello"}]}

        assert (
            _supplier_body_with_prompt_cache_key(
                body,
                "derived-key",
                requested_model="gpt-oss-120b",
                target_model="gpt-oss-120b",
                base_url="https://api.llm-gateway.popos-sf4.com/v1",
                supplier_protocol="openai",
            )
            is body
        )
        assert (
            _supplier_body_with_prompt_cache_key(
                body,
                "derived-key",
                requested_model="kimi-k2.7-code",
                target_model="kimi-for-coding",
                base_url="https://api.kimi.com/coding/v1",
                supplier_protocol="anthropic",
            )
            is body
        )
        assert (
            _supplier_body_with_prompt_cache_key(
                "not-a-dict",
                "derived-key",
                requested_model="kimi-k2.7-code",
                target_model="kimi-for-coding",
                base_url="https://api.kimi.com/coding/v1",
                supplier_protocol="openai",
            )
            == "not-a-dict"
        )


class TestRepeatMissDetection:
    def setup_method(self):
        ps._recent_prompt_cache_keys.clear()

    def test_first_sight_is_cold_no_warn(self):
        # Cold miss must not warn, but is recorded.
        assert _note_and_check_repeat_miss("k", cache_hit=False, now=100.0) is False

    def test_repeat_miss_warns(self):
        _note_and_check_repeat_miss("k", cache_hit=False, now=100.0)
        assert _note_and_check_repeat_miss("k", cache_hit=False, now=120.0) is True

    def test_repeat_hit_does_not_warn(self):
        _note_and_check_repeat_miss("k", cache_hit=False, now=100.0)
        assert _note_and_check_repeat_miss("k", cache_hit=True, now=120.0) is False

    def test_expired_repeat_is_cold_again(self):
        _note_and_check_repeat_miss("k", cache_hit=False, now=100.0)
        # Beyond the TTL window the prior sighting is pruned -> treated as cold, no warn.
        later = 100.0 + ps._CACHE_KEY_TTL_SECONDS + 1
        assert _note_and_check_repeat_miss("k", cache_hit=False, now=later) is False

    def test_none_key_never_warns(self):
        assert _note_and_check_repeat_miss(None, cache_hit=False, now=100.0) is False


class TestExactResponseCache:
    def test_gateway_cache_hit_marks_anthropic_usage_as_cache_read(self):
        body = {
            "type": "message",
            "usage": {"input_tokens": 5282, "output_tokens": 5},
        }

        with_usage = _body_with_gateway_response_cache_usage(body, 5282)

        assert with_usage["usage"]["cache_read_input_tokens"] == 5282
        assert "cache_read_input_tokens" not in body["usage"]

    def test_gateway_cache_hit_uses_provider_input_count_over_estimate(self):
        body = {
            "type": "message",
            "usage": {"input_tokens": 5282, "output_tokens": 5},
        }

        with_usage = _body_with_gateway_response_cache_usage(body, 5420)

        assert with_usage["usage"]["cache_read_input_tokens"] == 5282

    def test_gateway_cache_hit_marks_openai_responses_usage_as_cached(self):
        body = {
            "object": "response",
            "usage": {
                "input_tokens": 5282,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 5,
            },
        }

        with_usage = _body_with_gateway_response_cache_usage(body, 5282)

        assert with_usage["usage"]["input_tokens_details"]["cached_tokens"] == 5282

    def test_gateway_cache_hit_marks_openai_chat_usage_as_cached(self):
        body = {
            "object": "chat.completion",
            "usage": {
                "prompt_tokens": 5282,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens": 5,
            },
        }

        with_usage = _body_with_gateway_response_cache_usage(body, 5282)

        assert with_usage["usage"]["prompt_tokens_details"]["cached_tokens"] == 5282

    def test_allows_long_cross_protocol_gpt_responses_request(self):
        assert _exact_response_cache_allowed(
            request_protocol="anthropic",
            supplier_protocol="openai_responses",
            supplier_body={"model": "gpt-5.4", "input": "stable", "prompt_cache_key": "k"},
            method="POST",
            prompt_cache_key="k",
            requested_model="gpt-5.4",
            target_model="gpt-5.4",
            base_url="https://api.openai.com/v1",
            input_tokens=2048,
        ) is True

    def test_allows_long_cross_protocol_kimi_openai_request(self):
        assert _exact_response_cache_allowed(
            request_protocol="anthropic",
            supplier_protocol="openai",
            supplier_body={"model": "kimi-k2.7-code", "messages": [], "prompt_cache_key": "k"},
            method="POST",
            prompt_cache_key="k",
            requested_model="kimi-k2.7-code",
            target_model="kimi-k2.7-code",
            base_url="https://api.moonshot.ai/v1",
            input_tokens=2048,
        ) is True

    def test_rejects_native_or_stateful_or_short_requests(self):
        base = {
            "request_protocol": "anthropic",
            "supplier_protocol": "openai_responses",
            "supplier_body": {"model": "gpt-5.4", "input": "stable", "prompt_cache_key": "k"},
            "method": "POST",
            "prompt_cache_key": "k",
            "requested_model": "gpt-5.4",
            "target_model": "gpt-5.4",
            "base_url": "https://api.openai.com/v1",
            "input_tokens": 2048,
        }

        assert _exact_response_cache_allowed(**{**base, "request_protocol": "openai_responses"}) is False
        assert _exact_response_cache_allowed(**{**base, "input_tokens": 512}) is False
        assert _exact_response_cache_allowed(
            **{**base, "supplier_body": {**base["supplier_body"], "tools": []}}
        ) is False
        assert _exact_response_cache_allowed(
            **{**base, "supplier_body": {**base["supplier_body"], "store": True}}
        ) is False

    def test_cache_key_includes_api_key_provider_and_exact_supplier_body(self):
        base = {
            "api_key_id": 3,
            "method": "POST",
            "supplier_path": "/v1/responses",
            "requested_model": "gpt-5.4",
            "provider_id": 14,
            "provider_mapping_id": None,
            "target_model": "gpt-5.4",
            "supplier_body": {"model": "gpt-5.4", "input": "stable", "prompt_cache_key": "k"},
        }

        assert _exact_response_cache_key(**base) == _exact_response_cache_key(**base)
        assert _exact_response_cache_key(**base) != _exact_response_cache_key(
            **{**base, "api_key_id": 4}
        )
        assert _exact_response_cache_key(**base) != _exact_response_cache_key(
            **{**base, "supplier_body": {**base["supplier_body"], "max_output_tokens": 16}}
        )
