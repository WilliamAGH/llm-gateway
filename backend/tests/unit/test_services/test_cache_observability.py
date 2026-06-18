"""Unit tests for prompt-cache observability helpers in proxy_service."""

import app.services.proxy_service as ps
from app.services.proxy_service import (
    _body_with_prompt_cache_key,
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

    def test_does_not_derive_key_for_non_kimi_model(self):
        key = _prompt_cache_key_for_request(
            {"messages": [{"role": "user", "content": "hello"}]},
            "gpt-5.4",
            ["gpt-5.4"],
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

    def test_supplier_body_injects_only_for_openai_kimi_target(self):
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

    def test_supplier_body_does_not_inject_for_non_kimi_or_non_openai(self):
        body = {"messages": [{"role": "user", "content": "hello"}]}

        assert (
            _supplier_body_with_prompt_cache_key(
                body,
                "derived-key",
                requested_model="gpt-5.4",
                target_model="gpt-5.4",
                base_url="https://api.openai.com/v1",
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
