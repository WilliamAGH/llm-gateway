"""Unit tests for prompt-cache observability helpers in proxy_service."""

import app.services.proxy_service as ps
from app.services.proxy_service import (
    _note_and_check_repeat_miss,
    _request_has_cache_signal,
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
