import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api import auth as auth_api
from app.config import get_settings


def _build_auth_app() -> FastAPI:
    app = FastAPI()
    app.include_router(auth_api.router)
    return app


@pytest.fixture(autouse=True)
def _configure_auth_for_tests(monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    monkeypatch.setenv("ADMIN_TOKEN_TTL_SECONDS", "60")
    get_settings.cache_clear()

    # Keep thresholds small for deterministic tests.
    monkeypatch.setattr(auth_api, "_MAX_LOGIN_ATTEMPTS", 3)
    monkeypatch.setattr(auth_api, "_LOGIN_WINDOW_SECONDS", 60)
    monkeypatch.setattr(auth_api, "_LOGIN_BLOCK_SECONDS", 120)
    auth_api._reset_login_throttle_state()

    yield

    auth_api._reset_login_throttle_state()
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_login_rate_limit_blocks_after_repeated_failures():
    app = _build_auth_app()
    transport = ASGITransport(app=app, client=("10.0.0.1", 12345))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(2):
            response = await client.post(
                "/auth/login",
                json={"username": "admin", "password": "wrong-password"},
            )
            assert response.status_code == 401

        blocked = await client.post(
            "/auth/login",
            json={"username": "admin", "password": "wrong-password"},
        )
        assert blocked.status_code == 429
        assert blocked.headers.get("Retry-After") == "120"

        still_blocked = await client.post(
            "/auth/login",
            json={"username": "admin", "password": "secret"},
        )
        assert still_blocked.status_code == 429


@pytest.mark.asyncio
async def test_login_failure_counter_resets_after_success():
    app = _build_auth_app()
    transport = ASGITransport(app=app, client=("10.0.0.2", 23456))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        first_fail = await client.post(
            "/auth/login",
            json={"username": "admin", "password": "wrong-password"},
        )
        assert first_fail.status_code == 401

        success = await client.post(
            "/auth/login",
            json={"username": "admin", "password": "secret"},
        )
        assert success.status_code == 200
        assert "access_token" in success.json()

        # Counter should be reset, so this is a normal 401 (not an immediate 429).
        after_reset_fail = await client.post(
            "/auth/login",
            json={"username": "admin", "password": "wrong-password"},
        )
        assert after_reset_fail.status_code == 401


@pytest.mark.asyncio
async def test_login_rate_limit_is_scoped_per_ip():
    app = _build_auth_app()

    blocked_transport = ASGITransport(app=app, client=("10.0.0.3", 34567))
    async with AsyncClient(
        transport=blocked_transport, base_url="http://test"
    ) as blocked_client:
        for _ in range(3):
            await blocked_client.post(
                "/auth/login",
                json={"username": "admin", "password": "wrong-password"},
            )

        blocked_check = await blocked_client.post(
            "/auth/login",
            json={"username": "admin", "password": "wrong-password"},
        )
        assert blocked_check.status_code == 429

    other_ip_transport = ASGITransport(app=app, client=("10.0.0.4", 45678))
    async with AsyncClient(
        transport=other_ip_transport, base_url="http://test"
    ) as other_client:
        other_ip_response = await other_client.post(
            "/auth/login",
            json={"username": "admin", "password": "wrong-password"},
        )
        assert other_ip_response.status_code == 401


@pytest.mark.asyncio
async def test_block_remaining_seconds_rounds_up_subsecond(monkeypatch):
    auth_api._reset_login_throttle_state()
    auth_api._login_blocked_until["10.0.0.9"] = 1001.2
    monkeypatch.setattr(auth_api.time, "time", lambda: 1000.0)

    remaining = await auth_api._get_block_remaining_seconds("10.0.0.9")

    assert remaining == 2
