"""
Authentication API

Enables admin authentication when both ADMIN_USERNAME and ADMIN_PASSWORD are set:
- POST /auth/login: Exchange username and password for a token
- GET /auth/status: Check if enabled and authenticated
"""

from collections import deque
import asyncio
import time

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.common.admin_auth import create_admin_token, is_admin_auth_enabled, verify_admin_token
from app.config import get_settings


router = APIRouter(prefix="/auth", tags=["Auth"])

# Login brute-force mitigation (per client IP, in-memory).
_MAX_LOGIN_ATTEMPTS = 5
_LOGIN_WINDOW_SECONDS = 600
_LOGIN_BLOCK_SECONDS = 900

_login_failures: dict[str, deque[float]] = {}
_login_blocked_until: dict[str, float] = {}
_login_throttle_lock = asyncio.Lock()


class AuthStatusResponse(BaseModel):
    enabled: bool
    authenticated: bool


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip() or None
    return authorization.strip() or None


def _extract_client_ip(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _prune_login_throttle_state(now_ts: float) -> None:
    cutoff = now_ts - _LOGIN_WINDOW_SECONDS

    for client_ip, attempts in list(_login_failures.items()):
        while attempts and attempts[0] <= cutoff:
            attempts.popleft()
        if not attempts:
            _login_failures.pop(client_ip, None)

    for client_ip, blocked_until in list(_login_blocked_until.items()):
        if blocked_until <= now_ts:
            _login_blocked_until.pop(client_ip, None)


async def _get_block_remaining_seconds(client_ip: str) -> int:
    now_ts = time.time()
    async with _login_throttle_lock:
        _prune_login_throttle_state(now_ts)
        blocked_until = _login_blocked_until.get(client_ip)
        if not blocked_until:
            return 0
        remaining = int(blocked_until - now_ts)
        return max(1, remaining) if remaining > 0 else 0


async def _record_login_failure(client_ip: str) -> int:
    now_ts = time.time()
    async with _login_throttle_lock:
        _prune_login_throttle_state(now_ts)
        attempts = _login_failures.setdefault(client_ip, deque())
        attempts.append(now_ts)
        if len(attempts) >= _MAX_LOGIN_ATTEMPTS:
            _login_blocked_until[client_ip] = now_ts + _LOGIN_BLOCK_SECONDS
            _login_failures.pop(client_ip, None)
            return _LOGIN_BLOCK_SECONDS
        return 0


async def _clear_login_failure_state(client_ip: str) -> None:
    async with _login_throttle_lock:
        _login_failures.pop(client_ip, None)
        _login_blocked_until.pop(client_ip, None)


def _reset_login_throttle_state() -> None:
    """Test helper to clear in-memory login throttle state."""
    _login_failures.clear()
    _login_blocked_until.clear()


@router.get("/status", response_model=AuthStatusResponse)
async def auth_status(
    authorization: str = Header(None, description="Bearer token"),
    x_admin_token: str = Header(None, description="Admin token", alias="x-admin-token"),
):
    settings = get_settings()
    enabled = is_admin_auth_enabled(settings.ADMIN_USERNAME, settings.ADMIN_PASSWORD)
    if not enabled:
        return AuthStatusResponse(enabled=False, authenticated=True)

    token = x_admin_token or _extract_bearer_token(authorization)
    authenticated = bool(
        token
        and verify_admin_token(
            token=token,
            admin_username=settings.ADMIN_USERNAME or "",
            admin_password=settings.ADMIN_PASSWORD or "",
        )
    )
    return AuthStatusResponse(enabled=True, authenticated=authenticated)


@router.post("/login", response_model=LoginResponse)
async def login(data: LoginRequest, request: Request):
    settings = get_settings()
    if not is_admin_auth_enabled(settings.ADMIN_USERNAME, settings.ADMIN_PASSWORD):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin authentication is not enabled",
        )

    client_ip = _extract_client_ip(request)
    blocked_remaining = await _get_block_remaining_seconds(client_ip)
    if blocked_remaining > 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Please try again later.",
            headers={"Retry-After": str(blocked_remaining)},
        )

    if data.username != settings.ADMIN_USERNAME or data.password != settings.ADMIN_PASSWORD:
        blocked_for = await _record_login_failure(client_ip)
        if blocked_for > 0:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed login attempts. Please try again later.",
                headers={"Retry-After": str(blocked_for)},
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    await _clear_login_failure_state(client_ip)

    token = create_admin_token(
        admin_username=settings.ADMIN_USERNAME or "",
        admin_password=settings.ADMIN_PASSWORD or "",
        ttl_seconds=settings.ADMIN_TOKEN_TTL_SECONDS,
    )
    return LoginResponse(
        access_token=token,
        expires_in=settings.ADMIN_TOKEN_TTL_SECONDS,
    )


@router.post("/logout")
async def logout():
    return {"ok": True}
