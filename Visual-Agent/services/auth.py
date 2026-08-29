from __future__ import annotations

import asyncio
from functools import lru_cache

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from education.models import AuthUser
from utils.config import Settings, get_settings

_bearer = HTTPBearer(auto_error=False)
_DEV_USER_ID = "00000000-0000-4000-8000-000000000001"


class SupabaseJWTVerifier:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _verify_sync(self, token: str) -> dict:
        try:
            import jwt
        except ImportError as exc:  # pragma: no cover - deployment configuration
            raise RuntimeError("PyJWT[crypto] is required when auth is enabled") from exc

        jwks_url = (
            f"{self.settings.supabase_url}/auth/v1/.well-known/jwks.json"
        )
        signing_key = jwt.PyJWKClient(jwks_url).get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256", "EdDSA"],
            audience="authenticated",
            options={"require": ["sub", "exp"]},
        )

    async def verify(self, token: str) -> AuthUser:
        try:
            payload = await asyncio.to_thread(self._verify_sync, token)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired access token",
            ) from exc
        email = payload.get("email")
        return AuthUser(
            id=str(payload["sub"]),
            email=email,
            is_admin=bool(
                email and email.lower() in self.settings.admin_emails
            ),
            access_token=token,
        )


@lru_cache(maxsize=1)
def _verifier() -> SupabaseJWTVerifier:
    return SupabaseJWTVerifier(get_settings())


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> AuthUser:
    return await authenticate_access_token(
        credentials.credentials if credentials else None
    )


async def authenticate_access_token(token: str | None) -> AuthUser:
    """Authenticate HTTP or WebSocket clients with identical auth semantics."""
    settings = get_settings()
    if token:
        if not settings.supabase_enabled:
            if settings.auth_required:
                raise HTTPException(status_code=503, detail="Auth is not configured")
            return AuthUser(id=_DEV_USER_ID, email="dev@local", is_admin=True)
        return await _verifier().verify(token)

    if settings.auth_required:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return AuthUser(id=_DEV_USER_ID, email="dev@local", is_admin=True)


async def require_admin(user: AuthUser = Depends(get_current_user)) -> AuthUser:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
