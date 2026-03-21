"""Authentication — sign in via OAuth providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ── Types ──────────────────────────────────────────────────────────────

@dataclass
class AuthUser:
    """User information returned after authentication."""

    id: str = ""
    name: str | None = None
    email: str | None = None
    avatar_url: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuthUser:
        return cls(
            id=data.get("id", ""),
            name=data.get("name"),
            email=data.get("email"),
            avatar_url=data.get("avatar_url"),
        )


@dataclass
class AuthResponse:
    """Response from authentication endpoints."""

    token: str = ""
    user: AuthUser | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuthResponse:
        user_data = data.get("user", {})
        return cls(
            token=data.get("token", ""),
            user=AuthUser.from_dict(user_data) if user_data else None,
        )


# ── Functions ─────────────────────────────────────────────────────────

def auth_apple_sync(client: Any, id_token: str, name: str | None = None) -> AuthResponse:
    """Authenticate with Apple Sign-In (sync)."""
    body: dict[str, Any] = {"id_token": id_token}
    if name is not None:
        body["name"] = name
    data, _ = client._do_json("POST", "/qai/v1/auth/apple", body)
    return AuthResponse.from_dict(data)


async def auth_apple_async(client: Any, id_token: str, name: str | None = None) -> AuthResponse:
    """Authenticate with Apple Sign-In (async)."""
    body: dict[str, Any] = {"id_token": id_token}
    if name is not None:
        body["name"] = name
    data, _ = await client._do_json("POST", "/qai/v1/auth/apple", body)
    return AuthResponse.from_dict(data)
