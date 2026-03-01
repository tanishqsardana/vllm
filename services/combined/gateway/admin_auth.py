from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request
from jwt import InvalidTokenError, PyJWKClient, decode

from .auth import parse_bearer_token


@dataclass
class AdminPrincipal:
    identity: str
    mode: str
    claims: dict[str, Any] | None = None


class AdminAuthProvider:
    def __init__(
        self,
        *,
        mode: str,
        admin_token: str | None,
        jwks_url: str | None,
        issuer: str | None,
        audience: str | None,
        role_claim: str,
        groups_claim: str,
        admin_group: str | None,
    ) -> None:
        self.mode = mode
        self.admin_token = admin_token
        self.jwks_url = jwks_url
        self.issuer = issuer
        self.audience = audience
        self.role_claim = role_claim or "role"
        self.groups_claim = groups_claim or "groups"
        self.admin_group = admin_group
        self._jwks_client: PyJWKClient | None = None

        if self.mode == "oidc" and self.jwks_url:
            self._jwks_client = PyJWKClient(self.jwks_url)

    def auth_info(self) -> dict[str, Any]:
        return {
            "admin_auth_mode": self.mode,
            "requirements": {
                "needs_admin_token": self.mode == "static_token",
                "needs_jwt": self.mode == "oidc",
                "jwks_url_set": bool(self.jwks_url),
                "issuer_set": bool(self.issuer),
                "audience_set": bool(self.audience),
            },
        }

    def authenticate(self, request: Request) -> AdminPrincipal:
        if self.mode == "static_token":
            return self._authenticate_static(request)
        if self.mode == "oidc":
            return self._authenticate_oidc(request)

        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "type": "admin_unavailable",
                    "message": f"admin auth mode is unsupported: {self.mode}",
                }
            },
        )

    def _authenticate_static(self, request: Request) -> AdminPrincipal:
        if not self.admin_token:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "type": "admin_unavailable",
                        "message": "admin endpoints unavailable: ADMIN_TOKEN is not configured",
                    }
                },
            )

        provided = request.headers.get("x-admin-token")
        if not provided:
            raise HTTPException(
                status_code=401,
                detail={"error": {"type": "unauthorized", "message": "missing X-Admin-Token header"}},
            )

        if not hmac.compare_digest(self.admin_token, provided):
            raise HTTPException(
                status_code=401,
                detail={"error": {"type": "unauthorized", "message": "invalid admin token"}},
            )

        return AdminPrincipal(identity="static_admin", mode="static_token", claims=None)

    def _authenticate_oidc(self, request: Request) -> AdminPrincipal:
        missing: list[str] = []
        if not self.jwks_url:
            missing.append("JWKS_URL")
        if not self.issuer:
            missing.append("OIDC_ISSUER")
        if not self.audience:
            missing.append("OIDC_AUDIENCE")

        if missing:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "type": "admin_unavailable",
                        "message": f"oidc admin auth unavailable: missing {', '.join(missing)}",
                    }
                },
            )

        auth_header = request.headers.get("authorization")
        token = parse_bearer_token(auth_header)

        try:
            if self._jwks_client is None:
                self._jwks_client = PyJWKClient(str(self.jwks_url))
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            claims = decode(
                token,
                signing_key.key,
                algorithms=["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"],
                audience=self.audience,
                issuer=self.issuer,
            )
        except InvalidTokenError as exc:
            raise HTTPException(
                status_code=401,
                detail={"error": {"type": "unauthorized", "message": f"invalid admin jwt: {exc}"}},
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=401,
                detail={"error": {"type": "unauthorized", "message": f"admin jwt validation failed: {exc}"}},
            ) from exc

        has_admin_role = "admin" in _to_lower_set(claims.get(self.role_claim))

        has_admin_group = False
        if self.admin_group:
            has_admin_group = self.admin_group.strip().lower() in _to_lower_set(claims.get(self.groups_claim))

        if not has_admin_role and not has_admin_group:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": {
                        "type": "forbidden",
                        "message": (
                            "admin jwt is valid but missing required admin role/group "
                            f"({self.role_claim}=admin or {self.groups_claim} contains {self.admin_group})"
                        ),
                    }
                },
            )

        identity = str(claims.get("email") or claims.get("sub") or "unknown_admin")
        return AdminPrincipal(identity=identity, mode="oidc", claims=claims)


def _to_lower_set(value: Any) -> set[str]:
    out: set[str] = set()

    if value is None:
        return out

    if isinstance(value, str):
        split_values = [item.strip() for chunk in value.split(",") for item in chunk.split()]
        for item in split_values:
            if item:
                out.add(item.lower())
        return out

    if isinstance(value, (list, tuple, set)):
        for item in value:
            if item is None:
                continue
            text = str(item).strip().lower()
            if text:
                out.add(text)
        return out

    text = str(value).strip().lower()
    if text:
        out.add(text)
    return out
