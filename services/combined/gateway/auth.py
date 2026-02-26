from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Any

from fastapi import HTTPException

from .db import Database


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def generate_api_key() -> str:
    return f"cp_{secrets.token_urlsafe(32)}"


def parse_bearer_token(authorization_header: str | None) -> str:
    if not authorization_header:
        raise HTTPException(
            status_code=401,
            detail={"error": {"type": "unauthorized", "message": "missing Authorization header"}},
        )
    parts = authorization_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPException(
            status_code=401,
            detail={"error": {"type": "unauthorized", "message": "invalid bearer token"}},
        )
    return parts[1].strip()


def authenticate_tenant(api_key: str, db: Database) -> dict[str, Any] | None:
    candidate_hash = hash_api_key(api_key)
    rows = db.list_tenant_auth_rows()

    matched_row: dict[str, Any] | None = None
    for row in rows:
        if hmac.compare_digest(candidate_hash, row["api_key_hash"]):
            matched_row = row

    if matched_row is None:
        return None

    return {
        "tenant_id": matched_row["tenant_id"],
        "tenant_name": matched_row["tenant_name"],
        "max_concurrent": int(matched_row["max_concurrent"]),
        "rpm_limit": int(matched_row["rpm_limit"]),
        "tpm_limit": int(matched_row["tpm_limit"]),
        "max_context_tokens": int(matched_row["max_context_tokens"]),
        "max_output_tokens": int(matched_row["max_output_tokens"]),
    }


def verify_admin_token(configured_token: str | None, provided_token: str | None) -> None:
    if not configured_token:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "type": "admin_unavailable",
                    "message": "admin endpoints unavailable: ADMIN_TOKEN is not configured",
                }
            },
        )

    if not provided_token:
        raise HTTPException(
            status_code=401,
            detail={"error": {"type": "unauthorized", "message": "missing X-Admin-Token header"}},
        )

    if not hmac.compare_digest(configured_token, provided_token):
        raise HTTPException(
            status_code=401,
            detail={"error": {"type": "unauthorized", "message": "invalid admin token"}},
        )

