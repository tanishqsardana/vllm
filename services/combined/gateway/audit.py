from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from .db import Database


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def request_id_from_headers(headers: dict[str, str] | Any) -> str:
    candidate = None
    if hasattr(headers, "get"):
        candidate = headers.get("x-request-id")
    if candidate and str(candidate).strip():
        return str(candidate).strip()
    return str(uuid.uuid4())


def log_audit_event(
    *,
    db: Database,
    admin_identity: str,
    action: str,
    resource_type: str,
    resource_id: str,
    details: dict[str, Any] | None,
    request_id: str,
) -> dict[str, Any]:
    event_id = str(uuid.uuid4())
    ts = utc_now_iso()
    details_json = json.dumps(details or {}, separators=(",", ":"), default=str)

    return db.insert_audit_event(
        event_id=event_id,
        ts=ts,
        admin_identity=admin_identity,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details_json=details_json,
        request_id=request_id,
    )
