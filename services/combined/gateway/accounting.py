from __future__ import annotations

from datetime import timedelta
from typing import Any

from .db import Database
from .utils import extract_completion_text, percentile


def parse_window(window: str) -> int:
    mapping = {
        "1h": 3600,
        "24h": 24 * 3600,
        "7d": 7 * 24 * 3600,
    }
    if window not in mapping:
        raise ValueError("window must be one of 1h, 24h, 7d")
    return mapping[window]


def extract_usage(
    response_json: dict[str, Any] | None,
    prompt_tokens_estimate: int,
    estimator,
) -> tuple[int, int, int]:
    if isinstance(response_json, dict):
        usage = response_json.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
            return prompt_tokens, completion_tokens, total_tokens

        completion_text = extract_completion_text(response_json)
        completion_tokens = estimator.approx_tokens(completion_text)
        total_tokens = prompt_tokens_estimate + completion_tokens
        return prompt_tokens_estimate, completion_tokens, total_tokens

    return prompt_tokens_estimate, 0, prompt_tokens_estimate


def record_request(
    db: Database,
    ts_start: str,
    ts_end: str,
    latency_ms: int,
    tenant_id: str,
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    status_code: int,
    error_type: str | None,
    request_id: str,
) -> None:
    db.insert_request(
        ts_start=ts_start,
        ts_end=ts_end,
        latency_ms=latency_ms,
        tenant_id=tenant_id,
        model_id=model_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        status_code=status_code,
        error_type=error_type,
        request_id=request_id,
    )


def usage_rollup(db: Database, window: str) -> list[dict[str, Any]]:
    window_seconds = parse_window(window)

    threshold = (utc_now_dt() - timedelta(seconds=window_seconds)).isoformat()
    base_rows = db.list_usage_base(threshold)
    latencies = db.list_usage_latencies(threshold)

    out: list[dict[str, Any]] = []
    for row in base_rows:
        tenant_id = row["tenant_id"]
        tenant_latencies = latencies.get(tenant_id, [])
        out.append(
            {
                "tenant_id": tenant_id,
                "tenant_name": row["tenant_name"],
                "requests": int(row["requests"] or 0),
                "errors": int(row["errors"] or 0),
                "total_tokens": int(row["total_tokens"] or 0),
                "p50_latency_ms": percentile(tenant_latencies, 0.50),
                "p95_latency_ms": percentile(tenant_latencies, 0.95),
            }
        )

    return out


def utc_now_dt():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)
