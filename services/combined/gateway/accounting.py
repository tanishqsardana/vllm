from __future__ import annotations

from datetime import datetime, timedelta, timezone
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


def parse_budget_window(window: str) -> str:
    allowed = {"day", "week", "month"}
    if window not in allowed:
        raise ValueError("window must be one of day, week, month")
    return window


def window_start_for_budget(window: str, now: datetime | None = None) -> datetime:
    parsed = parse_budget_window(window)
    ts = now or utc_now_dt()
    ts = ts.astimezone(timezone.utc)
    base = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    if parsed == "day":
        return base
    if parsed == "week":
        return base - timedelta(days=base.weekday())
    return base.replace(day=1)


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
    seat_id: str | None,
    key_id: str | None,
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    status_code: int,
    error_type: str | None,
    request_id: str,
    gpu_seconds_est: float | None,
    cost_est: float | None,
) -> None:
    db.insert_request(
        ts_start=ts_start,
        ts_end=ts_end,
        latency_ms=latency_ms,
        tenant_id=tenant_id,
        seat_id=seat_id,
        key_id=key_id,
        model_id=model_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        status_code=status_code,
        error_type=error_type,
        request_id=request_id,
        gpu_seconds_est=gpu_seconds_est,
        cost_est=cost_est,
    )


def usage_rollup_tenants(db: Database, window: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
    window_seconds = parse_window(window)
    threshold = (utc_now_dt() - timedelta(seconds=window_seconds)).isoformat()
    base_rows = db.list_tenant_usage_base(threshold, tenant_id=tenant_id)
    latencies = db.list_tenant_usage_latencies(threshold, tenant_id=tenant_id)

    out: list[dict[str, Any]] = []
    for row in base_rows:
        current_tenant_id = row["tenant_id"]
        tenant_latencies = latencies.get(current_tenant_id, [])
        out.append(
            {
                "tenant_id": current_tenant_id,
                "tenant_name": row["tenant_name"],
                "requests": int(row["requests"] or 0),
                "errors": int(row["errors"] or 0),
                "prompt_tokens": int(row["prompt_tokens"] or 0),
                "completion_tokens": int(row["completion_tokens"] or 0),
                "total_tokens": int(row["total_tokens"] or 0),
                "p95_latency_ms": percentile(tenant_latencies, 0.95),
                "gpu_seconds_est_sum": float(row["gpu_seconds_est_sum"] or 0.0),
                "cost_est_sum": float(row["cost_est_sum"] or 0.0),
            }
        )
    return out


def usage_rollup_seats(db: Database, window: str, tenant_id: str) -> list[dict[str, Any]]:
    window_seconds = parse_window(window)
    threshold = (utc_now_dt() - timedelta(seconds=window_seconds)).isoformat()
    base_rows = db.list_seat_usage_base(threshold, tenant_id=tenant_id)
    latencies = db.list_seat_usage_latencies(threshold, tenant_id=tenant_id)

    out: list[dict[str, Any]] = []
    for row in base_rows:
        seat_key = row["seat_id"] if row["seat_id"] is not None else "__service__"
        seat_latencies = latencies.get(seat_key, [])
        out.append(
            {
                "seat_id": row["seat_id"],
                "seat_name": row["seat_name"],
                "role": row["role"],
                "requests": int(row["requests"] or 0),
                "errors": int(row["errors"] or 0),
                "prompt_tokens": int(row["prompt_tokens"] or 0),
                "completion_tokens": int(row["completion_tokens"] or 0),
                "total_tokens": int(row["total_tokens"] or 0),
                "p95_latency_ms": percentile(seat_latencies, 0.95),
                "gpu_seconds_est_sum": float(row["gpu_seconds_est_sum"] or 0.0),
                "cost_est_sum": float(row["cost_est_sum"] or 0.0),
            }
        )
    return out


def usage_rollup(db: Database, window: str) -> list[dict[str, Any]]:
    return usage_rollup_tenants(db, window)


def utc_now_dt() -> datetime:
    return datetime.now(timezone.utc)
