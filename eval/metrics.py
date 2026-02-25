from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import numpy as np


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), p))


def summarize_latency_ms(latencies_ms: List[float]) -> Dict[str, float]:
    return {
        "p50_ms": percentile(latencies_ms, 50),
        "p90_ms": percentile(latencies_ms, 90),
        "p95_ms": percentile(latencies_ms, 95),
        "p99_ms": percentile(latencies_ms, 99),
    }


def aggregate_request_metrics(records: List[Dict[str, object]], duration_s: float) -> Dict[str, float]:
    duration_s = max(duration_s, 1e-9)
    latencies = [_to_float(r.get("latency_ms")) for r in records]
    statuses = [int(_to_float(r.get("http_status"))) for r in records]
    completion_tokens = [_to_float(r.get("completion_tokens")) for r in records]

    total = len(records)
    success = sum(1 for status in statuses if status == 200)
    error_count = total - success
    total_completion_tokens = float(sum(completion_tokens))

    latency_summary = summarize_latency_ms(latencies)

    return {
        "request_count": total,
        "success_count": success,
        "error_count": error_count,
        "error_rate": (error_count / total) if total else 0.0,
        "rps": total / duration_s,
        "completion_tokens_per_sec": total_completion_tokens / duration_s,
        "total_completion_tokens": total_completion_tokens,
        **latency_summary,
    }


def p95_by_prompt_bucket(records: List[Dict[str, object]], bucket: str) -> float:
    latencies = [
        _to_float(r.get("latency_ms"))
        for r in records
        if str(r.get("prompt_bucket")) == bucket and int(_to_float(r.get("http_status"))) == 200
    ]
    return percentile(latencies, 95)


def latency_drift_percent(records: List[Dict[str, object]]) -> float:
    if len(records) < 10:
        return 0.0

    def parse_ts(ts: object) -> datetime:
        value = str(ts)
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)

    sorted_records = sorted(records, key=lambda r: parse_ts(r.get("start_time", "")))
    n = len(sorted_records)
    window = max(1, int(n * 0.1))

    first_window = sorted_records[:window]
    last_window = sorted_records[-window:]

    first_avg = float(np.mean([_to_float(r.get("latency_ms")) for r in first_window]))
    last_avg = float(np.mean([_to_float(r.get("latency_ms")) for r in last_window]))

    if first_avg <= 0:
        return 0.0
    return ((last_avg - first_avg) / first_avg) * 100.0


def memory_drift_percent_vram(
    total_mem_by_ts: List[Dict[str, float]],
    total_vram_mb: float,
    soak_duration_s: float,
) -> float:
    if not total_mem_by_ts or total_vram_mb <= 0:
        return 0.0

    total_mem_by_ts = sorted(total_mem_by_ts, key=lambda x: x["epoch_s"])
    start = total_mem_by_ts[0]["epoch_s"]
    end = total_mem_by_ts[-1]["epoch_s"]
    duration = max(soak_duration_s, end - start)

    first_window_s = min(300.0, duration / 2.0)
    last_window_s = min(300.0, duration / 2.0)

    first_vals = [
        item["total_mem_used_mb"]
        for item in total_mem_by_ts
        if item["epoch_s"] <= start + first_window_s
    ]
    last_vals = [
        item["total_mem_used_mb"]
        for item in total_mem_by_ts
        if item["epoch_s"] >= end - last_window_s
    ]

    if not first_vals or not last_vals:
        return 0.0

    first_avg = float(np.mean(first_vals))
    last_avg = float(np.mean(last_vals))
    drift_mb = last_avg - first_avg

    return (drift_mb / total_vram_mb) * 100.0
