"""Scrape and parse Dynamo frontend + worker Prometheus metrics into structured JSON."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .utils import json_log


_METRIC_LINE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)'
    r'(?:\{([^}]*)\})?'
    r'\s+(\S+)'
    r'(?:\s+\S+)?$'
)

_LABEL_PAIR = re.compile(r'(\w+)="([^"]*)"')


def _parse_prometheus_text(text: str) -> list[dict[str, Any]]:
    """Parse Prometheus exposition format into a list of metric dicts."""
    results: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE.match(line)
        if not match:
            continue
        name = match.group(1)
        raw_labels = match.group(2) or ""
        raw_value = match.group(3)

        labels: dict[str, str] = {}
        for lm in _LABEL_PAIR.finditer(raw_labels):
            labels[lm.group(1)] = lm.group(2)

        try:
            value = float(raw_value)
        except ValueError:
            continue

        results.append({"name": name, "labels": labels, "value": value})
    return results


def _find_metric(metrics: list[dict[str, Any]], name: str, labels: dict[str, str] | None = None) -> float | None:
    """Find first metric matching name and optional label filter."""
    for m in metrics:
        if m["name"] != name:
            continue
        if labels is not None:
            if not all(m["labels"].get(k) == v for k, v in labels.items()):
                continue
        return float(m["value"])
    return None


def _find_all(metrics: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    """Find all metrics whose name starts with prefix."""
    return [m for m in metrics if m["name"].startswith(prefix)]


def _extract_histogram_percentile(
    metrics: list[dict[str, Any]],
    name: str,
    percentile: float,
) -> float | None:
    """Approximate a percentile from histogram buckets."""
    bucket_name = f"{name}_bucket"
    buckets: list[tuple[float, float]] = []
    for m in metrics:
        if m["name"] != bucket_name:
            continue
        le = m["labels"].get("le")
        if le is None:
            continue
        try:
            le_val = float(le) if le != "+Inf" else float("inf")
        except ValueError:
            continue
        buckets.append((le_val, float(m["value"])))

    if not buckets:
        return None

    buckets.sort(key=lambda x: x[0])
    total = buckets[-1][1]
    if total == 0:
        return None

    target = total * percentile
    for le_val, count in buckets:
        if count >= target:
            return le_val
    return buckets[-1][0]


@dataclass
class DynamoMetricsScraper:
    """Scrapes Dynamo frontend and worker metrics endpoints."""

    frontend_url: str = "http://127.0.0.1:8001"
    worker_metrics_url: str = "http://127.0.0.1:8081"
    timeout_seconds: float = 3.0
    _client: httpx.AsyncClient | None = field(default=None, repr=False)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds, connect=2.0)
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _scrape(self, url: str) -> list[dict[str, Any]]:
        client = await self._get_client()
        try:
            resp = await client.get(f"{url}/metrics")
            if resp.status_code != 200:
                return []
            return _parse_prometheus_text(resp.text)
        except (httpx.HTTPError, Exception) as exc:
            json_log("dynamo_scrape_error", url=url, message=str(exc))
            return []

    async def scrape_frontend(self) -> list[dict[str, Any]]:
        return await self._scrape(self.frontend_url)

    async def scrape_worker(self) -> list[dict[str, Any]]:
        return await self._scrape(self.worker_metrics_url)

    async def get_dynamo_status(self) -> dict[str, Any]:
        """Return structured Dynamo topology and health status."""
        frontend_metrics = await self.scrape_frontend()
        worker_metrics = await self.scrape_worker()

        frontend_up = len(frontend_metrics) > 0
        worker_up = len(worker_metrics) > 0

        # Extract model config from frontend registration metrics
        model_config: dict[str, Any] = {}
        context_length = _find_metric(frontend_metrics, "dynamo_frontend_model_context_length")
        if context_length is not None:
            model_config["context_length"] = int(context_length)
        kv_block_size = _find_metric(frontend_metrics, "dynamo_frontend_model_kv_cache_block_size")
        if kv_block_size is not None:
            model_config["kv_cache_block_size"] = int(kv_block_size)
        max_batched = _find_metric(frontend_metrics, "dynamo_frontend_model_max_num_batched_tokens")
        if max_batched is not None:
            model_config["max_num_batched_tokens"] = int(max_batched)

        return {
            "ts": time.time(),
            "frontend": {
                "healthy": frontend_up,
                "url": self.frontend_url,
            },
            "worker": {
                "healthy": worker_up,
                "metrics_url": self.worker_metrics_url,
            },
            "model_config": model_config or None,
        }

    async def get_dynamo_metrics(self) -> dict[str, Any]:
        """Return structured inference metrics from Dynamo frontend + worker."""
        frontend_metrics = await self.scrape_frontend()
        worker_metrics = await self.scrape_worker()

        # --- Frontend metrics ---
        frontend: dict[str, Any] = {}
        inflight = _find_metric(frontend_metrics, "dynamo_frontend_inflight_requests")
        if inflight is not None:
            frontend["inflight_requests"] = int(inflight)

        queued = _find_metric(frontend_metrics, "dynamo_frontend_queued_requests")
        if queued is not None:
            frontend["queued_requests"] = int(queued)

        total_requests = _find_metric(frontend_metrics, "dynamo_frontend_requests_total")
        if total_requests is not None:
            frontend["total_requests"] = int(total_requests)

        disconnected = _find_metric(frontend_metrics, "dynamo_frontend_disconnected_clients")
        if disconnected is not None:
            frontend["disconnected_clients"] = int(disconnected)

        # TTFT percentiles
        ttft_p50 = _extract_histogram_percentile(
            frontend_metrics, "dynamo_frontend_time_to_first_token_seconds", 0.5
        )
        ttft_p95 = _extract_histogram_percentile(
            frontend_metrics, "dynamo_frontend_time_to_first_token_seconds", 0.95
        )
        ttft_p99 = _extract_histogram_percentile(
            frontend_metrics, "dynamo_frontend_time_to_first_token_seconds", 0.99
        )
        if any(v is not None for v in [ttft_p50, ttft_p95, ttft_p99]):
            frontend["ttft_seconds"] = {
                "p50": ttft_p50,
                "p95": ttft_p95,
                "p99": ttft_p99,
            }

        # ITL percentiles
        itl_p50 = _extract_histogram_percentile(
            frontend_metrics, "dynamo_frontend_inter_token_latency_seconds", 0.5
        )
        itl_p95 = _extract_histogram_percentile(
            frontend_metrics, "dynamo_frontend_inter_token_latency_seconds", 0.95
        )
        if any(v is not None for v in [itl_p50, itl_p95]):
            frontend["itl_seconds"] = {
                "p50": itl_p50,
                "p95": itl_p95,
            }

        # Request duration
        dur_p50 = _extract_histogram_percentile(
            frontend_metrics, "dynamo_frontend_request_duration_seconds", 0.5
        )
        dur_p95 = _extract_histogram_percentile(
            frontend_metrics, "dynamo_frontend_request_duration_seconds", 0.95
        )
        if any(v is not None for v in [dur_p50, dur_p95]):
            frontend["request_duration_seconds"] = {
                "p50": dur_p50,
                "p95": dur_p95,
            }

        # Cached tokens
        cached_p50 = _extract_histogram_percentile(
            frontend_metrics, "dynamo_frontend_cached_tokens", 0.5
        )
        if cached_p50 is not None:
            cached_sum = _find_metric(frontend_metrics, "dynamo_frontend_cached_tokens_sum")
            cached_count = _find_metric(frontend_metrics, "dynamo_frontend_cached_tokens_count")
            frontend["cached_tokens"] = {
                "sum": int(cached_sum) if cached_sum is not None else 0,
                "count": int(cached_count) if cached_count is not None else 0,
            }

        # --- Worker / vLLM engine metrics ---
        worker: dict[str, Any] = {}

        # vLLM request success/abort counts
        success_total = 0.0
        for m in worker_metrics:
            if m["name"] == "vllm:request_success_total":
                success_total += m["value"]
        if success_total > 0:
            worker["request_success_total"] = int(success_total)

        # KV cache utilization from vLLM
        gpu_cache_usage = _find_metric(worker_metrics, "vllm:gpu_cache_usage_perc")
        if gpu_cache_usage is not None:
            worker["gpu_cache_usage_percent"] = round(gpu_cache_usage * 100, 2)

        cpu_cache_usage = _find_metric(worker_metrics, "vllm:cpu_cache_usage_perc")
        if cpu_cache_usage is not None:
            worker["cpu_cache_usage_percent"] = round(cpu_cache_usage * 100, 2)

        # Running/waiting requests from vLLM
        num_running = _find_metric(worker_metrics, "vllm:num_requests_running")
        if num_running is not None:
            worker["requests_running"] = int(num_running)

        num_waiting = _find_metric(worker_metrics, "vllm:num_requests_waiting")
        if num_waiting is not None:
            worker["requests_waiting"] = int(num_waiting)

        # Dynamo component metrics
        component_requests = _find_metric(worker_metrics, "dynamo_component_requests_total")
        if component_requests is not None:
            worker["dynamo_component_requests_total"] = int(component_requests)

        component_inflight = _find_metric(worker_metrics, "dynamo_component_inflight_requests")
        if component_inflight is not None:
            worker["dynamo_component_inflight"] = int(component_inflight)

        return {
            "ts": time.time(),
            "frontend": frontend,
            "worker": worker,
        }
