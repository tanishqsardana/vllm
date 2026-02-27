from __future__ import annotations

from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

LATENCY_BUCKETS = (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40, 80)


class GatewayMetrics:
    def __init__(self, tenant_labels_enabled: bool = True) -> None:
        self.tenant_labels_enabled = tenant_labels_enabled
        self.registry = CollectorRegistry()

        self.gateway_http_requests_total = Counter(
            "gateway_http_requests_total",
            "Total HTTP requests handled by gateway.",
            ["route", "method", "status"],
            registry=self.registry,
        )
        self.gateway_http_request_duration_seconds = Histogram(
            "gateway_http_request_duration_seconds",
            "Gateway HTTP request duration.",
            ["route", "method"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.gateway_inflight_requests = Gauge(
            "gateway_inflight_requests",
            "In-flight HTTP requests in gateway.",
            registry=self.registry,
        )

        self.gateway_upstream_requests_total = Counter(
            "gateway_upstream_requests_total",
            "Gateway upstream requests to vLLM.",
            ["status"],
            registry=self.registry,
        )
        self.gateway_upstream_request_duration_seconds = Histogram(
            "gateway_upstream_request_duration_seconds",
            "Gateway upstream request duration.",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.gateway_engine_healthy = Gauge(
            "gateway_engine_healthy",
            "Engine health status as observed by gateway (1 healthy, 0 unhealthy).",
            registry=self.registry,
        )

        rejection_labels = ["reason", "tenant_id", "tenant_name"] if self.tenant_labels_enabled else ["reason"]
        self.gateway_rejections_total = Counter(
            "gateway_rejections_total",
            "Rejected or policy-enforced requests by reason.",
            rejection_labels,
            registry=self.registry,
        )

        token_labels = ["tenant_id", "tenant_name"] if self.tenant_labels_enabled else []
        self.gateway_prompt_tokens_total = Counter(
            "gateway_prompt_tokens_total",
            "Total prompt tokens accounted by gateway.",
            token_labels,
            registry=self.registry,
        )
        self.gateway_completion_tokens_total = Counter(
            "gateway_completion_tokens_total",
            "Total completion tokens accounted by gateway.",
            token_labels,
            registry=self.registry,
        )
        self.gateway_total_tokens_total = Counter(
            "gateway_total_tokens_total",
            "Total tokens accounted by gateway.",
            token_labels,
            registry=self.registry,
        )

        self.gateway_db_write_failures_total = Counter(
            "gateway_db_write_failures_total",
            "Number of request-accounting DB write failures.",
            registry=self.registry,
        )
        self.gateway_db_latency_seconds = Histogram(
            "gateway_db_latency_seconds",
            "Latency for writing request accounting records to DB.",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.budget_events_total = Counter(
            "budget_events_total",
            "Total budget threshold alert events.",
            ["threshold", "tenant_id"],
            registry=self.registry,
        )
        self.tenant_cost_estimated_usd = Gauge(
            "tenant_cost_estimated_usd",
            "Estimated tenant cost over the last 24h.",
            ["tenant_id"],
            registry=self.registry,
        )

        self.gpu_utilization_percent = Gauge(
            "gpu_utilization_percent",
            "GPU utilization percent per GPU.",
            ["gpu_index"],
            registry=self.registry,
        )
        self.gpu_memory_used_bytes = Gauge(
            "gpu_memory_used_bytes",
            "GPU memory used bytes per GPU.",
            ["gpu_index"],
            registry=self.registry,
        )
        self.gpu_memory_total_bytes = Gauge(
            "gpu_memory_total_bytes",
            "GPU memory total bytes per GPU.",
            ["gpu_index"],
            registry=self.registry,
        )
        self.gpu_power_watts = Gauge(
            "gpu_power_watts",
            "GPU power draw in watts per GPU.",
            ["gpu_index"],
            registry=self.registry,
        )
        self.gpu_temperature_celsius = Gauge(
            "gpu_temperature_celsius",
            "GPU temperature in celsius per GPU.",
            ["gpu_index"],
            registry=self.registry,
        )
        self._gpu_indices: set[str] = set()

    def render(self) -> bytes:
        return generate_latest(self.registry)

    def response(self) -> Response:
        return Response(content=self.render(), media_type=CONTENT_TYPE_LATEST)

    def inc_http_requests(self, route: str, method: str, status: int) -> None:
        self.gateway_http_requests_total.labels(route=route, method=method, status=str(status)).inc()

    def observe_http_duration(self, route: str, method: str, duration_seconds: float) -> None:
        self.gateway_http_request_duration_seconds.labels(route=route, method=method).observe(duration_seconds)

    def inc_inflight(self) -> None:
        self.gateway_inflight_requests.inc()

    def dec_inflight(self) -> None:
        self.gateway_inflight_requests.dec()

    def observe_upstream(self, status: str, duration_seconds: float) -> None:
        self.gateway_upstream_requests_total.labels(status=status).inc()
        self.gateway_upstream_request_duration_seconds.observe(duration_seconds)

    def set_engine_healthy(self, healthy: bool) -> None:
        self.gateway_engine_healthy.set(1 if healthy else 0)

    def inc_rejection(self, reason: str, tenant_id: str | None = None, tenant_name: str | None = None) -> None:
        if self.tenant_labels_enabled:
            self.gateway_rejections_total.labels(
                reason=reason,
                tenant_id=tenant_id or "unknown",
                tenant_name=tenant_name or "unknown",
            ).inc()
        else:
            self.gateway_rejections_total.labels(reason=reason).inc()

    def add_tokens(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        tenant_id: str | None,
        tenant_name: str | None,
    ) -> None:
        if self.tenant_labels_enabled:
            id_key = tenant_id or "unknown"
            name_key = tenant_name or "unknown"
            self.gateway_prompt_tokens_total.labels(tenant_id=id_key, tenant_name=name_key).inc(max(0, prompt_tokens))
            self.gateway_completion_tokens_total.labels(tenant_id=id_key, tenant_name=name_key).inc(
                max(0, completion_tokens)
            )
            self.gateway_total_tokens_total.labels(tenant_id=id_key, tenant_name=name_key).inc(max(0, total_tokens))
            return

        self.gateway_prompt_tokens_total.inc(max(0, prompt_tokens))
        self.gateway_completion_tokens_total.inc(max(0, completion_tokens))
        self.gateway_total_tokens_total.inc(max(0, total_tokens))

    def observe_db_write(self, duration_seconds: float, ok: bool) -> None:
        self.gateway_db_latency_seconds.observe(duration_seconds)
        if not ok:
            self.gateway_db_write_failures_total.inc()

    def inc_budget_event(self, threshold: str, tenant_id: str) -> None:
        self.budget_events_total.labels(threshold=threshold, tenant_id=tenant_id).inc()

    def set_tenant_cost_estimated_usd(self, tenant_id: str, cost_usd: float) -> None:
        self.tenant_cost_estimated_usd.labels(tenant_id=tenant_id).set(max(0.0, float(cost_usd)))

    def set_gpu_metrics(self, gpu_samples: list[dict[str, Any]]) -> None:
        current = {str(sample["gpu_index"]) for sample in gpu_samples}

        for removed_index in (self._gpu_indices - current):
            self.gpu_utilization_percent.remove(removed_index)
            self.gpu_memory_used_bytes.remove(removed_index)
            self.gpu_memory_total_bytes.remove(removed_index)
            self.gpu_power_watts.remove(removed_index)
            self.gpu_temperature_celsius.remove(removed_index)

        self._gpu_indices = current

        for sample in gpu_samples:
            idx = str(sample["gpu_index"])
            self.gpu_utilization_percent.labels(gpu_index=idx).set(float(sample.get("utilization_percent", 0.0)))
            self.gpu_memory_used_bytes.labels(gpu_index=idx).set(float(sample.get("memory_used_bytes", 0.0)))
            self.gpu_memory_total_bytes.labels(gpu_index=idx).set(float(sample.get("memory_total_bytes", 0.0)))
            self.gpu_power_watts.labels(gpu_index=idx).set(float(sample.get("power_watts", 0.0)))
            self.gpu_temperature_celsius.labels(gpu_index=idx).set(float(sample.get("temperature_celsius", 0.0)))
