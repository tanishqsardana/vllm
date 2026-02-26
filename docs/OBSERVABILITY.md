# Phase 3 Observability

Phase 3 adds Prometheus metrics to the Phase 2 gateway and provides an optional monitoring pack (Prometheus + Grafana) that runs outside the Runpod container.

Quick command reference:

- [PHASE3_RUN_COMMANDS.md](/Users/tanishqsardana/Documents/Startup/docs/PHASE3_RUN_COMMANDS.md)

## What Was Added

- Gateway metrics endpoint: `GET /metrics`
- HTTP, upstream, policy, token, DB, and GPU metrics
- `/metrics` protection with `X-Admin-Token` (same token as admin endpoints)
- Prometheus sample config with scrape headers
- Grafana provisioning (datasource + dashboard JSON)
- Optional `monitoring/compose.yaml` for local monitoring stack

## Metrics Endpoint Security

`/metrics` requires `X-Admin-Token: <ADMIN_TOKEN>`.

Behavior:

- Missing or wrong header: `401` with JSON error
- If `ADMIN_TOKEN` is not configured on gateway: `503` with `admin_unavailable`

Examples:

```bash
# should return 401
curl -i http://localhost:8000/metrics

# should return 200 text/plain; version=0.0.4
curl -sS -H "X-Admin-Token: $ADMIN_TOKEN" http://localhost:8000/metrics | head
```

## Prometheus Header Scraping

Prometheus can send custom headers via `http_headers` in `scrape_config`.

Important requirement:

- Prometheus version must be **>= 2.55** for this header config.

Sample config is provided in:

- `monitoring/prometheus/prometheus.yml`

Update these values before running:

- `targets`: set your Runpod proxy hostname (without `https://`)
- `ADMIN_TOKEN`: replace `${ADMIN_TOKEN}` with your real token (or generate config from template in your deployment process)

## Run Optional Monitoring Pack

From repo root:

```bash
docker compose -f monitoring/compose.yaml up -d --build
```

Endpoints:

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (`admin` / `admin`)

Grafana provisioning files:

- Datasource: `monitoring/grafana/provisioning/datasources/datasource.yml`
- Dashboard provider: `monitoring/grafana/provisioning/dashboards/dashboards.yml`
- Dashboard JSON: `monitoring/grafana/dashboards/vllm_controlplane_dashboard.json`

## Dashboard Coverage

Dashboard sections:

1. Health
- `gateway_engine_healthy`
- `gateway_inflight_requests`

2. Traffic + Latency
- RPS: `sum(rate(gateway_http_requests_total[1m]))`
- Gateway p95 latency from `gateway_http_request_duration_seconds_bucket`
- Upstream p95 latency from `gateway_upstream_request_duration_seconds_bucket`

3. Tokens
- `gateway_completion_tokens_total`
- `gateway_total_tokens_total`

4. Rejections
- `gateway_rejections_total` by reason
- `gateway_rejections_total` by tenant_name

5. GPU
- `gpu_utilization_percent`
- `gpu_memory_used_bytes / gpu_memory_total_bytes`
- `gpu_power_watts`
- `gpu_temperature_celsius`

Dashboard variable:

- `tenant_name` dropdown (supports multi-select + `All`) to filter token and rejection panels per tenant.

## Tenant Label Toggle

Env var:

- `METRICS_TENANT_LABELS=on|off` (default `on`)

When `off`:

- Tenant label is omitted from token/rejection metrics.
- Use this if you want lower metric cardinality.

## GPU Telemetry Backend

Polling interval:

- `GPU_METRICS_POLL_INTERVAL_SECONDS` (default `2`)

Collection order:

1. NVML via `pynvml`
2. Fallback to `nvidia-smi` CSV parsing

If neither is available, GPU metrics may be absent.

## Acceptance Checks

1. Metrics auth works

```bash
curl -i "$BASE_URL/metrics"
curl -sS -H "X-Admin-Token: $ADMIN_TOKEN" "$BASE_URL/metrics" | head
```

2. Core metrics present

```bash
curl -sS -H "X-Admin-Token: $ADMIN_TOKEN" "$BASE_URL/metrics" | rg "gateway_http_requests_total|gateway_rejections_total|gateway_engine_healthy"
```

3. GPU metrics present (if GPU visible)

```bash
curl -sS -H "X-Admin-Token: $ADMIN_TOKEN" "$BASE_URL/metrics" | rg "^gpu_"
```

4. Grafana provisioning loaded

- Start stack with `monitoring/compose.yaml`
- Open Grafana and verify dashboard `vLLM Control Plane Observability` is available automatically
