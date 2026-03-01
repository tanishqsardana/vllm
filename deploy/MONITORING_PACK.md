# Monitoring Pack (Prometheus + Grafana outside gateway container)

Run monitoring stack outside the serving container. Scrape gateway `/metrics` on port `8000`.

## Metrics Auth

- In `static_token` mode, Prometheus must send `X-Admin-Token`.
- In `oidc` mode, send `Authorization: Bearer <admin-jwt>`.

For demo simplicity, static mode is recommended for scrape auth.

## Prometheus Scrape Example (static token)

```yaml
scrape_configs:
  - job_name: "vllm_gateway"
    metrics_path: /metrics
    static_configs:
      - targets: ["gateway-host:8000"]
    authorization: {}
    headers:
      X-Admin-Token: "replace-with-admin-token"
```

If your Prometheus build does not support `headers`, use a reverse proxy (or service discovery relabeling approach) to inject auth headers.

## Grafana

- Point Grafana datasource to your Prometheus instance.
- Import existing dashboard JSON from `monitoring/grafana/dashboards/`.

## Quick Check

```bash
curl -sS -H "X-Admin-Token: <ADMIN_TOKEN>" http://gateway-host:8000/metrics | head
```
