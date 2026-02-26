# Phase 3 Run Commands

Copy-paste command reference for Phase 3 observability on Runpod.

## 1) Build and Push Gateway+Engine Image

```bash
cd /Users/tanishqsardana/Documents/Startup

docker login

IMAGE_REPO=docker.io/<dockerhub-user>/vllm-combined \
IMAGE_TAG=phase3-observability \
PUSH=1 \
TARGET_PLATFORM=linux/amd64 \
./scripts/build_combined_image.sh
```

## 2) Runpod Pod Env (minimum)

Set in Runpod pod:

- `MODEL_ID=<your-model>`
- `ADMIN_TOKEN=<strong-random-secret>`

Recommended:

- `DB_PATH=/workspace/data/controlplane.db`
- `HF_HOME=/workspace/cache/huggingface`
- `HUGGINGFACE_HUB_CACHE=/workspace/cache/huggingface/hub`
- `TRANSFORMERS_CACHE=/workspace/cache/huggingface/transformers`
- `VLLM_CACHE_ROOT=/workspace/cache/vllm`
- `METRICS_TENANT_LABELS=on`
- `GPU_METRICS_POLL_INTERVAL_SECONDS=2`

Expose port `8000`.

## 3) Basic Verification

```bash
BASE_URL="https://<runpod-proxy-host>"
ADMIN_TOKEN="<admin-token>"

curl -sS "$BASE_URL/livez"
curl -sS "$BASE_URL/healthz"
curl -sS "$BASE_URL/version"
```

## 4) Metrics Endpoint Verification

```bash
# should return 401
curl -i "$BASE_URL/metrics"

# should return Prometheus text output
curl -sS -H "X-Admin-Token: $ADMIN_TOKEN" "$BASE_URL/metrics" | head

# core metrics sanity
curl -sS -H "X-Admin-Token: $ADMIN_TOKEN" "$BASE_URL/metrics" \
  | rg "gateway_http_requests_total|gateway_rejections_total|gateway_engine_healthy|^gpu_"
```

## 5) Create Tenants and Set Limits

```bash
ADMIN_TOKEN=$ADMIN_TOKEN BASE_URL=$BASE_URL ./scripts/create_tenant.sh tenant-a
ADMIN_TOKEN=$ADMIN_TOKEN BASE_URL=$BASE_URL ./scripts/create_tenant.sh tenant-b

ADMIN_TOKEN=$ADMIN_TOKEN BASE_URL=$BASE_URL ./scripts/set_tenant_limits.sh tenant-a pro
ADMIN_TOKEN=$ADMIN_TOKEN BASE_URL=$BASE_URL ./scripts/set_tenant_limits.sh tenant-b pro
```

## 6) Prometheus + Grafana Monitoring Pack (outside Runpod)

Edit `monitoring/prometheus/prometheus.yml`:

- `targets`: Runpod proxy host only (no `https://`)
- `X-Admin-Token` value

Then run:

```bash
docker compose -f monitoring/compose.yaml up -d --build
```

Open:

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/admin)

Check target status:

- `http://localhost:9090/targets` must show `runpod_gateway` as `UP`.

## 7) Multi-Tenant Load Test

```bash
BASE_URL="https://<runpod-proxy-host>"
MODEL_ID="<model-id-from-/version>"
TENANT_A_KEY="<tenant-a-api-key>"
TENANT_B_KEY="<tenant-b-api-key>"

python3 old/scripts/throughput_benchmark.py \
  --base-url "$BASE_URL" \
  --api-key "$TENANT_A_KEY" \
  --models "$MODEL_ID" \
  --requests 100 \
  --concurrency 2 \
  --max-tokens 128 \
  --retries 3 \
  --retry-backoff-seconds 0.4 \
  --output-csv /tmp/tenant_a.csv \
  --prompt "Tenant A load test." &

python3 old/scripts/throughput_benchmark.py \
  --base-url "$BASE_URL" \
  --api-key "$TENANT_B_KEY" \
  --models "$MODEL_ID" \
  --requests 100 \
  --concurrency 2 \
  --max-tokens 128 \
  --retries 3 \
  --retry-backoff-seconds 0.4 \
  --output-csv /tmp/tenant_b.csv \
  --prompt "Tenant B load test." &

wait
```

## 8) Grafana Usage

Dashboard: `vLLM Control Plane Observability`

- Use `tenant_name` variable (multi-select + All) to filter token/rejection panels.
- Use time range `Last 15m` or `Last 30m` while testing.

## 9) Troubleshooting

If Grafana dashboard is stale after updates:

```bash
docker compose -f monitoring/compose.yaml down -v
docker compose -f monitoring/compose.yaml build --no-cache grafana
docker compose -f monitoring/compose.yaml up -d
```

If Prometheus has no data:

1. Verify `/metrics` with curl and admin token.
2. Verify `monitoring/prometheus/prometheus.yml` target/token.
3. Restart Prometheus:

```bash
docker compose -f monitoring/compose.yaml restart prometheus
```

4. Re-check `http://localhost:9090/targets`.
