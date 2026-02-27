# Economics Observability (Phase 4 MVP)

Phase 4 adds economic observability to the gateway without billing enforcement.

## What `gpu_seconds_est` means

`gpu_seconds_est` is a gateway-side estimate of GPU time per request.

- For successful requests, the gateway records:
  - `gpu_seconds_est = latency_ms / 1000.0`
- For rejected requests (`4xx`/`429`) the gateway stores `0` for `gpu_seconds_est` and `cost_est`.

This is an estimate for operations and budgeting, not true GPU kernel runtime.

## Cost estimation configuration

Cost estimation is controlled by:

- `GPU_HOURLY_RATE` (USD/hour, float)

Per successful request:

- `cost_est = gpu_seconds_est * (GPU_HOURLY_RATE / 3600.0)`

If `GPU_HOURLY_RATE=0` (default):

- `gpu_seconds_est` is still recorded for successful requests.
- `cost_est` is `0`.
- API responses that include cost rollups expose `cost_estimation_enabled: false`.

## Budgets are alerts only

Budgets are soft alerts. The gateway **does not block inference** when a budget is exceeded.

Threshold events are logged at 50%, 80%, and 100% of budget (if enabled).

## Admin API examples

Assume:

- `BASE_URL=http://localhost:8000`
- `ADMIN_TOKEN=<admin-token>`
- `TENANT_ID=<tenant-id>`

Create a seat:

```bash
curl -sS -X POST "$BASE_URL/admin/seats" \
  -H "X-Admin-Token: $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"'$TENANT_ID'","seat_name":"alice@example.com","role":"user"}'
```

Create a key for a seat:

```bash
curl -sS -X POST "$BASE_URL/admin/keys" \
  -H "X-Admin-Token: $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"'$TENANT_ID'","seat_id":"<seat-id>"}'
```

Set budget:

```bash
curl -sS -X PUT "$BASE_URL/admin/budgets/$TENANT_ID" \
  -H "X-Admin-Token: $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"window":"day","budget_usd":5.0}'
```

Seat usage rollup:

```bash
curl -sS "$BASE_URL/admin/usage/seats?window=24h&tenant_id=$TENANT_ID" \
  -H "X-Admin-Token: $ADMIN_TOKEN"
```

Tenant usage rollup:

```bash
curl -sS "$BASE_URL/admin/usage/tenants?window=24h" \
  -H "X-Admin-Token: $ADMIN_TOKEN"
```

Budget status:

```bash
curl -sS "$BASE_URL/admin/budget_status?window=day&tenant_id=$TENANT_ID" \
  -H "X-Admin-Token: $ADMIN_TOKEN"
```

## Smoke script

Run the phase-4 smoke script:

```bash
BASE_URL=http://localhost:8000 ADMIN_TOKEN=<admin-token> ./scripts/phase4_smoke.sh
```
