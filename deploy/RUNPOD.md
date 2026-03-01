# Runpod Deployment (Single Image)

This stack is a single container image running:

1. vLLM engine (`127.0.0.1:8001`, private)
2. Gateway control plane (`0.0.0.0:8000`, public)

No internal Docker/compose orchestration is required inside the container.

## Required Runtime Settings

Set at minimum:

- `MODEL_ID`
- `ADMIN_AUTH_MODE=static_token` and `ADMIN_TOKEN=<strong-secret>`

For OIDC placeholder mode:

- `ADMIN_AUTH_MODE=oidc`
- `JWKS_URL`
- `OIDC_ISSUER`
- `OIDC_AUDIENCE`
- `ADMIN_GROUP` (recommended)

Recommended:

- `DB_PATH=/data/controlplane.db`
- `GPU_HOURLY_RATE=<usd_per_hour>`
- `CONFIG_PATH=/app/config/config.yaml`

## Volumes

Mount persistent volumes:

- `/data` for SQLite + audit persistence
- `/cache` for model cache reuse

## Ports

Expose gateway port:

- `8000`

## Post-Deploy Validation

- `GET /healthz` should be `200` when ready
- `GET /ui` should return the control plane UI
- `GET /version` should include Phase 5 feature flags

## UI Connection

Open:

- `http://<runpod-endpoint>:8000/ui`

Use Connect screen:

- static mode: paste `ADMIN_TOKEN`
- oidc mode: paste admin JWT
