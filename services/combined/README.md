# Combined Service (Phase 2 Control Plane v1)

This image runs both processes inside one container:

1. vLLM OpenAI server (`127.0.0.1:8001`, internal only)
2. Gateway control plane (`0.0.0.0:8000`, public)

Gateway endpoints:

- `POST /v1/chat/completions`
- `GET /livez`
- `GET /healthz`
- `GET /version`
- `POST /admin/tenants`
- `GET /admin/tenants`
- `PATCH /admin/tenants/{id}`
- `GET /admin/usage?window=1h|24h|7d`

## Environment Variables

### Required

- `MODEL_ID`: model to serve

### Engine (vLLM)

- `DTYPE` (default: `bfloat16`)
- `TENSOR_PARALLEL` (default: `1`)
- `MAX_MODEL_LEN` (default: `8192`)
- `GPU_MEMORY_UTILIZATION` (default: `0.90`)
- `VLLM_PORT` (default: `8001`)
- `TRUST_REMOTE_CODE` (default: `0`)
- `EXTRA_VLLM_ARGS` (optional)

### Gateway / Control Plane

- `ADMIN_TOKEN` (required for admin endpoints)
- `DB_PATH` (default: `/data/controlplane.db`)
- `GLOBAL_MAX_CONCURRENT` (default: `128`)
- `MAX_BODY_BYTES` (default: `1048576`)
- `UPSTREAM_TIMEOUT_SECONDS` (default: `300`)
- `DEFAULT_MAX_CONCURRENT` (default: `4`)
- `DEFAULT_RPM_LIMIT` (default: `120`)
- `DEFAULT_TPM_LIMIT` (default: `120000`)
- `DEFAULT_MAX_CONTEXT_TOKENS` (default: `8192`)
- `DEFAULT_MAX_OUTPUT_TOKENS` (default: `512`)
- `BUILD_SHA` (default: `dev`)
- `BUILD_TIME` (default: `dev`)
- `GATEWAY_HOST` (default: `0.0.0.0`)
- `GATEWAY_PORT` (default: `8000`)
- `METRICS_TENANT_LABELS` (default: `on`; set `off` to drop `tenant_id` metric labels)
- `GPU_METRICS_POLL_INTERVAL_SECONDS` (default: `2`)

### Cache / model download compatibility (Phase 1 compatible)

- `HF_TOKEN` (optional)
- `HF_HOME` (default: `/cache/huggingface`)
- `HUGGINGFACE_HUB_CACHE` (default: `/cache/huggingface/hub`)
- `TRANSFORMERS_CACHE` (default: `/cache/huggingface/transformers`)
- `VLLM_CACHE_ROOT` (default: `/cache/vllm`)

## Local Run

Build:

```bash
./scripts/build_combined_image.sh
```

Run:

```bash
docker run --rm -it \
  --gpus all \
  -p 8000:8000 \
  -v model_cache:/cache \
  -v controlplane_data:/data \
  -e MODEL_ID=Qwen/Qwen2.5-3B-Instruct \
  -e ADMIN_TOKEN=replace-me \
  -e BUILD_SHA=dev \
  -e BUILD_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  vllm-combined:phase2
```

Smoke test:

```bash
ADMIN_TOKEN=replace-me BASE_URL=http://localhost:8000 ./scripts/phase2_smoke.sh
```

Tenant create helper:

```bash
ADMIN_TOKEN=replace-me BASE_URL=http://localhost:8000 ./scripts/create_tenant.sh tenant-a
```

Tenant limits helper:

```bash
ADMIN_TOKEN=replace-me BASE_URL=http://localhost:8000 ./scripts/set_tenant_limits.sh tenant-a standard
```

## Runpod (Single Image)

Publish and deploy one image, for example `docker.io/<user>/<repo>:phase2`, then configure env vars in Runpod:

- Required: `MODEL_ID`, `ADMIN_TOKEN`
- Recommended persistent volumes:
  - `/cache` for model cache reuse
  - `/data` for SQLite state persistence
- Expose only port `8000` publicly

`entrypoint.sh` launches vLLM first, waits readiness, then starts gateway.

Recommended Runpod env values:

- `MODEL_ID=<your-model>`
- `ADMIN_TOKEN=<strong-random-secret>`
- `DB_PATH=/workspace/data/controlplane.db`
- `HF_HOME=/workspace/cache/huggingface`
- `HUGGINGFACE_HUB_CACHE=/workspace/cache/huggingface/hub`
- `TRANSFORMERS_CACHE=/workspace/cache/huggingface/transformers`
- `VLLM_CACHE_ROOT=/workspace/cache/vllm`

## Example Tenant Inference Call

```bash
curl -sS http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer <TENANT_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ignored-by-gateway",
    "messages": [{"role": "user", "content": "Say hello in five words."}],
    "max_tokens": 64
  }'
```

## Example Admin Create Tenant

```bash
curl -sS http://localhost:8000/admin/tenants \
  -H "X-Admin-Token: <ADMIN_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_name": "tenant-a",
    "max_concurrent": 2,
    "rpm_limit": 120,
    "tpm_limit": 50000,
    "max_context_tokens": 8192,
    "max_output_tokens": 512
  }'
```

Or use helper:

```bash
ADMIN_TOKEN=<ADMIN_TOKEN> BASE_URL=http://localhost:8000 ./scripts/create_tenant.sh tenant-a
```

## Admin Token Rejection Behavior

- If `ADMIN_TOKEN` is not set in container env:
  - all admin endpoints return `503` with `error.type=admin_unavailable`
- If `ADMIN_TOKEN` is set but request is missing `X-Admin-Token`:
  - admin endpoints return `401` with `error.type=unauthorized`
- If `X-Admin-Token` is wrong:
  - admin endpoints return `401` with `error.type=unauthorized`

Quick checks:

```bash
curl -sS -i http://localhost:8000/admin/tenants
curl -sS -i http://localhost:8000/admin/tenants -H "X-Admin-Token: wrong-token"
```

`scripts/phase2_smoke.sh` now validates missing/invalid admin header rejections as part of the acceptance run.
It also supports explicit `ADMIN_TOKEN`-unset validation mode:

```bash
EXPECT_ADMIN_UNAVAILABLE=1 BASE_URL=http://localhost:8000 ./scripts/phase2_smoke.sh
```

## Setting Real Tenant Limits

Best method for now is profile-based patching plus targeted overrides:

```bash
ADMIN_TOKEN=<ADMIN_TOKEN> BASE_URL=<BASE_URL> ./scripts/set_tenant_limits.sh <tenant_id_or_name> standard
```

Profiles:

- `conservative`: `max_concurrent=1`, `rpm_limit=60`, `tpm_limit=30000`, `max_context_tokens=4096`, `max_output_tokens=256`
- `standard`: `max_concurrent=2`, `rpm_limit=180`, `tpm_limit=120000`, `max_context_tokens=8192`, `max_output_tokens=512`
- `pro`: `max_concurrent=4`, `rpm_limit=600`, `tpm_limit=400000`, `max_context_tokens=16384`, `max_output_tokens=1024`

Override any field per call:

```bash
MAX_CONCURRENT=3 RPM_LIMIT=240 TPM_LIMIT=180000 \
ADMIN_TOKEN=<ADMIN_TOKEN> BASE_URL=<BASE_URL> \
./scripts/set_tenant_limits.sh <tenant_id_or_name> standard
```

## 429 / Token Limit Troubleshooting

- `limit_concurrent`: tenant `max_concurrent` or global `GLOBAL_MAX_CONCURRENT` exceeded. Requests are rejected immediately (no queue).
- `limit_rpm`: request token bucket exceeded for current 60s window.
- `limit_tpm`: token bucket exceeded (gateway reserves `prompt_tokens + max_tokens` before forwarding).
- If you see frequent `limit_tpm`, increase `tpm_limit`, reduce `max_tokens`, or reduce prompt size.
- If prompts are rejected with `bad_input`, raise `max_context_tokens` or send shorter messages.

## Build Troubleshooting

- If build fails with `RuntimeError: Unknown runtime environment` while installing `vllm`, your build likely targeted `arm64` and pip attempted source build.
- Build for `linux/amd64`:

```bash
TARGET_PLATFORM=linux/amd64 ./scripts/build_combined_image.sh
```

- Direct command equivalent:

```bash
docker buildx build --platform linux/amd64 -f services/combined/Dockerfile -t vllm-combined:phase2 --load .
```

## Notes

- Tenant API keys are returned only once at creation. DB stores only SHA-256 hashes.
- RPM/TPM limiters are in-memory in Phase 2; use Redis later for multi-instance/shared state.
- SQLite tables are created automatically on startup if missing.

## Observability (Phase 3)

- `GET /metrics` is available in Prometheus exposition format.
- `GET /metrics` requires `X-Admin-Token` and uses the same token as admin endpoints.
- See full setup and dashboard docs in [docs/OBSERVABILITY.md](/Users/tanishqsardana/Documents/Startup/docs/OBSERVABILITY.md).

Quick check:

```bash
curl -i http://localhost:8000/metrics
curl -sS -H "X-Admin-Token: <ADMIN_TOKEN>" http://localhost:8000/metrics | head
```
