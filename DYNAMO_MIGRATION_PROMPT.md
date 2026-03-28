# Claude Code Prompt: Migrate Mirae Control Plane from Direct vLLM to NVIDIA Dynamo

## Context

This repo is **Mirae**, an enterprise control plane (governance/compliance/operations layer) for AI inference. It currently runs a single-container architecture where:

- A **gateway** (FastAPI, port 8000) handles: multi-tenant auth (RBAC with tenants → seats → API keys), rate limiting (RPM/TPM token buckets), request accounting (SQLite), cost estimation, budget alerts, audit trail, Prometheus metrics, and an admin UI.
- A **vLLM process** (port 8001, localhost-only) runs the actual inference engine.
- `entrypoint.sh` launches vLLM first, waits for readiness, then starts the gateway.
- The gateway proxies `/v1/chat/completions` requests to `http://127.0.0.1:{VLLM_PORT}/v1/chat/completions` after applying auth, rate limits, and policy checks.

## Goal

Migrate the inference layer from **direct vLLM process management** to **NVIDIA Dynamo** as the orchestration layer, using **Option A architecture**: Mirae's gateway stays on `:8000`, Dynamo's frontend runs on an internal port (default `:8001`). The gateway proxies to Dynamo's frontend instead of raw vLLM. **All gateway logic stays identical.**

Dynamo is an open-source inference orchestration framework (Apache 2.0, repo: `ai-dynamo/dynamo`). It provides:
- An OpenAI-compatible HTTP frontend (Rust-based, high performance)
- KV-aware routing, disaggregated prefill/decode, multi-node scaling
- Support for vLLM, SGLang, and TRT-LLM as backend engines

The key insight: Dynamo's capabilities are mostly transparent — disaggregated serving, KV routing, etc. happen based on deployment config, not API calls. The gateway doesn't need to know the internals.

## Dynamo Quick Reference

### Starting Dynamo locally (single-node, no etcd/NATS needed):

```bash
# Frontend (OpenAI-compat HTTP server)
python3 -m dynamo.frontend --http-port 8001 --store-kv file

# vLLM worker (registers with frontend automatically)
python3 -m dynamo.vllm --model Qwen/Qwen2.5-3B-Instruct --store-kv file \
  --kv-events-config '{"enable_kv_cache_events": false}'
```

### Prebuilt container:
```
nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1
```

### Key flags:
- `--store-kv file` — file-based discovery, avoids etcd/NATS dependency for single-node
- `--http-port` — frontend HTTP port (default 8000, we'll use 8001)
- `--kv-events-config '{"enable_kv_cache_events": false}'` — disables NATS requirement for vLLM worker
- `DYN_FILE_KV` env var — directory for file-based KV store (default: `$TMPDIR/dynamo_store_kv`)

### Health check:
Dynamo's frontend serves the same OpenAI-compat API. Health can be checked via:
- `GET /v1/models` (returns 200 with model data when ready)
- Sending a minimal chat completion request

### Usage/token info:
Dynamo passes through vLLM's usage response. The `usage` field in chat completion responses should contain `prompt_tokens`, `completion_tokens`, `total_tokens`. **This needs verification** — if Dynamo returns null for usage, the existing `TokenEstimator` fallback in `gateway/utils.py` handles it (char_length / 4 approximation).

## Files to Change

### 1. `services/combined/Dockerfile`

**Current**: Based on `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`, installs vLLM + gateway deps via pip.

**Change to**: Based on `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1`, which already includes Dynamo + vLLM. Install only the gateway-specific deps on top (FastAPI, uvicorn, httpx, prometheus_client, pynvml, pyyaml, PyJWT — but NOT vllm since it's already in the base image).

Keep:
- The `appuser` setup, `/app`, `/cache`, `/data` directory structure
- COPY of gateway code, config, profiles, UI
- `tini` entrypoint pattern
- All ENV vars for HF_HOME, cache paths, BUILD_SHA, BUILD_TIME

Add:
- `DYN_FILE_KV=/data/dynamo_kv` as default env var for Dynamo's file-based discovery store
- `DYNAMO_FRONTEND_PORT=8001` as default env var

Remove:
- The explicit `vllm==0.6.4.post1` pip install (comes from base image)
- The `transformers` pip install if already in base image (check and keep if not)

### 2. `services/combined/entrypoint.sh`

**Current**: Launches `python -m vllm.entrypoints.openai.api_server` with a long list of flags, waits for readiness, then starts the gateway.

**Change to**: Launch two Dynamo processes instead of one vLLM process:

```bash
# Process 1: Dynamo frontend (OpenAI-compat HTTP gateway, internal only)
python3 -m dynamo.frontend \
  --http-port "${DYNAMO_FRONTEND_PORT}" \
  --store-kv file &
DYNAMO_FRONTEND_PID=$!

# Process 2: Dynamo vLLM worker
python3 -m dynamo.vllm \
  --model "${MODEL_ID}" \
  --store-kv file \
  --kv-events-config '{"enable_kv_cache_events": false}' \
  ${DYNAMO_EXTRA_VLLM_ARGS:-} &
DYNAMO_WORKER_PID=$!
```

Update the readiness wait function to check Dynamo's frontend on `DYNAMO_FRONTEND_PORT` instead of `VLLM_PORT`. The check can use the same pattern (poll `/v1/models` or try a minimal completion).

Update the shutdown handler to kill both Dynamo processes.

Keep:
- All env var setup (HF_HOME, cache dirs, HF_TOKEN export)
- The `mkdir -p` for cache/data dirs
- Signal handling pattern (SIGTERM/SIGINT)
- The final `uvicorn gateway.main:app` launch for the Mirae gateway

Map existing vLLM flags to Dynamo equivalents where applicable:
- `--model` stays the same
- `--dtype`, `--tensor-parallel-size`, `--max-model-len`, `--gpu-memory-utilization` — pass these through to `dynamo.vllm` (it accepts standard vLLM flags)
- `--trust-remote-code` — pass through if enabled
- `--download-dir` — pass through as before
- Remove the `--host 127.0.0.1 --port ${VLLM_PORT}` flags (Dynamo handles worker binding internally)

New env vars to support:
- `DYNAMO_FRONTEND_PORT` (default: `8001`)
- `DYNAMO_EXTRA_VLLM_ARGS` (optional, replaces `EXTRA_VLLM_ARGS` for Dynamo-specific worker flags)
- `DYN_FILE_KV` (default: `/data/dynamo_kv`) — shared KV store path for frontend + worker

### 3. `services/combined/gateway/config_loader.py`

**Change**: Add `DYNAMO_FRONTEND_PORT` to the config with default `8001`. The `upstream_url` in Settings should be constructed as `http://127.0.0.1:{DYNAMO_FRONTEND_PORT}` instead of `http://127.0.0.1:{VLLM_PORT}`.

Keep `VLLM_PORT` in the config for backward compatibility but add `DYNAMO_FRONTEND_PORT` as the primary. The upstream URL should prefer `DYNAMO_FRONTEND_PORT` if set.

### 4. `services/combined/gateway/main.py`

**Change in `_probe_upstream`**: The health probe currently checks `{upstream_url}/health` and falls back to a minimal chat completion. Dynamo's frontend doesn't serve `/health` — it serves the OpenAI API directly. Update the probe to:
1. Try `GET {upstream_url}/v1/models` — if 200 and response has `data`, healthy.
2. Fall back to minimal chat completion as before.

This is likely a small change since the existing probe already has the completion fallback.

**Optional enhancement**: Add per-request priority hint injection in `chat_completions`. After applying tenant policy and before forwarding to Dynamo, inject Dynamo-compatible request hints based on tenant tier. This is NOT required for phase 1 but structure the code to make it easy later. For now, just add a comment like:

```python
# TODO: Inject Dynamo per-request hints (priority, cache_ttl) based on tenant policy
# Example: payload["extra_body"] = {"priority": tenant_priority_level}
```

### 5. `config/config.yaml`

Add `DYNAMO_FRONTEND_PORT: 8001` to the config file.

## Files That MUST NOT Change

All gateway business logic stays identical:
- `gateway/auth.py` — key resolution, bearer parsing
- `gateway/admin_auth.py` — OIDC/static admin auth
- `gateway/rate_limit.py` — token bucket rate limiting
- `gateway/limits.py` — concurrency gating
- `gateway/accounting.py` — usage rollups, budget evaluation
- `gateway/db.py` — SQLite schema and queries
- `gateway/metrics.py` — Prometheus metrics
- `gateway/gpu_metrics.py` — GPU telemetry polling
- `gateway/audit.py` — audit trail logging
- `gateway/schemas.py` — Pydantic models
- `gateway/utils.py` — token estimation, logging
- `gateway/middleware_metrics.py` — request metrics middleware
- `gateway/ui/` — admin UI (HTML/CSS/JS)
- `profiles/` — deployment profiles
- `scripts/` — smoke tests, tenant management
- `eval/` — evaluation harness
- `monitoring/` — Prometheus + Grafana stack

## Acceptance Criteria

1. The existing `scripts/phase2_smoke.sh` should pass against the Dynamo-backed stack (same tenant creation, auth, rate limiting, inference, 429 rejection behavior).
2. `GET /healthz` returns 200 when Dynamo frontend + worker are healthy.
3. `GET /version` still returns correct build info.
4. Token usage extraction works (verify `usage.prompt_tokens` and `usage.completion_tokens` in response).
5. `/metrics` endpoint still serves Prometheus metrics with all existing gateway metrics.
6. Admin UI at `/ui` works unchanged.

## What NOT to Build (phase 2+)

- Disaggregated prefill/decode worker pools (deployment config, not code)
- Per-tenant GPU isolation / dedicated worker pools
- Multi-node support (Dynamo handles this transparently)
- Dynamo-specific metrics scraping (beyond what gateway already collects)
- Per-request priority hint injection based on tenant tier
- Redis-backed rate limiting for multi-gateway instances
- Postgres migration from SQLite

## Notes

- The `--store-kv file` flag makes Dynamo work without etcd/NATS. Both frontend and worker must use this flag AND share the same `DYN_FILE_KV` directory.
- Dynamo's vLLM worker accepts most standard vLLM CLI flags (--model, --dtype, --tensor-parallel-size, etc.)
- The TP auto-detection logic in the current `entrypoint.sh` can likely be kept as-is — just pass the resolved TP value to `dynamo.vllm` instead of `vllm.entrypoints.openai.api_server`.
- GPU metrics polling via pynvml should work unchanged since it polls the GPU hardware directly, not vLLM.
