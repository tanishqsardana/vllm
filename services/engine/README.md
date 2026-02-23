# Engine (Phase 1 Inference Plane)

This service runs a single-model vLLM OpenAI-compatible server in Docker and exposes:

- `POST /v1/chat/completions`
- `GET /livez`
- `GET /healthz`
- `GET /version`

`/livez`, `/healthz`, and `/version` are served by a tiny in-container FastAPI gateway. `/v1/*` is proxied to vLLM.

## Required/Supported Environment Variables

- `MODEL_ID` (default: `Qwen/Qwen2.5-3B-Instruct`)
- `HF_TOKEN` (optional, required for gated models)
- `PORT` (default: `8000`)
- `HOST` (default: `0.0.0.0`)
- `DTYPE` (default: auto-detect: `bfloat16` if supported, else `float16`)
- `TENSOR_PARALLEL` (default: `1`)
- `MAX_MODEL_LEN` (default: `8192`)
- `GPU_MEMORY_UTILIZATION` (default: `0.90`)
- `BUILD_SHA` (default: `dev`)
- `BUILD_TIME` (default: `dev`)

Model/cache data is persisted to `/cache` in the container via a named Docker volume.

## Run

From repo root:

```bash
docker compose up -d --build
./scripts/smoke_test.sh
```

Check endpoints:

```bash
curl -sS http://localhost:8000/livez
curl -sS http://localhost:8000/healthz
curl -sS http://localhost:8000/version
```

## Change MODEL_ID and MAX_MODEL_LEN

Use environment overrides:

```bash
MODEL_ID=meta-llama/Llama-3.1-8B-Instruct MAX_MODEL_LEN=4096 docker compose up -d --build
```

Or set values in your shell/.env before running compose.

## Troubleshooting

- OOM on startup/inference:
  - Reduce `MAX_MODEL_LEN` (for example `4096` or `2048`)
  - Lower `GPU_MEMORY_UTILIZATION` (for example `0.80`)
  - Use a smaller model (`MODEL_ID`)
  - Set `DTYPE=float16` if bf16 is problematic on your GPU
- Gated/private model download fails:
  - Export `HF_TOKEN` with access to the model repo
- Readiness takes a long time:
  - First run downloads weights into the cache volume (`model_cache`), which can take several minutes
- `docker compose` cannot see GPU:
  - Verify NVIDIA Container Toolkit is installed and `nvidia-smi` works on host
