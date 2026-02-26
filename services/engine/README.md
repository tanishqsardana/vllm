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
- `AUTO_TP_FALLBACK` (default: `1`; auto-adjust invalid tensor parallel to nearest valid value)
- `TRUST_REMOTE_CODE` (default: `0`; set `1` to pass `--trust-remote-code` to vLLM)
- `ENABLE_AUTO_TOOL_CHOICE` (default: `0`; set `1` to pass `--enable-auto-tool-choice`)
- `TOOL_CALL_PARSER` (optional; passed as `--tool-call-parser <value>`)
- `REASONING_PARSER` (optional; passed as `--reasoning-parser <value>`)
- `EXTRA_VLLM_ARGS` (optional; extra args appended to vLLM server command)
- `BUILD_SHA` (default: `dev`)
- `BUILD_TIME` (default: `dev`)
- `VLLM_WORKER_MULTIPROC_METHOD` (default: `spawn`; recommended for tensor parallel > 1)
- `ENABLE_SSHD` (default: `0`; set `1` to start SSH daemon in container)
- `PUBLIC_KEY` (optional SSH public key; if set, SSH daemon auto-enables)
- `SSH_PORT` (default: `22`)

Model/cache data is persisted to `/cache` in the container via a Docker volume.

## Image-First Run (recommended for container VMs)

From repo root:

```bash
./scripts/build_image.sh
./scripts/run_image.sh
./scripts/smoke_test.sh
```

To publish an image:

```bash
IMAGE_REPO=ghcr.io/<org>/vllm-engine IMAGE_TAG=phase1 PUSH=1 ./scripts/build_image.sh
```

Then run that pushed image on your VM/runtime by setting environment variables and mounting `/cache`.
`build_image.sh` defaults to `TARGET_PLATFORM=linux/amd64` to match NVIDIA GPU runtimes.

## Docker Compose (optional local workflow)

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
MODEL_ID=meta-llama/Llama-3.1-8B-Instruct MAX_MODEL_LEN=4096 ./scripts/run_image.sh
```

Remote-code / tool parser example:

```bash
MODEL_ID=MiniMaxAI/MiniMax-M2.5 \
TRUST_REMOTE_CODE=1 \
TENSOR_PARALLEL=4 \
ENABLE_AUTO_TOOL_CHOICE=1 \
TOOL_CALL_PARSER=minimax_m2 \
REASONING_PARSER=minimax_m2_append_think \
SAFETENSORS_FAST_GPU=1 \
./scripts/run_image.sh
```

Or set values in your shell/.env before running compose/run scripts.

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
- Runpod/container VM deployment:
  - Build and push image with `./scripts/build_image.sh`
  - Configure container env vars (`MODEL_ID`, `MAX_MODEL_LEN`, etc.)
  - Mount persistent storage to `/cache` for model reuse
  - Expose container port `8000`
- Runpod SSH in custom templates:
  - Set `ENABLE_SSHD=1` and `PUBLIC_KEY=<your public key contents>`
  - Expose TCP port `22`
  - The image starts `sshd` and runs inference as `appuser`
- Build fails with `Unknown runtime environment` during `pip install vllm`:
  - This usually means you are building `arm64` and pip is attempting a source build
  - Build/push with `TARGET_PLATFORM=linux/amd64` (default in `build_image.sh`)
- Runtime fails with `Qwen2Tokenizer has no attribute all_special_tokens_extended`:
  - Rebuild and redeploy the latest image from this repo (includes a tokenizer compatibility shim)
  - Use `docker pull <your-image>:<tag>` on the runtime after pushing the rebuilt image
- Runtime fails with `tqdm_asyncio.__init__() got multiple values for keyword argument 'disable'`:
  - Rebuild and redeploy the latest image from this repo (includes a robust vLLM/hf-hub tqdm compatibility shim + pinned `huggingface_hub`/`tqdm`)
  - Ensure the runtime actually pulls the new tag before restarting
- Runtime returns persistent `503` on `/healthz` with tensor parallel > 1:
  - Check logs for `Cannot re-initialize CUDA in forked subprocess`
  - Set `VLLM_WORKER_MULTIPROC_METHOD=spawn` (default in latest image) and redeploy
- Runtime fails with `is not divisible by` after changing `TENSOR_PARALLEL`:
  - vLLM requires tensor parallel to divide both attention heads and padded vocab shard size
  - Latest image auto-adjusts invalid TP to nearest valid value (`AUTO_TP_FALLBACK=1`)
  - For `Qwen/Qwen2.5-7B-Instruct`, practical TP values are `1`, `2`, `4` (on 8 GPUs, use `4`)
