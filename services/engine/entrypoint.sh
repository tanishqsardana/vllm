#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_PORT="${VLLM_PORT:-8001}"

export HF_HOME="${HF_HOME:-/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/cache/vllm}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${VLLM_CACHE_ROOT}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

if [[ -z "${DTYPE:-}" ]]; then
  if python - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 1)
PY
  then
    DTYPE="bfloat16"
  else
    DTYPE="float16"
  fi
fi

echo "Starting vLLM with model=${MODEL_ID} dtype=${DTYPE} tensor_parallel=${TENSOR_PARALLEL}"

shutdown() {
  if [[ -n "${PROXY_PID:-}" ]] && kill -0 "${PROXY_PID}" 2>/dev/null; then
    kill -TERM "${PROXY_PID}" 2>/dev/null || true
  fi
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill -TERM "${VLLM_PID}" 2>/dev/null || true
  fi
}

handle_signal() {
  shutdown
  wait || true
  exit 0
}

trap handle_signal SIGTERM SIGINT

python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port "${VLLM_PORT}" \
  --model "${MODEL_ID}" \
  --dtype "${DTYPE}" \
  --tensor-parallel-size "${TENSOR_PARALLEL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --download-dir "${HF_HOME}" &
VLLM_PID=$!

python - <<'PY' &
import os

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

host = os.getenv("HOST", "0.0.0.0")
port = int(os.getenv("PORT", "8000"))
build_sha = os.getenv("BUILD_SHA", "dev")
build_time = os.getenv("BUILD_TIME", "dev")
upstream = f"http://127.0.0.1:{os.getenv('VLLM_PORT', '8001')}"

app = FastAPI(title="vLLM gateway", docs_url=None, redoc_url=None)


@app.get("/livez")
async def livez():
  return {"status": "alive"}


@app.get("/healthz")
async def healthz():
  timeout = httpx.Timeout(2.0, connect=2.0)
  async with httpx.AsyncClient(timeout=timeout) as client:
    try:
      resp = await client.get(f"{upstream}/v1/models")
      if resp.status_code == 200 and resp.json().get("data"):
        return {"status": "ready"}
    except (httpx.HTTPError, ValueError):
      pass
  return JSONResponse({"status": "not_ready"}, status_code=503)


@app.get("/version")
async def version():
  return {"build_sha": build_sha, "build_time": build_time}


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy_v1(path: str, request: Request):
  body = await request.body()
  headers = {
    k: v
    for k, v in request.headers.items()
    if k.lower() not in {"host", "content-length", "connection"}
  }
  url = f"{upstream}/v1/{path}"

  async with httpx.AsyncClient(timeout=None) as client:
    upstream_resp = await client.request(
      request.method,
      url,
      params=request.query_params,
      content=body,
      headers=headers,
    )

  response_headers = {}
  content_type = upstream_resp.headers.get("content-type")
  if content_type:
    response_headers["content-type"] = content_type

  return Response(
    content=upstream_resp.content,
    status_code=upstream_resp.status_code,
    headers=response_headers,
  )


uvicorn.run(app, host=host, port=port, log_level="info")
PY
PROXY_PID=$!

set +e
wait -n "${VLLM_PID}" "${PROXY_PID}"
EXIT_CODE=$?
set -e

shutdown
wait || true
exit "${EXIT_CODE}"
