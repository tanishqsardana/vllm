#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_PORT="${VLLM_PORT:-8001}"
AUTO_TP_FALLBACK="${AUTO_TP_FALLBACK:-1}"

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

TP_SELECTED="$(
python - <<'PY'
import math
import os
import sys

requested = int(os.getenv("TENSOR_PARALLEL", "1"))
if requested < 1:
    requested = 1

model_id = os.getenv("MODEL_ID", "")
hf_home = os.getenv("HF_HOME", "/cache/huggingface")
hf_token = os.getenv("HF_TOKEN") or None
auto_fallback = os.getenv("AUTO_TP_FALLBACK", "1") not in {"0", "false", "False", "no", "NO"}

gpu_count = 0
try:
    import torch  # type: ignore
    if torch.cuda.is_available():
        gpu_count = int(torch.cuda.device_count())
except Exception:
    gpu_count = 0

upper_bound = gpu_count if gpu_count > 0 else requested
upper_bound = max(upper_bound, requested)

heads = None
vocab_size = None
vocab_size_padded = None

try:
    from transformers import AutoConfig  # type: ignore

    cfg = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=False,
        cache_dir=hf_home,
        token=hf_token,
    )
    heads = getattr(cfg, "num_attention_heads", None)
    vocab_size = getattr(cfg, "vocab_size", None)
    if isinstance(vocab_size, int) and vocab_size > 0:
        # vLLM shards padded vocab for tensor-parallel embedding.
        vocab_size_padded = int(math.ceil(vocab_size / 128.0) * 128)
except Exception:
    pass

def is_valid(tp: int) -> bool:
    if tp < 1:
        return False
    if gpu_count > 0 and tp > gpu_count:
        return False
    if isinstance(heads, int) and heads > 0 and heads % tp != 0:
        return False
    if isinstance(vocab_size_padded, int) and vocab_size_padded > 0 and vocab_size_padded % tp != 0:
        return False
    return True

valid = [tp for tp in range(1, upper_bound + 1) if is_valid(tp)]
if not valid:
    valid = [1]

if requested in valid:
    print(requested)
    sys.exit(0)

if auto_fallback:
    lower = [tp for tp in valid if tp <= requested]
    selected = max(lower) if lower else min(valid)
    sys.stderr.write(
        f"[entrypoint] Requested TENSOR_PARALLEL={requested} is invalid for model={model_id}. "
        f"Using TENSOR_PARALLEL={selected}. Valid values (<= visible GPUs): {valid}\n"
    )
    print(selected)
    sys.exit(0)

sys.stderr.write(
    f"[entrypoint] Requested TENSOR_PARALLEL={requested} is invalid for model={model_id}. "
    f"Valid values (<= visible GPUs): {valid}\n"
)
sys.exit(2)
PY
)"
TENSOR_PARALLEL="${TP_SELECTED}"

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
