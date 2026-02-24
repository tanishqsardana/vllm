#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   IMAGE_REPO=vllm-engine IMAGE_TAG=phase1 ./scripts/run_image.sh
# Optional overrides:
#   MODEL_ID, HF_TOKEN, DTYPE, TENSOR_PARALLEL, MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION, BUILD_SHA, BUILD_TIME

IMAGE_REPO="${IMAGE_REPO:-vllm-engine}"
IMAGE_TAG="${IMAGE_TAG:-phase1}"
IMAGE="${IMAGE_REPO}:${IMAGE_TAG}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-engine}"
CACHE_VOLUME="${CACHE_VOLUME:-model_cache}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
DTYPE="${DTYPE:-}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
BUILD_SHA="${BUILD_SHA:-dev}"
BUILD_TIME="${BUILD_TIME:-dev}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  -p "${PORT}:8000" \
  -v "${CACHE_VOLUME}:/cache" \
  -e MODEL_ID="${MODEL_ID}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e PORT=8000 \
  -e HOST="${HOST}" \
  -e DTYPE="${DTYPE}" \
  -e TENSOR_PARALLEL="${TENSOR_PARALLEL}" \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  -e BUILD_SHA="${BUILD_SHA}" \
  -e BUILD_TIME="${BUILD_TIME}" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  --restart unless-stopped \
  "${IMAGE}"

echo "Container started: ${CONTAINER_NAME}"
echo "Image: ${IMAGE}"
echo "Health: curl -sS http://localhost:${PORT}/healthz"
