#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-}"
if [[ -z "${MODEL_ID}" ]]; then
  echo "MODEL_ID is required" >&2
  exit 1
fi

GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}"
GATEWAY_PORT="${GATEWAY_PORT:-8000}"
VLLM_HOST="127.0.0.1"
VLLM_PORT="${VLLM_PORT:-8001}"

DTYPE="${DTYPE:-bfloat16}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

VLLM_READY_TIMEOUT_SECONDS="${VLLM_READY_TIMEOUT_SECONDS:-600}"
VLLM_READY_POLL_SECONDS="${VLLM_READY_POLL_SECONDS:-2}"

export HF_HOME="${HF_HOME:-/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/cache/vllm}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${VLLM_CACHE_ROOT}" /data

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

shutdown() {
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

VLLM_CMD=(
  python -m vllm.entrypoints.openai.api_server
  --host "${VLLM_HOST}"
  --port "${VLLM_PORT}"
  --model "${MODEL_ID}"
  --dtype "${DTYPE}"
  --tensor-parallel-size "${TENSOR_PARALLEL}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --download-dir "${HF_HOME}"
)

if is_true "${TRUST_REMOTE_CODE}"; then
  VLLM_CMD+=(--trust-remote-code)
fi

if [[ -n "${EXTRA_VLLM_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( ${EXTRA_VLLM_ARGS} )
  VLLM_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[entrypoint] starting vLLM model=${MODEL_ID} on ${VLLM_HOST}:${VLLM_PORT}"
"${VLLM_CMD[@]}" &
VLLM_PID=$!

wait_for_vllm() {
  local deadline=$((SECONDS + VLLM_READY_TIMEOUT_SECONDS))
  local health_url="http://${VLLM_HOST}:${VLLM_PORT}/health"
  local completion_url="http://${VLLM_HOST}:${VLLM_PORT}/v1/chat/completions"

  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 3 "${health_url}" >/dev/null 2>&1; then
      return 0
    fi

    if curl -fsS --max-time 5 "${completion_url}" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"temperature\":0,\"max_tokens\":1}" \
      >/dev/null 2>&1; then
      return 0
    fi

    sleep "${VLLM_READY_POLL_SECONDS}"
  done

  return 1
}

if ! wait_for_vllm; then
  echo "[entrypoint] vLLM readiness failed after ${VLLM_READY_TIMEOUT_SECONDS}s" >&2
  shutdown
  wait || true
  exit 1
fi

echo "[entrypoint] vLLM is ready; starting gateway on ${GATEWAY_HOST}:${GATEWAY_PORT}"
set +e
python -m uvicorn gateway.main:app --host "${GATEWAY_HOST}" --port "${GATEWAY_PORT}" --log-level info
EXIT_CODE=$?
set -e
shutdown
wait || true
exit "${EXIT_CODE}"
