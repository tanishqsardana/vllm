#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-}"
if [[ -z "${MODEL_ID}" ]]; then
  echo "MODEL_ID is required" >&2
  exit 1
fi

GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}"
GATEWAY_PORT="${GATEWAY_PORT:-8000}"
DYNAMO_FRONTEND_PORT="${DYNAMO_FRONTEND_PORT:-8001}"

DTYPE="${DTYPE:-bfloat16}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
DYNAMO_EXTRA_VLLM_ARGS="${DYNAMO_EXTRA_VLLM_ARGS:-}"

DYNAMO_READY_TIMEOUT_SECONDS="${DYNAMO_READY_TIMEOUT_SECONDS:-600}"
DYNAMO_READY_POLL_SECONDS="${DYNAMO_READY_POLL_SECONDS:-2}"

export HF_HOME="${HF_HOME:-/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/cache/vllm}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export DYN_FILE_KV="${DYN_FILE_KV:-/data/dynamo_kv}"

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${VLLM_CACHE_ROOT}" /data "${DYN_FILE_KV}"

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
  if [[ -n "${DYNAMO_FRONTEND_PID:-}" ]] && kill -0 "${DYNAMO_FRONTEND_PID}" 2>/dev/null; then
    kill -TERM "${DYNAMO_FRONTEND_PID}" 2>/dev/null || true
  fi
  if [[ -n "${DYNAMO_WORKER_PID:-}" ]] && kill -0 "${DYNAMO_WORKER_PID}" 2>/dev/null; then
    kill -TERM "${DYNAMO_WORKER_PID}" 2>/dev/null || true
  fi
}

handle_signal() {
  shutdown
  wait || true
  exit 0
}

trap handle_signal SIGTERM SIGINT

# --- Dynamo frontend (OpenAI-compat HTTP server, internal only) ---
echo "[entrypoint] starting Dynamo frontend on port ${DYNAMO_FRONTEND_PORT}"
python3 -m dynamo.frontend \
  --http-port "${DYNAMO_FRONTEND_PORT}" \
  --store-kv file &
DYNAMO_FRONTEND_PID=$!

# --- Dynamo vLLM worker ---
WORKER_CMD=(
  python3 -m dynamo.vllm
  --model "${MODEL_ID}"
  --store-kv file
  --kv-events-config '{"enable_kv_cache_events": false}'
  --dtype "${DTYPE}"
  --tensor-parallel-size "${TENSOR_PARALLEL}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --download-dir "${HF_HOME}"
)

if is_true "${TRUST_REMOTE_CODE}"; then
  WORKER_CMD+=(--trust-remote-code)
fi

if [[ -n "${DYNAMO_EXTRA_VLLM_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( ${DYNAMO_EXTRA_VLLM_ARGS} )
  WORKER_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[entrypoint] starting Dynamo vLLM worker model=${MODEL_ID}"
"${WORKER_CMD[@]}" &
DYNAMO_WORKER_PID=$!

wait_for_dynamo() {
  local deadline=$((SECONDS + DYNAMO_READY_TIMEOUT_SECONDS))
  local models_url="http://127.0.0.1:${DYNAMO_FRONTEND_PORT}/v1/models"
  local completion_url="http://127.0.0.1:${DYNAMO_FRONTEND_PORT}/v1/chat/completions"

  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 3 "${models_url}" >/dev/null 2>&1; then
      return 0
    fi

    if curl -fsS --max-time 5 "${completion_url}" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"temperature\":0,\"max_tokens\":1}" \
      >/dev/null 2>&1; then
      return 0
    fi

    sleep "${DYNAMO_READY_POLL_SECONDS}"
  done

  return 1
}

if ! wait_for_dynamo; then
  echo "[entrypoint] Dynamo readiness failed after ${DYNAMO_READY_TIMEOUT_SECONDS}s" >&2
  shutdown
  wait || true
  exit 1
fi

echo "[entrypoint] Dynamo is ready; starting gateway on ${GATEWAY_HOST}:${GATEWAY_PORT}"
set +e
python -m uvicorn gateway.main:app --host "${GATEWAY_HOST}" --port "${GATEWAY_PORT}" --log-level info
EXIT_CODE=$?
set -e
shutdown
wait || true
exit "${EXIT_CODE}"
