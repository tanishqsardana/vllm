#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_preset> [base_url]"
  exit 1
fi

MODEL_PRESET="$1"
BASE_URL="${2:-http://localhost:8000}"
SEED="${SEED:-42}"
IMAGE_TAG="${IMAGE_TAG:-}"

slugify() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g'
}

PRIMARY_GPU="unknown-gpu"
if command -v nvidia-smi >/dev/null 2>&1; then
  PRIMARY_GPU_RAW="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr -d '\r')"
  if [[ -n "${PRIMARY_GPU_RAW}" ]]; then
    PRIMARY_GPU="$(slugify "${PRIMARY_GPU_RAW}")"
  fi
fi

TS="$(date -u +"%Y-%m-%d_%H%M")"
RUN_ID="${TS}_${PRIMARY_GPU}_$(slugify "${MODEL_PRESET}")"
if [[ -n "${IMAGE_TAG}" ]]; then
  RUN_ID+="_img-$(slugify "${IMAGE_TAG}")"
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

./scripts/collect_sysinfo.sh "${RUN_ID}"

CMD=(python3 eval/runner.py --model-preset "${MODEL_PRESET}" --base-url "${BASE_URL}" --run-id "${RUN_ID}" --seed "${SEED}")
if [[ -n "${NOTE:-}" ]]; then
  CMD+=(--note "${NOTE}")
fi

"${CMD[@]}"

echo "Completed run: ${RUN_ID}"
echo "Artifacts: results/${RUN_ID}"
