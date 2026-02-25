#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
PAUSE_BETWEEN="${PAUSE_BETWEEN:-1}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

mapfile -t PRESETS < <(awk '
  /^model_presets:/ { in_block=1; next }
  in_block && /^  [a-zA-Z0-9_-]+:/ {
    gsub(":", "", $1)
    print $1
  }
' eval/config/models.yaml)

if [[ ${#PRESETS[@]} -eq 0 ]]; then
  echo "No model presets found in eval/config/models.yaml"
  exit 1
fi

for PRESET in "${PRESETS[@]}"; do
  if [[ "${PAUSE_BETWEEN}" == "1" ]]; then
    echo "Switch to desired GPU machine for preset: ${PRESET}"
    read -r -p "Press Enter to run ${PRESET}..." _
  fi

  ./scripts/eval_run.sh "${PRESET}" "${BASE_URL}"
done

echo "Matrix complete."
