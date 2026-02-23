#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"
POLL_SECONDS="${POLL_SECONDS:-5}"
MODEL_ID="${MODEL_ID:-}"

deadline=$((SECONDS + TIMEOUT_SECONDS))

echo "Waiting for readiness at ${BASE_URL}/healthz (timeout=${TIMEOUT_SECONDS}s)..."
until curl -fsS "${BASE_URL}/healthz" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for /healthz"
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done
echo "Service is ready."

if [[ -z "${MODEL_ID}" ]]; then
  models_json="$(curl -fsS "${BASE_URL}/v1/models")"
  MODEL_ID="$(python - <<'PY' "${models_json}"
import json
import sys

payload = json.loads(sys.argv[1])
data = payload.get("data") or []
if not data or "id" not in data[0]:
    raise SystemExit("Could not determine model id from /v1/models")
print(data[0]["id"])
PY
)"
fi

echo "Using model: ${MODEL_ID}"
request_body="$(cat <<JSON
{
  "model": "${MODEL_ID}",
  "messages": [
    {"role": "user", "content": "Say hello in five words."}
  ],
  "temperature": 0,
  "max_tokens": 32
}
JSON
)"

tmp_response="$(mktemp)"
status_code="$(curl -sS -o "${tmp_response}" -w "%{http_code}" \
  -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "${request_body}")"

if [[ "${status_code}" != "200" ]]; then
  echo "Chat completions request failed with HTTP ${status_code}"
  cat "${tmp_response}"
  rm -f "${tmp_response}"
  exit 1
fi

python - <<'PY' "${tmp_response}"
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
content = (
    payload.get("choices", [{}])[0]
    .get("message", {})
    .get("content")
)
if not content:
    raise SystemExit("Missing choices[0].message.content in response")
snippet = str(content).replace("\n", " ")[:200]
print(f"Smoke test passed. Response snippet: {snippet}")
PY

rm -f "${tmp_response}"
