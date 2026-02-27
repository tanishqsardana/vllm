#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
ADMIN_TOKEN="${ADMIN_TOKEN:-}"
TENANT_NAME="${TENANT_NAME:-phase4-tenant}"
SEAT_NAME="${SEAT_NAME:-phase4-seat}"
WINDOW="${WINDOW:-24h}"
BUDGET_WINDOW="${BUDGET_WINDOW:-day}"
LOW_BUDGET_USD="${LOW_BUDGET_USD:-0.000001}"
MODEL_ID="${MODEL_ID:-}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
POLL_SECONDS="${POLL_SECONDS:-5}"

if [[ -z "${ADMIN_TOKEN}" ]]; then
  echo "ADMIN_TOKEN is required" >&2
  exit 1
fi

if [[ -z "${MODEL_ID}" ]]; then
  version_json="$(curl -fsS "${BASE_URL}/version")"
  MODEL_ID="$(python - <<'PY' "${version_json}"
import json
import sys
payload = json.loads(sys.argv[1])
print(payload.get("model_id") or "")
PY
)"
fi

if [[ -z "${MODEL_ID}" ]]; then
  echo "MODEL_ID not provided and /version did not return model_id" >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

echo "Waiting for ${BASE_URL}/healthz ..."
deadline=$((SECONDS + TIMEOUT_SECONDS))
until curl -fsS "${BASE_URL}/healthz" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for /healthz" >&2
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done
echo "Gateway is healthy."

call_admin() {
  local method="$1"
  local path="$2"
  local body="${3:-}"
  local out_file="$4"

  if [[ -n "${body}" ]]; then
    curl -sS -o "${out_file}" -w "%{http_code}" \
      -X "${method}" "${BASE_URL}${path}" \
      -H "Content-Type: application/json" \
      -H "X-Admin-Token: ${ADMIN_TOKEN}" \
      -d "${body}"
  else
    curl -sS -o "${out_file}" -w "%{http_code}" \
      -X "${method}" "${BASE_URL}${path}" \
      -H "X-Admin-Token: ${ADMIN_TOKEN}"
  fi
}

get_or_create_tenant() {
  local list_file="${tmp_dir}/tenants.json"
  local code
  code="$(call_admin GET "/admin/tenants" "" "${list_file}")"
  if [[ "${code}" != "200" ]]; then
    echo "Failed to list tenants: HTTP ${code}" >&2
    cat "${list_file}" >&2
    exit 1
  fi

  local existing_id
  existing_id="$(python - <<'PY' "${list_file}" "${TENANT_NAME}"
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
name = sys.argv[2]
for row in payload.get("data", []):
    if row.get("tenant_name") == name:
        print(row.get("tenant_id"))
        raise SystemExit(0)
print("")
PY
)"

  if [[ -n "${existing_id}" ]]; then
    echo "Using existing tenant: ${TENANT_NAME} (${existing_id})"
    TENANT_ID="${existing_id}"
    return
  fi

  local create_file="${tmp_dir}/tenant_create.json"
  local create_body
  create_body="$(printf '{"tenant_name":"%s"}' "${TENANT_NAME}")"
  code="$(call_admin POST "/admin/tenants" "${create_body}" "${create_file}")"
  if [[ "${code}" != "200" ]]; then
    echo "Failed to create tenant: HTTP ${code}" >&2
    cat "${create_file}" >&2
    exit 1
  fi

  TENANT_ID="$(python - <<'PY' "${create_file}"
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(payload["tenant_id"])
PY
)"
  echo "Created tenant: ${TENANT_NAME} (${TENANT_ID})"
}

create_seat() {
  local out_file="${tmp_dir}/seat_create.json"
  local body
  body="$(python - <<'PY' "${TENANT_ID}" "${SEAT_NAME}"
import json
import sys
print(json.dumps({"tenant_id": sys.argv[1], "seat_name": sys.argv[2], "role": "user"}))
PY
)"

  local code
  code="$(call_admin POST "/admin/seats" "${body}" "${out_file}")"
  if [[ "${code}" != "200" ]]; then
    echo "Failed to create seat: HTTP ${code}" >&2
    cat "${out_file}" >&2
    exit 1
  fi

  SEAT_ID="$(python - <<'PY' "${out_file}"
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(payload["seat_id"])
PY
)"
  echo "Created seat: ${SEAT_NAME} (${SEAT_ID})"
}

create_key_for_seat() {
  local out_file="${tmp_dir}/key_create.json"
  local body
  body="$(python - <<'PY' "${TENANT_ID}" "${SEAT_ID}"
import json
import sys
print(json.dumps({"tenant_id": sys.argv[1], "seat_id": sys.argv[2], "name": "phase4-smoke-key"}))
PY
)"

  local code
  code="$(call_admin POST "/admin/keys" "${body}" "${out_file}")"
  if [[ "${code}" != "200" ]]; then
    echo "Failed to create key: HTTP ${code}" >&2
    cat "${out_file}" >&2
    exit 1
  fi

  API_KEY="$(python - <<'PY' "${out_file}"
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(payload["api_key"])
PY
)"
  echo "Created key for seat ${SEAT_ID}"
}

run_inference() {
  local prompt="$1"
  local out_file="$2"
  local body
  body="$(python - <<'PY' "${MODEL_ID}" "${prompt}"
import json
import sys
print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": sys.argv[2]}],
    "temperature": 0,
    "max_tokens": 32,
}))
PY
)"

  curl -sS -o "${out_file}" -w "%{http_code}" \
    -X POST "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -d "${body}"
}

set_low_budget() {
  local out_file="${tmp_dir}/budget_put.json"
  local body
  body="$(python - <<'PY' "${BUDGET_WINDOW}" "${LOW_BUDGET_USD}"
import json
import sys
print(json.dumps({"window": sys.argv[1], "budget_usd": float(sys.argv[2])}))
PY
)"

  local code
  code="$(call_admin PUT "/admin/budgets/${TENANT_ID}" "${body}" "${out_file}")"
  if [[ "${code}" != "200" ]]; then
    echo "Failed to put budget: HTTP ${code}" >&2
    cat "${out_file}" >&2
    exit 1
  fi

  echo "Set low budget: ${LOW_BUDGET_USD} USD (${BUDGET_WINDOW})"
}

get_or_create_tenant
create_seat
create_key_for_seat

echo "Running 5 inference calls..."
for i in 1 2 3 4 5; do
  status="$(run_inference "Phase4 smoke request ${i}: return OK." "${tmp_dir}/infer_${i}.json")"
  if [[ "${status}" != "200" ]]; then
    echo "Inference ${i} failed: HTTP ${status}" >&2
    cat "${tmp_dir}/infer_${i}.json" >&2
    exit 1
  fi
  echo "  call ${i}: HTTP ${status}"
done

set_low_budget

echo "Triggering budget evaluation with one more request..."
extra_status="$(run_inference "Trigger budget evaluation and say OK." "${tmp_dir}/infer_trigger.json")"
if [[ "${extra_status}" != "200" ]]; then
  echo "Budget trigger inference failed: HTTP ${extra_status}" >&2
  cat "${tmp_dir}/infer_trigger.json" >&2
  exit 1
fi

usage_file="${tmp_dir}/usage_seats.json"
usage_code="$(call_admin GET "/admin/usage/seats?window=${WINDOW}&tenant_id=${TENANT_ID}" "" "${usage_file}")"
if [[ "${usage_code}" != "200" ]]; then
  echo "Failed to fetch /admin/usage/seats: HTTP ${usage_code}" >&2
  cat "${usage_file}" >&2
  exit 1
fi

budget_status_file="${tmp_dir}/budget_status.json"
budget_status_code="$(call_admin GET "/admin/budget_status?window=${BUDGET_WINDOW}&tenant_id=${TENANT_ID}" "" "${budget_status_file}")"
if [[ "${budget_status_code}" != "200" ]]; then
  echo "Failed to fetch /admin/budget_status: HTTP ${budget_status_code}" >&2
  cat "${budget_status_file}" >&2
  exit 1
fi

echo
python - <<'PY' "${usage_file}" "${budget_status_file}"
import json
import pathlib
import sys
usage = json.loads(pathlib.Path(sys.argv[1]).read_text())
budget = json.loads(pathlib.Path(sys.argv[2]).read_text())
print("=== /admin/usage/seats ===")
print(json.dumps(usage, indent=2))
print()
print("=== /admin/budget_status ===")
print(json.dumps(budget, indent=2))
PY

echo
echo "Phase 4 smoke completed successfully. tenant_id=${TENANT_ID} seat_id=${SEAT_ID}"
