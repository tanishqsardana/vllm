#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
ADMIN_TOKEN="${ADMIN_TOKEN:-}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"
POLL_SECONDS="${POLL_SECONDS:-5}"
MODEL_ID="${MODEL_ID:-}"
EXPECT_ADMIN_UNAVAILABLE="${EXPECT_ADMIN_UNAVAILABLE:-0}"

if [[ "${EXPECT_ADMIN_UNAVAILABLE}" != "1" && -z "${ADMIN_TOKEN}" ]]; then
  echo "ADMIN_TOKEN is required" >&2
  exit 1
fi

deadline=$((SECONDS + TIMEOUT_SECONDS))

echo "Waiting for ${BASE_URL}/healthz..."
until curl -fsS "${BASE_URL}/healthz" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for /healthz" >&2
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done
echo "Gateway is healthy."

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

assert_json_error_type() {
  local file_path="$1"
  local expected_error_type="$2"
  python - <<'PY' "${file_path}" "${expected_error_type}"
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
expected = sys.argv[2]
actual = ((payload.get("error") or {}).get("type"))
if actual != expected:
    raise SystemExit(f"expected error.type={expected}, got {actual}; payload={payload}")
PY
}

if [[ "${EXPECT_ADMIN_UNAVAILABLE}" == "1" ]]; then
  admin_unavailable_code="$(curl -sS -o "${tmp_dir}/admin_unavailable.json" -w "%{http_code}" \
    -X GET "${BASE_URL}/admin/tenants" \
    -H "X-Admin-Token: any-token")"
  if [[ "${admin_unavailable_code}" != "503" ]]; then
    echo "Expected admin endpoint to return 503 when ADMIN_TOKEN is not configured, got ${admin_unavailable_code}" >&2
    cat "${tmp_dir}/admin_unavailable.json" >&2
    exit 1
  fi
  assert_json_error_type "${tmp_dir}/admin_unavailable.json" "admin_unavailable"
  echo "Phase 2 admin-unavailable check passed."
  exit 0
fi

echo "Checking admin endpoint auth rejections..."
missing_admin_code="$(curl -sS -o "${tmp_dir}/admin_missing_header.json" -w "%{http_code}" \
  -X GET "${BASE_URL}/admin/tenants")"
if [[ "${missing_admin_code}" != "401" ]]; then
  echo "Expected admin request without X-Admin-Token to fail with 401, got ${missing_admin_code}" >&2
  cat "${tmp_dir}/admin_missing_header.json" >&2
  exit 1
fi
assert_json_error_type "${tmp_dir}/admin_missing_header.json" "unauthorized"

wrong_admin_code="$(curl -sS -o "${tmp_dir}/admin_wrong_header.json" -w "%{http_code}" \
  -X GET "${BASE_URL}/admin/tenants" \
  -H "X-Admin-Token: wrong-token")"
if [[ "${wrong_admin_code}" != "401" ]]; then
  echo "Expected admin request with invalid X-Admin-Token to fail with 401, got ${wrong_admin_code}" >&2
  cat "${tmp_dir}/admin_wrong_header.json" >&2
  exit 1
fi
assert_json_error_type "${tmp_dir}/admin_wrong_header.json" "unauthorized"

create_tenant() {
  local name="$1"
  local out_file="$2"
  local body
  body="$(printf '{"tenant_name":"%s"}' "${name}")"

  local code
  code="$(curl -sS -o "${out_file}" -w "%{http_code}" \
    -X POST "${BASE_URL}/admin/tenants" \
    -H "Content-Type: application/json" \
    -H "X-Admin-Token: ${ADMIN_TOKEN}" \
    -d "${body}")"

  if [[ "${code}" != "200" ]]; then
    echo "Failed creating tenant ${name}: HTTP ${code}" >&2
    cat "${out_file}" >&2
    exit 1
  fi
}

create_tenant "tenant-a-$(date +%s)" "${tmp_dir}/tenant_a.json"
create_tenant "tenant-b-$(date +%s)" "${tmp_dir}/tenant_b.json"

TENANT_A_ID="$(python - <<'PY' "${tmp_dir}/tenant_a.json"
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(payload["tenant_id"])
PY
)"
TENANT_A_KEY="$(python - <<'PY' "${tmp_dir}/tenant_a.json"
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(payload["api_key"])
PY
)"
TENANT_B_ID="$(python - <<'PY' "${tmp_dir}/tenant_b.json"
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(payload["tenant_id"])
PY
)"
TENANT_B_KEY="$(python - <<'PY' "${tmp_dir}/tenant_b.json"
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(payload["api_key"])
PY
)"

run_chat() {
  local api_key="$1"
  local prompt="$2"
  local max_tokens="$3"
  local out_file="$4"
  local request_body
  request_body="$(cat <<JSON
{
  "model": "${MODEL_ID}",
  "messages": [{"role": "user", "content": "${prompt}"}],
  "temperature": 0,
  "max_tokens": ${max_tokens}
}
JSON
)"

  curl -sS -o "${out_file}" -w "%{http_code}" \
    -X POST "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${api_key}" \
    -d "${request_body}"
}

a_status="$(run_chat "${TENANT_A_KEY}" "Say hello in five words." 32 "${tmp_dir}/a_single.json")"
b_status="$(run_chat "${TENANT_B_KEY}" "Say hello in five words." 32 "${tmp_dir}/b_single.json")"

if [[ "${a_status}" != "200" ]]; then
  echo "Tenant A initial request failed: HTTP ${a_status}" >&2
  cat "${tmp_dir}/a_single.json" >&2
  exit 1
fi

if [[ "${b_status}" != "200" ]]; then
  echo "Tenant B initial request failed: HTTP ${b_status}" >&2
  cat "${tmp_dir}/b_single.json" >&2
  exit 1
fi

patch_body='{"max_concurrent":1}'
patch_code="$(curl -sS -o "${tmp_dir}/patch_a.json" -w "%{http_code}" \
  -X PATCH "${BASE_URL}/admin/tenants/${TENANT_A_ID}" \
  -H "Content-Type: application/json" \
  -H "X-Admin-Token: ${ADMIN_TOKEN}" \
  -d "${patch_body}")"

if [[ "${patch_code}" != "200" ]]; then
  echo "Failed patching tenant A max_concurrent: HTTP ${patch_code}" >&2
  cat "${tmp_dir}/patch_a.json" >&2
  exit 1
fi

A_PROMPT="Write exactly 120 short bullet points about distributed systems contention and include no introduction."

for i in 1 2 3; do
  (
    run_chat "${TENANT_A_KEY}" "${A_PROMPT}" 512 "${tmp_dir}/a_burst_${i}.json" > "${tmp_dir}/a_burst_${i}.status"
  ) &
done

sleep 0.2
b_burst_status="$(run_chat "${TENANT_B_KEY}" "Respond with the word OK." 16 "${tmp_dir}/b_during_burst.json")"

wait

a_429_count=0
for i in 1 2 3; do
  status="$(cat "${tmp_dir}/a_burst_${i}.status")"
  if [[ "${status}" == "429" ]]; then
    a_429_count=$((a_429_count + 1))
  fi
done

if (( a_429_count < 2 )); then
  echo "Expected at least 2 tenant A requests to be rejected with 429, got ${a_429_count}" >&2
  for i in 1 2 3; do
    echo "tenant A burst request ${i}: HTTP $(cat "${tmp_dir}/a_burst_${i}.status")" >&2
    cat "${tmp_dir}/a_burst_${i}.json" >&2
  done
  exit 1
fi

if [[ "${b_burst_status}" != "200" ]]; then
  echo "Expected tenant B request during tenant A burst to succeed, got HTTP ${b_burst_status}" >&2
  cat "${tmp_dir}/b_during_burst.json" >&2
  exit 1
fi

echo "Phase 2 smoke passed."
echo "tenant_a_id=${TENANT_A_ID} tenant_b_id=${TENANT_B_ID} a_429_count=${a_429_count}"
