#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
ADMIN_TOKEN="${ADMIN_TOKEN:-}"
ADMIN_JWT="${ADMIN_JWT:-}"
TENANT_NAME="${TENANT_NAME:-phase5-tenant-$(date +%s)}"
SEAT_NAME="${SEAT_NAME:-phase5-seat}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

AUTH_INFO="$(curl -fsS "${BASE_URL}/admin/auth_info")"
ADMIN_AUTH_MODE="$(python - <<'PY' "${AUTH_INFO}"
import json, sys
print((json.loads(sys.argv[1]) or {}).get("admin_auth_mode") or "")
PY
)"

if [[ "${ADMIN_AUTH_MODE}" == "static_token" && -z "${ADMIN_TOKEN}" ]]; then
  echo "ADMIN_TOKEN is required for static_token mode" >&2
  exit 1
fi

if [[ "${ADMIN_AUTH_MODE}" == "oidc" && -z "${ADMIN_JWT}" ]]; then
  echo "ADMIN_JWT is required for oidc mode" >&2
  exit 1
fi

admin_curl() {
  local method="$1"
  local path="$2"
  local body="${3:-}"
  local out_file="$4"

  local -a cmd=(curl -sS -o "${out_file}" -w "%{http_code}" -X "${method}" "${BASE_URL}${path}")

  if [[ "${ADMIN_AUTH_MODE}" == "static_token" ]]; then
    cmd+=( -H "X-Admin-Token: ${ADMIN_TOKEN}" )
  else
    cmd+=( -H "Authorization: Bearer ${ADMIN_JWT}" )
  fi

  if [[ -n "${body}" ]]; then
    cmd+=( -H "Content-Type: application/json" -d "${body}" )
  fi

  "${cmd[@]}"
}

echo "Checking /ui ..."
ui_code="$(curl -sS -o "${TMP_DIR}/ui.html" -w "%{http_code}" "${BASE_URL}/ui")"
if [[ "${ui_code}" != "200" ]]; then
  echo "/ui expected HTTP 200, got ${ui_code}" >&2
  cat "${TMP_DIR}/ui.html" >&2
  exit 1
fi

echo "Checking /version fields ..."
version_json="$(curl -fsS "${BASE_URL}/version")"
python - <<'PY' "${version_json}"
import json, sys
payload = json.loads(sys.argv[1])
required = [
    "build_sha",
    "build_time",
    "model_id",
    "vllm_version",
    "ui_enabled",
    "metrics_enabled",
    "admin_auth_mode",
    "profiles_enabled",
]
missing = [k for k in required if k not in payload]
if missing:
    raise SystemExit(f"missing /version fields: {missing}")
print("/version fields OK")
PY

echo "Creating tenant ..."
tenant_body="$(python - <<'PY' "${TENANT_NAME}"
import json, sys
print(json.dumps({"tenant_name": sys.argv[1]}))
PY
)"
tenant_code="$(admin_curl POST "/admin/tenants" "${tenant_body}" "${TMP_DIR}/tenant.json")"
if [[ "${tenant_code}" != "200" ]]; then
  echo "Tenant creation failed: HTTP ${tenant_code}" >&2
  cat "${TMP_DIR}/tenant.json" >&2
  exit 1
fi
TENANT_ID="$(python - <<'PY' "${TMP_DIR}/tenant.json"
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["tenant_id"])
PY
)"

echo "Creating seat ..."
seat_body="$(python - <<'PY' "${TENANT_ID}" "${SEAT_NAME}"
import json, sys
print(json.dumps({"tenant_id": sys.argv[1], "seat_name": sys.argv[2], "role": "user"}))
PY
)"
seat_code="$(admin_curl POST "/admin/seats" "${seat_body}" "${TMP_DIR}/seat.json")"
if [[ "${seat_code}" != "200" ]]; then
  echo "Seat creation failed: HTTP ${seat_code}" >&2
  cat "${TMP_DIR}/seat.json" >&2
  exit 1
fi
SEAT_ID="$(python - <<'PY' "${TMP_DIR}/seat.json"
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["seat_id"])
PY
)"

echo "Creating key ..."
key_body="$(python - <<'PY' "${TENANT_ID}" "${SEAT_ID}"
import json, sys
print(json.dumps({"tenant_id": sys.argv[1], "seat_id": sys.argv[2]}))
PY
)"
key_code="$(admin_curl POST "/admin/keys" "${key_body}" "${TMP_DIR}/key.json")"
if [[ "${key_code}" != "200" ]]; then
  echo "Key creation failed: HTTP ${key_code}" >&2
  cat "${TMP_DIR}/key.json" >&2
  exit 1
fi
KEY_ID="$(python - <<'PY' "${TMP_DIR}/key.json"
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["key_id"])
PY
)"

echo "Verifying list endpoints include created entities ..."
tenants_code="$(admin_curl GET "/admin/tenants" "" "${TMP_DIR}/tenants_list.json")"
seats_code="$(admin_curl GET "/admin/seats?tenant_id=${TENANT_ID}" "" "${TMP_DIR}/seats_list.json")"
keys_code="$(admin_curl GET "/admin/keys?tenant_id=${TENANT_ID}" "" "${TMP_DIR}/keys_list.json")"

if [[ "${tenants_code}" != "200" || "${seats_code}" != "200" || "${keys_code}" != "200" ]]; then
  echo "List endpoint failure: tenants=${tenants_code} seats=${seats_code} keys=${keys_code}" >&2
  exit 1
fi

python - <<'PY' "${TMP_DIR}/tenants_list.json" "${TMP_DIR}/seats_list.json" "${TMP_DIR}/keys_list.json" "${TENANT_ID}" "${SEAT_ID}" "${KEY_ID}"
import json, pathlib, sys

tenants = json.loads(pathlib.Path(sys.argv[1]).read_text()).get("data", [])
seats = json.loads(pathlib.Path(sys.argv[2]).read_text()).get("data", [])
keys = json.loads(pathlib.Path(sys.argv[3]).read_text()).get("data", [])

tenant_id = sys.argv[4]
seat_id = sys.argv[5]
key_id = sys.argv[6]

if not any(t.get("tenant_id") == tenant_id for t in tenants):
    raise SystemExit("tenant not found in /admin/tenants")
if not any(s.get("seat_id") == seat_id for s in seats):
    raise SystemExit("seat not found in /admin/seats")
if not any(k.get("key_id") == key_id for k in keys):
    raise SystemExit("key not found in /admin/keys")

print("Entity verification OK")
PY

echo "Phase 5 smoke passed. tenant_id=${TENANT_ID} seat_id=${SEAT_ID} key_id=${KEY_ID}"
