#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
ADMIN_TOKEN="${ADMIN_TOKEN:-}"
ADMIN_JWT="${ADMIN_JWT:-}"
TENANT_NAME="${TENANT_NAME:-audit-smoke-tenant-$(date +%s)}"
SEAT_NAME="${SEAT_NAME:-audit-smoke-seat}"

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

echo "Creating tenant for audit mutation ..."
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

echo "Performing audited admin mutation (create seat) ..."
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

echo "Checking /admin/audit ..."
audit_code="$(admin_curl GET "/admin/audit?window=24h&tenant_id=${TENANT_ID}" "" "${TMP_DIR}/audit.json")"
if [[ "${audit_code}" != "200" ]]; then
  echo "Audit fetch failed: HTTP ${audit_code}" >&2
  cat "${TMP_DIR}/audit.json" >&2
  exit 1
fi

python - <<'PY' "${TMP_DIR}/audit.json" "${SEAT_ID}"
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
seat_id = sys.argv[2]
rows = payload.get("data", [])
if not rows:
    raise SystemExit("audit log is empty")

matched = [r for r in rows if r.get("resource_type") == "seat" and r.get("resource_id") == seat_id]
if not matched:
    raise SystemExit(f"expected seat audit event for {seat_id} not found")

print("Audit verification OK")
PY

echo "Audit smoke passed. tenant_id=${TENANT_ID} seat_id=${SEAT_ID}"
