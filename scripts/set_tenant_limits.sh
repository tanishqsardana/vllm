#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
ADMIN_TOKEN="${ADMIN_TOKEN:-}"
TENANT_REF="${1:-}"
PROFILE="${2:-standard}"

if [[ -z "${ADMIN_TOKEN}" ]]; then
  echo "ADMIN_TOKEN is required" >&2
  exit 1
fi

if [[ -z "${TENANT_REF}" ]]; then
  cat >&2 <<'USAGE'
Usage:
  BASE_URL=http://localhost:8000 ADMIN_TOKEN=<token> ./scripts/set_tenant_limits.sh <tenant_id_or_name> [profile]

Profiles:
  conservative  max_concurrent=1  rpm_limit=60   tpm_limit=30000  max_context_tokens=4096  max_output_tokens=256
  standard      max_concurrent=2  rpm_limit=180  tpm_limit=120000 max_context_tokens=8192  max_output_tokens=512
  pro           max_concurrent=4  rpm_limit=600  tpm_limit=400000 max_context_tokens=16384 max_output_tokens=1024

Optional overrides via env vars:
  MAX_CONCURRENT RPM_LIMIT TPM_LIMIT MAX_CONTEXT_TOKENS MAX_OUTPUT_TOKENS
USAGE
  exit 1
fi

case "${PROFILE}" in
  conservative)
    default_max_concurrent=1
    default_rpm_limit=60
    default_tpm_limit=30000
    default_max_context_tokens=4096
    default_max_output_tokens=256
    ;;
  standard)
    default_max_concurrent=2
    default_rpm_limit=180
    default_tpm_limit=120000
    default_max_context_tokens=8192
    default_max_output_tokens=512
    ;;
  pro)
    default_max_concurrent=4
    default_rpm_limit=600
    default_tpm_limit=400000
    default_max_context_tokens=16384
    default_max_output_tokens=1024
    ;;
  *)
    echo "Unknown profile: ${PROFILE}. Use conservative|standard|pro." >&2
    exit 1
    ;;
esac

max_concurrent="${MAX_CONCURRENT:-${default_max_concurrent}}"
rpm_limit="${RPM_LIMIT:-${default_rpm_limit}}"
tpm_limit="${TPM_LIMIT:-${default_tpm_limit}}"
max_context_tokens="${MAX_CONTEXT_TOKENS:-${default_max_context_tokens}}"
max_output_tokens="${MAX_OUTPUT_TOKENS:-${default_max_output_tokens}}"

tenants_tmp="$(mktemp)"
patch_tmp="$(mktemp)"
trap 'rm -f "${tenants_tmp}" "${patch_tmp}"' EXIT

list_code="$(curl -sS -o "${tenants_tmp}" -w "%{http_code}" \
  -X GET "${BASE_URL}/admin/tenants" \
  -H "X-Admin-Token: ${ADMIN_TOKEN}")"

if [[ "${list_code}" != "200" ]]; then
  echo "Failed to list tenants: HTTP ${list_code}" >&2
  cat "${tenants_tmp}" >&2
  exit 1
fi

tenant_id="$(python - <<'PY' "${tenants_tmp}" "${TENANT_REF}"
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
ref = sys.argv[2]

rows = payload.get("data") or []
match = None
for row in rows:
    tid = row.get("tenant_id")
    tname = row.get("tenant_name")
    if ref == tid or ref == tname:
        match = tid
        break

if not match:
    raise SystemExit(f"tenant not found by id/name: {ref}")

print(match)
PY
)"

patch_body="$(python - <<'PY' "${max_concurrent}" "${rpm_limit}" "${tpm_limit}" "${max_context_tokens}" "${max_output_tokens}"
import json
import sys

body = {
    "max_concurrent": int(sys.argv[1]),
    "rpm_limit": int(sys.argv[2]),
    "tpm_limit": int(sys.argv[3]),
    "max_context_tokens": int(sys.argv[4]),
    "max_output_tokens": int(sys.argv[5]),
}
print(json.dumps(body))
PY
)"

patch_code="$(curl -sS -o "${patch_tmp}" -w "%{http_code}" \
  -X PATCH "${BASE_URL}/admin/tenants/${tenant_id}" \
  -H "Content-Type: application/json" \
  -H "X-Admin-Token: ${ADMIN_TOKEN}" \
  -d "${patch_body}")"

if [[ "${patch_code}" != "200" ]]; then
  echo "Failed setting limits for tenant ${TENANT_REF}: HTTP ${patch_code}" >&2
  cat "${patch_tmp}" >&2
  exit 1
fi

python - <<'PY' "${patch_tmp}" "${PROFILE}"
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
profile = sys.argv[2]
out = {
    "applied_profile": profile,
    "tenant_id": payload["tenant_id"],
    "tenant_name": payload["tenant_name"],
    "max_concurrent": payload["max_concurrent"],
    "rpm_limit": payload["rpm_limit"],
    "tpm_limit": payload["tpm_limit"],
    "max_context_tokens": payload["max_context_tokens"],
    "max_output_tokens": payload["max_output_tokens"],
}
print(json.dumps(out, indent=2))
PY
