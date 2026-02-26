#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
ADMIN_TOKEN="${ADMIN_TOKEN:-}"
TENANT_NAME="${1:-tenant-$(date +%s)}"

if [[ -z "${ADMIN_TOKEN}" ]]; then
  echo "ADMIN_TOKEN is required" >&2
  exit 1
fi

MAX_CONCURRENT="${MAX_CONCURRENT:-}"
RPM_LIMIT="${RPM_LIMIT:-}"
TPM_LIMIT="${TPM_LIMIT:-}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-}"

request_body="$(python - <<'PY' "${TENANT_NAME}" "${MAX_CONCURRENT}" "${RPM_LIMIT}" "${TPM_LIMIT}" "${MAX_CONTEXT_TOKENS}" "${MAX_OUTPUT_TOKENS}"
import json
import sys

body = {"tenant_name": sys.argv[1]}
keys = [
    ("max_concurrent", sys.argv[2]),
    ("rpm_limit", sys.argv[3]),
    ("tpm_limit", sys.argv[4]),
    ("max_context_tokens", sys.argv[5]),
    ("max_output_tokens", sys.argv[6]),
]
for key, value in keys:
    if value:
        body[key] = int(value)
print(json.dumps(body))
PY
)"

tmp_response="$(mktemp)"
status_code="$(curl -sS -o "${tmp_response}" -w "%{http_code}" \
  -X POST "${BASE_URL}/admin/tenants" \
  -H "Content-Type: application/json" \
  -H "X-Admin-Token: ${ADMIN_TOKEN}" \
  -d "${request_body}")"

if [[ "${status_code}" != "200" ]]; then
  echo "Tenant creation failed with HTTP ${status_code}" >&2
  cat "${tmp_response}" >&2
  rm -f "${tmp_response}"
  exit 1
fi

python - <<'PY' "${tmp_response}"
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(json.dumps({
    "tenant_id": payload["tenant_id"],
    "tenant_name": payload["tenant_name"],
    "api_key": payload["api_key"],
    "max_concurrent": payload["max_concurrent"],
    "rpm_limit": payload["rpm_limit"],
    "tpm_limit": payload["tpm_limit"],
    "max_context_tokens": payload["max_context_tokens"],
    "max_output_tokens": payload["max_output_tokens"],
}, indent=2))
PY

rm -f "${tmp_response}"
