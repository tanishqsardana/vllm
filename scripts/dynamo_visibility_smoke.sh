#!/usr/bin/env bash
set -euo pipefail

# Usage: ADMIN_TOKEN=<token> BASE_URL=<url> ./scripts/dynamo_visibility_smoke.sh

BASE_URL="${BASE_URL:-http://localhost:8000}"
ADMIN_TOKEN="${ADMIN_TOKEN:-}"

if [[ -z "${ADMIN_TOKEN}" ]]; then
  echo "ADMIN_TOKEN is required" >&2
  exit 1
fi

echo "=== Dynamo Visibility Smoke Test ==="
echo "Base URL: ${BASE_URL}"
echo

# 1. Check /admin/dynamo/status
echo "--- /admin/dynamo/status ---"
status_code="$(curl -sS -o /tmp/dynamo_status.json -w "%{http_code}" \
  "${BASE_URL}/admin/dynamo/status" \
  -H "X-Admin-Token: ${ADMIN_TOKEN}")"

if [[ "${status_code}" != "200" ]]; then
  echo "FAIL: HTTP ${status_code}"
  cat /tmp/dynamo_status.json
  exit 1
fi

python3 -c "
import json
data = json.load(open('/tmp/dynamo_status.json'))
print(json.dumps(data, indent=2))
fe = data.get('frontend', {})
wk = data.get('worker', {})
print()
print(f'Frontend healthy: {fe.get(\"healthy\")}')
print(f'Worker healthy:   {wk.get(\"healthy\")}')
mc = data.get('model_config')
if mc:
    print(f'Model config:     {mc}')
else:
    print('Model config:     (none reported yet)')
"
echo

# 2. Check /admin/dynamo/metrics
echo "--- /admin/dynamo/metrics ---"
metrics_code="$(curl -sS -o /tmp/dynamo_metrics.json -w "%{http_code}" \
  "${BASE_URL}/admin/dynamo/metrics" \
  -H "X-Admin-Token: ${ADMIN_TOKEN}")"

if [[ "${metrics_code}" != "200" ]]; then
  echo "FAIL: HTTP ${metrics_code}"
  cat /tmp/dynamo_metrics.json
  exit 1
fi

python3 -c "
import json
data = json.load(open('/tmp/dynamo_metrics.json'))
print(json.dumps(data, indent=2))

fe = data.get('frontend', {})
wk = data.get('worker', {})
print()
print('Frontend:')
if 'inflight_requests' in fe:
    print(f'  Inflight:  {fe[\"inflight_requests\"]}')
if 'total_requests' in fe:
    print(f'  Total:     {fe[\"total_requests\"]}')
if 'ttft_seconds' in fe:
    ttft = fe['ttft_seconds']
    print(f'  TTFT p50:  {ttft.get(\"p50\", \"n/a\")}s  p95: {ttft.get(\"p95\", \"n/a\")}s')
if 'itl_seconds' in fe:
    itl = fe['itl_seconds']
    print(f'  ITL p50:   {itl.get(\"p50\", \"n/a\")}s  p95: {itl.get(\"p95\", \"n/a\")}s')

print('Worker:')
if 'gpu_cache_usage_percent' in wk:
    print(f'  GPU cache: {wk[\"gpu_cache_usage_percent\"]}%')
if 'requests_running' in wk:
    print(f'  Running:   {wk[\"requests_running\"]}')
if 'requests_waiting' in wk:
    print(f'  Waiting:   {wk[\"requests_waiting\"]}')
if not fe and not wk:
    print('  (no metrics yet — send some requests first)')
"
echo

# 3. Auth rejection check
echo "--- Auth check (no token → 401) ---"
noauth_code="$(curl -sS -o /dev/null -w "%{http_code}" \
  "${BASE_URL}/admin/dynamo/status")"
if [[ "${noauth_code}" == "401" ]]; then
  echo "OK: unauthenticated request correctly rejected (${noauth_code})"
else
  echo "WARN: expected 401, got ${noauth_code}"
fi

echo
echo "=== Dynamo visibility smoke passed ==="
