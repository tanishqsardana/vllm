#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
API_KEY="${API_KEY:-}"
TOTAL_REQUESTS="${TOTAL_REQUESTS:-100}"
CONCURRENCY="${CONCURRENCY:-20}"
MAX_TOKENS="${MAX_TOKENS:-64}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-60}"
PROMPT="${PROMPT:-Return exactly: OK}"
MODEL="${MODEL:-ignored-by-gateway}"

if [[ -z "${API_KEY}" ]]; then
  echo "API_KEY is required" >&2
  echo "Usage: BASE_URL=http://localhost:8000 API_KEY=<tenant_or_seat_key> ./scripts/rapid_concurrency_test.sh" >&2
  exit 1
fi

if ! [[ "${TOTAL_REQUESTS}" =~ ^[0-9]+$ ]] || (( TOTAL_REQUESTS < 1 )); then
  echo "TOTAL_REQUESTS must be a positive integer" >&2
  exit 1
fi

if ! [[ "${CONCURRENCY}" =~ ^[0-9]+$ ]] || (( CONCURRENCY < 1 )); then
  echo "CONCURRENCY must be a positive integer" >&2
  exit 1
fi

if ! [[ "${MAX_TOKENS}" =~ ^[0-9]+$ ]] || (( MAX_TOKENS < 1 )); then
  echo "MAX_TOKENS must be a positive integer" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

run_one() {
  local idx="$1"
  local resp_file="${TMP_DIR}/resp_${idx}.json"
  local req_file="${TMP_DIR}/req_${idx}.json"
  local meta_file="${TMP_DIR}/meta_${idx}.json"

  python - <<'PY' "${req_file}" "${MODEL}" "${PROMPT}" "${idx}" "${MAX_TOKENS}"
import json
import pathlib
import sys

out_path = pathlib.Path(sys.argv[1])
model = sys.argv[2]
prompt = sys.argv[3]
idx = int(sys.argv[4])
max_tokens = int(sys.argv[5])

payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": f"{prompt}\n\nrequest_id={idx}"},
    ],
    "temperature": 0,
    "max_tokens": max_tokens,
}
out_path.write_text(json.dumps(payload), encoding="utf-8")
PY

  local status_time=""
  local curl_exit=0

  set +e
  status_time="$(curl -sS --max-time "${REQUEST_TIMEOUT_S}" \
    -o "${resp_file}" -w "%{http_code} %{time_total}" \
    -X POST "${BASE_URL}/v1/chat/completions" \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "Content-Type: application/json" \
    --data-binary "@${req_file}")"
  curl_exit=$?
  set -e

  local http_status="0"
  local time_s="0"
  if [[ "${curl_exit}" -eq 0 ]]; then
    http_status="$(awk '{print $1}' <<<"${status_time}")"
    time_s="$(awk '{print $2}' <<<"${status_time}")"
  fi

  python - <<'PY' "${meta_file}" "${idx}" "${http_status}" "${time_s}" "${curl_exit}" "${resp_file}"
import json
import pathlib
import sys

meta_path = pathlib.Path(sys.argv[1])
idx = int(sys.argv[2])
http_status = int(sys.argv[3])
latency_s = float(sys.argv[4])
curl_exit = int(sys.argv[5])
resp_path = pathlib.Path(sys.argv[6])

error_type = ""
error_message = ""
response_snippet = ""

if resp_path.exists():
    raw = resp_path.read_text(encoding="utf-8", errors="ignore")
    response_snippet = raw.strip().replace("\n", " ")[:280]
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            if isinstance(payload.get("error"), dict):
                err = payload.get("error") or {}
                error_type = str(err.get("type") or "")
                error_message = str(err.get("message") or "")
    except Exception:
        pass

if curl_exit != 0:
    error_type = error_type or "curl_error"

row = {
    "idx": idx,
    "http_status": http_status,
    "latency_ms": round(latency_s * 1000.0, 2),
    "curl_exit": curl_exit,
    "error_type": error_type,
    "error_message": error_message,
    "response_snippet": response_snippet,
}
meta_path.write_text(json.dumps(row), encoding="utf-8")
PY
}

export -f run_one
export BASE_URL API_KEY TOTAL_REQUESTS CONCURRENCY MAX_TOKENS REQUEST_TIMEOUT_S PROMPT MODEL TMP_DIR

echo "Running rapid test: requests=${TOTAL_REQUESTS} concurrency=${CONCURRENCY} base=${BASE_URL}"
seq 1 "${TOTAL_REQUESTS}" | xargs -n 1 -P "${CONCURRENCY}" -I{} bash -c 'run_one "$@"' _ {}

python - <<'PY' "${TMP_DIR}" "${TOTAL_REQUESTS}" "${CONCURRENCY}"
import json
import pathlib
import statistics
import sys
from collections import Counter

base = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
concurrency = int(sys.argv[3])

rows = []
for path in sorted(base.glob("meta_*.json")):
    rows.append(json.loads(path.read_text(encoding="utf-8")))

if len(rows) != expected:
    print(f"WARN: expected {expected} results but found {len(rows)}")

status_counts = Counter(str(r.get("http_status", 0)) for r in rows)
error_counts = Counter((r.get("error_type") or "") for r in rows)
latencies = [float(r.get("latency_ms", 0.0)) for r in rows if float(r.get("latency_ms", 0.0)) > 0]
latencies_ok = [
    float(r.get("latency_ms", 0.0))
    for r in rows
    if int(r.get("http_status", 0)) == 200 and float(r.get("latency_ms", 0.0)) > 0
]


def pctl(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    idx = max(0, min(len(values) - 1, int(round((len(values) - 1) * p))))
    return values[idx]

success = sum(1 for r in rows if int(r.get("http_status", 0)) == 200)
rate_429 = sum(1 for r in rows if int(r.get("http_status", 0)) == 429)
non_200 = len(rows) - success

print("\n=== Rapid Concurrency Summary ===")
print(f"total_requests: {len(rows)}")
print(f"concurrency: {concurrency}")
print(f"success_200: {success}")
print(f"rejected_429: {rate_429}")
print(f"non_200: {non_200}")
print("status_counts:")
for k, v in sorted(status_counts.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 9999):
    print(f"  {k}: {v}")

if latencies:
    print("latency_ms_all:")
    print(f"  p50: {pctl(latencies, 0.50):.2f}")
    print(f"  p95: {pctl(latencies, 0.95):.2f}")
    print(f"  max: {max(latencies):.2f}")

if latencies_ok:
    print("latency_ms_200_only:")
    print(f"  p50: {pctl(latencies_ok, 0.50):.2f}")
    print(f"  p95: {pctl(latencies_ok, 0.95):.2f}")
    print(f"  max: {max(latencies_ok):.2f}")

non_empty_errors = {k: v for k, v in error_counts.items() if k}
if non_empty_errors:
    print("error_type_counts:")
    for k, v in sorted(non_empty_errors.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v}")

failed = [r for r in rows if int(r.get("http_status", 0)) != 200]
if failed:
    print("sample_failures:")
    for row in failed[:10]:
        print(
            f"  idx={row['idx']} status={row['http_status']} "
            f"error_type={row.get('error_type') or '-'} "
            f"msg={(row.get('error_message') or row.get('response_snippet') or '')[:120]}"
        )

out_path = base / "summary.json"
out_path.write_text(
    json.dumps(
        {
            "total_requests": len(rows),
            "concurrency": concurrency,
            "success_200": success,
            "rejected_429": rate_429,
            "status_counts": dict(status_counts),
            "error_type_counts": dict(non_empty_errors),
            "p50_latency_ms_all": pctl(latencies, 0.50),
            "p95_latency_ms_all": pctl(latencies, 0.95),
            "p50_latency_ms_200": pctl(latencies_ok, 0.50),
            "p95_latency_ms_200": pctl(latencies_ok, 0.95),
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(f"\nsummary_json: {out_path}")
PY
