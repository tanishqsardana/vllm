#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx


PROMPTS: list[str] = [
    "Explain recursion with one simple example.",
    "Write a concise apology email for a delayed shipment.",
    "List three healthy breakfasts for busy mornings.",
    "Summarize TCP vs UDP in practical terms.",
    "Draft a short product blurb for wireless headphones.",
    "Provide five backend interview questions with brief rationale.",
    "Explain what vector databases are used for.",
    "Write a release note for bug fixes and stability improvements.",
    "Generate a SQL query to find top 10 customers by revenue.",
    "Create a two-day travel plan for San Francisco.",
    "Explain CAP theorem for a product engineer.",
    "Draft a customer support reply for password reset issues.",
    "Give a simple regex for validating basic email formats.",
    "Write an onboarding checklist for new engineers.",
    "Compare unit tests and integration tests with examples.",
    "Write a commit message for retry logic on API timeouts.",
    "Suggest three names for a fintech budgeting app.",
    "Explain deadlocks and common prevention strategies.",
    "Give a brief page-load performance improvement plan.",
    "Draft a Slack maintenance announcement for tonight.",
]

TOKEN_LIMITS: list[int] = [
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    72,
    80,
    96,
    112,
    128,
    160,
    192,
    224,
    256,
    320,
    384,
    448,
    512,
]


@dataclass
class Tenant:
    tenant_id: str
    tenant_name: str
    api_key: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(round((len(sorted_values) - 1) * p))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return float(sorted_values[idx])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a mixed prompt/token workload for 1 or 2 tenants and report tenant cost usage."
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--admin-token", default="")
    parser.add_argument("--duration-s", type=int, default=300)
    parser.add_argument("--tenants", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument("--usage-window", default="1h", choices=["1h", "24h", "7d"])
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name-prefix", default="cost-run")
    return parser.parse_args()


async def admin_request(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    admin_token: str,
    method: str,
    path: str,
    json_body: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    response = await client.request(
        method=method,
        url=f"{base_url}{path}",
        headers={"X-Admin-Token": admin_token},
        json=json_body,
    )
    try:
        payload = response.json()
    except Exception:
        payload = {"raw_text": response.text}
    return response.status_code, payload


async def fetch_model_id(client: httpx.AsyncClient, base_url: str) -> str:
    try:
        models_resp = await client.get(f"{base_url}/v1/models")
        if models_resp.status_code == 200:
            payload = models_resp.json()
            data = payload.get("data", [])
            if isinstance(data, list):
                for row in data:
                    if isinstance(row, dict) and isinstance(row.get("id"), str) and row.get("id"):
                        return str(row["id"])
    except Exception:
        pass

    try:
        version_resp = await client.get(f"{base_url}/version")
        if version_resp.status_code == 200:
            payload = version_resp.json()
            model_id = payload.get("model_id")
            if isinstance(model_id, str) and model_id:
                return model_id
    except Exception:
        pass

    return "ignored-by-gateway"


async def ensure_healthy(client: httpx.AsyncClient, base_url: str) -> None:
    live = await client.get(f"{base_url}/livez")
    health = await client.get(f"{base_url}/healthz")
    if live.status_code != 200 or health.status_code != 200:
        raise SystemExit(
            f"Gateway not healthy: /livez={live.status_code} /healthz={health.status_code}"
        )


async def create_tenant(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    admin_token: str,
    tenant_name: str,
    max_output_tokens: int,
) -> Tenant:
    body = {
        "tenant_name": tenant_name,
        "max_concurrent": 64,
        "rpm_limit": 120000,
        "tpm_limit": 5000000,
        "max_context_tokens": 8192,
        "max_output_tokens": max_output_tokens,
    }
    status, payload = await admin_request(
        client,
        base_url=base_url,
        admin_token=admin_token,
        method="POST",
        path="/admin/tenants",
        json_body=body,
    )
    if status != 200:
        raise SystemExit(f"Failed to create tenant {tenant_name}: HTTP {status} payload={payload}")

    try:
        return Tenant(
            tenant_id=str(payload["tenant_id"]),
            tenant_name=str(payload["tenant_name"]),
            api_key=str(payload["api_key"]),
        )
    except Exception as exc:
        raise SystemExit(f"Malformed tenant create response for {tenant_name}: {payload}") from exc


def extract_usage_tokens(payload: Any) -> tuple[int, int, int]:
    if not isinstance(payload, dict):
        return (0, 0, 0)
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return (0, 0, 0)
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    return (prompt, completion, total)


async def run_worker(
    *,
    worker_id: int,
    client: httpx.AsyncClient,
    base_url: str,
    tenants: list[Tenant],
    model_id: str,
    end_mono: float,
    seed: int,
    records: list[dict[str, Any]],
) -> None:
    rng = random.Random(seed + worker_id * 10007)
    while time.monotonic() < end_mono:
        tenant = tenants[rng.randrange(len(tenants))]
        prompt_idx = rng.randrange(len(PROMPTS))
        token_limit = TOKEN_LIMITS[rng.randrange(len(TOKEN_LIMITS))]
        req_id = uuid.uuid4().hex[:12]
        prompt = PROMPTS[prompt_idx]

        body = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n\n"
                        f"Constraints: keep response concise. "
                        f"req_id={req_id} prompt_id={prompt_idx} token_limit={token_limit}"
                    ),
                },
            ],
            "temperature": 0,
            "top_p": 1,
            "stream": False,
            "max_tokens": token_limit,
        }

        status = 0
        error_type = ""
        response_snippet = ""
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        start = time.perf_counter()
        try:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {tenant.api_key}"},
                json=body,
            )
            status = response.status_code
            try:
                payload = response.json()
            except Exception:
                payload = {"raw_text": response.text}

            if isinstance(payload, dict):
                err = payload.get("error")
                if isinstance(err, dict):
                    error_type = str(err.get("type") or "")
                prompt_tokens, completion_tokens, total_tokens = extract_usage_tokens(payload)

            if status != 200:
                response_snippet = response.text.strip().replace("\n", " ")[:220]
        except Exception as exc:
            error_type = exc.__class__.__name__
            response_snippet = str(exc).replace("\n", " ")[:220]
        latency_ms = (time.perf_counter() - start) * 1000.0

        records.append(
            {
                "ts_utc": utc_now_iso(),
                "worker_id": worker_id,
                "tenant_id": tenant.tenant_id,
                "tenant_name": tenant.tenant_name,
                "req_id": req_id,
                "prompt_id": prompt_idx,
                "token_limit": token_limit,
                "http_status": status,
                "error_type": error_type,
                "latency_ms": round(latency_ms, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "response_snippet": response_snippet,
            }
        )


def summarize_local(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(r.get("latency_ms", 0.0)) for r in records if float(r.get("latency_ms", 0.0)) > 0]
    latencies_200 = [
        float(r.get("latency_ms", 0.0))
        for r in records
        if int(r.get("http_status", 0)) == 200 and float(r.get("latency_ms", 0.0)) > 0
    ]
    status_counts: dict[str, int] = {}
    for row in records:
        key = str(int(row.get("http_status", 0)))
        status_counts[key] = status_counts.get(key, 0) + 1

    ok = sum(1 for r in records if int(r.get("http_status", 0)) == 200)
    return {
        "requests": len(records),
        "success_200": ok,
        "errors_non_200": len(records) - ok,
        "success_rate": (ok / len(records)) if records else 0.0,
        "status_counts": status_counts,
        "latency_ms_p50_all": round(pct(latencies, 0.50), 2),
        "latency_ms_p95_all": round(pct(latencies, 0.95), 2),
        "latency_ms_p50_200": round(pct(latencies_200, 0.50), 2),
        "latency_ms_p95_200": round(pct(latencies_200, 0.95), 2),
        "prompt_tokens_local_sum": int(sum(int(r.get("prompt_tokens", 0)) for r in records)),
        "completion_tokens_local_sum": int(sum(int(r.get("completion_tokens", 0)) for r in records)),
        "total_tokens_local_sum": int(sum(int(r.get("total_tokens", 0)) for r in records)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Tenant Cost Run")
    lines.append("")
    lines.append(f"- Run ID: `{summary['run_id']}`")
    lines.append(f"- Base URL: `{summary['base_url']}`")
    lines.append(f"- Duration (s): `{summary['duration_s']}`")
    lines.append(f"- Tenant count: `{summary['tenant_count']}`")
    lines.append(f"- Concurrency: `{summary['concurrency']}`")
    lines.append(f"- Prompt variants: `{summary['prompt_variants']}`")
    lines.append(f"- Token limit variants: `{summary['token_limit_variants']}`")
    lines.append(f"- Cost estimation enabled: `{summary['cost_estimation_enabled']}`")
    lines.append("")
    lines.append("## Combined (Admin Usage)")
    lines.append("")
    combined = summary["combined_admin"]
    lines.append(f"- Requests: `{combined['requests']}`")
    lines.append(f"- Errors: `{combined['errors']}`")
    lines.append(f"- Prompt tokens: `{combined['prompt_tokens']}`")
    lines.append(f"- Completion tokens: `{combined['completion_tokens']}`")
    lines.append(f"- Total tokens: `{combined['total_tokens']}`")
    lines.append(f"- GPU seconds est: `{combined['gpu_seconds_est_sum']:.4f}`")
    lines.append(f"- Cost USD est: `{combined['cost_est_sum']:.6f}`")
    lines.append(f"- Cost per 1K tokens (USD): `{combined['cost_per_1k_tokens_usd']:.6f}`")
    lines.append("")
    lines.append("## Tenant Breakdown")
    lines.append("")
    for tenant in summary["tenants"]:
        lines.append(f"### {tenant['tenant_name']} ({tenant['tenant_id']})")
        local = tenant["local"]
        admin = tenant["admin_usage"]
        lines.append(f"- Local requests: `{local['requests']}` (200: `{local['success_200']}`)")
        lines.append(f"- Local p50/p95 latency (ms): `{local['latency_ms_p50_all']}` / `{local['latency_ms_p95_all']}`")
        lines.append(f"- Admin requests/errors: `{admin['requests']}` / `{admin['errors']}`")
        lines.append(
            f"- Admin tokens (prompt/completion/total): "
            f"`{admin['prompt_tokens']}` / `{admin['completion_tokens']}` / `{admin['total_tokens']}`"
        )
        lines.append(f"- Admin cost USD est: `{admin['cost_est_sum']:.6f}`")
        lines.append("")

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


async def main() -> None:
    args = parse_args()
    admin_token = args.admin_token or ""
    if not admin_token:
        raise SystemExit("ADMIN_TOKEN is required (pass --admin-token or set env before invocation).")
    if args.tenants not in {1, 2}:
        raise SystemExit("--tenants must be 1 or 2")
    if args.duration_s <= 0:
        raise SystemExit("--duration-s must be > 0")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be > 0")

    base_url = args.base_url.rstrip("/")
    now = datetime.now(timezone.utc)
    run_id = now.strftime("cost_%Y-%m-%d_%H%M%S")
    run_dir = (Path(args.results_root).resolve() / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    timeout = httpx.Timeout(args.request_timeout_s, connect=10.0)
    records: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        await ensure_healthy(client, base_url)
        model_id = await fetch_model_id(client, base_url)

        stamp = now.strftime("%Y%m%d%H%M%S")
        tenants: list[Tenant] = []
        for idx in range(args.tenants):
            tenant_name = f"{args.name_prefix}-{stamp}-{idx + 1}"
            tenant = await create_tenant(
                client,
                base_url=base_url,
                admin_token=admin_token,
                tenant_name=tenant_name,
                max_output_tokens=max(TOKEN_LIMITS),
            )
            tenants.append(tenant)

        start_utc = utc_now_iso()
        start_mono = time.monotonic()
        end_mono = start_mono + float(args.duration_s)

        tasks = [
            asyncio.create_task(
                run_worker(
                    worker_id=worker_id,
                    client=client,
                    base_url=base_url,
                    tenants=tenants,
                    model_id=model_id,
                    end_mono=end_mono,
                    seed=args.seed,
                    records=records,
                )
            )
            for worker_id in range(args.concurrency)
        ]
        await asyncio.gather(*tasks)
        end_utc = utc_now_iso()
        duration_actual_s = max(0.001, time.monotonic() - start_mono)

        usage_paths = [
            f"/admin/usage/tenants?window={quote(args.usage_window, safe='')}",
            f"/admin/usage?window={quote(args.usage_window, safe='')}",
        ]
        usage_status = 0
        usage_payload: Any = {}
        usage_endpoint = ""
        attempts: list[dict[str, Any]] = []
        for path in usage_paths:
            usage_status, usage_payload = await admin_request(
                client,
                base_url=base_url,
                admin_token=admin_token,
                method="GET",
                path=path,
            )
            attempts.append({"path": path, "status": usage_status})
            if usage_status == 200 and isinstance(usage_payload, dict):
                usage_endpoint = path
                break
            if usage_status in {404, 405}:
                continue
            raise SystemExit(
                f"Failed to fetch usage rollup: HTTP {usage_status} path={path} payload={usage_payload}"
            )
        if usage_status != 200 or not isinstance(usage_payload, dict):
            raise SystemExit(
                f"Failed to fetch usage rollup on all known endpoints. attempts={attempts} last_payload={usage_payload}"
            )

    usage_rows = usage_payload.get("data", []) if isinstance(usage_payload, dict) else []
    if not isinstance(usage_rows, list):
        usage_rows = []
    cost_estimation_enabled = usage_payload.get("cost_estimation_enabled")
    if not isinstance(cost_estimation_enabled, bool):
        cost_estimation_enabled = any(
            float(row.get("cost_est_sum", 0.0)) > 0.0 for row in usage_rows if isinstance(row, dict)
        )
    rows_by_tenant = {
        str(row.get("tenant_id")): row for row in usage_rows if isinstance(row, dict)
    }
    local_by_tenant: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        tid = str(rec.get("tenant_id"))
        local_by_tenant.setdefault(tid, []).append(rec)

    tenant_summaries: list[dict[str, Any]] = []
    combined_admin = {
        "requests": 0,
        "errors": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "gpu_seconds_est_sum": 0.0,
        "cost_est_sum": 0.0,
        "cost_per_1k_tokens_usd": 0.0,
    }

    tenant_manifest = [
        {"tenant_id": t.tenant_id, "tenant_name": t.tenant_name} for t in tenants
    ]
    tenant_ids = {t["tenant_id"] for t in tenant_manifest}

    for tenant in tenant_manifest:
        tenant_id = tenant["tenant_id"]
        local_rows = local_by_tenant.get(tenant_id, [])
        local_summary = summarize_local(local_rows)

        admin_row = rows_by_tenant.get(tenant_id, {})
        admin_summary = {
            "requests": int(admin_row.get("requests", 0)),
            "errors": int(admin_row.get("errors", 0)),
            "prompt_tokens": int(admin_row.get("prompt_tokens", 0)),
            "completion_tokens": int(admin_row.get("completion_tokens", 0)),
            "total_tokens": int(admin_row.get("total_tokens", 0)),
            "p95_latency_ms": float(admin_row.get("p95_latency_ms", 0.0)),
            "gpu_seconds_est_sum": float(admin_row.get("gpu_seconds_est_sum", 0.0)),
            "cost_est_sum": float(admin_row.get("cost_est_sum", 0.0)),
        }

        combined_admin["requests"] += admin_summary["requests"]
        combined_admin["errors"] += admin_summary["errors"]
        combined_admin["prompt_tokens"] += admin_summary["prompt_tokens"]
        combined_admin["completion_tokens"] += admin_summary["completion_tokens"]
        combined_admin["total_tokens"] += admin_summary["total_tokens"]
        combined_admin["gpu_seconds_est_sum"] += admin_summary["gpu_seconds_est_sum"]
        combined_admin["cost_est_sum"] += admin_summary["cost_est_sum"]

        tenant_summaries.append(
            {
                "tenant_id": tenant_id,
                "tenant_name": tenant["tenant_name"],
                "local": local_summary,
                "admin_usage": admin_summary,
            }
        )

    if combined_admin["total_tokens"] > 0:
        combined_admin["cost_per_1k_tokens_usd"] = (
            combined_admin["cost_est_sum"] / combined_admin["total_tokens"]
        ) * 1000.0

    all_local = summarize_local(records)
    summary = {
        "run_id": run_id,
        "base_url": base_url,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "duration_s": round(duration_actual_s, 3),
        "tenant_count": args.tenants,
        "concurrency": args.concurrency,
        "usage_window": args.usage_window,
        "seed": args.seed,
        "prompt_variants": len(PROMPTS),
        "token_limit_variants": len(TOKEN_LIMITS),
        "token_limits": TOKEN_LIMITS,
        "cost_estimation_enabled": cost_estimation_enabled,
        "usage_endpoint_used": usage_endpoint,
        "model_id": model_id,
        "all_local": all_local,
        "combined_admin": combined_admin,
        "tenants": tenant_summaries,
    }

    filtered_usage_rows = [row for row in usage_rows if str(row.get("tenant_id")) in tenant_ids]
    artifacts = {
        "summary_json": run_dir / "summary.json",
        "summary_md": run_dir / "summary.md",
        "requests_csv": run_dir / "requests.csv",
        "usage_tenants_json": run_dir / "usage_tenants.json",
        "tenant_manifest_json": run_dir / "tenants.json",
    }

    artifacts["summary_json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_summary(artifacts["summary_md"], summary)
    write_csv(artifacts["requests_csv"], records)
    artifacts["usage_tenants_json"].write_text(
        json.dumps(
            {
                "window": usage_payload.get("window", args.usage_window),
                "cost_estimation_enabled": cost_estimation_enabled,
                "usage_endpoint_used": usage_endpoint,
                "data": filtered_usage_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    artifacts["tenant_manifest_json"].write_text(json.dumps(tenant_manifest, indent=2), encoding="utf-8")

    print(f"Run ID: {run_id}")
    print(f"Model ID: {model_id}")
    print(f"Duration seconds: {summary['duration_s']}")
    print(f"Prompt variants: {len(PROMPTS)}")
    print(f"Token limit variants: {len(TOKEN_LIMITS)}")
    print(f"Cost estimation enabled: {summary['cost_estimation_enabled']}")
    print("")
    print("Combined admin usage (selected tenants):")
    print(
        json.dumps(
            summary["combined_admin"],
            indent=2,
        )
    )
    print("")
    print("Artifacts:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
