from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from .dynamo_metrics import DynamoMetricsScraper

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from .accounting import (
    extract_usage,
    parse_budget_window,
    parse_window,
    record_request,
    usage_rollup_seats,
    usage_rollup_tenants,
    window_start_for_budget,
)
from .admin_auth import AdminAuthProvider, AdminPrincipal
from .audit import log_audit_event, request_id_from_headers
from .auth import generate_api_key, hash_api_key, parse_bearer_token, resolve_api_key
from .config_loader import load_config
from .db import Database
from .gpu_metrics import GPUMetricsPoller
from .limits import ConcurrencyManager
from .metrics import GatewayMetrics
from .middleware_metrics import GatewayMetricsMiddleware
from .rate_limit import TenantRateLimiter
from .schemas import (
    AuditEventPublic,
    AuditResponse,
    AuthInfoResponse,
    BudgetEventPublic,
    BudgetStatusResponse,
    ChatCompletionPayload,
    GatewayVersion,
    KeyCreateRequest,
    KeyCreateResponse,
    KeyPublic,
    ProfilePublic,
    SeatCreateRequest,
    SeatPatchRequest,
    SeatPublic,
    TenantBudgetPublic,
    TenantBudgetPutRequest,
    TenantCreateRequest,
    TenantCreateResponse,
    TenantPatchRequest,
    TenantPublic,
    UsageResponse,
    UsageSeatsResponse,
    UsageTenantsResponse,
)
from .utils import TokenEstimator, configure_logging, extract_prompt_text, json_log, latency_ms, to_iso, utc_now


@dataclass
class Settings:
    config_path: str
    model_id: str
    db_path: str
    admin_auth_mode: str
    admin_token: str | None
    jwks_url: str | None
    oidc_issuer: str | None
    oidc_audience: str | None
    role_claim: str
    groups_claim: str
    admin_group: str | None
    global_max_concurrent: int
    max_body_bytes: int
    upstream_url: str
    upstream_timeout_seconds: float
    build_sha: str
    build_time: str
    vllm_version: str
    default_max_concurrent: int
    default_rpm_limit: int
    default_tpm_limit: int
    default_max_context_tokens: int
    default_max_output_tokens: int
    trust_remote_code: bool
    metrics_tenant_labels: bool
    gpu_metrics_poll_interval_seconds: float
    gpu_hourly_rate: float
    metrics_enabled: bool
    ui_enabled: bool
    profiles_enabled: bool
    profiles_path: str
    ui_path: str

    @classmethod
    def from_env(cls) -> "Settings":
        cfg = load_config()

        try:
            vllm_version = version("vllm")
        except PackageNotFoundError:
            vllm_version = "unknown"

        def _clean_optional(value: str | None) -> str | None:
            if value is None:
                return None
            text = value.strip()
            return text or None

        return cls(
            config_path=str(cfg["CONFIG_PATH"]),
            model_id=str(cfg["MODEL_ID"]),
            db_path=str(cfg["DB_PATH"]),
            admin_auth_mode=str(cfg["ADMIN_AUTH_MODE"]),
            admin_token=_clean_optional(cfg.get("ADMIN_TOKEN")),
            jwks_url=_clean_optional(cfg.get("JWKS_URL")),
            oidc_issuer=_clean_optional(cfg.get("OIDC_ISSUER")),
            oidc_audience=_clean_optional(cfg.get("OIDC_AUDIENCE")),
            role_claim=str(cfg.get("ROLE_CLAIM") or "role"),
            groups_claim=str(cfg.get("GROUPS_CLAIM") or "groups"),
            admin_group=_clean_optional(cfg.get("ADMIN_GROUP")),
            global_max_concurrent=int(cfg["GLOBAL_MAX_CONCURRENT"]),
            max_body_bytes=int(cfg["MAX_BODY_BYTES"]),
            upstream_url=f"http://127.0.0.1:{int(cfg.get('DYNAMO_FRONTEND_PORT') or cfg['VLLM_PORT'])}",
            upstream_timeout_seconds=float(cfg["UPSTREAM_TIMEOUT_SECONDS"]),
            build_sha=str(cfg["BUILD_SHA"]),
            build_time=str(cfg["BUILD_TIME"]),
            vllm_version=vllm_version,
            default_max_concurrent=int(cfg["DEFAULT_MAX_CONCURRENT"]),
            default_rpm_limit=int(cfg["DEFAULT_RPM_LIMIT"]),
            default_tpm_limit=int(cfg["DEFAULT_TPM_LIMIT"]),
            default_max_context_tokens=int(cfg["DEFAULT_MAX_CONTEXT_TOKENS"]),
            default_max_output_tokens=int(cfg["DEFAULT_MAX_OUTPUT_TOKENS"]),
            trust_remote_code=bool(cfg["TRUST_REMOTE_CODE"]),
            metrics_tenant_labels=str(cfg["METRICS_TENANT_LABELS"]) == "on",
            gpu_metrics_poll_interval_seconds=float(cfg["GPU_METRICS_POLL_INTERVAL_SECONDS"]),
            gpu_hourly_rate=float(cfg["GPU_HOURLY_RATE"]),
            metrics_enabled=bool(cfg["METRICS_ENABLED"]),
            ui_enabled=bool(cfg["UI_ENABLED"]),
            profiles_enabled=bool(cfg["PROFILES_ENABLED"]),
            profiles_path=str(cfg["PROFILES_PATH"]),
            ui_path=str(cfg["UI_PATH"]),
        )


app = FastAPI(title="control-plane-gateway", docs_url=None, redoc_url=None)
app.add_middleware(GatewayMetricsMiddleware)


def error_response(status_code: int, error_type: str, message: str, request_id: str | None = None) -> JSONResponse:
    headers = {"x-request-id": request_id} if request_id else None
    return JSONResponse(
        status_code=status_code,
        content={"error": {"type": error_type, "message": message}},
        headers=headers,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"type": "http_error", "message": str(exc.detail)}},
    )


def _effective_limits(settings: Settings, body: TenantCreateRequest) -> dict[str, int]:
    return {
        "max_concurrent": body.max_concurrent if body.max_concurrent is not None else settings.default_max_concurrent,
        "rpm_limit": body.rpm_limit if body.rpm_limit is not None else settings.default_rpm_limit,
        "tpm_limit": body.tpm_limit if body.tpm_limit is not None else settings.default_tpm_limit,
        "max_context_tokens": body.max_context_tokens
        if body.max_context_tokens is not None
        else settings.default_max_context_tokens,
        "max_output_tokens": body.max_output_tokens
        if body.max_output_tokens is not None
        else settings.default_max_output_tokens,
    }


def _resolve_requested_max_tokens(payload: dict[str, Any], tenant_max_output_tokens: int) -> tuple[int, int]:
    requested_max = payload.get("max_tokens")
    if requested_max is None:
        requested_max = payload.get("max_completion_tokens")
    if requested_max is None:
        requested_max = tenant_max_output_tokens

    if not isinstance(requested_max, int) or requested_max <= 0:
        raise ValueError("max_tokens must be a positive integer")

    clamped = min(requested_max, tenant_max_output_tokens)
    return requested_max, clamped


def _cost_estimation_enabled(settings: Settings) -> bool:
    return settings.gpu_hourly_rate > 0.0


def _estimate_gpu_and_cost(settings: Settings, latency_ms_value: int, status_code: int) -> tuple[float, float]:
    if status_code >= 400:
        return 0.0, 0.0

    gpu_seconds_est = max(0.0, float(latency_ms_value) / 1000.0)
    if settings.gpu_hourly_rate <= 0.0:
        return gpu_seconds_est, 0.0

    cost_est = gpu_seconds_est * (settings.gpu_hourly_rate / 3600.0)
    return gpu_seconds_est, max(0.0, cost_est)


async def _check_admin(request: Request) -> AdminPrincipal:
    provider: AdminAuthProvider = app.state.admin_auth
    return provider.authenticate(request)


def _audit(
    *,
    request: Request,
    principal: AdminPrincipal,
    action: str,
    resource_type: str,
    resource_id: str,
    details: dict[str, Any] | None,
) -> None:
    db: Database = app.state.db
    req_id = request_id_from_headers(request.headers)
    log_audit_event(
        db=db,
        admin_identity=principal.identity,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        request_id=req_id,
    )


def _resolve_profile_path(settings: Settings) -> Path:
    return Path(settings.profiles_path).expanduser().resolve()


def _load_profiles(settings: Settings) -> dict[str, dict[str, Any]]:
    if not settings.profiles_enabled:
        return {}

    profile_dir = _resolve_profile_path(settings)
    if not profile_dir.exists() or not profile_dir.is_dir():
        json_log("profiles_dir_missing", profiles_path=str(profile_dir))
        return {}

    profiles: dict[str, dict[str, Any]] = {}
    for path in sorted(profile_dir.glob("*.yaml")):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("profile must be a mapping")

            normalized = _normalize_profile_payload(path.stem, payload)
            profile = ProfilePublic(**normalized)
            profiles[profile.name] = profile.model_dump()
        except Exception as exc:
            json_log("profile_load_failed", file=str(path), message=str(exc))
    return profiles


def _normalize_profile_payload(fallback_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["name"] = str(payload.get("name") or fallback_name)

    suggested_defaults = payload.get("suggested_defaults")
    if suggested_defaults is None:
        suggested_defaults = {}
    if not isinstance(suggested_defaults, dict):
        raise ValueError("suggested_defaults must be a mapping")

    tenant_limits = suggested_defaults.get("tenant_limits") or suggested_defaults.get("default_tenant_limits")
    if tenant_limits is None and isinstance(payload.get("default_tenant_limits"), dict):
        tenant_limits = payload.get("default_tenant_limits")

    merged_defaults: dict[str, Any] = {}
    if isinstance(suggested_defaults.get("global_max_concurrent"), (int, float)):
        merged_defaults["global_max_concurrent"] = int(suggested_defaults["global_max_concurrent"])
    elif isinstance(payload.get("global_max_concurrent"), (int, float)):
        merged_defaults["global_max_concurrent"] = int(payload["global_max_concurrent"])

    if isinstance(suggested_defaults.get("gpu_hourly_rate"), (int, float)):
        merged_defaults["gpu_hourly_rate"] = float(suggested_defaults["gpu_hourly_rate"])
    elif isinstance(payload.get("gpu_hourly_rate"), (int, float)):
        merged_defaults["gpu_hourly_rate"] = float(payload["gpu_hourly_rate"])

    if isinstance(tenant_limits, dict):
        merged_defaults["tenant_limits"] = tenant_limits

    normalized["suggested_defaults"] = merged_defaults or None
    return normalized


def _profiles_or_404(settings: Settings) -> dict[str, dict[str, Any]]:
    if not settings.profiles_enabled:
        raise HTTPException(status_code=404, detail={"error": {"type": "not_found", "message": "profiles disabled"}})
    return app.state.profiles


def _generate_runpod_snippet(settings: Settings, profile: dict[str, Any]) -> tuple[str, str, list[str]]:
    suggested = profile.get("suggested_defaults") or {}
    tenant_defaults = (suggested.get("tenant_limits") or {}) if isinstance(suggested, dict) else {}
    gpu_hourly_rate = suggested.get("gpu_hourly_rate") if isinstance(suggested, dict) else None
    if gpu_hourly_rate is None:
        gpu_hourly_rate = settings.gpu_hourly_rate

    run_lines = [
        f"MODEL_ID={profile['model_id']}",
        f"DTYPE={profile['dtype']}",
        f"MAX_MODEL_LEN={profile['max_model_len']}",
        f"TENSOR_PARALLEL={profile['tensor_parallel']}",
        f"GPU_MEMORY_UTILIZATION={profile['gpu_memory_utilization']}",
        f"GLOBAL_MAX_CONCURRENT={suggested.get('global_max_concurrent', settings.global_max_concurrent)}",
        f"DEFAULT_MAX_CONCURRENT={tenant_defaults.get('max_concurrent', settings.default_max_concurrent)}",
        f"DEFAULT_RPM_LIMIT={tenant_defaults.get('rpm_limit', settings.default_rpm_limit)}",
        f"DEFAULT_TPM_LIMIT={tenant_defaults.get('tpm_limit', settings.default_tpm_limit)}",
        f"DEFAULT_MAX_CONTEXT_TOKENS={tenant_defaults.get('max_context_tokens', settings.default_max_context_tokens)}",
        f"DEFAULT_MAX_OUTPUT_TOKENS={tenant_defaults.get('max_output_tokens', settings.default_max_output_tokens)}",
        f"ADMIN_AUTH_MODE={settings.admin_auth_mode}",
        "ADMIN_TOKEN=<set-if-static-token-mode>",
        f"DB_PATH={settings.db_path}",
        f"GPU_HOURLY_RATE={gpu_hourly_rate}",
    ]

    if settings.admin_auth_mode == "oidc":
        run_lines.extend(
            [
                "JWKS_URL=<your-jwks-url>",
                "OIDC_ISSUER=<your-issuer>",
                "OIDC_AUDIENCE=<your-audience>",
                f"ROLE_CLAIM={settings.role_claim}",
                f"GROUPS_CLAIM={settings.groups_claim}",
                f"ADMIN_GROUP={settings.admin_group or '<your-admin-group>'}",
            ]
        )

    if settings.admin_auth_mode == "static_token":
        auth_header = "-H \"X-Admin-Token: ${ADMIN_TOKEN}\""
    else:
        auth_header = "-H \"Authorization: Bearer ${ADMIN_JWT}\""

    bootstrap_lines = [
        "BASE_URL=http://<gateway-host>:8000",
        "# Step 1: create first tenant",
        (
            "curl -sS -X POST \"${BASE_URL}/admin/tenants\" "
            f"{auth_header} "
            "-H \"Content-Type: application/json\" "
            "-d '{\"tenant_name\":\"demo-tenant\"}'"
        ),
        "# Save tenant_id from response as TENANT_ID",
        "# Step 2: create a seat",
        (
            "curl -sS -X POST \"${BASE_URL}/admin/seats\" "
            f"{auth_header} "
            "-H \"Content-Type: application/json\" "
            "-d '{\"tenant_id\":\"<TENANT_ID>\",\"seat_name\":\"demo-seat\",\"role\":\"user\"}'"
        ),
        "# Save seat_id from response as SEAT_ID",
        "# Step 3: create seat-bound API key",
        (
            "curl -sS -X POST \"${BASE_URL}/admin/keys\" "
            f"{auth_header} "
            "-H \"Content-Type: application/json\" "
            "-d '{\"tenant_id\":\"<TENANT_ID>\",\"seat_id\":\"<SEAT_ID>\"}'"
        ),
    ]

    notes = [
        "Profiles are packaging presets only; they do not provision GPUs.",
        "Expose gateway port 8000 and persist /data + /cache volumes.",
        "In static_token mode use ADMIN_TOKEN header; in oidc mode send Bearer JWT.",
    ]

    return "\n".join(run_lines), "\n".join(bootstrap_lines), notes


async def _probe_upstream() -> bool:
    client: httpx.AsyncClient = app.state.http_client
    settings: Settings = app.state.settings

    try:
        resp = await client.get(f"{settings.upstream_url}/v1/models")
        if resp.status_code == 200:
            body = resp.json()
            if isinstance(body, dict) and body.get("data"):
                return True
    except (httpx.HTTPError, ValueError):
        pass

    try:
        payload = {
            "model": settings.model_id,
            "messages": [{"role": "user", "content": "ping"}],
            "temperature": 0,
            "max_tokens": 1,
        }
        resp = await client.post(f"{settings.upstream_url}/v1/chat/completions", json=payload)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def _record_request_with_metrics(
    *,
    metrics: GatewayMetrics,
    db: Database,
    ts_start: str,
    ts_end: str,
    latency_ms_value: int,
    tenant_id: str,
    seat_id: str | None,
    key_id: str | None,
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    status_code: int,
    error_type: str | None,
    request_id: str,
    gpu_seconds_est: float | None,
    cost_est: float | None,
) -> None:
    start = time.perf_counter()
    ok = False
    try:
        record_request(
            db=db,
            ts_start=ts_start,
            ts_end=ts_end,
            latency_ms=latency_ms_value,
            tenant_id=tenant_id,
            seat_id=seat_id,
            key_id=key_id,
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            status_code=status_code,
            error_type=error_type,
            request_id=request_id,
            gpu_seconds_est=gpu_seconds_est,
            cost_est=cost_est,
        )
        ok = True
    finally:
        metrics.observe_db_write(duration_seconds=max(0.0, time.perf_counter() - start), ok=ok)


def _refresh_tenant_cost_metric(db: Database, metrics: GatewayMetrics, tenant_id: str) -> None:
    threshold = to_iso(utc_now().replace(microsecond=0) - timedelta(hours=24))
    cost_24h = db.sum_tenant_cost_since(tenant_id, threshold)
    metrics.set_tenant_cost_estimated_usd(tenant_id, cost_24h)


def _evaluate_budget_for_tenant(db: Database, metrics: GatewayMetrics, tenant_id: str) -> None:
    budget = db.get_tenant_budget(tenant_id)
    if budget is None:
        return

    try:
        window = parse_budget_window(str(budget["window"]))
    except ValueError:
        return

    window_start_dt = window_start_for_budget(window)
    window_start = to_iso(window_start_dt)
    spend = db.sum_tenant_cost_since(tenant_id, window_start)
    budget_usd = float(budget["budget_usd"])

    checks = [
        ("50", 0.50, bool(budget.get("warn_50", True))),
        ("80", 0.80, bool(budget.get("warn_80", True))),
        ("100", 1.00, bool(budget.get("warn_100", True))),
    ]

    for threshold, ratio, enabled in checks:
        if not enabled:
            continue
        if spend + 1e-12 < budget_usd * ratio:
            continue
        if db.budget_event_exists(tenant_id=tenant_id, window=window, window_start=window_start, threshold=threshold):
            continue

        db.insert_budget_event(
            event_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            window=window,
            window_start=window_start,
            threshold=threshold,
            cost_usd=spend,
            budget_usd=budget_usd,
        )
        metrics.inc_budget_event(threshold=threshold, tenant_id=tenant_id)
        json_log(
            "budget_alert",
            tenant_id=tenant_id,
            window=window,
            window_start=window_start,
            threshold=threshold,
            cost_usd=spend,
            budget_usd=budget_usd,
        )


def _budget_status_payload(db: Database, settings: Settings, tenant_id: str, requested_window: str | None) -> dict[str, Any]:
    budget = db.get_tenant_budget(tenant_id)
    if budget is None:
        raise HTTPException(status_code=404, detail={"error": {"type": "not_found", "message": "budget not found"}})

    if requested_window is None:
        window = parse_budget_window(str(budget["window"]))
    else:
        window = parse_budget_window(requested_window)

    window_start_dt = window_start_for_budget(window)
    window_start = to_iso(window_start_dt)
    spend = db.sum_tenant_cost_since(tenant_id, window_start)
    budget_usd = float(budget["budget_usd"])
    ratio = 0.0 if budget_usd <= 0 else (spend / budget_usd)

    warn_50 = bool(budget.get("warn_50", True))
    warn_80 = bool(budget.get("warn_80", True))
    warn_100 = bool(budget.get("warn_100", True))

    crossed: list[str] = []
    if warn_50 and ratio >= 0.5:
        crossed.append("50")
    if warn_80 and ratio >= 0.8:
        crossed.append("80")
    if warn_100 and ratio >= 1.0:
        crossed.append("100")

    return {
        "tenant_id": tenant_id,
        "window": window,
        "window_start": window_start_dt,
        "budget_usd": budget_usd,
        "cost_usd": spend,
        "budget_ratio": ratio,
        "thresholds_crossed": crossed,
        "cost_estimation_enabled": _cost_estimation_enabled(settings),
    }


def _resolve_ui_file(settings: Settings, asset_path: str | None = None) -> Path:
    ui_root = Path(settings.ui_path).expanduser().resolve()
    if asset_path is None or asset_path.strip() == "":
        candidate = (ui_root / "index.html").resolve()
    else:
        candidate = (ui_root / asset_path).resolve()

    if ui_root != candidate and ui_root not in candidate.parents:
        raise HTTPException(status_code=404, detail={"error": {"type": "not_found", "message": "asset not found"}})
    return candidate


@app.on_event("startup")
async def startup() -> None:
    configure_logging()

    settings = Settings.from_env()
    db = Database(settings.db_path)
    db.ensure_schema()

    timeout = httpx.Timeout(settings.upstream_timeout_seconds, connect=5.0)
    http_client = httpx.AsyncClient(timeout=timeout)

    estimator = TokenEstimator(
        model_id=settings.model_id,
        hf_token=os.getenv("HF_TOKEN"),
        trust_remote_code=settings.trust_remote_code,
    )

    metrics = GatewayMetrics(tenant_labels_enabled=settings.metrics_tenant_labels)
    metrics.set_engine_healthy(False)

    gpu_metrics_poller = GPUMetricsPoller(
        metrics=metrics,
        poll_interval_seconds=settings.gpu_metrics_poll_interval_seconds,
    )
    await gpu_metrics_poller.start()

    admin_auth = AdminAuthProvider(
        mode=settings.admin_auth_mode,
        admin_token=settings.admin_token,
        jwks_url=settings.jwks_url,
        issuer=settings.oidc_issuer,
        audience=settings.oidc_audience,
        role_claim=settings.role_claim,
        groups_claim=settings.groups_claim,
        admin_group=settings.admin_group,
    )

    app.state.settings = settings
    app.state.db = db
    app.state.http_client = http_client
    app.state.rate_limiter = TenantRateLimiter()
    app.state.concurrency = ConcurrencyManager(settings.global_max_concurrent)
    app.state.estimator = estimator
    app.state.metrics = metrics
    app.state.gpu_metrics_poller = gpu_metrics_poller
    app.state.admin_auth = admin_auth
    app.state.profiles = _load_profiles(settings)

    dynamo_worker_metrics_port = int(os.getenv("DYNAMO_WORKER_METRICS_PORT", "8081"))
    dynamo_scraper = DynamoMetricsScraper(
        frontend_url=settings.upstream_url,
        worker_metrics_url=f"http://127.0.0.1:{dynamo_worker_metrics_port}",
    )
    app.state.dynamo_scraper = dynamo_scraper

    json_log(
        "gateway_started",
        model_id=settings.model_id,
        config_path=settings.config_path,
        db_path=settings.db_path,
        admin_auth_mode=settings.admin_auth_mode,
        global_max_concurrent=settings.global_max_concurrent,
        metrics_tenant_labels=settings.metrics_tenant_labels,
        metrics_enabled=settings.metrics_enabled,
        ui_enabled=settings.ui_enabled,
        profiles_enabled=settings.profiles_enabled,
        profiles_path=settings.profiles_path,
        gpu_metrics_poll_interval_seconds=settings.gpu_metrics_poll_interval_seconds,
        gpu_hourly_rate=settings.gpu_hourly_rate,
        dynamo_frontend_url=settings.upstream_url,
        dynamo_worker_metrics_port=dynamo_worker_metrics_port,
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    client: httpx.AsyncClient = app.state.http_client
    db: Database = app.state.db
    gpu_metrics_poller: GPUMetricsPoller = app.state.gpu_metrics_poller

    await gpu_metrics_poller.stop()

    dynamo_scraper: DynamoMetricsScraper = app.state.dynamo_scraper
    await dynamo_scraper.close()

    await client.aclose()
    db.close()


@app.get("/livez")
async def livez() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/healthz")
async def healthz() -> JSONResponse:
    db: Database = app.state.db
    metrics: GatewayMetrics = app.state.metrics

    db_ok = db.health_check()
    upstream_ok = await _probe_upstream()
    metrics.set_engine_healthy(upstream_ok)

    if db_ok and upstream_ok:
        return JSONResponse({"status": "ready"}, status_code=200)

    return JSONResponse(
        {
            "status": "not_ready",
            "db_ok": db_ok,
            "upstream_ok": upstream_ok,
        },
        status_code=503,
    )


@app.get("/version", response_model=GatewayVersion)
async def gateway_version() -> GatewayVersion:
    settings: Settings = app.state.settings
    return GatewayVersion(
        build_sha=settings.build_sha,
        build_time=settings.build_time,
        model_id=settings.model_id,
        vllm_version=settings.vllm_version,
        ui_enabled=settings.ui_enabled,
        metrics_enabled=settings.metrics_enabled,
        admin_auth_mode=settings.admin_auth_mode,
        profiles_enabled=settings.profiles_enabled,
        global_max_concurrent=settings.global_max_concurrent,
        gpu_hourly_rate=settings.gpu_hourly_rate,
    )


@app.get("/admin/auth_info", response_model=AuthInfoResponse)
async def admin_auth_info() -> AuthInfoResponse:
    provider: AdminAuthProvider = app.state.admin_auth
    payload = provider.auth_info()
    return AuthInfoResponse(**payload)


@app.get("/metrics")
async def metrics_endpoint(request: Request) -> Response:
    settings: Settings = app.state.settings
    metrics: GatewayMetrics = app.state.metrics

    if not settings.metrics_enabled:
        return JSONResponse(
            status_code=404,
            content={"error": {"type": "not_found", "message": "metrics endpoint is disabled"}},
        )

    await _check_admin(request)
    return metrics.response()


@app.get("/ui")
async def ui_index() -> Response:
    settings: Settings = app.state.settings
    if not settings.ui_enabled:
        raise HTTPException(status_code=404, detail={"error": {"type": "not_found", "message": "ui disabled"}})

    target = _resolve_ui_file(settings)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail={"error": {"type": "not_found", "message": "ui not found"}})
    return FileResponse(target)


@app.get("/ui/{asset_path:path}")
async def ui_assets(asset_path: str) -> Response:
    settings: Settings = app.state.settings
    if not settings.ui_enabled:
        raise HTTPException(status_code=404, detail={"error": {"type": "not_found", "message": "ui disabled"}})

    target = _resolve_ui_file(settings, asset_path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail={"error": {"type": "not_found", "message": "asset not found"}})
    return FileResponse(target)


@app.post("/admin/tenants", response_model=TenantCreateResponse)
async def create_tenant(body: TenantCreateRequest, request: Request) -> TenantCreateResponse:
    principal = await _check_admin(request)

    settings: Settings = app.state.settings
    db: Database = app.state.db
    concurrency: ConcurrencyManager = app.state.concurrency

    limits = _effective_limits(settings, body)
    tenant_id = str(uuid.uuid4())
    key_id = str(uuid.uuid4())
    api_key = generate_api_key()
    key_hash = hash_api_key(api_key)

    try:
        tenant = db.create_tenant(
            tenant_id=tenant_id,
            tenant_name=body.tenant_name,
            api_key_hash=key_hash,
            max_concurrent=limits["max_concurrent"],
            rpm_limit=limits["rpm_limit"],
            tpm_limit=limits["tpm_limit"],
            max_context_tokens=limits["max_context_tokens"],
            max_output_tokens=limits["max_output_tokens"],
        )
        db.create_api_key(
            key_id=key_id,
            tenant_id=tenant_id,
            seat_id=None,
            key_hash=key_hash,
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": {"type": "conflict", "message": f"tenant already exists: {exc}"}},
        ) from exc

    await concurrency.set_tenant_limit(tenant_id, limits["max_concurrent"])

    _audit(
        request=request,
        principal=principal,
        action="tenant_create",
        resource_type="tenant",
        resource_id=tenant_id,
        details={
            "tenant_id": tenant_id,
            "tenant_name": body.tenant_name,
            "limits": limits,
        },
    )

    return TenantCreateResponse(**tenant, api_key=api_key)


@app.get("/admin/tenants")
async def list_tenants(request: Request) -> dict[str, list[TenantPublic]]:
    await _check_admin(request)

    db: Database = app.state.db
    tenants = [TenantPublic(**row) for row in db.list_tenants()]
    return {"data": tenants}

@app.get("/admin/tenants/{tenant_id}", response_model=TenantPublic)
async def get_tenant_by_id(tenant_id: str, request: Request) -> TenantPublic:
    await _check_admin(request)

    db: Database = app.state.db
    tenant = db.get_tenant(tenant_id)
    if tenant is None:
        raise HTTPException(
            status_code=404,
            detail={"error": {"type": "not_found", "message": "tenant not found"}},
        )
    return TenantPublic(**tenant)
    
@app.patch("/admin/tenants/{tenant_id}", response_model=TenantPublic)
async def patch_tenant(
    tenant_id: str,
    body: TenantPatchRequest,
    request: Request,
) -> TenantPublic:
    principal = await _check_admin(request)

    db: Database = app.state.db
    concurrency: ConcurrencyManager = app.state.concurrency

    updates = body.model_dump(exclude_none=True)

    try:
        tenant = db.patch_tenant(tenant_id, updates)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": {"type": "conflict", "message": f"tenant update conflict: {exc}"}},
        ) from exc

    if tenant is None:
        raise HTTPException(status_code=404, detail={"error": {"type": "not_found", "message": "tenant not found"}})

    if "max_concurrent" in updates:
        await concurrency.set_tenant_limit(tenant_id, int(tenant["max_concurrent"]))

    _audit(
        request=request,
        principal=principal,
        action="tenant_update_limits",
        resource_type="tenant",
        resource_id=tenant_id,
        details={"tenant_id": tenant_id, "updates": updates},
    )

    return TenantPublic(**tenant)


@app.post("/admin/seats", response_model=SeatPublic)
async def create_seat(body: SeatCreateRequest, request: Request) -> SeatPublic:
    principal = await _check_admin(request)

    db: Database = app.state.db
    if db.get_tenant(body.tenant_id) is None:
        raise HTTPException(status_code=404, detail={"error": {"type": "not_found", "message": "tenant not found"}})

    try:
        seat = db.create_seat(
            seat_id=str(uuid.uuid4()),
            tenant_id=body.tenant_id,
            seat_name=body.seat_name,
            role=body.role,
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": {"type": "conflict", "message": f"seat creation conflict: {exc}"}},
        ) from exc

    _audit(
        request=request,
        principal=principal,
        action="seat_create",
        resource_type="seat",
        resource_id=seat["seat_id"],
        details={
            "tenant_id": body.tenant_id,
            "seat_id": seat["seat_id"],
            "seat_name": body.seat_name,
            "role": body.role,
        },
    )

    return SeatPublic(**seat)


@app.get("/admin/seats")
async def list_seats(
    request: Request,
    tenant_id: str | None = Query(default=None),
) -> dict[str, list[SeatPublic]]:
    await _check_admin(request)

    db: Database = app.state.db
    seats = [SeatPublic(**row) for row in db.list_seats(tenant_id=tenant_id)]
    return {"data": seats}


@app.patch("/admin/seats/{seat_id}", response_model=SeatPublic)
async def patch_seat(
    seat_id: str,
    body: SeatPatchRequest,
    request: Request,
) -> SeatPublic:
    principal = await _check_admin(request)

    db: Database = app.state.db
    updates = body.model_dump(exclude_none=True)

    try:
        seat = db.patch_seat(seat_id, updates)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": {"type": "conflict", "message": f"seat update conflict: {exc}"}},
        ) from exc

    if seat is None:
        raise HTTPException(status_code=404, detail={"error": {"type": "not_found", "message": "seat not found"}})

    action = "seat_update"
    if "is_active" in updates:
        action = "seat_activated" if bool(updates["is_active"]) else "seat_deactivated"

    _audit(
        request=request,
        principal=principal,
        action=action,
        resource_type="seat",
        resource_id=seat_id,
        details={"tenant_id": seat["tenant_id"], "seat_id": seat_id, "updates": updates},
    )

    return SeatPublic(**seat)


@app.post("/admin/keys", response_model=KeyCreateResponse)
async def create_key(body: KeyCreateRequest, request: Request) -> KeyCreateResponse:
    principal = await _check_admin(request)

    db: Database = app.state.db
    tenant = db.get_tenant(body.tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail={"error": {"type": "not_found", "message": "tenant not found"}})

    if body.seat_id is not None:
        seat = db.get_seat(body.seat_id)
        if seat is None or seat["tenant_id"] != body.tenant_id:
            raise HTTPException(status_code=404, detail={"error": {"type": "not_found", "message": "seat not found"}})

    api_key = generate_api_key()
    key_hash = hash_api_key(api_key)
    key_id = str(uuid.uuid4())

    try:
        key_row = db.create_api_key(
            key_id=key_id,
            tenant_id=body.tenant_id,
            seat_id=body.seat_id,
            key_hash=key_hash,
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": {"type": "conflict", "message": f"key creation conflict: {exc}"}},
        ) from exc

    _audit(
        request=request,
        principal=principal,
        action="key_create",
        resource_type="key",
        resource_id=key_id,
        details={"tenant_id": body.tenant_id, "seat_id": body.seat_id, "key_id": key_id},
    )

    return KeyCreateResponse(
        key_id=key_row["key_id"],
        tenant_id=key_row["tenant_id"],
        seat_id=key_row["seat_id"],
        api_key=api_key,
        created_at=key_row["created_at"],
    )


@app.get("/admin/keys")
async def list_keys(
    request: Request,
    tenant_id: str | None = Query(default=None),
) -> dict[str, list[KeyPublic]]:
    await _check_admin(request)

    db: Database = app.state.db
    keys = [KeyPublic(**row) for row in db.list_api_keys(tenant_id=tenant_id)]
    return {"data": keys}


@app.post("/admin/keys/{key_id}/revoke", response_model=KeyPublic)
async def revoke_key(key_id: str, request: Request) -> KeyPublic:
    principal = await _check_admin(request)

    db: Database = app.state.db
    key_row = db.revoke_api_key(key_id)
    if key_row is None:
        raise HTTPException(status_code=404, detail={"error": {"type": "not_found", "message": "key not found"}})

    _audit(
        request=request,
        principal=principal,
        action="key_revoke",
        resource_type="key",
        resource_id=key_id,
        details={"tenant_id": key_row["tenant_id"], "key_id": key_id},
    )

    return KeyPublic(**key_row)


@app.get("/admin/usage", response_model=UsageResponse)
async def tenant_usage(
    request: Request,
    window: str = "24h",
) -> UsageResponse:
    await _check_admin(request)

    db: Database = app.state.db
    try:
        parse_window(window)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": {"type": "bad_input", "message": str(exc)}}) from exc

    data = usage_rollup_tenants(db, window)
    return UsageResponse(window=window, data=data)


@app.get("/admin/usage/tenants", response_model=UsageTenantsResponse)
async def tenant_usage_rollup(
    request: Request,
    window: str = "24h",
) -> UsageTenantsResponse:
    await _check_admin(request)

    settings: Settings = app.state.settings
    db: Database = app.state.db
    try:
        parse_window(window)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": {"type": "bad_input", "message": str(exc)}}) from exc

    data = usage_rollup_tenants(db, window)
    return UsageTenantsResponse(
        window=window,
        cost_estimation_enabled=_cost_estimation_enabled(settings),
        data=data,
    )


@app.get("/admin/usage/seats", response_model=UsageSeatsResponse)
async def seat_usage_rollup(
    request: Request,
    tenant_id: str,
    window: str = "24h",
) -> UsageSeatsResponse:
    await _check_admin(request)

    settings: Settings = app.state.settings
    db: Database = app.state.db

    try:
        parse_window(window)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": {"type": "bad_input", "message": str(exc)}}) from exc

    if db.get_tenant(tenant_id) is None:
        raise HTTPException(status_code=404, detail={"error": {"type": "not_found", "message": "tenant not found"}})

    data = usage_rollup_seats(db, window=window, tenant_id=tenant_id)
    return UsageSeatsResponse(
        window=window,
        tenant_id=tenant_id,
        cost_estimation_enabled=_cost_estimation_enabled(settings),
        data=data,
    )


@app.put("/admin/budgets/{tenant_id}", response_model=TenantBudgetPublic)
async def put_budget(
    tenant_id: str,
    body: TenantBudgetPutRequest,
    request: Request,
) -> TenantBudgetPublic:
    principal = await _check_admin(request)

    db: Database = app.state.db
    if db.get_tenant(tenant_id) is None:
        raise HTTPException(status_code=404, detail={"error": {"type": "not_found", "message": "tenant not found"}})

    try:
        parse_budget_window(body.window)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": {"type": "bad_input", "message": str(exc)}}) from exc

    existing_budget = db.get_tenant_budget(tenant_id)
    budget = db.upsert_tenant_budget(tenant_id=tenant_id, window=body.window, budget_usd=body.budget_usd)

    _audit(
        request=request,
        principal=principal,
        action="budget_update" if existing_budget else "budget_set",
        resource_type="budget",
        resource_id=tenant_id,
        details={"tenant_id": tenant_id, "window": body.window, "budget_usd": body.budget_usd},
    )

    return TenantBudgetPublic(**budget)


@app.get("/admin/budgets")
async def list_budgets(
    request: Request,
    tenant_id: str | None = Query(default=None),
) -> dict[str, list[TenantBudgetPublic]]:
    await _check_admin(request)

    db: Database = app.state.db
    budgets = [TenantBudgetPublic(**row) for row in db.list_tenant_budgets(tenant_id=tenant_id)]
    return {"data": budgets}


@app.get("/admin/budget_status", response_model=BudgetStatusResponse)
async def budget_status(
    request: Request,
    tenant_id: str,
    window: str | None = Query(default=None),
) -> BudgetStatusResponse:
    await _check_admin(request)

    db: Database = app.state.db
    settings: Settings = app.state.settings

    if db.get_tenant(tenant_id) is None:
        raise HTTPException(status_code=404, detail={"error": {"type": "not_found", "message": "tenant not found"}})

    try:
        payload = _budget_status_payload(db=db, settings=settings, tenant_id=tenant_id, requested_window=window)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": {"type": "bad_input", "message": str(exc)}}) from exc

    return BudgetStatusResponse(**payload)


@app.get("/admin/budget_events")
async def budget_events(
    request: Request,
    tenant_id: str | None = Query(default=None),
    window: str | None = Query(default=None),
) -> dict[str, list[BudgetEventPublic]]:
    await _check_admin(request)

    db: Database = app.state.db
    if window is not None:
        try:
            parse_budget_window(window)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": {"type": "bad_input", "message": str(exc)}}) from exc

    rows = db.list_budget_events(tenant_id=tenant_id, window=window)
    return {"data": [BudgetEventPublic(**row) for row in rows]}


@app.get("/admin/audit", response_model=AuditResponse)
async def admin_audit(
    request: Request,
    window: str = "24h",
    tenant_id: str | None = Query(default=None),
) -> AuditResponse:
    await _check_admin(request)

    db: Database = app.state.db
    try:
        window_seconds = parse_window(window)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": {"type": "bad_input", "message": str(exc)}}) from exc

    threshold = to_iso(utc_now() - timedelta(seconds=window_seconds))
    rows = db.list_audit_events(threshold_ts=threshold, tenant_id=tenant_id)
    return AuditResponse(window=window, data=[AuditEventPublic(**row) for row in rows])


@app.get("/admin/profiles")
async def list_profiles(request: Request) -> dict[str, list[ProfilePublic]]:
    await _check_admin(request)
    settings: Settings = app.state.settings
    profiles = _profiles_or_404(settings)
    data = [ProfilePublic(**profiles[name]) for name in sorted(profiles)]
    return {"data": data}


@app.get("/admin/profiles/{name}", response_model=ProfilePublic)
async def get_profile(name: str, request: Request) -> ProfilePublic:
    principal = await _check_admin(request)
    settings: Settings = app.state.settings
    profiles = _profiles_or_404(settings)

    profile = profiles.get(name)
    if profile is None:
        raise HTTPException(status_code=404, detail={"error": {"type": "not_found", "message": "profile not found"}})

    _audit(
        request=request,
        principal=principal,
        action="profile_viewed",
        resource_type="profile",
        resource_id=name,
        details={"profile_name": name},
    )

    return ProfilePublic(**profile)


@app.post("/admin/profiles/{name}/generate_runpod")
async def generate_profile_runpod(name: str, request: Request) -> dict[str, Any]:
    principal = await _check_admin(request)
    settings: Settings = app.state.settings
    profiles = _profiles_or_404(settings)

    profile = profiles.get(name)
    if profile is None:
        raise HTTPException(status_code=404, detail={"error": {"type": "not_found", "message": "profile not found"}})

    run_snippet, bootstrap_snippet, notes = _generate_runpod_snippet(settings, profile)

    _audit(
        request=request,
        principal=principal,
        action="profile_generated",
        resource_type="profile",
        resource_id=name,
        details={"profile_name": name, "target": "runpod"},
    )

    return {
        "profile": ProfilePublic(**profile),
        "admin_auth_mode": settings.admin_auth_mode,
        "runpod_env_snippet": run_snippet,
        "bootstrap_snippet": bootstrap_snippet,
        "notes": notes,
    }

@app.get("/admin/dynamo/status")
async def dynamo_status(request: Request) -> dict:
    """Dynamo topology: frontend health, worker health, model config from registration."""
    await _check_admin(request)
    scraper: DynamoMetricsScraper = app.state.dynamo_scraper
    return await scraper.get_dynamo_status()


@app.get("/admin/dynamo/metrics")
async def dynamo_metrics(request: Request) -> dict:
    """Live Dynamo inference metrics: TTFT, ITL, inflight, cache usage, worker stats."""
    await _check_admin(request)
    scraper: DynamoMetricsScraper = app.state.dynamo_scraper
    return await scraper.get_dynamo_metrics()
    
@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    settings: Settings = app.state.settings
    db: Database = app.state.db
    client: httpx.AsyncClient = app.state.http_client
    limiter: TenantRateLimiter = app.state.rate_limiter
    concurrency: ConcurrencyManager = app.state.concurrency
    estimator: TokenEstimator = app.state.estimator
    metrics: GatewayMetrics = app.state.metrics

    start = utc_now()
    ts_start = to_iso(start)
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())

    try:
        api_key = parse_bearer_token(request.headers.get("authorization"))
    except HTTPException as exc:
        metrics.inc_rejection(reason="auth", tenant_id=None, tenant_name=None)
        return error_response(exc.status_code, "unauthorized", exc.detail["error"]["message"], request_id=request_id)

    auth_row = resolve_api_key(api_key, db)
    if auth_row is None:
        metrics.inc_rejection(reason="auth", tenant_id=None, tenant_name=None)
        return error_response(401, "unauthorized", "invalid API key", request_id=request_id)

    tenant_id = auth_row["tenant_id"]
    tenant_name = auth_row["tenant_name"]
    seat_id = auth_row.get("seat_id")
    key_id = auth_row.get("key_id")

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    global_acquired = False
    tenant_acquired = False
    tpm_reserved = 0

    def finalize_error(code: int, etype: str, message: str, rejection_reason: str | None = None) -> JSONResponse:
        nonlocal total_tokens
        total_tokens = prompt_tokens + completion_tokens

        if rejection_reason is not None:
            metrics.inc_rejection(reason=rejection_reason, tenant_id=tenant_id, tenant_name=tenant_name)

        end = utc_now()
        ts_end = to_iso(end)
        elapsed_ms = latency_ms(start, end)

        gpu_seconds_est, cost_est = _estimate_gpu_and_cost(settings=settings, latency_ms_value=elapsed_ms, status_code=code)

        _record_request_with_metrics(
            metrics=metrics,
            db=db,
            ts_start=ts_start,
            ts_end=ts_end,
            latency_ms_value=elapsed_ms,
            tenant_id=tenant_id,
            seat_id=seat_id,
            key_id=key_id,
            model_id=settings.model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            status_code=code,
            error_type=etype,
            request_id=request_id,
            gpu_seconds_est=gpu_seconds_est,
            cost_est=cost_est,
        )

        json_log(
            "request_complete",
            tenant_id=tenant_id,
            seat_id=seat_id,
            key_id=key_id,
            request_id=request_id,
            latency_ms=elapsed_ms,
            status=code,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error_type=etype,
            gpu_seconds_est=gpu_seconds_est,
            cost_est=cost_est,
        )
        return error_response(code, etype, message, request_id=request_id)

    # Tenant-mismatch check comes first so callers can't enumerate whether a key
    # is revoked/inactive on a different tenant (information leak prevention).
    expected_tenant_id = request.headers.get("x-expected-tenant-id")
    if expected_tenant_id and expected_tenant_id != tenant_id:
        return finalize_error(403, "unauthorized", "tenant mismatch: API key does not belong to the selected tenant", rejection_reason="auth")

    if auth_row.get("revoked_at") is not None:
        return finalize_error(401, "unauthorized", "API key revoked", rejection_reason="auth")
    if not bool(auth_row.get("tenant_is_active")):
        return finalize_error(401, "unauthorized", "tenant inactive", rejection_reason="auth")
    if not bool(auth_row.get("seat_is_active")):
        return finalize_error(401, "unauthorized", "seat inactive", rejection_reason="auth")

    try:
        body = await request.body()
        if len(body) > settings.max_body_bytes:
            return finalize_error(
                413,
                "bad_input",
                f"request body exceeds MAX_BODY_BYTES={settings.max_body_bytes}",
                rejection_reason="body_too_large",
            )

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return finalize_error(400, "bad_input", "request body must be valid JSON")

        if not isinstance(payload, dict):
            return finalize_error(400, "bad_input", "request body must be a JSON object")

        try:
            ChatCompletionPayload(**payload)
        except Exception as exc:
            return finalize_error(400, "bad_input", f"invalid request payload: {exc}")

        prompt_text = extract_prompt_text(payload)
        prompt_tokens = estimator.approx_tokens(prompt_text)

        if prompt_tokens > int(auth_row["max_context_tokens"]):
            return finalize_error(
                400,
                "bad_input",
                f"prompt tokens exceed max_context_tokens ({prompt_tokens} > {auth_row['max_context_tokens']})",
                rejection_reason="max_context",
            )

        try:
            requested_max_tokens, clamped_max_tokens = _resolve_requested_max_tokens(
                payload,
                int(auth_row["max_output_tokens"]),
            )
        except ValueError as exc:
            return finalize_error(400, "bad_input", str(exc), rejection_reason="max_output")

        if requested_max_tokens > int(auth_row["max_output_tokens"]):
            metrics.inc_rejection(reason="max_output", tenant_id=tenant_id, tenant_name=tenant_name)

        payload["max_tokens"] = clamped_max_tokens
        if "max_completion_tokens" in payload:
            payload["max_completion_tokens"] = clamped_max_tokens

        payload["model"] = settings.model_id

        global_acquired = await concurrency.try_acquire_global()
        if not global_acquired:
            return finalize_error(
                429,
                "limit_concurrent",
                "global concurrent request limit exceeded",
                rejection_reason="limit_concurrent",
            )

        tenant_acquired = await concurrency.try_acquire_tenant(tenant_id, int(auth_row["max_concurrent"]))
        if not tenant_acquired:
            return finalize_error(
                429,
                "limit_concurrent",
                "tenant concurrent request limit exceeded",
                rejection_reason="limit_concurrent",
            )

        rpm_ok = await limiter.allow_request(tenant_id, int(auth_row["rpm_limit"]))
        if not rpm_ok:
            return finalize_error(429, "limit_rpm", "tenant rpm_limit exceeded", rejection_reason="limit_rpm")

        tpm_reserved = prompt_tokens + clamped_max_tokens
        tpm_ok = await limiter.reserve_tokens(tenant_id, int(auth_row["tpm_limit"]), tpm_reserved)
        if not tpm_ok:
            return finalize_error(429, "limit_tpm", "tenant tpm_limit exceeded", rejection_reason="limit_tpm")

        forward_headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in {"host", "content-length", "connection", "authorization"}
        }
        forward_headers["x-request-id"] = request_id

        # TODO: Inject Dynamo per-request hints (priority, cache_ttl) based on tenant policy
        # Example: payload["extra_body"] = {"priority": tenant_priority_level}

        upstream_response: httpx.Response
        upstream_start = time.perf_counter()
        try:
            upstream_response = await client.post(
                f"{settings.upstream_url}/v1/chat/completions",
                params=request.query_params,
                json=payload,
                headers=forward_headers,
            )
        except httpx.TimeoutException:
            metrics.observe_upstream(status="timeout", duration_seconds=max(0.0, time.perf_counter() - upstream_start))
            metrics.set_engine_healthy(False)
            if tpm_reserved > prompt_tokens:
                await limiter.refund_tokens(tenant_id, tpm_reserved - prompt_tokens)
            return finalize_error(504, "timeout", "upstream request timed out")
        except httpx.HTTPError as exc:
            metrics.observe_upstream(status="error", duration_seconds=max(0.0, time.perf_counter() - upstream_start))
            metrics.set_engine_healthy(False)
            if tpm_reserved > prompt_tokens:
                await limiter.refund_tokens(tenant_id, tpm_reserved - prompt_tokens)
            return finalize_error(502, "upstream_5xx", f"upstream request failed: {exc}")

        metrics.observe_upstream(
            status=str(upstream_response.status_code),
            duration_seconds=max(0.0, time.perf_counter() - upstream_start),
        )
        metrics.set_engine_healthy(upstream_response.status_code < 500)

        response_json: dict[str, Any] | None = None
        content_type = upstream_response.headers.get("content-type", "")
        if "application/json" in content_type.lower():
            try:
                response_json = upstream_response.json()
            except ValueError:
                response_json = None

        prompt_tokens, completion_tokens, total_tokens = extract_usage(response_json, prompt_tokens, estimator)

        if tpm_reserved > total_tokens:
            await limiter.refund_tokens(tenant_id, tpm_reserved - total_tokens)

        status_code = upstream_response.status_code
        error_type = "upstream_5xx" if status_code >= 500 else ("bad_input" if status_code >= 400 else None)

        end = utc_now()
        ts_end = to_iso(end)
        elapsed_ms = latency_ms(start, end)
        gpu_seconds_est, cost_est = _estimate_gpu_and_cost(
            settings=settings,
            latency_ms_value=elapsed_ms,
            status_code=status_code,
        )

        _record_request_with_metrics(
            metrics=metrics,
            db=db,
            ts_start=ts_start,
            ts_end=ts_end,
            latency_ms_value=elapsed_ms,
            tenant_id=tenant_id,
            seat_id=seat_id,
            key_id=key_id,
            model_id=settings.model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            status_code=status_code,
            error_type=error_type,
            request_id=request_id,
            gpu_seconds_est=gpu_seconds_est,
            cost_est=cost_est,
        )

        if status_code < 400:
            metrics.add_tokens(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                tenant_id=tenant_id,
                tenant_name=tenant_name,
            )
            _evaluate_budget_for_tenant(db=db, metrics=metrics, tenant_id=tenant_id)
            _refresh_tenant_cost_metric(db=db, metrics=metrics, tenant_id=tenant_id)

        json_log(
            "request_complete",
            tenant_id=tenant_id,
            seat_id=seat_id,
            key_id=key_id,
            request_id=request_id,
            latency_ms=elapsed_ms,
            status=status_code,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error_type=error_type,
            gpu_seconds_est=gpu_seconds_est,
            cost_est=cost_est,
        )

        response_headers: dict[str, str] = {"x-request-id": request_id}
        upstream_content_type = upstream_response.headers.get("content-type")
        if upstream_content_type:
            response_headers["content-type"] = upstream_content_type

        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            headers=response_headers,
        )
    finally:
        if tenant_acquired:
            await concurrency.release_tenant(tenant_id)
        if global_acquired:
            await concurrency.release_global()
