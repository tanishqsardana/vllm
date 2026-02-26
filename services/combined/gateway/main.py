from __future__ import annotations

import hmac
import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from importlib.metadata import PackageNotFoundError, version

from .accounting import extract_usage, parse_window, record_request, usage_rollup
from .auth import authenticate_tenant, generate_api_key, hash_api_key, parse_bearer_token, verify_admin_token
from .db import Database
from .gpu_metrics import GPUMetricsPoller
from .limits import ConcurrencyManager
from .metrics import GatewayMetrics
from .middleware_metrics import GatewayMetricsMiddleware
from .rate_limit import TenantRateLimiter
from .schemas import (
    ChatCompletionPayload,
    GatewayVersion,
    TenantCreateRequest,
    TenantCreateResponse,
    TenantPatchRequest,
    TenantPublic,
    UsageResponse,
)
from .utils import TokenEstimator, configure_logging, extract_prompt_text, json_log, latency_ms, to_iso, utc_now


@dataclass
class Settings:
    model_id: str
    db_path: str
    admin_token: str | None
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

    @classmethod
    def from_env(cls) -> "Settings":
        model_id = os.getenv("MODEL_ID")
        if not model_id:
            raise RuntimeError("MODEL_ID is required for gateway")

        try:
            vllm_version = version("vllm")
        except PackageNotFoundError:
            vllm_version = "unknown"

        metrics_tenant_labels = os.getenv("METRICS_TENANT_LABELS", "on").strip().lower()
        if metrics_tenant_labels not in {"on", "off"}:
            raise RuntimeError("METRICS_TENANT_LABELS must be one of: on, off")

        return cls(
            model_id=model_id,
            db_path=os.getenv("DB_PATH", "/data/controlplane.db"),
            admin_token=os.getenv("ADMIN_TOKEN"),
            global_max_concurrent=int(os.getenv("GLOBAL_MAX_CONCURRENT", "128")),
            max_body_bytes=int(os.getenv("MAX_BODY_BYTES", str(1024 * 1024))),
            upstream_url=f"http://127.0.0.1:{os.getenv('VLLM_PORT', '8001')}",
            upstream_timeout_seconds=float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "300")),
            build_sha=os.getenv("BUILD_SHA", "dev"),
            build_time=os.getenv("BUILD_TIME", "dev"),
            vllm_version=vllm_version,
            default_max_concurrent=int(os.getenv("DEFAULT_MAX_CONCURRENT", "4")),
            default_rpm_limit=int(os.getenv("DEFAULT_RPM_LIMIT", "120")),
            default_tpm_limit=int(os.getenv("DEFAULT_TPM_LIMIT", "120000")),
            default_max_context_tokens=int(os.getenv("DEFAULT_MAX_CONTEXT_TOKENS", "8192")),
            default_max_output_tokens=int(os.getenv("DEFAULT_MAX_OUTPUT_TOKENS", "512")),
            trust_remote_code=os.getenv("TRUST_REMOTE_CODE", "0").lower() in {"1", "true", "yes", "on"},
            metrics_tenant_labels=metrics_tenant_labels == "on",
            gpu_metrics_poll_interval_seconds=float(os.getenv("GPU_METRICS_POLL_INTERVAL_SECONDS", "2")),
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

    app.state.settings = settings
    app.state.db = db
    app.state.http_client = http_client
    app.state.rate_limiter = TenantRateLimiter()
    app.state.concurrency = ConcurrencyManager(settings.global_max_concurrent)
    app.state.estimator = estimator
    app.state.metrics = metrics
    app.state.gpu_metrics_poller = gpu_metrics_poller

    json_log(
        "gateway_started",
        model_id=settings.model_id,
        db_path=settings.db_path,
        global_max_concurrent=settings.global_max_concurrent,
        metrics_tenant_labels=settings.metrics_tenant_labels,
        gpu_metrics_poll_interval_seconds=settings.gpu_metrics_poll_interval_seconds,
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    client: httpx.AsyncClient = app.state.http_client
    db: Database = app.state.db
    gpu_metrics_poller: GPUMetricsPoller = app.state.gpu_metrics_poller

    await gpu_metrics_poller.stop()
    await client.aclose()
    db.close()


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


async def _check_admin(x_admin_token: str | None) -> None:
    settings: Settings = app.state.settings
    verify_admin_token(settings.admin_token, x_admin_token)


async def _probe_upstream() -> bool:
    client: httpx.AsyncClient = app.state.http_client
    settings: Settings = app.state.settings

    try:
        resp = await client.get(f"{settings.upstream_url}/health")
        if resp.status_code == 200:
            return True
    except httpx.HTTPError:
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
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    status_code: int,
    error_type: str | None,
    request_id: str,
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
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            status_code=status_code,
            error_type=error_type,
            request_id=request_id,
        )
        ok = True
    finally:
        metrics.observe_db_write(duration_seconds=max(0.0, time.perf_counter() - start), ok=ok)


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
    )


@app.get("/metrics")
async def metrics_endpoint(x_admin_token: str | None = Header(default=None)) -> Response:
    settings: Settings = app.state.settings
    metrics: GatewayMetrics = app.state.metrics

    if not settings.admin_token:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "type": "admin_unavailable",
                    "message": "metrics unavailable: ADMIN_TOKEN is not configured",
                }
            },
        )

    if not x_admin_token:
        return JSONResponse(
            status_code=401,
            content={"error": {"type": "unauthorized", "message": "missing X-Admin-Token header"}},
        )

    if not hmac.compare_digest(settings.admin_token, x_admin_token):
        return JSONResponse(
            status_code=401,
            content={"error": {"type": "unauthorized", "message": "invalid admin token"}},
        )

    return metrics.response()


@app.post("/admin/tenants", response_model=TenantCreateResponse)
async def create_tenant(body: TenantCreateRequest, x_admin_token: str | None = Header(default=None)) -> TenantCreateResponse:
    await _check_admin(x_admin_token)

    settings: Settings = app.state.settings
    db: Database = app.state.db
    concurrency: ConcurrencyManager = app.state.concurrency

    limits = _effective_limits(settings, body)
    tenant_id = str(uuid.uuid4())
    api_key = generate_api_key()

    try:
        tenant = db.create_tenant(
            tenant_id=tenant_id,
            tenant_name=body.tenant_name,
            api_key_hash=hash_api_key(api_key),
            max_concurrent=limits["max_concurrent"],
            rpm_limit=limits["rpm_limit"],
            tpm_limit=limits["tpm_limit"],
            max_context_tokens=limits["max_context_tokens"],
            max_output_tokens=limits["max_output_tokens"],
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": {"type": "conflict", "message": f"tenant already exists: {exc}"}},
        ) from exc

    await concurrency.set_tenant_limit(tenant_id, limits["max_concurrent"])

    return TenantCreateResponse(**tenant, api_key=api_key)


@app.get("/admin/tenants")
async def list_tenants(x_admin_token: str | None = Header(default=None)) -> dict[str, list[TenantPublic]]:
    await _check_admin(x_admin_token)

    db: Database = app.state.db
    tenants = [TenantPublic(**row) for row in db.list_tenants()]
    return {"data": tenants}


@app.patch("/admin/tenants/{tenant_id}", response_model=TenantPublic)
async def patch_tenant(
    tenant_id: str,
    body: TenantPatchRequest,
    x_admin_token: str | None = Header(default=None),
) -> TenantPublic:
    await _check_admin(x_admin_token)

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

    return TenantPublic(**tenant)


@app.get("/admin/usage", response_model=UsageResponse)
async def tenant_usage(
    window: str = "24h",
    x_admin_token: str | None = Header(default=None),
) -> UsageResponse:
    await _check_admin(x_admin_token)

    db: Database = app.state.db
    try:
        parse_window(window)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": {"type": "bad_input", "message": str(exc)}}) from exc

    data = usage_rollup(db, window)
    return UsageResponse(window=window, data=data)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str | None = Header(default=None)) -> Response:
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
        api_key = parse_bearer_token(authorization)
    except HTTPException as exc:
        metrics.inc_rejection(reason="auth", tenant_id=None, tenant_name=None)
        return error_response(exc.status_code, "unauthorized", exc.detail["error"]["message"], request_id=request_id)

    tenant = authenticate_tenant(api_key, db)
    if tenant is None:
        metrics.inc_rejection(reason="auth", tenant_id=None, tenant_name=None)
        return error_response(401, "unauthorized", "invalid API key", request_id=request_id)

    tenant_id = tenant["tenant_id"]
    tenant_name = tenant["tenant_name"]
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
        _record_request_with_metrics(
            metrics=metrics,
            db=db,
            ts_start=ts_start,
            ts_end=ts_end,
            latency_ms_value=elapsed_ms,
            tenant_id=tenant_id,
            model_id=settings.model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            status_code=code,
            error_type=etype,
            request_id=request_id,
        )
        json_log(
            "request_complete",
            tenant_id=tenant_id,
            request_id=request_id,
            latency_ms=elapsed_ms,
            status=code,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error_type=etype,
        )
        return error_response(code, etype, message, request_id=request_id)

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

        if prompt_tokens > int(tenant["max_context_tokens"]):
            return finalize_error(
                400,
                "bad_input",
                f"prompt tokens exceed max_context_tokens ({prompt_tokens} > {tenant['max_context_tokens']})",
                rejection_reason="max_context",
            )

        try:
            requested_max_tokens, clamped_max_tokens = _resolve_requested_max_tokens(
                payload,
                int(tenant["max_output_tokens"]),
            )
        except ValueError as exc:
            return finalize_error(400, "bad_input", str(exc), rejection_reason="max_output")

        if requested_max_tokens > int(tenant["max_output_tokens"]):
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

        tenant_acquired = await concurrency.try_acquire_tenant(tenant_id, int(tenant["max_concurrent"]))
        if not tenant_acquired:
            return finalize_error(
                429,
                "limit_concurrent",
                "tenant concurrent request limit exceeded",
                rejection_reason="limit_concurrent",
            )

        rpm_ok = await limiter.allow_request(tenant_id, int(tenant["rpm_limit"]))
        if not rpm_ok:
            return finalize_error(429, "limit_rpm", "tenant rpm_limit exceeded", rejection_reason="limit_rpm")

        tpm_reserved = prompt_tokens + clamped_max_tokens
        tpm_ok = await limiter.reserve_tokens(tenant_id, int(tenant["tpm_limit"]), tpm_reserved)
        if not tpm_ok:
            return finalize_error(429, "limit_tpm", "tenant tpm_limit exceeded", rejection_reason="limit_tpm")

        forward_headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in {"host", "content-length", "connection", "authorization"}
        }
        forward_headers["x-request-id"] = request_id

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
        _record_request_with_metrics(
            metrics=metrics,
            db=db,
            ts_start=ts_start,
            ts_end=ts_end,
            latency_ms_value=elapsed_ms,
            tenant_id=tenant_id,
            model_id=settings.model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            status_code=status_code,
            error_type=error_type,
            request_id=request_id,
        )

        if status_code < 400:
            metrics.add_tokens(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                tenant_id=tenant_id,
                tenant_name=tenant_name,
            )

        json_log(
            "request_complete",
            tenant_id=tenant_id,
            request_id=request_id,
            latency_ms=elapsed_ms,
            status=status_code,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error_type=error_type,
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
