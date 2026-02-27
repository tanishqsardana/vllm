from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class TenantLimitsMixin(BaseModel):
    max_concurrent: int = Field(default=4, ge=1)
    rpm_limit: int = Field(default=120, ge=1)
    tpm_limit: int = Field(default=120000, ge=1)
    max_context_tokens: int = Field(default=8192, ge=1)
    max_output_tokens: int = Field(default=512, ge=1)


class TenantCreateRequest(BaseModel):
    tenant_name: str = Field(min_length=1, max_length=128)
    max_concurrent: int | None = Field(default=None, ge=1)
    rpm_limit: int | None = Field(default=None, ge=1)
    tpm_limit: int | None = Field(default=None, ge=1)
    max_context_tokens: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)


class TenantPatchRequest(BaseModel):
    tenant_name: str | None = Field(default=None, min_length=1, max_length=128)
    max_concurrent: int | None = Field(default=None, ge=1)
    rpm_limit: int | None = Field(default=None, ge=1)
    tpm_limit: int | None = Field(default=None, ge=1)
    max_context_tokens: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    is_active: bool | None = None


class TenantPublic(BaseModel):
    tenant_id: str
    tenant_name: str
    max_concurrent: int
    rpm_limit: int
    tpm_limit: int
    max_context_tokens: int
    max_output_tokens: int
    is_active: bool
    created_at: datetime
    updated_at: datetime


class TenantCreateResponse(TenantPublic):
    api_key: str


class UsageTenantRow(BaseModel):
    tenant_id: str
    tenant_name: str
    requests: int
    errors: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    p95_latency_ms: int
    gpu_seconds_est_sum: float
    cost_est_sum: float


class UsageResponse(BaseModel):
    window: Literal["1h", "24h", "7d"]
    data: list[UsageTenantRow]


class UsageTenantsResponse(BaseModel):
    window: Literal["1h", "24h", "7d"]
    cost_estimation_enabled: bool
    data: list[UsageTenantRow]


class UsageSeatRow(BaseModel):
    seat_id: str | None
    seat_name: str | None
    role: str
    requests: int
    errors: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    p95_latency_ms: int
    gpu_seconds_est_sum: float
    cost_est_sum: float


class UsageSeatsResponse(BaseModel):
    window: Literal["1h", "24h", "7d"]
    tenant_id: str
    cost_estimation_enabled: bool
    data: list[UsageSeatRow]


class SeatCreateRequest(BaseModel):
    tenant_id: str
    seat_name: str | None = Field(default=None, max_length=320)
    role: Literal["admin", "user", "service"] = "user"


class SeatPatchRequest(BaseModel):
    seat_name: str | None = Field(default=None, max_length=320)
    role: Literal["admin", "user", "service"] | None = None
    is_active: bool | None = None


class SeatPublic(BaseModel):
    seat_id: str
    tenant_id: str
    seat_name: str | None
    role: str
    is_active: bool
    created_at: datetime


class KeyCreateRequest(BaseModel):
    tenant_id: str
    seat_id: str | None = None
    name: str | None = Field(default=None, max_length=128)


class KeyCreateResponse(BaseModel):
    key_id: str
    tenant_id: str
    seat_id: str | None
    api_key: str
    created_at: datetime


class KeyPublic(BaseModel):
    key_id: str
    tenant_id: str
    seat_id: str | None
    seat_name: str | None
    role: str | None
    created_at: datetime
    revoked_at: datetime | None


class TenantBudgetPutRequest(BaseModel):
    window: Literal["day", "week", "month"]
    budget_usd: float = Field(gt=0)


class TenantBudgetPublic(BaseModel):
    tenant_id: str
    window: Literal["day", "week", "month"]
    budget_usd: float
    warn_50: bool
    warn_80: bool
    warn_100: bool
    updated_at: datetime


class BudgetStatusResponse(BaseModel):
    tenant_id: str
    window: Literal["day", "week", "month"]
    window_start: datetime
    budget_usd: float
    cost_usd: float
    budget_ratio: float
    thresholds_crossed: list[Literal["50", "80", "100"]]
    cost_estimation_enabled: bool


class ErrorBody(BaseModel):
    type: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class GatewayVersion(BaseModel):
    build_sha: str
    build_time: str
    model_id: str
    vllm_version: str


class ChatCompletionPayload(BaseModel):
    model: str | None = None
    messages: Any | None = None
    prompt: Any | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    stream: bool | None = None
