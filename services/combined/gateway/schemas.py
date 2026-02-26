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


class TenantPublic(BaseModel):
    tenant_id: str
    tenant_name: str
    max_concurrent: int
    rpm_limit: int
    tpm_limit: int
    max_context_tokens: int
    max_output_tokens: int
    created_at: datetime
    updated_at: datetime


class TenantCreateResponse(TenantPublic):
    api_key: str


class UsageRow(BaseModel):
    tenant_id: str
    tenant_name: str
    requests: int
    errors: int
    total_tokens: int
    p50_latency_ms: int
    p95_latency_ms: int


class UsageResponse(BaseModel):
    window: Literal["1h", "24h", "7d"]
    data: list[UsageRow]


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

