from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


@dataclass
class TokenBucket:
    capacity: float
    refill_per_second: float
    tokens: float
    updated_at: float

    @classmethod
    def with_capacity(cls, capacity: int) -> "TokenBucket":
        now = time.monotonic()
        cap = float(max(capacity, 0))
        refill = cap / 60.0 if cap > 0 else 0.0
        return cls(capacity=cap, refill_per_second=refill, tokens=cap, updated_at=now)

    def resize(self, capacity: int) -> None:
        cap = float(max(capacity, 0))
        self._refill()
        self.capacity = cap
        self.refill_per_second = cap / 60.0 if cap > 0 else 0.0
        self.tokens = min(self.tokens, self.capacity)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = max(0.0, now - self.updated_at)
        self.updated_at = now
        if self.refill_per_second <= 0:
            self.tokens = min(self.tokens, self.capacity)
            return
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)

    def consume(self, amount: int) -> bool:
        if amount <= 0:
            return True
        self._refill()
        if self.tokens < amount:
            return False
        self.tokens -= amount
        return True

    def refund(self, amount: int) -> None:
        if amount <= 0:
            return
        self._refill()
        self.tokens = min(self.capacity, self.tokens + amount)


class TenantRateLimiter:
    def __init__(self) -> None:
        self._rpm_buckets: dict[str, TokenBucket] = {}
        self._tpm_buckets: dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()

    async def allow_request(self, tenant_id: str, rpm_limit: int) -> bool:
        async with self._lock:
            bucket = self._rpm_buckets.get(tenant_id)
            if bucket is None:
                bucket = TokenBucket.with_capacity(rpm_limit)
                self._rpm_buckets[tenant_id] = bucket
            elif int(bucket.capacity) != rpm_limit:
                bucket.resize(rpm_limit)
            return bucket.consume(1)

    async def reserve_tokens(self, tenant_id: str, tpm_limit: int, tokens: int) -> bool:
        async with self._lock:
            bucket = self._tpm_buckets.get(tenant_id)
            if bucket is None:
                bucket = TokenBucket.with_capacity(tpm_limit)
                self._tpm_buckets[tenant_id] = bucket
            elif int(bucket.capacity) != tpm_limit:
                bucket.resize(tpm_limit)
            return bucket.consume(tokens)

    async def refund_tokens(self, tenant_id: str, tokens: int) -> None:
        if tokens <= 0:
            return
        async with self._lock:
            bucket = self._tpm_buckets.get(tenant_id)
            if bucket is None:
                return
            bucket.refund(tokens)

