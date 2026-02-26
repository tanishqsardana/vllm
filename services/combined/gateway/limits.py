from __future__ import annotations

import asyncio


class NonBlockingSemaphoreGate:
    """
    Non-blocking gate backed by asyncio.Semaphore semantics.
    We mutate available permits under a lock so acquire is immediate (reject instead of queue).
    """

    def __init__(self, limit: int) -> None:
        self.limit = max(0, int(limit))
        self.semaphore = asyncio.Semaphore(self.limit)
        self.inflight = 0
        self._lock = asyncio.Lock()

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self.limit <= 0 or self.semaphore._value <= 0:  # noqa: SLF001
                return False
            self.semaphore._value -= 1  # noqa: SLF001
            self.inflight += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            if self.inflight > 0:
                self.inflight -= 1
            target_available = max(0, self.limit - self.inflight)
            self.semaphore._value = target_available  # noqa: SLF001

    async def set_limit(self, new_limit: int) -> None:
        async with self._lock:
            self.limit = max(0, int(new_limit))
            target_available = max(0, self.limit - self.inflight)
            self.semaphore._value = target_available  # noqa: SLF001


class ConcurrencyManager:
    def __init__(self, global_max_concurrent: int) -> None:
        self._global_gate = NonBlockingSemaphoreGate(global_max_concurrent)
        self._tenant_gates: dict[str, NonBlockingSemaphoreGate] = {}
        self._tenant_lock = asyncio.Lock()

    async def try_acquire_global(self) -> bool:
        return await self._global_gate.try_acquire()

    async def release_global(self) -> None:
        await self._global_gate.release()

    async def set_global_limit(self, new_limit: int) -> None:
        await self._global_gate.set_limit(new_limit)

    async def try_acquire_tenant(self, tenant_id: str, tenant_limit: int) -> bool:
        gate = await self._get_or_create_tenant_gate(tenant_id, tenant_limit)
        await gate.set_limit(tenant_limit)
        return await gate.try_acquire()

    async def release_tenant(self, tenant_id: str) -> None:
        gate = self._tenant_gates.get(tenant_id)
        if gate is None:
            return
        await gate.release()

    async def set_tenant_limit(self, tenant_id: str, tenant_limit: int) -> None:
        gate = await self._get_or_create_tenant_gate(tenant_id, tenant_limit)
        await gate.set_limit(tenant_limit)

    async def _get_or_create_tenant_gate(self, tenant_id: str, tenant_limit: int) -> NonBlockingSemaphoreGate:
        gate = self._tenant_gates.get(tenant_id)
        if gate is not None:
            return gate

        async with self._tenant_lock:
            gate = self._tenant_gates.get(tenant_id)
            if gate is None:
                gate = NonBlockingSemaphoreGate(tenant_limit)
                self._tenant_gates[tenant_id] = gate
            return gate

