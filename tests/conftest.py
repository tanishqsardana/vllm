"""
Shared fixtures and helpers for the gateway integration test suite.

Configuration via environment variables:
  BASE_URL        Gateway base URL (default: http://localhost:8000)
  ADMIN_TOKEN     Admin token (required unless EXPECT_ADMIN_UNAVAILABLE=1)
  WAIT_TIMEOUT    Seconds to wait for gateway health (default: 60)
  POLL_INTERVAL   Poll interval while waiting (default: 2)
  SKIP_INFERENCE  Set to "1" to skip tests that need a live vLLM upstream
"""
from __future__ import annotations

import os
import time
from typing import Any

import pytest
import requests

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
WAIT_TIMEOUT = int(os.environ.get("WAIT_TIMEOUT", "60"))
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "2"))
SKIP_INFERENCE = os.environ.get("SKIP_INFERENCE", "0") == "1"


class GatewayClient:
    """Thin wrapper around requests.Session with convenience methods for the gateway API."""

    def __init__(self, base_url: str, admin_token: str) -> None:
        self.base = base_url
        self.token = admin_token
        self._session = requests.Session()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _admin_headers(self) -> dict[str, str]:
        return {"X-Admin-Token": self.token}

    def url(self, path: str) -> str:
        return f"{self.base}{path}"

    # ── unauthenticated ───────────────────────────────────────────────────────

    def get(self, path: str, **kwargs: Any) -> requests.Response:
        return self._session.get(self.url(path), **kwargs)

    # ── admin ─────────────────────────────────────────────────────────────────

    def admin_get(self, path: str, **kwargs: Any) -> requests.Response:
        return self._session.get(self.url(path), headers=self._admin_headers(), **kwargs)

    def admin_post(self, path: str, json: Any = None, **kwargs: Any) -> requests.Response:
        return self._session.post(
            self.url(path),
            json=json,
            headers=self._admin_headers(),
            **kwargs,
        )

    def admin_patch(self, path: str, json: Any = None, **kwargs: Any) -> requests.Response:
        return self._session.patch(
            self.url(path),
            json=json,
            headers=self._admin_headers(),
            **kwargs,
        )

    def admin_put(self, path: str, json: Any = None, **kwargs: Any) -> requests.Response:
        return self._session.put(
            self.url(path),
            json=json,
            headers=self._admin_headers(),
            **kwargs,
        )

    # ── inference ─────────────────────────────────────────────────────────────

    def inference(
        self,
        api_key: str,
        prompt: str = "Say OK.",
        max_tokens: int = 8,
        extra_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        headers = {
            "Authorization": f"Bearer {api_key}",
            **(extra_headers or {}),
        }
        return self._session.post(
            self.url("/v1/chat/completions"),
            json={
                "model": "placeholder",  # gateway overwrites with its own model_id
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            headers=headers,
            **kwargs,
        )


# ── session fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client() -> GatewayClient:
    return GatewayClient(BASE_URL, ADMIN_TOKEN)


@pytest.fixture(scope="session", autouse=True)
def wait_for_gateway(client: GatewayClient) -> None:
    """Block until /healthz returns 200 or WAIT_TIMEOUT is reached."""
    deadline = time.time() + WAIT_TIMEOUT
    while True:
        try:
            r = client.get("/healthz")
            if r.status_code == 200:
                return
        except requests.exceptions.ConnectionError:
            pass
        if time.time() > deadline:
            pytest.fail(
                f"Gateway at {BASE_URL} did not become healthy within {WAIT_TIMEOUT}s"
            )
        time.sleep(POLL_INTERVAL)


@pytest.fixture(scope="session")
def model_id(client: GatewayClient) -> str:
    r = client.get("/version")
    assert r.status_code == 200
    mid = r.json().get("model_id", "")
    if not mid:
        pytest.skip("model_id not available in /version — is the engine running?")
    return mid


@pytest.fixture(scope="session")
def tenant_a(client: GatewayClient) -> dict[str, str]:
    """Long-lived tenant used by most tests — created once per session."""
    ts = int(time.time() * 1000)
    r = client.admin_post("/admin/tenants", json={"tenant_name": f"test-tenant-a-{ts}"})
    assert r.status_code == 200, f"tenant_a creation failed: {r.text}"
    data = r.json()
    return {
        "id": data["tenant_id"],
        "key": data["api_key"],
        "name": data["tenant_name"],
    }


@pytest.fixture(scope="session")
def tenant_b(client: GatewayClient) -> dict[str, str]:
    """Second tenant for isolation / cross-tenant tests."""
    ts = int(time.time() * 1000) + 1
    r = client.admin_post("/admin/tenants", json={"tenant_name": f"test-tenant-b-{ts}"})
    assert r.status_code == 200, f"tenant_b creation failed: {r.text}"
    data = r.json()
    return {
        "id": data["tenant_id"],
        "key": data["api_key"],
        "name": data["tenant_name"],
    }
