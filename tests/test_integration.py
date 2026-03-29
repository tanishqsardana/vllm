"""
Gateway integration test suite.

Covers:
  - Health + version endpoint
  - Admin auth enforcement (missing/wrong/correct token)
  - Tenant lifecycle: create, list, get, patch, deactivate/reactivate
  - Seat lifecycle: create, list, patch, deactivate/reactivate, missing-tenant guard
  - Key lifecycle: service key, seat key, revoke, list
  - Inference auth: no key, invalid key, valid, revoked, inactive tenant, inactive seat
  - Tenant-mismatch header enforcement (X-Expected-Tenant-Id)
  - Concurrency limiting: burst triggers 429, other tenant unaffected
  - Usage rollup: requests appear in /admin/usage/tenants, all windows accepted
  - Audit log: CRUD events present, all windows accepted
  - Budget: PUT, GET budget_status, window validation

Tests that require a live vLLM upstream are guarded by @pytest.mark.inference
and skipped automatically when SKIP_INFERENCE=1 (or ADMIN_TOKEN is absent).

Run:
  pytest tests/ -v
  SKIP_INFERENCE=1 pytest tests/ -v   # admin-plane only
"""
from __future__ import annotations

import threading
import time

import pytest
import requests

from conftest import ADMIN_TOKEN, BASE_URL, SKIP_INFERENCE

# Marker applied to every test that sends a real inference request
inference = pytest.mark.skipif(SKIP_INFERENCE, reason="SKIP_INFERENCE=1")


# ─── Health + Version ─────────────────────────────────────────────────────────

class TestHealthAndVersion:
    def test_healthz_returns_200(self, client):
        r = client.get("/healthz")
        assert r.status_code == 200

    def test_version_returns_200(self, client):
        r = client.get("/version")
        assert r.status_code == 200

    def test_version_has_required_fields(self, client):
        r = client.get("/version")
        data = r.json()
        required = [
            "build_sha",
            "build_time",
            "model_id",
            "vllm_version",
            "ui_enabled",
            "admin_auth_mode",
        ]
        missing = [f for f in required if f not in data]
        assert not missing, f"Missing /version fields: {missing}"

    def test_version_requires_no_auth(self, client):
        # Version should be publicly readable
        r = requests.get(f"{BASE_URL}/version")
        assert r.status_code == 200

    def test_ui_endpoint_returns_200(self, client):
        r = client.get("/ui")
        assert r.status_code == 200


# ─── Admin Auth Enforcement ───────────────────────────────────────────────────

class TestAdminAuth:
    def test_missing_token_is_401(self, client):
        r = requests.get(f"{BASE_URL}/admin/tenants")
        assert r.status_code == 401
        body = r.json()
        assert body["error"]["type"] == "unauthorized"

    def test_wrong_token_is_401(self, client):
        r = requests.get(
            f"{BASE_URL}/admin/tenants",
            headers={"X-Admin-Token": "definitely-wrong-token"},
        )
        assert r.status_code == 401
        assert r.json()["error"]["type"] == "unauthorized"

    def test_correct_token_is_200(self, client):
        r = client.admin_get("/admin/tenants")
        assert r.status_code == 200

    def test_wrong_token_on_post_is_401(self, client):
        r = requests.post(
            f"{BASE_URL}/admin/tenants",
            json={"tenant_name": "should-fail"},
            headers={"X-Admin-Token": "bad-token"},
        )
        assert r.status_code == 401


# ─── Tenant Lifecycle ─────────────────────────────────────────────────────────

class TestTenantLifecycle:
    def test_create_returns_api_key_with_prefix(self, tenant_a):
        assert tenant_a["id"], "tenant_id is empty"
        assert tenant_a["key"].startswith("cp_"), "api_key should start with cp_"

    def test_list_tenants_includes_created(self, client, tenant_a):
        r = client.admin_get("/admin/tenants")
        assert r.status_code == 200
        ids = {t["tenant_id"] for t in r.json().get("data", [])}
        assert tenant_a["id"] in ids

    def test_get_tenant_by_id(self, client, tenant_a):
        r = client.admin_get(f"/admin/tenants/{tenant_a['id']}")
        assert r.status_code == 200
        data = r.json()
        assert data["tenant_id"] == tenant_a["id"]
        assert data["tenant_name"] == tenant_a["name"]
        assert data["is_active"] is True

    def test_get_nonexistent_tenant_is_404(self, client):
        r = client.admin_get("/admin/tenants/does-not-exist-xyz")
        assert r.status_code == 404
        assert r.json()["error"]["type"] == "not_found"

    def test_patch_rpm_limit(self, client, tenant_a):
        r = client.admin_patch(f"/admin/tenants/{tenant_a['id']}", json={"rpm_limit": 1234})
        assert r.status_code == 200
        assert r.json()["rpm_limit"] == 1234

    def test_patch_max_concurrent(self, client, tenant_a):
        r = client.admin_patch(f"/admin/tenants/{tenant_a['id']}", json={"max_concurrent": 10})
        assert r.status_code == 200
        assert r.json()["max_concurrent"] == 10

    def test_create_with_explicit_limits(self, client):
        ts = int(time.time() * 1000)
        r = client.admin_post("/admin/tenants", json={
            "tenant_name": f"test-explicit-limits-{ts}",
            "max_concurrent": 3,
            "rpm_limit": 77,
            "tpm_limit": 55_000,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["max_concurrent"] == 3
        assert data["rpm_limit"] == 77
        assert data["tpm_limit"] == 55_000

    def test_deactivate_tenant(self, client):
        ts = int(time.time() * 1000)
        r = client.admin_post("/admin/tenants", json={"tenant_name": f"test-deactivate-{ts}"})
        assert r.status_code == 200
        tid = r.json()["tenant_id"]

        r = client.admin_patch(f"/admin/tenants/{tid}", json={"is_active": False})
        assert r.status_code == 200
        assert r.json()["is_active"] is False

    def test_reactivate_tenant(self, client):
        ts = int(time.time() * 1000)
        r = client.admin_post("/admin/tenants", json={"tenant_name": f"test-reactivate-{ts}"})
        assert r.status_code == 200
        tid = r.json()["tenant_id"]

        client.admin_patch(f"/admin/tenants/{tid}", json={"is_active": False})

        r = client.admin_patch(f"/admin/tenants/{tid}", json={"is_active": True})
        assert r.status_code == 200
        assert r.json()["is_active"] is True

    def test_duplicate_name_is_409(self, client, tenant_a):
        r = client.admin_post("/admin/tenants", json={"tenant_name": tenant_a["name"]})
        assert r.status_code == 409
        assert r.json()["error"]["type"] == "conflict"


# ─── Seat Lifecycle ───────────────────────────────────────────────────────────

class TestSeatLifecycle:
    @pytest.fixture(scope="class")
    def seat(self, client, tenant_a):
        ts = int(time.time() * 1000)
        r = client.admin_post("/admin/seats", json={
            "tenant_id": tenant_a["id"],
            "seat_name": f"test-seat-{ts}",
            "role": "user",
        })
        assert r.status_code == 200, r.text
        return r.json()

    def test_seat_has_expected_fields(self, seat, tenant_a):
        assert seat["seat_id"]
        assert seat["tenant_id"] == tenant_a["id"]
        assert seat["role"] == "user"
        assert seat["is_active"] is True

    def test_list_seats_includes_created(self, client, tenant_a, seat):
        r = client.admin_get(f"/admin/seats?tenant_id={tenant_a['id']}")
        assert r.status_code == 200
        seat_ids = {s["seat_id"] for s in r.json().get("data", [])}
        assert seat["seat_id"] in seat_ids

    def test_deactivate_seat(self, client, seat):
        r = client.admin_patch(f"/admin/seats/{seat['seat_id']}", json={"is_active": False})
        assert r.status_code == 200
        assert r.json()["is_active"] is False

    def test_reactivate_seat(self, client, seat):
        r = client.admin_patch(f"/admin/seats/{seat['seat_id']}", json={"is_active": True})
        assert r.status_code == 200
        assert r.json()["is_active"] is True

    def test_seat_for_nonexistent_tenant_is_404(self, client):
        r = client.admin_post("/admin/seats", json={
            "tenant_id": "nonexistent-tenant-id",
            "seat_name": "orphan",
            "role": "user",
        })
        assert r.status_code == 404
        assert r.json()["error"]["type"] == "not_found"

    def test_admin_role_seat(self, client, tenant_a):
        ts = int(time.time() * 1000)
        r = client.admin_post("/admin/seats", json={
            "tenant_id": tenant_a["id"],
            "seat_name": f"test-admin-seat-{ts}",
            "role": "admin",
        })
        assert r.status_code == 200
        assert r.json()["role"] == "admin"


# ─── Key Lifecycle ────────────────────────────────────────────────────────────

class TestKeyLifecycle:
    @pytest.fixture(scope="class")
    def service_key(self, client, tenant_a):
        r = client.admin_post("/admin/keys", json={
            "tenant_id": tenant_a["id"],
            "name": "test-service-key",
        })
        assert r.status_code == 200, r.text
        return r.json()

    @pytest.fixture(scope="class")
    def seat_and_key(self, client, tenant_a):
        ts = int(time.time() * 1000)
        seat_r = client.admin_post("/admin/seats", json={
            "tenant_id": tenant_a["id"],
            "seat_name": f"test-key-seat-{ts}",
            "role": "user",
        })
        assert seat_r.status_code == 200
        sid = seat_r.json()["seat_id"]

        key_r = client.admin_post("/admin/keys", json={
            "tenant_id": tenant_a["id"],
            "seat_id": sid,
            "name": "test-seat-bound-key",
        })
        assert key_r.status_code == 200
        return {"seat_id": sid, **key_r.json()}

    def test_service_key_has_cp_prefix(self, service_key):
        assert service_key["api_key"].startswith("cp_")

    def test_service_key_has_no_seat_id(self, service_key):
        assert service_key["seat_id"] is None

    def test_seat_key_linked_to_seat(self, seat_and_key):
        assert seat_and_key["seat_id"]
        assert seat_and_key["api_key"].startswith("cp_")

    def test_list_keys_includes_service_key(self, client, tenant_a, service_key):
        r = client.admin_get(f"/admin/keys?tenant_id={tenant_a['id']}")
        assert r.status_code == 200
        ids = {k["key_id"] for k in r.json().get("data", [])}
        assert service_key["key_id"] in ids

    def test_list_keys_includes_seat_key(self, client, tenant_a, seat_and_key):
        r = client.admin_get(f"/admin/keys?tenant_id={tenant_a['id']}")
        ids = {k["key_id"] for k in r.json().get("data", [])}
        assert seat_and_key["key_id"] in ids

    def test_revoke_key_sets_revoked_at(self, client, tenant_a):
        r = client.admin_post("/admin/keys", json={
            "tenant_id": tenant_a["id"],
            "name": "revoke-target",
        })
        assert r.status_code == 200
        kid = r.json()["key_id"]

        r = client.admin_post(f"/admin/keys/{kid}/revoke")
        assert r.status_code == 200

        r = client.admin_get(f"/admin/keys?tenant_id={tenant_a['id']}")
        keys = r.json().get("data", [])
        match = next((k for k in keys if k["key_id"] == kid), None)
        assert match is not None, "Revoked key not found in key list"
        assert match["revoked_at"] is not None, "revoked_at should be set after revocation"


# ─── Inference Auth ───────────────────────────────────────────────────────────

class TestInferenceAuth:
    @inference
    def test_no_key_is_401(self, client):
        r = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
        )
        assert r.status_code == 401

    @inference
    def test_invalid_key_is_401(self, client):
        r = client.inference("cp_thiskeyisnotvalid000000000000")
        assert r.status_code == 401

    @inference
    def test_valid_key_succeeds(self, client, tenant_a):
        r = client.inference(tenant_a["key"], prompt="Say OK in one word.", max_tokens=4)
        assert r.status_code == 200
        data = r.json()
        assert "choices" in data
        assert data["choices"][0]["message"]["content"]

    @inference
    def test_revoked_key_is_401(self, client, tenant_a):
        r = client.admin_post("/admin/keys", json={
            "tenant_id": tenant_a["id"],
            "name": "revoke-infer-test",
        })
        assert r.status_code == 200
        throwaway = r.json()

        client.admin_post(f"/admin/keys/{throwaway['key_id']}/revoke")

        r = client.inference(throwaway["api_key"])
        assert r.status_code == 401
        assert "revoked" in r.json()["error"]["message"].lower()

    @inference
    def test_inactive_tenant_is_401(self, client):
        ts = int(time.time() * 1000)
        r = client.admin_post("/admin/tenants", json={"tenant_name": f"test-infer-inactive-{ts}"})
        assert r.status_code == 200
        tid = r.json()["tenant_id"]
        key = r.json()["api_key"]

        client.admin_patch(f"/admin/tenants/{tid}", json={"is_active": False})

        r = client.inference(key)
        assert r.status_code == 401
        assert "inactive" in r.json()["error"]["message"].lower()

    @inference
    def test_inactive_seat_key_is_401(self, client):
        ts = int(time.time() * 1000)
        t = client.admin_post("/admin/tenants", json={"tenant_name": f"test-seat-infer-{ts}"})
        tid = t.json()["tenant_id"]

        s = client.admin_post("/admin/seats", json={
            "tenant_id": tid, "seat_name": "deactivate-me", "role": "user",
        })
        sid = s.json()["seat_id"]

        k = client.admin_post("/admin/keys", json={
            "tenant_id": tid, "seat_id": sid, "name": "seat-infer-key",
        })
        api_key = k.json()["api_key"]

        # Confirm it works while seat is active
        r = client.inference(api_key, prompt="Say OK.", max_tokens=4)
        assert r.status_code == 200, f"Expected 200 before deactivation, got {r.status_code}: {r.text}"

        # Deactivate seat
        client.admin_patch(f"/admin/seats/{sid}", json={"is_active": False})

        # Now should be rejected
        r = client.inference(api_key)
        assert r.status_code == 401
        assert "seat" in r.json()["error"]["message"].lower()

    @inference
    def test_tenant_mismatch_header_is_403(self, client, tenant_a, tenant_b):
        r = client.inference(
            tenant_a["key"],
            extra_headers={"X-Expected-Tenant-Id": tenant_b["id"]},
        )
        assert r.status_code == 403
        assert "mismatch" in r.json()["error"]["message"].lower()

    @inference
    def test_correct_expected_tenant_id_passes(self, client, tenant_a):
        r = client.inference(
            tenant_a["key"],
            extra_headers={"X-Expected-Tenant-Id": tenant_a["id"]},
        )
        assert r.status_code == 200


# ─── Concurrency Limiting ─────────────────────────────────────────────────────

class TestConcurrencyLimiting:
    @inference
    def test_burst_triggers_429(self, client, tenant_b):
        """Set max_concurrent=1, fire 3 simultaneous long requests → ≥2 get 429."""
        client.admin_patch(f"/admin/tenants/{tenant_b['id']}", json={"max_concurrent": 1})

        results: dict[int, int] = {}

        def do_req(idx: int) -> None:
            r = client.inference(
                tenant_b["key"],
                prompt="Write a very long detailed technical essay on distributed systems, at least 500 words.",
                max_tokens=256,
            )
            results[idx] = r.status_code

        threads = [threading.Thread(target=do_req, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        count_429 = sum(1 for s in results.values() if s == 429)
        assert count_429 >= 2, (
            f"Expected ≥2 requests to get 429 (max_concurrent=1, 3 burst), "
            f"got statuses: {results}"
        )

    @inference
    def test_other_tenant_unaffected_during_burst(self, client, tenant_a, tenant_b):
        """While tenant_b is at max, tenant_a requests should still get 200."""
        client.admin_patch(f"/admin/tenants/{tenant_b['id']}", json={"max_concurrent": 1})

        blocker_started = threading.Event()
        blocker_done = threading.Event()

        def blocker() -> None:
            blocker_started.set()
            client.inference(
                tenant_b["key"],
                prompt="Write an extremely long essay. Be verbose.",
                max_tokens=512,
            )
            blocker_done.set()

        t = threading.Thread(target=blocker)
        t.start()
        blocker_started.wait(timeout=5)
        time.sleep(0.1)  # let the blocking request actually reach the gateway

        r = client.inference(tenant_a["key"], prompt="Say OK.", max_tokens=4)
        assert r.status_code == 200, (
            f"Tenant A was affected by Tenant B's concurrency limit: "
            f"HTTP {r.status_code} — {r.text}"
        )

        blocker_done.wait(timeout=60)
        t.join(timeout=5)


# ─── RPM Rate Limiting ────────────────────────────────────────────────────────

class TestRpmLimiting:
    @inference
    def test_rpm_limit_triggers_429(self, client):
        """Set rpm_limit=2, send 4 rapid requests → at least 1 should be 429."""
        ts = int(time.time() * 1000)
        r = client.admin_post("/admin/tenants", json={
            "tenant_name": f"test-rpm-{ts}",
            "rpm_limit": 2,
        })
        assert r.status_code == 200
        key = r.json()["api_key"]

        statuses = []
        for _ in range(4):
            r = client.inference(key, prompt="Say OK.", max_tokens=4)
            statuses.append(r.status_code)

        assert 429 in statuses, (
            f"Expected at least one 429 with rpm_limit=2 after 4 requests, "
            f"got statuses: {statuses}"
        )


# ─── Usage Rollup ─────────────────────────────────────────────────────────────

class TestUsageRollup:
    @pytest.mark.parametrize("window", ["1h", "24h", "7d", "30d"])
    def test_usage_windows_accepted(self, client, window):
        r = client.admin_get(f"/admin/usage/tenants?window={window}")
        assert r.status_code == 200, f"window={window!r} failed: {r.text}"

    def test_invalid_window_is_400(self, client):
        r = client.admin_get("/admin/usage/tenants?window=99years")
        assert r.status_code == 400

    def test_usage_response_has_data_key(self, client):
        r = client.admin_get("/admin/usage/tenants?window=24h")
        assert r.status_code == 200
        assert "data" in r.json()

    @inference
    def test_inference_appears_in_usage(self, client, tenant_a):
        r = client.inference(tenant_a["key"], prompt="Say usage test.", max_tokens=8)
        assert r.status_code == 200

        time.sleep(1)  # allow accounting write to flush

        r = client.admin_get("/admin/usage/tenants?window=1h")
        assert r.status_code == 200
        rows = r.json().get("data", [])
        row = next((u for u in rows if u["tenant_id"] == tenant_a["id"]), None)
        assert row is not None, "Tenant A not found in /admin/usage/tenants"
        assert row["requests"] >= 1, f"Expected ≥1 request in usage, got {row['requests']}"

    @inference
    def test_usage_has_token_counts(self, client, tenant_a):
        r = client.admin_get("/admin/usage/tenants?window=1h")
        rows = r.json().get("data", [])
        row = next((u for u in rows if u["tenant_id"] == tenant_a["id"]), None)
        if row is None:
            pytest.skip("No usage row for tenant_a yet")
        assert row.get("total_tokens", 0) >= 0


# ─── Audit Log ────────────────────────────────────────────────────────────────

class TestAuditLog:
    @pytest.mark.parametrize("window", ["1h", "24h", "7d", "30d"])
    def test_audit_windows_accepted(self, client, window):
        r = client.admin_get(f"/admin/audit?window={window}")
        assert r.status_code == 200, f"audit window={window!r} failed: {r.text}"

    def test_invalid_audit_window_is_400(self, client):
        r = client.admin_get("/admin/audit?window=badwindow")
        assert r.status_code == 400

    def test_tenant_create_is_audited(self, client):
        ts = int(time.time() * 1000)
        r = client.admin_post("/admin/tenants", json={"tenant_name": f"test-audit-tenant-{ts}"})
        assert r.status_code == 200
        tid = r.json()["tenant_id"]

        r = client.admin_get(f"/admin/audit?window=1h&tenant_id={tid}")
        assert r.status_code == 200
        events = r.json().get("data", [])
        assert any(
            e["action"] == "tenant_create" and e["resource_id"] == tid
            for e in events
        ), f"tenant_create not in audit log for {tid}"

    def test_seat_create_is_audited(self, client, tenant_a):
        ts = int(time.time() * 1000)
        r = client.admin_post("/admin/seats", json={
            "tenant_id": tenant_a["id"],
            "seat_name": f"audit-seat-{ts}",
            "role": "user",
        })
        assert r.status_code == 200
        sid = r.json()["seat_id"]

        r = client.admin_get(f"/admin/audit?window=1h&tenant_id={tenant_a['id']}")
        events = r.json().get("data", [])
        assert any(
            e["action"] == "seat_create" and e["resource_id"] == sid
            for e in events
        ), f"seat_create not in audit log for seat {sid}"

    def test_key_revoke_is_audited(self, client, tenant_a):
        r = client.admin_post("/admin/keys", json={
            "tenant_id": tenant_a["id"],
            "name": "audit-revoke-key",
        })
        assert r.status_code == 200
        kid = r.json()["key_id"]

        client.admin_post(f"/admin/keys/{kid}/revoke")

        r = client.admin_get(f"/admin/audit?window=1h&tenant_id={tenant_a['id']}")
        events = r.json().get("data", [])
        assert any(
            e["action"] == "key_revoke" and e["resource_id"] == kid
            for e in events
        ), f"key_revoke not in audit log for key {kid}"

    def test_tenant_patch_is_audited(self, client, tenant_a):
        client.admin_patch(f"/admin/tenants/{tenant_a['id']}", json={"rpm_limit": 500})

        r = client.admin_get(f"/admin/audit?window=1h&tenant_id={tenant_a['id']}")
        events = r.json().get("data", [])
        assert any(
            e["action"] == "tenant_update" and e["resource_id"] == tenant_a["id"]
            for e in events
        ), "tenant_update not in audit log"

    def test_audit_data_has_required_fields(self, client, tenant_a):
        r = client.admin_get(f"/admin/audit?window=1h&tenant_id={tenant_a['id']}")
        events = r.json().get("data", [])
        if not events:
            pytest.skip("No audit events yet for tenant_a")
        event = events[0]
        for field in ["action", "resource_type", "resource_id", "created_at"]:
            assert field in event, f"Audit event missing field: {field}"


# ─── Budget ───────────────────────────────────────────────────────────────────

class TestBudget:
    @pytest.fixture(scope="class")
    def budgeted_tenant(self, client):
        ts = int(time.time() * 1000)
        r = client.admin_post("/admin/tenants", json={"tenant_name": f"test-budget-{ts}"})
        assert r.status_code == 200
        return r.json()

    def test_put_budget_returns_200(self, client, budgeted_tenant):
        r = client.admin_put(
            f"/admin/budgets/{budgeted_tenant['tenant_id']}",
            json={"window": "month", "budget_usd": 50.0},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["budget_usd"] == 50.0
        assert data["window"] == "month"

    def test_budget_status_has_required_fields(self, client, budgeted_tenant):
        # ensure budget is set first
        client.admin_put(
            f"/admin/budgets/{budgeted_tenant['tenant_id']}",
            json={"window": "month", "budget_usd": 25.0},
        )
        r = client.admin_get(f"/admin/budget_status?tenant_id={budgeted_tenant['tenant_id']}")
        assert r.status_code == 200
        data = r.json()
        for field in ["tenant_id", "budget_usd", "cost_usd", "budget_ratio"]:
            assert field in data, f"Missing budget_status field: {field}"

    def test_budget_status_reflects_set_amount(self, client, budgeted_tenant):
        client.admin_put(
            f"/admin/budgets/{budgeted_tenant['tenant_id']}",
            json={"window": "day", "budget_usd": 99.0},
        )
        r = client.admin_get(
            f"/admin/budget_status?tenant_id={budgeted_tenant['tenant_id']}&window=day"
        )
        assert r.status_code == 200
        assert r.json()["budget_usd"] == 99.0

    def test_budget_status_ratio_is_non_negative(self, client, budgeted_tenant):
        r = client.admin_get(f"/admin/budget_status?tenant_id={budgeted_tenant['tenant_id']}")
        assert r.status_code == 200
        assert r.json()["budget_ratio"] >= 0.0

    def test_budget_status_nonexistent_tenant_is_404(self, client):
        r = client.admin_get("/admin/budget_status?tenant_id=ghost-tenant-xyz")
        assert r.status_code == 404
        assert r.json()["error"]["type"] == "not_found"

    def test_update_budget_amount(self, client, budgeted_tenant):
        tid = budgeted_tenant["tenant_id"]
        client.admin_put(f"/admin/budgets/{tid}", json={"window": "month", "budget_usd": 10.0})
        client.admin_put(f"/admin/budgets/{tid}", json={"window": "month", "budget_usd": 200.0})

        r = client.admin_get(f"/admin/budget_status?tenant_id={tid}&window=month")
        assert r.status_code == 200
        assert r.json()["budget_usd"] == 200.0

    @pytest.mark.parametrize("window", ["day", "week", "month"])
    def test_budget_windows_accepted(self, client, budgeted_tenant, window):
        tid = budgeted_tenant["tenant_id"]
        r = client.admin_put(
            f"/admin/budgets/{tid}",
            json={"window": window, "budget_usd": 1.0},
        )
        assert r.status_code == 200, f"PUT budget window={window!r} failed: {r.text}"
