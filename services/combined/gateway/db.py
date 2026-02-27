from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        db_parent = Path(db_path).expanduser().resolve().parent
        db_parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def health_check(self) -> bool:
        try:
            with self._lock:
                self._conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    def ensure_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tenants (
                    tenant_id TEXT PRIMARY KEY,
                    tenant_name TEXT NOT NULL UNIQUE,
                    api_key_hash TEXT NOT NULL UNIQUE,
                    max_concurrent INTEGER NOT NULL,
                    rpm_limit INTEGER NOT NULL,
                    tpm_limit INTEGER NOT NULL,
                    max_context_tokens INTEGER NOT NULL,
                    max_output_tokens INTEGER NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_start TEXT NOT NULL,
                    ts_end TEXT NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    tenant_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    status_code INTEGER NOT NULL,
                    error_type TEXT,
                    request_id TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS seats (
                    seat_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    seat_name TEXT,
                    role TEXT NOT NULL DEFAULT 'user',
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (tenant_id) REFERENCES tenants (tenant_id)
                );

                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    seat_id TEXT,
                    key_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    revoked_at TEXT,
                    FOREIGN KEY (tenant_id) REFERENCES tenants (tenant_id),
                    FOREIGN KEY (seat_id) REFERENCES seats (seat_id)
                );

                CREATE TABLE IF NOT EXISTS tenant_budgets (
                    tenant_id TEXT PRIMARY KEY,
                    window TEXT NOT NULL,
                    budget_usd REAL NOT NULL,
                    warn_50 INTEGER NOT NULL DEFAULT 1,
                    warn_80 INTEGER NOT NULL DEFAULT 1,
                    warn_100 INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (tenant_id) REFERENCES tenants (tenant_id)
                );

                CREATE TABLE IF NOT EXISTS budget_events (
                    event_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    window TEXT NOT NULL,
                    window_start TEXT NOT NULL,
                    threshold TEXT NOT NULL,
                    cost_usd REAL NOT NULL,
                    budget_usd REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (tenant_id) REFERENCES tenants (tenant_id)
                );

                CREATE INDEX IF NOT EXISTS idx_requests_tenant_ts ON requests (tenant_id, ts_start);
                CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests (ts_start);
                CREATE INDEX IF NOT EXISTS idx_api_keys_tenant ON api_keys (tenant_id);
                CREATE INDEX IF NOT EXISTS idx_api_keys_seat ON api_keys (seat_id);
                CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys (key_hash);
                CREATE INDEX IF NOT EXISTS idx_api_keys_revoked ON api_keys (revoked_at);
                CREATE INDEX IF NOT EXISTS idx_seats_tenant ON seats (tenant_id);
                CREATE INDEX IF NOT EXISTS idx_budget_events_window ON budget_events (tenant_id, window, window_start);
                """
            )

            self._add_column_if_missing("tenants", "is_active INTEGER NOT NULL DEFAULT 1")
            self._add_column_if_missing("requests", "seat_id TEXT")
            self._add_column_if_missing("requests", "key_id TEXT")
            self._add_column_if_missing("requests", "gpu_seconds_est REAL")
            self._add_column_if_missing("requests", "cost_est REAL")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_seat_ts ON requests (seat_id, ts_start)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_key_ts ON requests (key_id, ts_start)")
            self._migrate_legacy_tenant_keys()
            self._conn.commit()

    def _add_column_if_missing(self, table_name: str, column_definition: str) -> None:
        column_name = column_definition.split(" ", 1)[0]
        try:
            self._conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {column_definition}")
            return
        except sqlite3.OperationalError:
            pass

        if not self._column_exists(table_name, column_name):
            self._conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_definition}")

    def _column_exists(self, table_name: str, column_name: str) -> bool:
        rows = self._conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return any(row["name"] == column_name for row in rows)

    def _migrate_legacy_tenant_keys(self) -> None:
        tenant_rows = self._conn.execute(
            """
            SELECT tenant_id, api_key_hash, created_at
            FROM tenants
            WHERE api_key_hash IS NOT NULL AND api_key_hash <> ''
            """
        ).fetchall()
        for row in tenant_rows:
            exists = self._conn.execute(
                """
                SELECT 1
                FROM api_keys
                WHERE tenant_id = ? AND key_hash = ?
                LIMIT 1
                """,
                (row["tenant_id"], row["api_key_hash"]),
            ).fetchone()
            if exists is not None:
                continue

            created_at = row["created_at"] or utc_now_iso()
            self._conn.execute(
                """
                INSERT INTO api_keys (key_id, tenant_id, seat_id, key_hash, created_at, revoked_at)
                VALUES (?, ?, NULL, ?, ?, NULL)
                """,
                (str(uuid.uuid4()), row["tenant_id"], row["api_key_hash"], created_at),
            )

    @staticmethod
    def _tenant_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "tenant_id": row["tenant_id"],
            "tenant_name": row["tenant_name"],
            "max_concurrent": row["max_concurrent"],
            "rpm_limit": row["rpm_limit"],
            "tpm_limit": row["tpm_limit"],
            "max_context_tokens": row["max_context_tokens"],
            "max_output_tokens": row["max_output_tokens"],
            "is_active": bool(row["is_active"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_tenants(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT tenant_id, tenant_name, max_concurrent, rpm_limit, tpm_limit,
                       max_context_tokens, max_output_tokens, is_active, created_at, updated_at
                FROM tenants
                ORDER BY created_at ASC
                """
            ).fetchall()
        return [self._tenant_row_to_dict(row) for row in rows]

    def get_tenant(self, tenant_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT tenant_id, tenant_name, max_concurrent, rpm_limit, tpm_limit,
                       max_context_tokens, max_output_tokens, is_active, created_at, updated_at
                FROM tenants
                WHERE tenant_id = ?
                """,
                (tenant_id,),
            ).fetchone()
        if row is None:
            return None
        return self._tenant_row_to_dict(row)

    def list_tenant_auth_rows(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT tenant_id, tenant_name, api_key_hash, max_concurrent, rpm_limit, tpm_limit,
                       max_context_tokens, max_output_tokens, is_active
                FROM tenants
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_api_key_auth_rows(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    k.key_id,
                    k.tenant_id,
                    k.seat_id,
                    k.key_hash,
                    k.created_at,
                    k.revoked_at,
                    t.tenant_name,
                    t.max_concurrent,
                    t.rpm_limit,
                    t.tpm_limit,
                    t.max_context_tokens,
                    t.max_output_tokens,
                    t.is_active AS tenant_is_active,
                    s.seat_id AS seat_ref_id,
                    s.seat_name,
                    s.role,
                    s.is_active AS seat_is_active
                FROM api_keys k
                JOIN tenants t ON t.tenant_id = k.tenant_id
                LEFT JOIN seats s ON s.seat_id = k.seat_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def create_tenant(
        self,
        tenant_id: str,
        tenant_name: str,
        api_key_hash: str,
        max_concurrent: int,
        rpm_limit: int,
        tpm_limit: int,
        max_context_tokens: int,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        ts = utc_now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO tenants (
                    tenant_id, tenant_name, api_key_hash,
                    max_concurrent, rpm_limit, tpm_limit,
                    max_context_tokens, max_output_tokens, is_active,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    tenant_id,
                    tenant_name,
                    api_key_hash,
                    max_concurrent,
                    rpm_limit,
                    tpm_limit,
                    max_context_tokens,
                    max_output_tokens,
                    ts,
                    ts,
                ),
            )
            self._conn.commit()
        tenant = self.get_tenant(tenant_id)
        if tenant is None:
            raise RuntimeError("failed to create tenant")
        return tenant

    def patch_tenant(self, tenant_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        if not updates:
            return self.get_tenant(tenant_id)

        allowed_fields = {
            "tenant_name",
            "max_concurrent",
            "rpm_limit",
            "tpm_limit",
            "max_context_tokens",
            "max_output_tokens",
            "is_active",
        }
        filtered = {k: v for k, v in updates.items() if k in allowed_fields and v is not None}
        if not filtered:
            return self.get_tenant(tenant_id)

        set_parts = [f"{field} = ?" for field in filtered]
        params: list[Any] = list(filtered.values())
        set_parts.append("updated_at = ?")
        params.append(utc_now_iso())
        params.append(tenant_id)

        query = f"UPDATE tenants SET {', '.join(set_parts)} WHERE tenant_id = ?"

        with self._lock:
            cur = self._conn.execute(query, params)
            self._conn.commit()
            if cur.rowcount == 0:
                return None
        return self.get_tenant(tenant_id)

    def insert_request(
        self,
        ts_start: str,
        ts_end: str,
        latency_ms: int,
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
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO requests (
                    ts_start, ts_end, latency_ms, tenant_id, model_id,
                    prompt_tokens, completion_tokens, total_tokens,
                    status_code, error_type, request_id,
                    seat_id, key_id, gpu_seconds_est, cost_est
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_start,
                    ts_end,
                    latency_ms,
                    tenant_id,
                    model_id,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    status_code,
                    error_type,
                    request_id,
                    seat_id,
                    key_id,
                    gpu_seconds_est,
                    cost_est,
                ),
            )
            self._conn.commit()

    @staticmethod
    def _seat_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "seat_id": row["seat_id"],
            "tenant_id": row["tenant_id"],
            "seat_name": row["seat_name"],
            "role": row["role"],
            "is_active": bool(row["is_active"]),
            "created_at": row["created_at"],
        }

    def get_seat(self, seat_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT seat_id, tenant_id, seat_name, role, is_active, created_at
                FROM seats
                WHERE seat_id = ?
                """,
                (seat_id,),
            ).fetchone()
        if row is None:
            return None
        return self._seat_row_to_dict(row)

    def create_seat(
        self,
        seat_id: str,
        tenant_id: str,
        seat_name: str | None,
        role: str,
    ) -> dict[str, Any]:
        ts = utc_now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO seats (seat_id, tenant_id, seat_name, role, is_active, created_at)
                VALUES (?, ?, ?, ?, 1, ?)
                """,
                (seat_id, tenant_id, seat_name, role, ts),
            )
            self._conn.commit()
        seat = self.get_seat(seat_id)
        if seat is None:
            raise RuntimeError("failed to create seat")
        return seat

    def list_seats(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if tenant_id is None:
                rows = self._conn.execute(
                    """
                    SELECT seat_id, tenant_id, seat_name, role, is_active, created_at
                    FROM seats
                    ORDER BY created_at ASC
                    """
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT seat_id, tenant_id, seat_name, role, is_active, created_at
                    FROM seats
                    WHERE tenant_id = ?
                    ORDER BY created_at ASC
                    """,
                    (tenant_id,),
                ).fetchall()
        return [self._seat_row_to_dict(row) for row in rows]

    def patch_seat(self, seat_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        if not updates:
            return self.get_seat(seat_id)

        allowed_fields = {"seat_name", "role", "is_active"}
        filtered = {k: v for k, v in updates.items() if k in allowed_fields and v is not None}
        if not filtered:
            return self.get_seat(seat_id)

        set_parts = [f"{field} = ?" for field in filtered]
        params: list[Any] = list(filtered.values())
        params.append(seat_id)

        query = f"UPDATE seats SET {', '.join(set_parts)} WHERE seat_id = ?"
        with self._lock:
            cur = self._conn.execute(query, params)
            self._conn.commit()
            if cur.rowcount == 0:
                return None
        return self.get_seat(seat_id)

    @staticmethod
    def _api_key_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "key_id": row["key_id"],
            "tenant_id": row["tenant_id"],
            "seat_id": row["seat_id"],
            "seat_name": row["seat_name"],
            "role": row["role"],
            "created_at": row["created_at"],
            "revoked_at": row["revoked_at"],
        }

    def get_api_key(self, key_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT k.key_id, k.tenant_id, k.seat_id, s.seat_name, s.role, k.created_at, k.revoked_at
                FROM api_keys k
                LEFT JOIN seats s ON s.seat_id = k.seat_id
                WHERE k.key_id = ?
                """,
                (key_id,),
            ).fetchone()
        if row is None:
            return None
        return self._api_key_row_to_dict(row)

    def create_api_key(
        self,
        key_id: str,
        tenant_id: str,
        seat_id: str | None,
        key_hash: str,
    ) -> dict[str, Any]:
        ts = utc_now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO api_keys (key_id, tenant_id, seat_id, key_hash, created_at, revoked_at)
                VALUES (?, ?, ?, ?, ?, NULL)
                """,
                (key_id, tenant_id, seat_id, key_hash, ts),
            )
            self._conn.commit()
        key_row = self.get_api_key(key_id)
        if key_row is None:
            raise RuntimeError("failed to create api key")
        return key_row

    def list_api_keys(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if tenant_id is None:
                rows = self._conn.execute(
                    """
                    SELECT k.key_id, k.tenant_id, k.seat_id, s.seat_name, s.role, k.created_at, k.revoked_at
                    FROM api_keys k
                    LEFT JOIN seats s ON s.seat_id = k.seat_id
                    ORDER BY k.created_at ASC
                    """
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT k.key_id, k.tenant_id, k.seat_id, s.seat_name, s.role, k.created_at, k.revoked_at
                    FROM api_keys k
                    LEFT JOIN seats s ON s.seat_id = k.seat_id
                    WHERE k.tenant_id = ?
                    ORDER BY k.created_at ASC
                    """,
                    (tenant_id,),
                ).fetchall()
        return [self._api_key_row_to_dict(row) for row in rows]

    def revoke_api_key(self, key_id: str) -> dict[str, Any] | None:
        now = utc_now_iso()
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE api_keys
                SET revoked_at = COALESCE(revoked_at, ?)
                WHERE key_id = ?
                """,
                (now, key_id),
            )
            self._conn.commit()
            if cur.rowcount == 0:
                return None
        return self.get_api_key(key_id)

    def list_usage_base(self, threshold_ts: str) -> list[dict[str, Any]]:
        return self.list_tenant_usage_base(threshold_ts)

    def list_tenant_usage_base(self, threshold_ts: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if tenant_id is None:
                rows = self._conn.execute(
                    """
                    SELECT
                        t.tenant_id AS tenant_id,
                        t.tenant_name AS tenant_name,
                        COUNT(r.id) AS requests,
                        SUM(CASE WHEN r.status_code >= 400 THEN 1 ELSE 0 END) AS errors,
                        COALESCE(SUM(r.prompt_tokens), 0) AS prompt_tokens,
                        COALESCE(SUM(r.completion_tokens), 0) AS completion_tokens,
                        COALESCE(SUM(r.total_tokens), 0) AS total_tokens,
                        COALESCE(SUM(r.gpu_seconds_est), 0.0) AS gpu_seconds_est_sum,
                        COALESCE(SUM(r.cost_est), 0.0) AS cost_est_sum
                    FROM tenants t
                    LEFT JOIN requests r
                        ON t.tenant_id = r.tenant_id AND r.ts_start >= ?
                    GROUP BY t.tenant_id, t.tenant_name
                    ORDER BY t.tenant_name ASC
                    """,
                    (threshold_ts,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT
                        t.tenant_id AS tenant_id,
                        t.tenant_name AS tenant_name,
                        COUNT(r.id) AS requests,
                        SUM(CASE WHEN r.status_code >= 400 THEN 1 ELSE 0 END) AS errors,
                        COALESCE(SUM(r.prompt_tokens), 0) AS prompt_tokens,
                        COALESCE(SUM(r.completion_tokens), 0) AS completion_tokens,
                        COALESCE(SUM(r.total_tokens), 0) AS total_tokens,
                        COALESCE(SUM(r.gpu_seconds_est), 0.0) AS gpu_seconds_est_sum,
                        COALESCE(SUM(r.cost_est), 0.0) AS cost_est_sum
                    FROM tenants t
                    LEFT JOIN requests r
                        ON t.tenant_id = r.tenant_id AND r.ts_start >= ?
                    WHERE t.tenant_id = ?
                    GROUP BY t.tenant_id, t.tenant_name
                    ORDER BY t.tenant_name ASC
                    """,
                    (threshold_ts, tenant_id),
                ).fetchall()
        return [dict(row) for row in rows]

    def list_usage_latencies(self, threshold_ts: str) -> dict[str, list[int]]:
        return self.list_tenant_usage_latencies(threshold_ts)

    def list_tenant_usage_latencies(self, threshold_ts: str, tenant_id: str | None = None) -> dict[str, list[int]]:
        with self._lock:
            if tenant_id is None:
                rows = self._conn.execute(
                    """
                    SELECT tenant_id, latency_ms
                    FROM requests
                    WHERE ts_start >= ?
                    ORDER BY latency_ms ASC
                    """,
                    (threshold_ts,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT tenant_id, latency_ms
                    FROM requests
                    WHERE ts_start >= ? AND tenant_id = ?
                    ORDER BY latency_ms ASC
                    """,
                    (threshold_ts, tenant_id),
                ).fetchall()

        out: dict[str, list[int]] = {}
        for row in rows:
            out.setdefault(row["tenant_id"], []).append(int(row["latency_ms"]))
        return out

    def list_seat_usage_base(self, threshold_ts: str, tenant_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    r.seat_id AS seat_id,
                    s.seat_name AS seat_name,
                    COALESCE(s.role, CASE WHEN r.seat_id IS NULL THEN 'service' ELSE 'user' END) AS role,
                    COUNT(r.id) AS requests,
                    SUM(CASE WHEN r.status_code >= 400 THEN 1 ELSE 0 END) AS errors,
                    COALESCE(SUM(r.prompt_tokens), 0) AS prompt_tokens,
                    COALESCE(SUM(r.completion_tokens), 0) AS completion_tokens,
                    COALESCE(SUM(r.total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(r.gpu_seconds_est), 0.0) AS gpu_seconds_est_sum,
                    COALESCE(SUM(r.cost_est), 0.0) AS cost_est_sum
                FROM requests r
                LEFT JOIN seats s ON s.seat_id = r.seat_id
                WHERE r.ts_start >= ? AND r.tenant_id = ?
                GROUP BY r.seat_id, s.seat_name, s.role
                ORDER BY seat_name ASC
                """,
                (threshold_ts, tenant_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_seat_usage_latencies(self, threshold_ts: str, tenant_id: str) -> dict[str, list[int]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT seat_id, latency_ms
                FROM requests
                WHERE ts_start >= ? AND tenant_id = ?
                ORDER BY latency_ms ASC
                """,
                (threshold_ts, tenant_id),
            ).fetchall()

        out: dict[str, list[int]] = {}
        for row in rows:
            key = row["seat_id"] if row["seat_id"] is not None else "__service__"
            out.setdefault(key, []).append(int(row["latency_ms"]))
        return out

    @staticmethod
    def _budget_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "tenant_id": row["tenant_id"],
            "window": row["window"],
            "budget_usd": float(row["budget_usd"]),
            "warn_50": bool(row["warn_50"]),
            "warn_80": bool(row["warn_80"]),
            "warn_100": bool(row["warn_100"]),
            "updated_at": row["updated_at"],
        }

    def upsert_tenant_budget(
        self,
        tenant_id: str,
        window: str,
        budget_usd: float,
        warn_50: bool = True,
        warn_80: bool = True,
        warn_100: bool = True,
    ) -> dict[str, Any]:
        ts = utc_now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO tenant_budgets (tenant_id, window, budget_usd, warn_50, warn_80, warn_100, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id) DO UPDATE SET
                    window = excluded.window,
                    budget_usd = excluded.budget_usd,
                    warn_50 = excluded.warn_50,
                    warn_80 = excluded.warn_80,
                    warn_100 = excluded.warn_100,
                    updated_at = excluded.updated_at
                """,
                (tenant_id, window, budget_usd, int(warn_50), int(warn_80), int(warn_100), ts),
            )
            self._conn.commit()
        budget = self.get_tenant_budget(tenant_id)
        if budget is None:
            raise RuntimeError("failed to upsert tenant budget")
        return budget

    def get_tenant_budget(self, tenant_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT tenant_id, window, budget_usd, warn_50, warn_80, warn_100, updated_at
                FROM tenant_budgets
                WHERE tenant_id = ?
                """,
                (tenant_id,),
            ).fetchone()
        if row is None:
            return None
        return self._budget_row_to_dict(row)

    def list_tenant_budgets(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if tenant_id is None:
                rows = self._conn.execute(
                    """
                    SELECT tenant_id, window, budget_usd, warn_50, warn_80, warn_100, updated_at
                    FROM tenant_budgets
                    ORDER BY tenant_id ASC
                    """
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT tenant_id, window, budget_usd, warn_50, warn_80, warn_100, updated_at
                    FROM tenant_budgets
                    WHERE tenant_id = ?
                    ORDER BY tenant_id ASC
                    """,
                    (tenant_id,),
                ).fetchall()
        return [self._budget_row_to_dict(row) for row in rows]

    def sum_tenant_cost_since(self, tenant_id: str, threshold_ts: str) -> float:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COALESCE(SUM(cost_est), 0.0) AS total_cost
                FROM requests
                WHERE tenant_id = ? AND ts_start >= ?
                """,
                (tenant_id, threshold_ts),
            ).fetchone()
        if row is None:
            return 0.0
        return float(row["total_cost"] or 0.0)

    def budget_event_exists(self, tenant_id: str, window: str, window_start: str, threshold: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT 1
                FROM budget_events
                WHERE tenant_id = ? AND window = ? AND window_start = ? AND threshold = ?
                LIMIT 1
                """,
                (tenant_id, window, window_start, threshold),
            ).fetchone()
        return row is not None

    def insert_budget_event(
        self,
        event_id: str,
        tenant_id: str,
        window: str,
        window_start: str,
        threshold: str,
        cost_usd: float,
        budget_usd: float,
    ) -> dict[str, Any]:
        ts = utc_now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO budget_events (
                    event_id, tenant_id, window, window_start, threshold, cost_usd, budget_usd, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (event_id, tenant_id, window, window_start, threshold, cost_usd, budget_usd, ts),
            )
            self._conn.commit()
        return {
            "event_id": event_id,
            "tenant_id": tenant_id,
            "window": window,
            "window_start": window_start,
            "threshold": threshold,
            "cost_usd": float(cost_usd),
            "budget_usd": float(budget_usd),
            "created_at": ts,
        }

    def list_budget_events(
        self,
        tenant_id: str,
        window: str | None = None,
        window_start: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            if window is None and window_start is None:
                rows = self._conn.execute(
                    """
                    SELECT event_id, tenant_id, window, window_start, threshold, cost_usd, budget_usd, created_at
                    FROM budget_events
                    WHERE tenant_id = ?
                    ORDER BY created_at ASC
                    """,
                    (tenant_id,),
                ).fetchall()
            elif window is not None and window_start is not None:
                rows = self._conn.execute(
                    """
                    SELECT event_id, tenant_id, window, window_start, threshold, cost_usd, budget_usd, created_at
                    FROM budget_events
                    WHERE tenant_id = ? AND window = ? AND window_start = ?
                    ORDER BY created_at ASC
                    """,
                    (tenant_id, window, window_start),
                ).fetchall()
            elif window is not None:
                rows = self._conn.execute(
                    """
                    SELECT event_id, tenant_id, window, window_start, threshold, cost_usd, budget_usd, created_at
                    FROM budget_events
                    WHERE tenant_id = ? AND window = ?
                    ORDER BY created_at ASC
                    """,
                    (tenant_id, window),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT event_id, tenant_id, window, window_start, threshold, cost_usd, budget_usd, created_at
                    FROM budget_events
                    WHERE tenant_id = ? AND window_start = ?
                    ORDER BY created_at ASC
                    """,
                    (tenant_id, window_start),
                ).fetchall()
        return [dict(row) for row in rows]
