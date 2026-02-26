from __future__ import annotations

import sqlite3
import threading
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
                    request_id TEXT NOT NULL,
                    FOREIGN KEY (tenant_id) REFERENCES tenants (tenant_id)
                );

                CREATE INDEX IF NOT EXISTS idx_requests_tenant_ts ON requests (tenant_id, ts_start);
                CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests (ts_start);
                """
            )
            self._conn.commit()

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
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_tenants(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT tenant_id, tenant_name, max_concurrent, rpm_limit, tpm_limit,
                       max_context_tokens, max_output_tokens, created_at, updated_at
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
                       max_context_tokens, max_output_tokens, created_at, updated_at
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
                       max_context_tokens, max_output_tokens
                FROM tenants
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
                    max_context_tokens, max_output_tokens,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        status_code: int,
        error_type: str | None,
        request_id: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO requests (
                    ts_start, ts_end, latency_ms, tenant_id, model_id,
                    prompt_tokens, completion_tokens, total_tokens,
                    status_code, error_type, request_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )
            self._conn.commit()

    def list_usage_base(self, threshold_ts: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    t.tenant_id AS tenant_id,
                    t.tenant_name AS tenant_name,
                    COUNT(r.id) AS requests,
                    SUM(CASE WHEN r.status_code >= 400 THEN 1 ELSE 0 END) AS errors,
                    COALESCE(SUM(r.total_tokens), 0) AS total_tokens
                FROM tenants t
                LEFT JOIN requests r
                    ON t.tenant_id = r.tenant_id AND r.ts_start >= ?
                GROUP BY t.tenant_id, t.tenant_name
                ORDER BY t.tenant_name ASC
                """,
                (threshold_ts,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_usage_latencies(self, threshold_ts: str) -> dict[str, list[int]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT tenant_id, latency_ms
                FROM requests
                WHERE ts_start >= ?
                ORDER BY latency_ms ASC
                """,
                (threshold_ts,),
            ).fetchall()

        out: dict[str, list[int]] = {}
        for row in rows:
            out.setdefault(row["tenant_id"], []).append(int(row["latency_ms"]))
        return out

