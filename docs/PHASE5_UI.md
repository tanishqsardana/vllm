# Phase 5 UI Guide

The gateway now serves a demo-ready control plane at `/ui`.

## Connect Flow

1. Open `http://<gateway-host>:8000/ui`.
2. Set **Target Base URL** (defaults to current origin).
3. Click **Detect** to read `/admin/auth_info`.
4. Enter credentials:
   - `static_token` mode: provide `X-Admin-Token` value.
   - `oidc` mode: provide admin JWT.
5. Optionally set Grafana URL.
6. Click **Connect**.

The UI stores only local browser state (`base_url`, token/JWT, grafana URL) in `localStorage`.
No admin secrets are persisted server-side.

## Tabs

- **Overview**: `/version`, `/healthz`, quick counts, token/cost summaries.
- **Tenants**: create + patch limits + usage summary windowed by `1h|24h|7d`.
- **Seats**: create seats and activate/deactivate seats.
- **Keys**: create/revoke keys; key secret shown once in modal.
- **Budgets**: set budget (`day|week|month`) + budget status + recent events.
- **Usage**: per-tenant and per-seat rollups with p95 latency, tokens, cost estimates.
- **Quick Test**: sends `POST /v1/chat/completions` with provided tenant key.
- **Profiles**: list/view blueprint profiles and generate Runpod/bootstrap snippets.

## API Notes

- Admin requests use either:
  - `X-Admin-Token` (static mode), or
  - `Authorization: Bearer <JWT>` (OIDC mode).
- Auth mode detection: `GET /admin/auth_info`.
- Audit trail endpoint: `GET /admin/audit?window=24h&tenant_id=<optional>`.
