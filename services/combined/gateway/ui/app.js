const STORAGE_KEY = "phase5_control_plane_ui_v1";

const state = {
  connected: false,
  baseUrl: window.location.origin,
  authMode: null,
  adminToken: "",
  adminJwt: "",
  grafanaUrl: "",
  requirements: null,
  tenants: [],
  seats: [],
  keys: [],
  profiles: [],
  selectedProfile: null,
  recentKey: "",
  activeTab: "overview",
};

const TENANT_LIMIT_PRESETS = {
  conservative: {
    max_concurrent: 1,
    rpm_limit: 60,
    tpm_limit: 30000,
    max_context_tokens: 4096,
    max_output_tokens: 256,
  },
  standard: {
    max_concurrent: 2,
    rpm_limit: 180,
    tpm_limit: 120000,
    max_context_tokens: 8192,
    max_output_tokens: 512,
  },
  pro: {
    max_concurrent: 4,
    rpm_limit: 600,
    tpm_limit: 400000,
    max_context_tokens: 16384,
    max_output_tokens: 1024,
  },
};

function byId(id) {
  return document.getElementById(id);
}

function showToast(message, isError = false) {
  const el = byId("toast");
  el.textContent = message;
  el.classList.remove("hidden");
  el.style.background = isError ? "#5f1d1d" : "#102a47";
  window.setTimeout(() => el.classList.add("hidden"), 2500);
}

function normalizeBaseUrl(url) {
  return String(url || "").trim().replace(/\/+$/, "");
}

function saveState() {
  const payload = {
    baseUrl: state.baseUrl,
    authMode: state.authMode,
    adminToken: state.adminToken,
    adminJwt: state.adminJwt,
    grafanaUrl: state.grafanaUrl,
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}

function clearState() {
  localStorage.removeItem(STORAGE_KEY);
  state.connected = false;
  state.authMode = null;
  state.adminToken = "";
  state.adminJwt = "";
  state.grafanaUrl = "";
  state.recentKey = "";
}

function loadState() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    byId("base-url").value = window.location.origin;
    return;
  }
  try {
    const parsed = JSON.parse(raw);
    state.baseUrl = normalizeBaseUrl(parsed.baseUrl || window.location.origin);
    state.authMode = parsed.authMode || null;
    state.adminToken = parsed.adminToken || "";
    state.adminJwt = parsed.adminJwt || "";
    state.grafanaUrl = parsed.grafanaUrl || "";
    byId("base-url").value = state.baseUrl;
    byId("grafana-url").value = state.grafanaUrl;
    byId("admin-token").value = state.adminToken;
    byId("admin-jwt").value = state.adminJwt;
  } catch (_e) {
    byId("base-url").value = window.location.origin;
  }
}

function setAuthModeUI(mode, requirements) {
  state.authMode = mode;
  state.requirements = requirements;

  byId("auth-mode-readout").textContent = mode || "Unknown";

  const reqs = requirements || {};
  byId("auth-mode-requirements").textContent = `token=${reqs.needs_admin_token ? "yes" : "no"} jwt=${reqs.needs_jwt ? "yes" : "no"} jwks=${reqs.jwks_url_set ? "set" : "unset"} issuer=${reqs.issuer_set ? "set" : "unset"} audience=${reqs.audience_set ? "set" : "unset"}`;

  byId("admin-token-row").classList.toggle("hidden", mode !== "static_token");
  byId("admin-jwt-row").classList.toggle("hidden", mode !== "oidc");
}

async function fetchJson(url, options = {}) {
  const res = await fetch(url, options);
  const text = await res.text();
  let payload = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch (_e) {
    payload = text;
  }

  if (!res.ok) {
    const errorMessage = payload && payload.error && payload.error.message ? payload.error.message : `HTTP ${res.status}`;
    const err = new Error(errorMessage);
    err.status = res.status;
    err.payload = payload;
    throw err;
  }

  return payload;
}

async function detectAuthMode(baseUrl) {
  const payload = await fetchJson(`${baseUrl}/admin/auth_info`);
  setAuthModeUI(payload.admin_auth_mode, payload.requirements);
  return payload;
}

function adminHeaders(includeContentType = false) {
  const headers = {};
  if (includeContentType) {
    headers["Content-Type"] = "application/json";
  }

  if (state.authMode === "static_token") {
    headers["X-Admin-Token"] = state.adminToken;
  } else if (state.authMode === "oidc") {
    headers["Authorization"] = `Bearer ${state.adminJwt}`;
  }
  return headers;
}

async function adminRequest(path, { method = "GET", body = null } = {}) {
  const headers = adminHeaders(body !== null);
  return fetchJson(`${state.baseUrl}${path}`, {
    method,
    headers,
    body: body !== null ? JSON.stringify(body) : undefined,
  });
}

async function publicRequest(path) {
  return fetchJson(`${state.baseUrl}${path}`);
}

function setConnectedUI(connected) {
  state.connected = connected;
  byId("connect-screen").classList.toggle("hidden", connected);
  byId("app-shell").classList.toggle("hidden", !connected);
  byId("connection-badge").textContent = connected
    ? `${state.baseUrl} (${state.authMode})`
    : "Disconnected";
}

function setSelectOptions(selectId, options, preferredValue = null) {
  const select = byId(selectId);
  if (!select) return;

  const current = preferredValue ?? select.value;
  select.innerHTML = "";

  options.forEach((opt) => {
    const el = document.createElement("option");
    el.value = String(opt.value);
    el.textContent = opt.label;
    select.appendChild(el);
  });

  if (options.some((o) => String(o.value) === String(current))) {
    select.value = String(current);
  }
}

function renderTable(tableId, columns, rows) {
  const table = byId(tableId);
  if (!table) return;

  const head = `<thead><tr>${columns.map((c) => `<th>${c.label}</th>`).join("")}</tr></thead>`;
  const bodyRows = rows
    .map((row) => {
      const cells = columns
        .map((c) => {
          const value = c.render ? c.render(row) : row[c.key];
          return `<td>${value ?? ""}</td>`;
        })
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");

  table.innerHTML = `${head}<tbody>${bodyRows}</tbody>`;
}

function asCurrency(v) {
  return Number(v || 0).toFixed(4);
}

function asInt(v) {
  return Number(v || 0).toLocaleString();
}

function refreshTenantSelects() {
  const options = state.tenants.map((t) => ({ value: t.tenant_id, label: `${t.tenant_name} (${t.tenant_id.slice(0, 8)})` }));
  const ids = [
    "tenant-edit-id",
    "seats-tenant",
    "keys-tenant",
    "budget-tenant",
    "usage-seat-tenant",
  ];
  ids.forEach((id) => setSelectOptions(id, options));
}

async function loadTenants() {
  const payload = await adminRequest("/admin/tenants");
  state.tenants = payload.data || [];
  refreshTenantSelects();
}

async function loadSeats(tenantId = null) {
  const query = tenantId ? `?tenant_id=${encodeURIComponent(tenantId)}` : "";
  const payload = await adminRequest(`/admin/seats${query}`);
  state.seats = payload.data || [];

  const seatTenant = byId("keys-tenant").value || tenantId;
  const forTenant = state.seats.filter((s) => s.tenant_id === seatTenant);
  const seatOptions = [{ value: "", label: "service (no seat)" }].concat(
    forTenant.map((s) => ({ value: s.seat_id, label: `${s.seat_name || "unnamed"} (${s.role})` })),
  );
  setSelectOptions("keys-seat", seatOptions);
}

async function loadKeys(tenantId = null) {
  const query = tenantId ? `?tenant_id=${encodeURIComponent(tenantId)}` : "";
  const payload = await adminRequest(`/admin/keys${query}`);
  state.keys = payload.data || [];
}

async function renderOverview() {
  const [versionInfo, health, tenants, seats, usage1h, usage24h] = await Promise.all([
    publicRequest("/version"),
    publicRequest("/healthz"),
    adminRequest("/admin/tenants"),
    adminRequest("/admin/seats"),
    adminRequest("/admin/usage/tenants?window=1h"),
    adminRequest("/admin/usage/tenants?window=24h"),
  ]);

  const tokens1h = (usage1h.data || []).reduce((acc, row) => acc + Number(row.total_tokens || 0), 0);
  const cost24h = (usage24h.data || []).reduce((acc, row) => acc + Number(row.cost_est_sum || 0), 0);

  const kpis = [
    { label: "Gateway", value: health.status || "unknown" },
    { label: "Tenants", value: String((tenants.data || []).length) },
    { label: "Seats", value: String((seats.data || []).length) },
    { label: "Tokens (1h)", value: asInt(tokens1h) },
    { label: "Cost (24h USD)", value: asCurrency(cost24h) },
    { label: "UI Enabled", value: String(versionInfo.ui_enabled) },
    { label: "Metrics Enabled", value: String(versionInfo.metrics_enabled) },
    { label: "Auth Mode", value: versionInfo.admin_auth_mode },
  ];

  byId("overview-kpis").innerHTML = kpis
    .map((k) => `<div class="kpi"><div class="label">${k.label}</div><div class="value">${k.value}</div></div>`)
    .join("");

  byId("overview-health").textContent = JSON.stringify({ version: versionInfo, health }, null, 2);
}

function fillTenantEditForm() {
  const tenantId = byId("tenant-edit-id").value;
  const tenant = state.tenants.find((t) => t.tenant_id === tenantId);
  if (!tenant) return;

  byId("tenant-preset").value = "";
  byId("tenant-edit-max-concurrent").value = tenant.max_concurrent;
  byId("tenant-edit-rpm").value = tenant.rpm_limit;
  byId("tenant-edit-tpm").value = tenant.tpm_limit;
  byId("tenant-edit-context").value = tenant.max_context_tokens;
  byId("tenant-edit-output").value = tenant.max_output_tokens;
  byId("tenant-edit-active").value = String(tenant.is_active);
}

function applyTenantPresetToForm() {
  const presetName = byId("tenant-preset").value;
  const preset = TENANT_LIMIT_PRESETS[presetName];
  if (!preset) {
    showToast("Select a preset", true);
    return null;
  }

  byId("tenant-edit-max-concurrent").value = preset.max_concurrent;
  byId("tenant-edit-rpm").value = preset.rpm_limit;
  byId("tenant-edit-tpm").value = preset.tpm_limit;
  byId("tenant-edit-context").value = preset.max_context_tokens;
  byId("tenant-edit-output").value = preset.max_output_tokens;
  showToast(`Preset applied: ${presetName}`);
  return presetName;
}

async function saveTenantEditForm() {
  const tenantId = byId("tenant-edit-id").value;
  if (!tenantId) {
    showToast("Select a tenant", true);
    return;
  }

  const body = {
    max_concurrent: Number(byId("tenant-edit-max-concurrent").value),
    rpm_limit: Number(byId("tenant-edit-rpm").value),
    tpm_limit: Number(byId("tenant-edit-tpm").value),
    max_context_tokens: Number(byId("tenant-edit-context").value),
    max_output_tokens: Number(byId("tenant-edit-output").value),
    is_active: byId("tenant-edit-active").value === "true",
  };

  await adminRequest(`/admin/tenants/${tenantId}`, { method: "PATCH", body });
  await loadTenants();
  fillTenantEditForm();
  await renderTenantSummary();
}

async function renderTenantSummary() {
  const windowValue = byId("tenant-summary-window").value;
  const payload = await adminRequest(`/admin/usage/tenants?window=${encodeURIComponent(windowValue)}`);
  renderTable(
    "tenant-summary-table",
    [
      { label: "Tenant", render: (r) => `${r.tenant_name}<br/><span class='subtle small'>${r.tenant_id}</span>` },
      { label: "Requests", render: (r) => asInt(r.requests) },
      { label: "Errors", render: (r) => asInt(r.errors) },
      { label: "Prompt", render: (r) => asInt(r.prompt_tokens) },
      { label: "Completion", render: (r) => asInt(r.completion_tokens) },
      { label: "Total", render: (r) => asInt(r.total_tokens) },
      { label: "P95(ms)", render: (r) => asInt(r.p95_latency_ms) },
      { label: "GPU Sec", render: (r) => Number(r.gpu_seconds_est_sum || 0).toFixed(2) },
      { label: "Cost USD", render: (r) => asCurrency(r.cost_est_sum) },
    ],
    payload.data || [],
  );
}

async function renderSeats() {
  const tenantId = byId("seats-tenant").value;
  await loadSeats(tenantId);
  const rows = state.seats.filter((s) => !tenantId || s.tenant_id === tenantId);
  renderTable(
    "seats-table",
    [
      { label: "Seat", render: (r) => `${r.seat_name || "unnamed"}<br/><span class='subtle small'>${r.seat_id}</span>` },
      { label: "Role", key: "role" },
      { label: "Active", render: (r) => String(r.is_active) },
      { label: "Created", key: "created_at" },
      {
        label: "Action",
        render: (r) => `<button class='button button-ghost seat-toggle' data-seat-id='${r.seat_id}' data-next='${!r.is_active}'>${r.is_active ? "Deactivate" : "Activate"}</button>`,
      },
    ],
    rows,
  );
}

async function renderKeys() {
  const tenantId = byId("keys-tenant").value;
  await loadKeys(tenantId);
  renderTable(
    "keys-table",
    [
      { label: "Key ID", key: "key_id" },
      { label: "Seat", render: (r) => r.seat_name || r.seat_id || "service" },
      { label: "Role", render: (r) => r.role || "service" },
      { label: "Created", key: "created_at" },
      { label: "Revoked", render: (r) => r.revoked_at || "" },
      {
        label: "Action",
        render: (r) =>
          r.revoked_at
            ? ""
            : `<button class='button button-danger key-revoke' data-key-id='${r.key_id}'>Revoke</button>`,
      },
    ],
    state.keys,
  );
}

async function renderBudgets() {
  const tenantId = byId("budget-tenant").value;
  const windowValue = byId("budget-window").value;
  if (!tenantId) {
    byId("budget-status").textContent = "Select a tenant.";
    return;
  }

  try {
    const [status, events] = await Promise.all([
      adminRequest(`/admin/budget_status?tenant_id=${encodeURIComponent(tenantId)}&window=${encodeURIComponent(windowValue)}`),
      adminRequest(`/admin/budget_events?tenant_id=${encodeURIComponent(tenantId)}`),
    ]);
    byId("budget-status").textContent = JSON.stringify(status, null, 2);

    renderTable(
      "budget-events-table",
      [
        { label: "Threshold", key: "threshold" },
        { label: "Window", key: "window" },
        { label: "Window Start", key: "window_start" },
        { label: "Cost USD", render: (r) => asCurrency(r.cost_usd) },
        { label: "Budget USD", render: (r) => asCurrency(r.budget_usd) },
        { label: "Created", key: "created_at" },
      ],
      events.data || [],
    );
  } catch (err) {
    byId("budget-status").textContent = `No budget set or unavailable: ${err.message}`;
    renderTable("budget-events-table", [{ label: "Message", key: "message" }], []);
  }
}

async function renderUsage() {
  const windowValue = byId("usage-window").value;
  const tenantUsage = await adminRequest(`/admin/usage/tenants?window=${encodeURIComponent(windowValue)}`);
  renderTable(
    "usage-tenants-table",
    [
      { label: "Tenant", render: (r) => `${r.tenant_name}<br/><span class='subtle small'>${r.tenant_id}</span>` },
      { label: "Requests", render: (r) => asInt(r.requests) },
      { label: "Errors", render: (r) => asInt(r.errors) },
      { label: "Prompt", render: (r) => asInt(r.prompt_tokens) },
      { label: "Completion", render: (r) => asInt(r.completion_tokens) },
      { label: "Total", render: (r) => asInt(r.total_tokens) },
      { label: "P95(ms)", render: (r) => asInt(r.p95_latency_ms) },
      { label: "GPU Sec", render: (r) => Number(r.gpu_seconds_est_sum || 0).toFixed(2) },
      { label: "Cost USD", render: (r) => asCurrency(r.cost_est_sum) },
    ],
    tenantUsage.data || [],
  );

  await renderSeatUsage();
}

async function renderSeatUsage() {
  const tenantId = byId("usage-seat-tenant").value;
  const windowValue = byId("usage-window").value;
  if (!tenantId) {
    renderTable("usage-seats-table", [{ label: "Message", key: "message" }], []);
    return;
  }

  const seatUsage = await adminRequest(
    `/admin/usage/seats?tenant_id=${encodeURIComponent(tenantId)}&window=${encodeURIComponent(windowValue)}`,
  );

  renderTable(
    "usage-seats-table",
    [
      { label: "Seat", render: (r) => r.seat_name || r.seat_id || "service" },
      { label: "Role", key: "role" },
      { label: "Requests", render: (r) => asInt(r.requests) },
      { label: "Errors", render: (r) => asInt(r.errors) },
      { label: "Prompt", render: (r) => asInt(r.prompt_tokens) },
      { label: "Completion", render: (r) => asInt(r.completion_tokens) },
      { label: "Total", render: (r) => asInt(r.total_tokens) },
      { label: "P95(ms)", render: (r) => asInt(r.p95_latency_ms) },
      { label: "GPU Sec", render: (r) => Number(r.gpu_seconds_est_sum || 0).toFixed(2) },
      { label: "Cost USD", render: (r) => asCurrency(r.cost_est_sum) },
    ],
    seatUsage.data || [],
  );
}

async function renderProfiles() {
  const payload = await adminRequest("/admin/profiles");
  state.profiles = payload.data || [];
  const container = byId("profiles-list");

  if (!state.profiles.length) {
    container.innerHTML = "<p class='subtle'>No profiles found.</p>";
    byId("profile-details").textContent = "";
    return;
  }

  if (!state.selectedProfile) {
    state.selectedProfile = state.profiles[0].name;
  }

  container.innerHTML = state.profiles
    .map(
      (p) =>
        `<div class='profile-item'><button class='button ${
          p.name === state.selectedProfile ? "" : "button-ghost"
        } profile-select' data-profile='${p.name}'>${p.name}</button></div>`,
    )
    .join("");

  await loadProfileDetail(state.selectedProfile);
}

async function loadProfileDetail(name) {
  const profile = await adminRequest(`/admin/profiles/${encodeURIComponent(name)}`);
  state.selectedProfile = profile.name;
  byId("profile-details").textContent = JSON.stringify(profile, null, 2);
}

async function refreshTab(tab) {
  if (!state.connected) return;

  try {
    if (tab === "overview") {
      await renderOverview();
      return;
    }

    if (tab === "tenants") {
      await loadTenants();
      fillTenantEditForm();
      await renderTenantSummary();
      return;
    }

    if (tab === "seats") {
      await loadTenants();
      await renderSeats();
      return;
    }

    if (tab === "keys") {
      await loadTenants();
      await loadSeats(byId("keys-tenant").value || null);
      await renderKeys();
      return;
    }

    if (tab === "budgets") {
      await loadTenants();
      await renderBudgets();
      return;
    }

    if (tab === "usage") {
      await loadTenants();
      await renderUsage();
      return;
    }

    if (tab === "profiles") {
      await renderProfiles();
    }
  } catch (err) {
    showToast(err.message || "Failed to refresh", true);
  }
}

function activateTab(tabName) {
  state.activeTab = tabName;
  document.querySelectorAll(".tab").forEach((el) => el.classList.toggle("active", el.dataset.tab === tabName));
  document
    .querySelectorAll(".tab-panel")
    .forEach((el) => el.classList.toggle("hidden", el.id !== `tab-${tabName}`));

  refreshTab(tabName);
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    showToast("Copied");
  } catch (_e) {
    showToast("Copy failed", true);
  }
}

function openKeyModal(apiKey) {
  state.recentKey = apiKey;
  byId("key-modal-value").value = apiKey;
  byId("key-modal").showModal();
}

async function connect() {
  const baseUrl = normalizeBaseUrl(byId("base-url").value);
  const grafanaUrl = normalizeBaseUrl(byId("grafana-url").value);
  byId("connect-error").textContent = "";

  try {
    const authInfo = await detectAuthMode(baseUrl);
    const mode = authInfo.admin_auth_mode;

    const token = byId("admin-token").value.trim();
    const jwt = byId("admin-jwt").value.trim();

    if (mode === "static_token" && !token) {
      throw new Error("Admin token is required in static_token mode");
    }
    if (mode === "oidc" && !jwt) {
      throw new Error("Admin JWT is required in oidc mode");
    }

    state.baseUrl = baseUrl;
    state.grafanaUrl = grafanaUrl;
    state.adminToken = token;
    state.adminJwt = jwt;

    await publicRequest("/version");
    await adminRequest("/admin/tenants");

    saveState();
    setConnectedUI(true);
    showToast("Connected");
    await refreshTab(state.activeTab);
  } catch (err) {
    byId("connect-error").textContent = err.message || "Connection failed";
  }
}

function wireEvents() {
  byId("detect-auth").addEventListener("click", async () => {
    try {
      await detectAuthMode(normalizeBaseUrl(byId("base-url").value));
      showToast("Auth mode detected");
    } catch (err) {
      showToast(err.message || "Unable to detect auth", true);
    }
  });

  byId("connect-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    await connect();
  });

  byId("disconnect").addEventListener("click", () => {
    clearState();
    setConnectedUI(false);
    showToast("Disconnected");
  });

  byId("open-grafana").addEventListener("click", () => {
    if (!state.grafanaUrl) {
      showToast("Grafana URL not configured", true);
      return;
    }
    window.open(state.grafanaUrl, "_blank", "noopener,noreferrer");
  });

  byId("tabs").addEventListener("click", (e) => {
    const target = e.target.closest("button[data-tab]");
    if (!target) return;
    activateTab(target.dataset.tab);
  });

  byId("tenant-edit-id").addEventListener("change", fillTenantEditForm);

  byId("tenant-create-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const body = { tenant_name: byId("tenant-create-name").value.trim() };

    const maybeInts = [
      ["max_concurrent", "tenant-create-max-concurrent"],
      ["rpm_limit", "tenant-create-rpm"],
      ["tpm_limit", "tenant-create-tpm"],
      ["max_context_tokens", "tenant-create-context"],
      ["max_output_tokens", "tenant-create-output"],
    ];
    maybeInts.forEach(([key, id]) => {
      const value = byId(id).value;
      if (value) body[key] = Number(value);
    });

    try {
      const payload = await adminRequest("/admin/tenants", { method: "POST", body });
      showToast(`Tenant created: ${payload.tenant_name}`);
      byId("tenant-create-form").reset();
      await loadTenants();
      fillTenantEditForm();
      await renderTenantSummary();
    } catch (err) {
      showToast(err.message, true);
    }
  });

  byId("tenant-edit-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      await saveTenantEditForm();
      showToast("Tenant updated");
    } catch (err) {
      showToast(err.message, true);
    }
  });

  byId("tenant-preset-apply").addEventListener("click", () => {
    applyTenantPresetToForm();
  });

  byId("tenant-preset-save").addEventListener("click", async () => {
    try {
      const applied = applyTenantPresetToForm();
      if (!applied) return;
      await saveTenantEditForm();
      showToast(`Tenant updated with preset: ${applied}`);
    } catch (err) {
      showToast(err.message, true);
    }
  });

  byId("tenant-summary-refresh").addEventListener("click", () => renderTenantSummary());

  byId("seats-tenant").addEventListener("change", () => renderSeats());
  byId("seat-create-btn").addEventListener("click", async () => {
    const tenantId = byId("seats-tenant").value;
    if (!tenantId) {
      showToast("Select a tenant", true);
      return;
    }

    const body = {
      tenant_id: tenantId,
      seat_name: byId("seat-create-name").value || null,
      role: byId("seat-create-role").value,
    };

    try {
      await adminRequest("/admin/seats", { method: "POST", body });
      showToast("Seat created");
      byId("seat-create-name").value = "";
      await renderSeats();
    } catch (err) {
      showToast(err.message, true);
    }
  });

  byId("seats-table").addEventListener("click", async (e) => {
    const btn = e.target.closest(".seat-toggle");
    if (!btn) return;
    try {
      await adminRequest(`/admin/seats/${btn.dataset.seatId}`, {
        method: "PATCH",
        body: { is_active: btn.dataset.next === "true" },
      });
      showToast("Seat updated");
      await renderSeats();
    } catch (err) {
      showToast(err.message, true);
    }
  });

  byId("keys-tenant").addEventListener("change", async () => {
    await loadSeats(byId("keys-tenant").value || null);
    await renderKeys();
  });

  byId("key-create-btn").addEventListener("click", async () => {
    const tenantId = byId("keys-tenant").value;
    if (!tenantId) {
      showToast("Select a tenant", true);
      return;
    }

    const seatId = byId("keys-seat").value;
    const body = { tenant_id: tenantId, seat_id: seatId || null };

    try {
      const payload = await adminRequest("/admin/keys", { method: "POST", body });
      showToast("Key created");
      openKeyModal(payload.api_key);
      await renderKeys();
    } catch (err) {
      showToast(err.message, true);
    }
  });

  byId("keys-table").addEventListener("click", async (e) => {
    const btn = e.target.closest(".key-revoke");
    if (!btn) return;
    try {
      await adminRequest(`/admin/keys/${btn.dataset.keyId}/revoke`, { method: "POST" });
      showToast("Key revoked");
      await renderKeys();
    } catch (err) {
      showToast(err.message, true);
    }
  });

  byId("budget-save").addEventListener("click", async () => {
    const tenantId = byId("budget-tenant").value;
    const windowValue = byId("budget-window").value;
    const budgetUsd = Number(byId("budget-usd").value);

    if (!tenantId || !budgetUsd) {
      showToast("Tenant and budget are required", true);
      return;
    }

    try {
      await adminRequest(`/admin/budgets/${tenantId}`, {
        method: "PUT",
        body: { window: windowValue, budget_usd: budgetUsd },
      });
      showToast("Budget saved");
      await renderBudgets();
    } catch (err) {
      showToast(err.message, true);
    }
  });

  byId("budget-refresh").addEventListener("click", () => renderBudgets());
  byId("budget-tenant").addEventListener("change", () => renderBudgets());
  byId("budget-window").addEventListener("change", () => renderBudgets());

  byId("usage-refresh").addEventListener("click", () => renderUsage());
  byId("usage-seat-refresh").addEventListener("click", () => renderSeatUsage());
  byId("usage-seat-tenant").addEventListener("change", () => renderSeatUsage());

  byId("quick-run").addEventListener("click", async () => {
    const apiKey = byId("quick-key").value.trim();
    const prompt = byId("quick-prompt").value;
    const maxTokens = Number(byId("quick-max-tokens").value) || 64;
    if (!apiKey) {
      showToast("API key is required", true);
      return;
    }

    try {
      const payload = await fetchJson(`${state.baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: "ignored-by-gateway",
          messages: [{ role: "user", content: prompt }],
          max_tokens: maxTokens,
        }),
      });
      byId("quick-result").textContent = JSON.stringify(payload, null, 2);
    } catch (err) {
      byId("quick-result").textContent = `Error: ${err.message}\n${JSON.stringify(err.payload || {}, null, 2)}`;
    }
  });

  byId("quick-use-recent").addEventListener("click", () => {
    if (!state.recentKey) {
      showToast("No recent key in this session", true);
      return;
    }
    byId("quick-key").value = state.recentKey;
    showToast("Recent key loaded");
  });

  byId("profiles-list").addEventListener("click", async (e) => {
    const btn = e.target.closest(".profile-select");
    if (!btn) return;
    try {
      await loadProfileDetail(btn.dataset.profile);
      await renderProfiles();
    } catch (err) {
      showToast(err.message, true);
    }
  });

  byId("profile-generate").addEventListener("click", async () => {
    if (!state.selectedProfile) {
      showToast("Select a profile", true);
      return;
    }
    try {
      const payload = await adminRequest(`/admin/profiles/${encodeURIComponent(state.selectedProfile)}/generate_runpod`, {
        method: "POST",
      });
      byId("profile-runpod").value = payload.runpod_env_snippet || "";
      byId("profile-bootstrap").value = payload.bootstrap_snippet || "";
      showToast("Snippets generated");
    } catch (err) {
      showToast(err.message, true);
    }
  });

  byId("copy-runpod").addEventListener("click", () => copyText(byId("profile-runpod").value));
  byId("copy-bootstrap").addEventListener("click", () => copyText(byId("profile-bootstrap").value));

  byId("key-modal-copy").addEventListener("click", () => copyText(byId("key-modal-value").value));
  byId("key-modal-use").addEventListener("click", () => {
    byId("quick-key").value = byId("key-modal-value").value;
    byId("key-modal").close();
    activateTab("quick-test");
  });
  byId("key-modal-close").addEventListener("click", () => byId("key-modal").close());
}

async function tryAutoConnect() {
  if (!state.authMode) return;
  const token = state.authMode === "static_token" ? state.adminToken : state.adminJwt;
  if (!token) return;

  try {
    await detectAuthMode(state.baseUrl);
    await adminRequest("/admin/tenants");
    setConnectedUI(true);
    await refreshTab(state.activeTab);
    showToast("Session restored");
  } catch (_err) {
    setConnectedUI(false);
  }
}

async function init() {
  loadState();
  wireEvents();

  if (state.authMode) {
    setAuthModeUI(state.authMode, state.requirements);
  }

  setConnectedUI(false);
  await tryAutoConnect();
}

window.addEventListener("DOMContentLoaded", init);
