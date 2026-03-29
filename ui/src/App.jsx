import { useState, useEffect, useCallback, useRef } from "react";

/* ═══════════════════ THEME ═══════════════════ */
const T = {
  bg: "#f8fafc", card: "#ffffff", border: "#e2e8f0", borderStrong: "#cbd5e1",
  text: "#0f172a", textSec: "#475569", textTert: "#94a3b8",
  accent: "#0d9488", accentHover: "#0f766e", accentLight: "#f0fdfa",
  accent2: "#6366f1", accent2Light: "#eef2ff",
  success: "#16a34a", successBg: "#f0fdf4", successBorder: "#bbf7d0",
  warning: "#d97706", warningBg: "#fffbeb", warningBorder: "#fde68a",
  danger: "#dc2626", dangerBg: "#fef2f2", dangerBorder: "#fecaca",
  info: "#0284c7", infoBg: "#f0f9ff", infoBorder: "#bae6fd",
  blueBg: "#f0f9ff",
};

/* ═══════════════════ STORAGE ═══════════════════ */
const STORAGE_KEY = "mirae_cp_v3";
function loadSaved() { try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); } catch { return {}; } }
function saveCreds(obj) { localStorage.setItem(STORAGE_KEY, JSON.stringify(obj)); }

/* ═══════════════════ API ═══════════════════ */
async function apiFetch(baseUrl, path, { token, method = "GET", body } = {}) {
  const headers = {};
  if (token) headers["X-Admin-Token"] = token;
  if (body) headers["Content-Type"] = "application/json";
  const res = await fetch(`${baseUrl}${path}`, { method, headers, body: body ? JSON.stringify(body) : undefined });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      const raw = j.detail || j.error || j.message || msg;
      msg = typeof raw === "string" ? raw : JSON.stringify(raw);
    } catch {}
    throw new Error(msg);
  }
  if (res.status === 204) return {};
  return res.json();
}

/* ═══════════════════ HELPERS ═══════════════════ */
function normalizeTenants(raw) {
  let arr;
  if (Array.isArray(raw)) arr = raw;
  else if (raw?.tenants) arr = raw.tenants;
  else if (raw?.items) arr = raw.items;
  else if (raw?.data) arr = raw.data;
  else arr = [];
  return arr.map(t => ({
    ...t,
    id: t.id || t.tenant_id,
    name: t.name || t.tenant_name || t.display_name || t.id || t.tenant_id || "—",
  }));
}
function normalizeKeys(raw) {
  let arr;
  if (Array.isArray(raw)) arr = raw;
  else if (raw?.keys) arr = raw.keys;
  else if (raw?.items) arr = raw.items;
  else if (raw?.data) arr = raw.data;
  else arr = [];
  return arr.map(k => ({ ...k, id: k.id || k.key_id }));
}

/* ═══════════════════ TABS ═══════════════════ */
const TABS = [
  { id: "overview",  label: "Overview",       icon: "M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" },
  { id: "live",      label: "Live",           icon: "M13 10V3L4 14h7v7l9-11h-7z" },
  { id: "dynamo",    label: "Infrastructure", icon: "M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" },
  { id: "tenants",   label: "Tenants",        icon: "M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-2 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" },
  { id: "keys",      label: "Keys & Seats",   icon: "M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" },
  { id: "policy",    label: "Policies",       icon: "M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" },
  { id: "audit",     label: "Audit",          icon: "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" },
  { id: "usage",     label: "Usage",          icon: "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" },
  { id: "test",      label: "Test",           icon: "M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" },
];

/* ═══════════════════ PRIMITIVE COMPONENTS ═══════════════════ */
function SVG({ d, size = 16, color }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color || "currentColor"} strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round"><path d={d} /></svg>;
}
function Spinner({ size = 14 }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} style={{ animation: "spin 0.8s linear infinite" }}><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" /></svg>;
}
function Badge({ children, variant = "default" }) {
  const styles = {
    default: { bg: T.bg, border: T.border, color: T.textSec },
    info:    { bg: T.infoBg, border: T.infoBorder, color: T.info },
    success: { bg: T.successBg, border: T.successBorder, color: T.success },
    warning: { bg: T.warningBg, border: T.warningBorder, color: T.warning },
    danger:  { bg: T.dangerBg, border: T.dangerBorder, color: T.danger },
    accent:  { bg: T.accentLight, border: "#99f6e4", color: T.accent },
    purple:  { bg: T.accent2Light, border: "#c7d2fe", color: T.accent2 },
  };
  const s = styles[variant] || styles.default;
  return <span style={{ display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11, fontWeight: 600, padding: "2px 7px", borderRadius: 5, background: s.bg, border: `1px solid ${s.border}`, color: s.color }}>{children}</span>;
}
function Btn({ children, onClick, variant = "primary", disabled, small, style: extra = {} }) {
  const base = { display: "inline-flex", alignItems: "center", gap: 6, border: "none", borderRadius: 8, cursor: disabled ? "not-allowed" : "pointer", fontWeight: 600, fontFamily: "inherit", opacity: disabled ? 0.55 : 1, transition: "all 0.15s", ...(small ? { fontSize: 12, padding: "5px 10px" } : { fontSize: 13, padding: "8px 14px" }), ...extra };
  const variants = {
    primary: { background: T.accent, color: "#fff" },
    secondary: { background: T.bg, color: T.textSec, border: `1px solid ${T.border}` },
    danger: { background: T.dangerBg, color: T.danger, border: `1px solid ${T.dangerBorder}` },
    ghost: { background: "none", color: T.textSec },
  };
  return <button style={{ ...base, ...variants[variant] }} onClick={onClick} disabled={disabled}>{children}</button>;
}
function Input({ value, onChange, placeholder, type = "text", style: extra = {} }) {
  return <input type={type} value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder} style={{ width: "100%", padding: "8px 11px", fontSize: 13, border: `1px solid ${T.border}`, borderRadius: 8, outline: "none", background: T.card, color: T.text, fontFamily: "inherit", boxSizing: "border-box", ...extra }} />;
}
function Select({ value, onChange, children, style: extra = {} }) {
  return <select value={value} onChange={e => onChange(e.target.value)} style={{ padding: "8px 11px", fontSize: 13, border: `1px solid ${T.border}`, borderRadius: 8, background: T.card, color: T.text, fontFamily: "inherit", cursor: "pointer", ...extra }}>{children}</select>;
}
function Card({ children, style: extra = {} }) {
  return <div style={{ background: T.card, border: `1px solid ${T.border}`, borderRadius: 12, padding: "18px 20px", ...extra }}>{children}</div>;
}
function SectionLabel({ children }) {
  return <div style={{ fontSize: 11, fontWeight: 700, color: T.textTert, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>{children}</div>;
}
function Anim({ children }) { return <div style={{ animation: "fadeIn 0.25s ease-out" }}>{children}</div>; }
function Toast({ message, type, onDone }) {
  useEffect(() => { const t = setTimeout(onDone, 3500); return () => clearTimeout(t); }, []);
  return <div style={{ position: "fixed", bottom: 24, right: 24, zIndex: 999, background: type === "error" ? T.dangerBg : T.successBg, border: `1px solid ${type === "error" ? T.dangerBorder : T.successBorder}`, borderRadius: 10, padding: "10px 16px", fontSize: 13, color: type === "error" ? T.danger : T.success, boxShadow: "0 8px 24px rgba(0,0,0,0.1)", animation: "slideUp 0.2s ease-out", maxWidth: 480, wordBreak: "break-all" }}>{message}</div>;
}
function fmtTime(ts) { try { return new Date(ts).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" }); } catch { return ts; } }

/* ═══════════════════ MODAL ═══════════════════ */
function Modal({ title, onClose, children, width = 480 }) {
  useEffect(() => {
    const handler = (e) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onClose]);
  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(15,23,42,0.45)", zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center", padding: 20 }} onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div style={{ background: T.card, borderRadius: 14, width: "100%", maxWidth: width, boxShadow: "0 20px 60px rgba(0,0,0,0.18)", animation: "fadeIn 0.2s ease-out" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "16px 20px", borderBottom: `1px solid ${T.border}` }}>
          <span style={{ fontWeight: 700, fontSize: 15, color: T.text }}>{title}</span>
          <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: T.textTert, display: "flex", padding: 2 }}>
            <SVG d="M6 18L18 6M6 6l12 12" size={18} />
          </button>
        </div>
        <div style={{ padding: "20px" }}>{children}</div>
      </div>
    </div>
  );
}

/* ═══════════════════ API KEY COPY MODAL ═══════════════════ */
function ApiKeyCopyModal({ apiKey, onClose }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(apiKey).then(() => { setCopied(true); setTimeout(() => setCopied(false), 2500); });
  };
  return (
    <Modal title="API Key Created" onClose={onClose} width={520}>
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <div style={{ background: T.warningBg, border: `1px solid ${T.warningBorder}`, borderRadius: 8, padding: "10px 14px", display: "flex", gap: 10, alignItems: "flex-start" }}>
          <SVG d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" size={16} color={T.warning} />
          <span style={{ fontSize: 12, color: T.warning, fontWeight: 500 }}>This key is shown only once. Copy it now — you won't be able to retrieve it again.</span>
        </div>
        <div>
          <SectionLabel>Your API Key</SectionLabel>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <div style={{ flex: 1, background: T.bg, border: `1px solid ${T.border}`, borderRadius: 8, padding: "10px 12px", fontFamily: "'JetBrains Mono', 'Fira Code', monospace", fontSize: 12, color: T.text, wordBreak: "break-all", userSelect: "all" }}>
              {apiKey}
            </div>
            <Btn variant={copied ? "secondary" : "primary"} onClick={copy} style={{ whiteSpace: "nowrap", flexShrink: 0 }}>
              {copied ? <><SVG d="M5 13l4 4L19 7" size={13} /> Copied!</> : <><SVG d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" size={13} /> Copy Key</>}
            </Btn>
          </div>
        </div>
        <div style={{ display: "flex", justifyContent: "flex-end", paddingTop: 4 }}>
          <Btn variant="secondary" onClick={onClose}>I've saved this key — close</Btn>
        </div>
      </div>
    </Modal>
  );
}

/* ═══════════════════ CONNECT ═══════════════════ */
function ConnectScreen({ onConnect }) {
  const saved = loadSaved();
  const [url, setUrl] = useState(saved.baseUrl || ""); const [token, setToken] = useState(saved.token || ""); const [error, setError] = useState(""); const [loading, setLoading] = useState(false);
  const go = async (e) => { e.preventDefault(); setError(""); setLoading(true); const base = url.replace(/\/+$/, ""); try { await apiFetch(base, "/version"); await apiFetch(base, "/admin/tenants", { token }); saveCreds({ baseUrl: base, token }); onConnect(base, token); } catch (e) { setError(e.message); } finally { setLoading(false); } };
  return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", padding: 24, background: `linear-gradient(155deg, ${T.accentLight} 0%, ${T.bg} 35%, ${T.blueBg} 100%)` }}>
      <div style={{ width: "100%", maxWidth: 400, animation: "fadeIn 0.4s ease-out" }}>
        <div style={{ textAlign: "center", marginBottom: 28 }}>
          <div style={{ display: "inline-flex", alignItems: "center", gap: 7, background: T.card, border: `1px solid ${T.border}`, borderRadius: 10, padding: "7px 16px", marginBottom: 14, boxShadow: "0 1px 3px rgba(0,0,0,0.04)" }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: T.accent }} /><span style={{ fontSize: 15, fontWeight: 700, color: T.text }}>mirae</span>
          </div>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: T.text, margin: "0 0 4px" }}>Control Plane</h1>
          <p style={{ fontSize: 13, color: T.textTert, margin: 0 }}>Production-ready inference. Enterprise-grade control.</p>
        </div>
        <Card>
          <form onSubmit={go} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div><SectionLabel>Gateway URL</SectionLabel><Input value={url} onChange={setUrl} placeholder="https://your-pod.proxy.runpod.net" /></div>
            <div><SectionLabel>Admin Token</SectionLabel><Input value={token} onChange={setToken} type="password" placeholder="sk-admin-…" /></div>
            {error && <div style={{ background: T.dangerBg, border: `1px solid ${T.dangerBorder}`, borderRadius: 8, padding: "8px 12px", fontSize: 12, color: T.danger }}>{error}</div>}
            <Btn onClick={go} disabled={loading} style={{ width: "100%", justifyContent: "center", padding: "10px" }}>{loading ? <><Spinner /> Connecting…</> : "Connect"}</Btn>
          </form>
        </Card>
        <style>{`@keyframes spin{to{transform:rotate(360deg)}} @keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}`}</style>
      </div>
    </div>
  );
}

/* ═══════════════════ OVERVIEW TAB [FIX 8: version fields] ═══════════════════ */
function OverviewTab({ baseUrl, token }) {
  const [data, setData] = useState(null); const [loading, setLoading] = useState(true);
  useEffect(() => {
    Promise.all([
      apiFetch(baseUrl, "/admin/tenants", { token }).catch(() => []),
      apiFetch(baseUrl, "/version").catch(() => ({})),
    ]).then(([tenantsRaw, version]) => {
      const tenants = normalizeTenants(tenantsRaw);
      const usage = {
        total_requests: tenants.reduce((s, t) => s + (t.total_requests || 0), 0),
        total_tokens: tenants.reduce((s, t) => s + (t.total_tokens || 0), 0),
      };
      setData({ tenants, usage, version });
      setLoading(false);
    });
  }, []);
  if (loading) return <div style={{ padding: 40, textAlign: "center", color: T.textTert }}><Spinner size={20} /></div>;
  const tenants = data?.tenants || []; const usage = data?.usage || {}; const version = data?.version || {};
  const kpis = [
    { label: "Active Tenants", value: tenants.length, icon: "M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-2 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4", accent: T.accent },
    { label: "Total API Keys", value: tenants.reduce((s, t) => s + (t.api_key_count || 0), 0), icon: "M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z", accent: T.accent2 },
    { label: "Total Requests", value: (usage.total_requests || 0).toLocaleString(), icon: "M13 10V3L4 14h7v7l9-11h-7z", accent: T.info },
    { label: "Total Tokens", value: (usage.total_tokens || 0).toLocaleString(), icon: "M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z", accent: T.success },
  ];
  return (
    <Anim>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 14, marginBottom: 20 }}>
        {kpis.map(k => (
          <Card key={k.label}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div><div style={{ fontSize: 11, fontWeight: 600, color: T.textTert, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 6 }}>{k.label}</div><div style={{ fontSize: 26, fontWeight: 700, color: T.text }}>{k.value}</div></div>
              <div style={{ background: k.accent + "18", borderRadius: 8, padding: 8 }}><SVG d={k.icon} size={18} color={k.accent} /></div>
            </div>
          </Card>
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <Card>
          <SectionLabel>Recent Tenants</SectionLabel>
          {tenants.length === 0 ? <div style={{ fontSize: 13, color: T.textTert, padding: "12px 0" }}>No tenants yet</div> : (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {tenants.slice(0, 6).map(t => (
                <div key={t.id} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 0", borderBottom: `1px solid ${T.border}` }}>
                  <div><div style={{ fontSize: 13, fontWeight: 600, color: T.text }}>{t.name}</div><div style={{ fontSize: 11, color: T.textTert }}>{t.id}</div></div>
                  <Badge variant="success">Active</Badge>
                </div>
              ))}
            </div>
          )}
        </Card>
        <Card>
          <SectionLabel>System Info</SectionLabel>
          {/* FIX 8: Use correct field names from backend GatewayVersion */}
          {[
            ["Build SHA", version.build_sha || "—"],
            ["Model", version.model_id || "—"],
            ["Admin Auth", version.admin_auth_mode || "—"],
            ["Max Concurrent", version.global_max_concurrent ?? "—"],
            ["GPU Rate", version.gpu_hourly_rate ? `$${version.gpu_hourly_rate}/hr` : "—"],
          ].map(([k, v]) => (
            <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "7px 0", borderBottom: `1px solid ${T.border}`, fontSize: 13 }}>
              <span style={{ color: T.textSec }}>{k}</span><span style={{ color: T.text, fontWeight: 600 }}>{v}</span>
            </div>
          ))}
        </Card>
      </div>
    </Anim>
  );
}

/* ═══════════════════ LIVE TAB [FIX 4: nested response parsing] ═══════════════════ */
function LiveTab({ baseUrl, token }) {
  const [metrics, setMetrics] = useState(null); const [history, setHistory] = useState([]); const histRef = useRef([]);
  const [error, setError] = useState(null);
  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const m = await apiFetch(baseUrl, "/admin/dynamo/metrics", { token });
        if (!active) return;
        setMetrics(m); setError(null);
        /* FIX 4: Backend returns { frontend: { inflight_requests, queued_requests, ... }, worker: { requests_running, requests_waiting, ... } } */
        const fe = m.frontend || {};
        const wk = m.worker || {};
        const point = {
          t: Date.now(),
          inflight: (fe.inflight_requests ?? 0) + (wk.requests_running ?? 0),
          queued: (fe.queued_requests ?? 0) + (wk.requests_waiting ?? 0),
          tps: fe.total_requests ?? 0,
        };
        histRef.current = [...histRef.current.slice(-59), point];
        setHistory([...histRef.current]);
      } catch (e) { if (active) setError(e.message); }
    };
    poll(); const id = setInterval(poll, 3000);
    return () => { active = false; clearInterval(id); };
  }, [baseUrl, token]);

  /* FIX 4: Read from nested frontend/worker sub-objects */
  const getMetricVal = (m, key) => {
    const fe = m.frontend || {};
    const wk = m.worker || {};
    switch (key) {
      case "inflight": return (fe.inflight_requests ?? 0) + (wk.requests_running ?? 0);
      case "queued": return (fe.queued_requests ?? 0) + (wk.requests_waiting ?? 0);
      case "tps": return fe.total_requests ?? 0;
      default: return 0;
    }
  };

  const Sparkline = ({ data, key_, color }) => {
    if (data.length < 2) return <div style={{ height: 36, background: T.bg, borderRadius: 6 }} />;
    const vals = data.map(d => d[key_] ?? 0); const max = Math.max(...vals, 1);
    const pts = vals.map((v, i) => `${(i / (vals.length - 1)) * 200},${36 - (v / max) * 34}`).join(" ");
    return <svg viewBox={`0 0 200 36`} style={{ width: "100%", height: 36 }}><polyline points={pts} fill="none" stroke={color} strokeWidth={1.5} /><polyline points={`0,36 ${pts} 200,36`} fill={color + "20"} stroke="none" /></svg>;
  };
  return (
    <Anim>
      {error && <div style={{ background: T.warningBg, border: `1px solid ${T.warningBorder}`, borderRadius: 8, padding: "8px 14px", fontSize: 12, color: T.warning, marginBottom: 14 }}>Metrics endpoint error: {error} — retrying every 3s</div>}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 14, marginBottom: 20 }}>
        {[
          { label: "In-flight", key_: "inflight", color: T.accent, icon: "M13 10V3L4 14h7v7l9-11h-7z" },
          { label: "Queued", key_: "queued", color: T.accent2, icon: "M4 6h16M4 10h16M4 14h16M4 18h16" },
          { label: "Total Requests", key_: "tps", color: T.info, icon: "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" },
        ].map(kpi => (
          <Card key={kpi.label}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: T.textTert, textTransform: "uppercase", letterSpacing: "0.05em" }}>{kpi.label}</span>
              <div style={{ display: "flex", gap: 4, alignItems: "center" }}><div style={{ width: 6, height: 6, borderRadius: "50%", background: metrics ? T.success : T.warning, animation: "pulse 2s infinite" }} /><span style={{ fontSize: 10, color: T.textTert }}>{metrics ? "LIVE" : "…"}</span></div>
            </div>
            <div style={{ fontSize: 28, fontWeight: 700, color: T.text, marginBottom: 8 }}>{metrics ? getMetricVal(metrics, kpi.key_) : "—"}</div>
            <Sparkline data={history} key_={kpi.key_} color={kpi.color} />
          </Card>
        ))}
      </div>
      {!metrics && !error && <div style={{ textAlign: "center", padding: 40, color: T.textTert, fontSize: 13 }}>Connecting to live metrics…<br /><span style={{ fontSize: 11 }}>Polling /admin/dynamo/metrics every 3s</span></div>}
    </Anim>
  );
}

/* ═══════════════════ INFRASTRUCTURE TAB [FIX 7: correct endpoints] ═══════════════════ */
function DynamoTab({ baseUrl, token }) {
  const [status, setStatus] = useState(null); const [metrics, setMetrics] = useState(null); const [loading, setLoading] = useState(true);
  useEffect(() => {
    const poll = async () => {
      try {
        /* FIX 7: Use /admin/dynamo/status and /admin/dynamo/metrics instead of non-existent /frontend/stats, /worker/stats */
        const [st, mt] = await Promise.all([
          apiFetch(baseUrl, "/admin/dynamo/status", { token }).catch(() => null),
          apiFetch(baseUrl, "/admin/dynamo/metrics", { token }).catch(() => null),
        ]);
        setStatus(st); setMetrics(mt); setLoading(false);
      } catch { setLoading(false); }
    };
    poll(); const id = setInterval(poll, 5000); return () => clearInterval(id);
  }, []);

  const fe = metrics?.frontend || {};
  const wk = metrics?.worker || {};
  const feHealthy = status?.frontend?.healthy;
  const wkHealthy = status?.worker?.healthy;
  const modelCfg = status?.model_config || {};

  const StatRow = ({ label, value, unit, tooltip }) => (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "9px 0", borderBottom: `1px solid ${T.border}` }}>
      <div>
        <span style={{ fontSize: 13, color: T.textSec }}>{label}</span>
        {tooltip && <div style={{ fontSize: 11, color: T.textTert, marginTop: 1 }}>{tooltip}</div>}
      </div>
      <span style={{ fontSize: 13, fontWeight: 700, color: T.text }}>{value ?? "—"}{unit && <span style={{ fontSize: 11, fontWeight: 400, color: T.textTert, marginLeft: 3 }}>{unit}</span>}</span>
    </div>
  );

  return (
    <Anim>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        {/* Dynamo Frontend */}
        <Card>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
            <div>
              <div style={{ fontWeight: 700, fontSize: 15, color: T.text, marginBottom: 3 }}>Dynamo Frontend</div>
              <div style={{ fontSize: 12, color: T.textTert }}>HTTP router · port 8001</div>
            </div>
            <Badge variant={feHealthy ? "success" : "warning"}>{feHealthy ? "Online" : "Polling…"}</Badge>
          </div>
          <div style={{ background: T.accentLight, border: `1px solid #99f6e4`, borderRadius: 8, padding: "8px 12px", marginBottom: 14, fontSize: 12, color: T.accent }}>
            Routes incoming requests, manages KV store discovery, and handles request migration across workers.
          </div>
          <StatRow label="Requests In-flight" value={fe.inflight_requests} tooltip="Requests currently being routed" />
          <StatRow label="Queued Requests" value={fe.queued_requests} tooltip="Requests waiting to be routed" />
          <StatRow label="Total Requests" value={fe.total_requests} tooltip="Cumulative requests handled" />
          <StatRow label="Disconnected Clients" value={fe.disconnected_clients} />
          <StatRow label="KV Store Backend" value="File" tooltip="Discovery backend (--store-kv file)" />
          {fe.ttft_seconds && (
            <div style={{ marginTop: 14 }}>
              <SectionLabel>Time to First Token</SectionLabel>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
                {[["p50", fe.ttft_seconds?.p50], ["p95", fe.ttft_seconds?.p95], ["p99", fe.ttft_seconds?.p99]].map(([p, v]) => (
                  <div key={p} style={{ background: T.bg, borderRadius: 8, padding: "10px", textAlign: "center", border: `1px solid ${T.border}` }}>
                    <div style={{ fontSize: 10, fontWeight: 700, color: T.textTert, textTransform: "uppercase", marginBottom: 4 }}>{p}</div>
                    <div style={{ fontSize: 18, fontWeight: 700, color: T.text }}>{v != null ? (v * 1000).toFixed(0) : "—"}</div>
                    <div style={{ fontSize: 10, color: T.textTert }}>ms</div>
                  </div>
                ))}
              </div>
            </div>
          )}
          {fe.request_duration_seconds && (
            <div style={{ marginTop: 14 }}>
              <SectionLabel>Request Duration</SectionLabel>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                {[["p50", fe.request_duration_seconds?.p50], ["p95", fe.request_duration_seconds?.p95]].map(([p, v]) => (
                  <div key={p} style={{ background: T.bg, borderRadius: 8, padding: "10px", textAlign: "center", border: `1px solid ${T.border}` }}>
                    <div style={{ fontSize: 10, fontWeight: 700, color: T.textTert, textTransform: "uppercase", marginBottom: 4 }}>{p}</div>
                    <div style={{ fontSize: 18, fontWeight: 700, color: T.text }}>{v != null ? (v * 1000).toFixed(0) : "—"}</div>
                    <div style={{ fontSize: 10, color: T.textTert }}>ms</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </Card>

        {/* vLLM Worker */}
        <Card>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
            <div>
              <div style={{ fontWeight: 700, fontSize: 15, color: T.text, marginBottom: 3 }}>vLLM Worker</div>
              <div style={{ fontSize: 12, color: T.textTert }}>GPU inference process · metrics port 8081</div>
            </div>
            <Badge variant={wkHealthy ? "success" : "warning"}>{wkHealthy ? "Online" : "Polling…"}</Badge>
          </div>
          <div style={{ background: T.accent2Light, border: `1px solid #c7d2fe`, borderRadius: 8, padding: "8px 12px", marginBottom: 14, fontSize: 12, color: T.accent2 }}>
            Loads and runs the model on GPU. Manages KV cache, tokenization, and generation.
          </div>
          <StatRow label="GPU KV Cache Usage" value={wk.gpu_cache_usage_percent != null ? `${wk.gpu_cache_usage_percent.toFixed(1)}%` : null} tooltip="Fraction of GPU KV cache in use" />
          <StatRow label="CPU Cache Usage" value={wk.cpu_cache_usage_percent != null ? `${wk.cpu_cache_usage_percent.toFixed(1)}%` : null} />
          <StatRow label="Requests Running" value={wk.requests_running} tooltip="Sequences currently being generated" />
          <StatRow label="Requests Waiting" value={wk.requests_waiting} tooltip="Requests queued for GPU time" />
          <StatRow label="Success Total" value={wk.request_success_total} tooltip="Cumulative successful requests" />
          {modelCfg && Object.keys(modelCfg).length > 0 && (
            <div style={{ marginTop: 14 }}>
              <SectionLabel>Model Config</SectionLabel>
              <StatRow label="Context Length" value={modelCfg.context_length} />
              <StatRow label="KV Block Size" value={modelCfg.kv_cache_block_size} />
              <StatRow label="Max Batched Tokens" value={modelCfg.max_num_batched_tokens} />
            </div>
          )}
        </Card>
      </div>
    </Anim>
  );
}

/* ═══════════════════ TENANTS TAB [FIX 5: correct field names] ═══════════════════ */
function TenantsTab({ baseUrl, token, toast }) {
  const [tenants, setTenants] = useState([]); const [loading, setLoading] = useState(true);
  const [showAdd, setShowAdd] = useState(false);
  const [form, setForm] = useState({
    name: "", tier: "starter",
    max_concurrent: "", rpm_limit: "", tpm_limit: "",
    budget_limit: "", budget_window: "month",
    max_context_tokens: "", max_output_tokens: "",
  });
  const [saving, setSaving] = useState(false);
  const [revealKey, setRevealKey] = useState(null);

  const load = () => apiFetch(baseUrl, "/admin/tenants", { token }).then(r => setTenants(normalizeTenants(r))).catch(() => {}).finally(() => setLoading(false));
  useEffect(() => { load(); }, []);

  const TIERS = ["starter", "standard", "pro", "enterprise"];
  const PRESETS = {
    starter:    { max_concurrent: 5,   rpm_limit: 60,   tpm_limit: 50000,    max_context_tokens: 8192,   max_output_tokens: 1024 },
    standard:   { max_concurrent: 20,  rpm_limit: 300,  tpm_limit: 200000,   max_context_tokens: 32768,  max_output_tokens: 4096 },
    pro:        { max_concurrent: 50,  rpm_limit: 1000, tpm_limit: 1000000,  max_context_tokens: 131072, max_output_tokens: 8192 },
    enterprise: { max_concurrent: 200, rpm_limit: 5000, tpm_limit: 10000000, max_context_tokens: 131072, max_output_tokens: 8192 },
  };

  const applyTier = (tier) => {
    const p = PRESETS[tier];
    setForm(f => ({ ...f, tier, ...Object.fromEntries(Object.entries(p).map(([k, v]) => [k, String(v)])) }));
  };

  const add = async () => {
    if (!form.name.trim()) { toast("Name is required", "error"); return; }
    setSaving(true);
    try {
      /* FIX 5: Use backend field names (rpm_limit, tpm_limit) not requests_per_minute/tokens_per_minute */
      const body = { tenant_name: form.name.trim() };
      const numFields = ["max_concurrent", "rpm_limit", "tpm_limit", "max_context_tokens", "max_output_tokens"];
      numFields.forEach(k => { if (form[k]) body[k] = Number(form[k]); });

      const res = await apiFetch(baseUrl, "/admin/tenants", { token, method: "POST", body });
      const tenantId = res.tenant_id;
      const key = res.api_key || res.key || res.token;

      /* FIX 5: Budget is a separate endpoint, not part of tenant creation */
      if (form.budget_limit && tenantId) {
        await apiFetch(baseUrl, `/admin/budgets/${tenantId}`, {
          token, method: "PUT",
          body: { window: form.budget_window, budget_usd: Number(form.budget_limit) }
        }).catch(() => {}); // non-fatal
      }

      setShowAdd(false);
      if (key) setRevealKey(key); else toast("Tenant created");
      setForm({ name: "", tier: "starter", max_concurrent: "", rpm_limit: "", tpm_limit: "", budget_limit: "", budget_window: "month", max_context_tokens: "", max_output_tokens: "" });
      load();
    } catch (e) { toast(e.message, "error"); } finally { setSaving(false); }
  };

  return (
    <Anim>
      {revealKey && <ApiKeyCopyModal apiKey={revealKey} onClose={() => setRevealKey(null)} />}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 17, fontWeight: 700, color: T.text }}>Tenants <span style={{ fontSize: 13, fontWeight: 400, color: T.textTert }}>({tenants.length})</span></h2>
        <Btn onClick={() => setShowAdd(true)}><SVG d="M12 4v16m8-8H4" size={14} /> Add Tenant</Btn>
      </div>

      {showAdd && (
        <Modal title="Add Tenant" onClose={() => setShowAdd(false)} width={560}>
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div style={{ gridColumn: "1/-1" }}>
                <SectionLabel>Tenant Name *</SectionLabel>
                <Input value={form.name} onChange={v => setForm(f => ({ ...f, name: v }))} placeholder="Acme Financial" />
              </div>
              <div style={{ gridColumn: "1/-1" }}>
                <SectionLabel>Tier Preset</SectionLabel>
                <div style={{ display: "flex", gap: 6 }}>
                  {TIERS.map(t => (
                    <button key={t} onClick={() => applyTier(t)} style={{ flex: 1, padding: "6px 0", fontSize: 12, fontWeight: 600, border: `1px solid ${form.tier === t ? T.accent : T.border}`, background: form.tier === t ? T.accentLight : T.bg, color: form.tier === t ? T.accent : T.textSec, borderRadius: 7, cursor: "pointer", textTransform: "capitalize", fontFamily: "inherit" }}>{t}</button>
                  ))}
                </div>
              </div>
            </div>
            <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: 10, padding: "14px 16px" }}>
              <SectionLabel>Rate Limits</SectionLabel>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
                {/* FIX 5: Use correct backend field names */}
                {[
                  ["Max Concurrent", "max_concurrent", "reqs"],
                  ["Requests/min", "rpm_limit", "rpm"],
                  ["Tokens/min", "tpm_limit", "tpm"],
                ].map(([label, key, unit]) => (
                  <div key={key}>
                    <div style={{ fontSize: 11, color: T.textTert, marginBottom: 4 }}>{label} <span style={{ color: T.textTert }}>({unit})</span></div>
                    <Input value={form[key]} onChange={v => setForm(f => ({ ...f, [key]: v }))} placeholder="—" type="number" />
                  </div>
                ))}
              </div>
            </div>
            <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: 10, padding: "14px 16px" }}>
              <SectionLabel>Token Limits</SectionLabel>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                {[["Max Context Tokens", "max_context_tokens"], ["Max Output Tokens", "max_output_tokens"]].map(([label, key]) => (
                  <div key={key}>
                    <div style={{ fontSize: 11, color: T.textTert, marginBottom: 4 }}>{label}</div>
                    <Input value={form[key]} onChange={v => setForm(f => ({ ...f, [key]: v }))} placeholder="—" type="number" />
                  </div>
                ))}
              </div>
            </div>
            <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: 10, padding: "14px 16px" }}>
              <SectionLabel>Spend Policy</SectionLabel>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                <div>
                  <div style={{ fontSize: 11, color: T.textTert, marginBottom: 4 }}>Budget Cap ($)</div>
                  <Input value={form.budget_limit} onChange={v => setForm(f => ({ ...f, budget_limit: v }))} placeholder="e.g. 500" type="number" />
                </div>
                <div>
                  <div style={{ fontSize: 11, color: T.textTert, marginBottom: 4 }}>Window</div>
                  <Select value={form.budget_window} onChange={v => setForm(f => ({ ...f, budget_window: v }))}>
                    <option value="day">Daily</option>
                    <option value="week">Weekly</option>
                    <option value="month">Monthly</option>
                  </Select>
                </div>
              </div>
            </div>
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, paddingTop: 4 }}>
              <Btn variant="secondary" onClick={() => setShowAdd(false)}>Cancel</Btn>
              <Btn onClick={add} disabled={saving}>{saving ? <><Spinner /> Creating…</> : "Create Tenant"}</Btn>
            </div>
          </div>
        </Modal>
      )}

      {loading ? <div style={{ textAlign: "center", padding: 40 }}><Spinner size={20} /></div> : (
        <Card style={{ padding: 0 }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead><tr>{["Name", "ID", "Concurrency", "RPM", "TPM", "Status"].map(h => <th key={h} style={{ textAlign: "left", padding: "10px 14px", fontSize: 11, fontWeight: 700, color: T.textTert, textTransform: "uppercase", letterSpacing: "0.04em", borderBottom: `1px solid ${T.border}`, background: T.bg }}>{h}</th>)}</tr></thead>
            <tbody>{tenants.map(t => (
              <tr key={t.id || t.tenant_id} style={{ borderBottom: `1px solid ${T.border}` }}>
                <td style={{ padding: "10px 14px", fontWeight: 600, color: T.text }}>{t.name}</td>
                <td style={{ padding: "10px 14px", color: T.textTert, fontFamily: "monospace", fontSize: 11 }}>{(t.id || "")?.slice(0, 12)}…</td>
                <td style={{ padding: "10px 14px", color: T.textSec }}>{t.max_concurrent ?? "—"}</td>
                <td style={{ padding: "10px 14px", color: T.textSec }}>{t.rpm_limit ?? "—"}</td>
                <td style={{ padding: "10px 14px", color: T.textSec }}>{t.tpm_limit ?? "—"}</td>
                <td style={{ padding: "10px 14px" }}><Badge variant={t.is_active !== false ? "success" : "danger"}>{t.is_active !== false ? "Active" : "Inactive"}</Badge></td>
              </tr>
            ))}</tbody>
          </table>
          {tenants.length === 0 && <div style={{ textAlign: "center", padding: "30px 0", color: T.textTert, fontSize: 13 }}>No tenants yet — add one to get started</div>}
        </Card>
      )}
    </Anim>
  );
}

/* ═══════════════════ KEYS TAB ═══════════════════ */
function KeysTab({ baseUrl, token, toast }) {
  const [tenants, setTenants] = useState([]); const [keys, setKeys] = useState([]);
  const [seats, setSeats] = useState([]);
  const [selTenant, setSelTenant] = useState(""); const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false); const [keyName, setKeyName] = useState("");
  const [keySeatId, setKeySeatId] = useState("");
  const [revealKey, setRevealKey] = useState(null);

  // Seat creation state
  const [showAddSeat, setShowAddSeat] = useState(false);
  const [seatForm, setSeatForm] = useState({ seat_name: "", role: "user" });
  const [creatingSeat, setCreatingSeat] = useState(false);

  const loadTenants = () => apiFetch(baseUrl, "/admin/tenants", { token }).then(r => { const t = normalizeTenants(r); setTenants(t); if (t.length && !selTenant) setSelTenant(t[0].id); }).catch(() => {});
  const loadKeys = (tid) => {
    if (!tid) return;
    apiFetch(baseUrl, `/admin/keys?tenant_id=${tid}`, { token })
      .then(r => setKeys(normalizeKeys(r)))
      .catch(() => setKeys([]))
      .finally(() => setLoading(false));
  };
  const loadSeats = (tid) => {
    if (!tid) return;
    apiFetch(baseUrl, `/admin/seats?tenant_id=${tid}`, { token })
      .then(r => { const arr = Array.isArray(r) ? r : (r.seats || r.data || r.items || []); setSeats(arr); })
      .catch(() => setSeats([]));
  };
  useEffect(() => { loadTenants(); }, []);
  useEffect(() => { if (selTenant) { loadKeys(selTenant); loadSeats(selTenant); setKeySeatId(""); } }, [selTenant]);

  const create = async () => {
    if (!selTenant) return;
    setCreating(true);
    try {
      const kn = keyName || `key-${Date.now()}`;
      const body = { tenant_id: selTenant, name: kn };
      if (keySeatId) body.seat_id = keySeatId;
      const res = await apiFetch(baseUrl, `/admin/keys`, { token, method: "POST", body });
      const key = res.api_key || res.key || res.raw_key || res.token;
      if (key && typeof key === "string") { setRevealKey(key); } else { toast("Key created (check keys list)", "info"); }
      setKeyName(""); setKeySeatId(""); loadKeys(selTenant);
    } catch (e) { toast(e.message, "error"); } finally { setCreating(false); }
  };

  const revoke = async (kid) => {
    if (!confirm("Revoke this key? This cannot be undone.")) return;
    try {
      await apiFetch(baseUrl, `/admin/keys/${kid}/revoke`, { token, method: "POST" });
      toast("Key revoked"); loadKeys(selTenant);
    } catch (e) { toast(e.message, "error"); }
  };

  const createSeat = async () => {
    if (!selTenant || !seatForm.seat_name.trim()) { toast("Seat name is required", "error"); return; }
    setCreatingSeat(true);
    try {
      await apiFetch(baseUrl, "/admin/seats", { token, method: "POST", body: { tenant_id: selTenant, seat_name: seatForm.seat_name.trim(), role: seatForm.role } });
      toast("Seat created");
      setSeatForm({ seat_name: "", role: "user" });
      setShowAddSeat(false);
      loadSeats(selTenant);
    } catch (e) { toast(e.message, "error"); } finally { setCreatingSeat(false); }
  };

  const deactivateSeat = async (sid, active) => {
    try {
      await apiFetch(baseUrl, `/admin/seats/${sid}`, { token, method: "PATCH", body: { is_active: !active } });
      toast(active ? "Seat deactivated" : "Seat activated");
      loadSeats(selTenant);
    } catch (e) { toast(e.message, "error"); }
  };

  const activeSeatOptions = seats.filter(s => s.is_active !== false);

  return (
    <Anim>
      {revealKey && <ApiKeyCopyModal apiKey={revealKey} onClose={() => setRevealKey(null)} />}
      <div style={{ display: "grid", gridTemplateColumns: "220px 1fr", gap: 16 }}>
        <Card>
          <SectionLabel>Tenant</SectionLabel>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {tenants.map(t => (
              <button key={t.id} onClick={() => setSelTenant(t.id)} style={{ textAlign: "left", padding: "8px 10px", borderRadius: 8, background: selTenant === t.id ? T.accentLight : "none", border: `1px solid ${selTenant === t.id ? T.accent + "40" : "transparent"}`, cursor: "pointer", fontSize: 13, fontWeight: 600, color: selTenant === t.id ? T.accent : T.text, fontFamily: "inherit" }}>{t.name}</button>
            ))}
          </div>
        </Card>
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {/* Seats section */}
          <Card style={{ padding: 0 }}>
            <div style={{ padding: "12px 16px", borderBottom: `1px solid ${T.border}`, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <span style={{ fontSize: 13, fontWeight: 600, color: T.text }}>Seats</span>
                <span style={{ fontSize: 12, color: T.textTert, marginLeft: 8 }}>({seats.length})</span>
                <span style={{ fontSize: 11, color: T.textTert, marginLeft: 8 }}>· A seat represents a named user or service identity. Keys tied to a seat inherit its role.</span>
              </div>
              <Btn small onClick={() => setShowAddSeat(v => !v)}><SVG d="M12 4v16m8-8H4" size={12} /> Add Seat</Btn>
            </div>
            {showAddSeat && (
              <div style={{ padding: "12px 16px", borderBottom: `1px solid ${T.border}`, display: "flex", gap: 8, alignItems: "flex-end", background: T.accentLight + "60" }}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 11, color: T.textTert, marginBottom: 4 }}>Seat Name</div>
                  <Input value={seatForm.seat_name} onChange={v => setSeatForm(f => ({ ...f, seat_name: v }))} placeholder="e.g. alice@company.com" />
                </div>
                <div>
                  <div style={{ fontSize: 11, color: T.textTert, marginBottom: 4 }}>Role</div>
                  <Select value={seatForm.role} onChange={v => setSeatForm(f => ({ ...f, role: v }))}>
                    <option value="user">user</option>
                    <option value="admin">admin</option>
                    <option value="service">service</option>
                  </Select>
                </div>
                <Btn onClick={createSeat} disabled={creatingSeat}>{creatingSeat ? <><Spinner /> Saving…</> : "Save"}</Btn>
                <Btn variant="ghost" onClick={() => setShowAddSeat(false)}>Cancel</Btn>
              </div>
            )}
            {seats.length === 0 ? (
              <div style={{ textAlign: "center", padding: "18px 0", color: T.textTert, fontSize: 13 }}>No seats yet — seats are optional named identities (users or services)</div>
            ) : (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead><tr>{["Name", "Role", "Status", ""].map(h => <th key={h} style={{ textAlign: "left", padding: "7px 16px", fontSize: 11, fontWeight: 700, color: T.textTert, textTransform: "uppercase", letterSpacing: "0.04em", background: T.bg, borderBottom: `1px solid ${T.border}` }}>{h}</th>)}</tr></thead>
                <tbody>{seats.map(s => (
                  <tr key={s.seat_id} style={{ borderBottom: `1px solid ${T.border}` }}>
                    <td style={{ padding: "9px 16px", fontWeight: 600, color: T.text }}>{s.seat_name || "—"}</td>
                    <td style={{ padding: "9px 16px" }}><Badge variant={s.role === "admin" ? "warning" : s.role === "service" ? "purple" : "default"}>{s.role}</Badge></td>
                    <td style={{ padding: "9px 16px" }}>{s.is_active !== false ? <Badge variant="success">Active</Badge> : <Badge variant="danger">Inactive</Badge>}</td>
                    <td style={{ padding: "9px 16px" }}><Btn variant={s.is_active !== false ? "danger" : "primary"} small onClick={() => deactivateSeat(s.seat_id, s.is_active !== false)}>{s.is_active !== false ? "Deactivate" : "Activate"}</Btn></td>
                  </tr>
                ))}</tbody>
              </table>
            )}
          </Card>

          {/* API Keys section */}
          <Card>
            <SectionLabel>New API Key</SectionLabel>
            <div style={{ display: "flex", gap: 8, alignItems: "flex-end" }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 11, color: T.textTert, marginBottom: 4 }}>Key Name (optional)</div>
                <Input value={keyName} onChange={setKeyName} placeholder="e.g. production-key" />
              </div>
              <div style={{ minWidth: 180 }}>
                <div style={{ fontSize: 11, color: T.textTert, marginBottom: 4 }}>Assign to Seat (optional)</div>
                <Select value={keySeatId} onChange={setKeySeatId} style={{ width: "100%" }}>
                  <option value="">— Service key (no seat) —</option>
                  {activeSeatOptions.map(s => <option key={s.seat_id} value={s.seat_id}>{s.seat_name} ({s.role})</option>)}
                </Select>
              </div>
              <Btn onClick={create} disabled={creating || !selTenant}>{creating ? <><Spinner /> Creating…</> : <><SVG d="M12 4v16m8-8H4" size={13} /> Create Key</>}</Btn>
            </div>
            <div style={{ marginTop: 8, fontSize: 11, color: T.textTert }}>
              A <strong>service key</strong> is not tied to any seat — use it for server-to-server or automated access. A <strong>seat key</strong> is linked to a named identity and inherits that seat's role and active status.
            </div>
          </Card>
          <Card style={{ padding: 0 }}>
            <div style={{ padding: "12px 16px", borderBottom: `1px solid ${T.border}` }}>
              <span style={{ fontSize: 13, fontWeight: 600, color: T.text }}>API Keys</span>
              <span style={{ fontSize: 12, color: T.textTert, marginLeft: 8 }}>({keys.length})</span>
            </div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead><tr>{["Name / Seat", "Role", "Created", "Status", ""].map(h => <th key={h} style={{ textAlign: "left", padding: "8px 16px", fontSize: 11, fontWeight: 700, color: T.textTert, textTransform: "uppercase", letterSpacing: "0.04em", background: T.bg, borderBottom: `1px solid ${T.border}` }}>{h}</th>)}</tr></thead>
              <tbody>{keys.map(k => (
                <tr key={k.id} style={{ borderBottom: `1px solid ${T.border}` }}>
                  <td style={{ padding: "10px 16px", fontWeight: 600, color: T.text }}>
                    {k.seat_name ? <span>{k.seat_name} <span style={{ fontSize: 11, fontWeight: 400, color: T.textTert }}>(seat)</span></span> : k.name || <span style={{ color: T.textTert, fontWeight: 400, fontStyle: "italic" }}>service key</span>}
                  </td>
                  <td style={{ padding: "10px 16px" }}>{k.role ? <Badge variant={k.role === "admin" ? "warning" : k.role === "service" ? "purple" : "default"}>{k.role}</Badge> : <span style={{ color: T.textTert }}>—</span>}</td>
                  <td style={{ padding: "10px 16px", color: T.textSec }}>{k.created_at ? fmtTime(k.created_at) : "—"}</td>
                  <td style={{ padding: "10px 16px" }}>{k.revoked_at ? <Badge variant="danger">Revoked</Badge> : <Badge variant="success">Active</Badge>}</td>
                  <td style={{ padding: "10px 16px" }}>{!k.revoked_at && <Btn variant="danger" small onClick={() => revoke(k.id)}>Revoke</Btn>}</td>
                </tr>
              ))}</tbody>
            </table>
            {keys.length === 0 && !loading && <div style={{ textAlign: "center", padding: "24px 0", color: T.textTert, fontSize: 13 }}>No keys for this tenant</div>}
          </Card>
        </div>
      </div>
    </Anim>
  );
}

/* ═══════════════════ POLICY TAB [FIX 6: split tenant PATCH + budget PUT] ═══════════════════ */
function PolicyTab({ baseUrl, token, toast }) {
  const [tenants, setTenants] = useState([]); const [selTenant, setSelTenant] = useState("");
  const [policy, setPolicy] = useState(null); const [saving, setSaving] = useState(false); const [loading, setLoading] = useState(false);
  useEffect(() => { apiFetch(baseUrl, "/admin/tenants", { token }).then(r => { const t = normalizeTenants(r); setTenants(t); if (t.length) setSelTenant(t[0].id); }).catch(() => {}); }, []);

  useEffect(() => {
    if (!selTenant) return; setLoading(true);
    /* FIX 6: Use correct backend field names (rpm_limit, tpm_limit) */
    const extractPolicy = (t) => ({
      max_concurrent: t.max_concurrent ?? "",
      rpm_limit: t.rpm_limit ?? "",
      tpm_limit: t.tpm_limit ?? "",
      max_context_tokens: t.max_context_tokens ?? "",
      max_output_tokens: t.max_output_tokens ?? "",
      budget_limit: "", budget_window: "month",
      budget_alert_threshold: "80",
    });
    /* FIX 6: Use the new GET /admin/tenants/{id} endpoint (added to backend) */
    apiFetch(baseUrl, `/admin/tenants/${selTenant}`, { token })
      .then(t => setPolicy(extractPolicy(t)))
      .catch(() => {
        const found = tenants.find(t => t.id === selTenant);
        setPolicy(found ? extractPolicy(found) : { max_concurrent: "", rpm_limit: "", tpm_limit: "", max_context_tokens: "", max_output_tokens: "", budget_limit: "", budget_window: "month", budget_alert_threshold: "80" });
      })
      .finally(() => setLoading(false));
    /* FIX 6: Fetch budget separately from /admin/budgets?tenant_id= */
    apiFetch(baseUrl, `/admin/budgets?tenant_id=${selTenant}`, { token })
      .then(r => {
        const budgets = r.data || [];
        if (budgets.length > 0) {
          setPolicy(p => p ? { ...p, budget_limit: budgets[0].budget_usd || "", budget_window: budgets[0].window || "month" } : p);
        }
      }).catch(() => {});
  }, [selTenant, tenants]);

  /* FIX 6: Split save into tenant PATCH + budget PUT */
  const save = async () => {
    setSaving(true);
    try {
      const tenantBody = {};
      if (policy.max_concurrent !== "") tenantBody.max_concurrent = Number(policy.max_concurrent);
      if (policy.rpm_limit !== "") tenantBody.rpm_limit = Number(policy.rpm_limit);
      if (policy.tpm_limit !== "") tenantBody.tpm_limit = Number(policy.tpm_limit);
      if (policy.max_context_tokens !== "") tenantBody.max_context_tokens = Number(policy.max_context_tokens);
      if (policy.max_output_tokens !== "") tenantBody.max_output_tokens = Number(policy.max_output_tokens);

      if (Object.keys(tenantBody).length > 0) {
        await apiFetch(baseUrl, `/admin/tenants/${selTenant}`, { token, method: "PATCH", body: tenantBody });
      }
      if (policy.budget_limit) {
        await apiFetch(baseUrl, `/admin/budgets/${selTenant}`, {
          token, method: "PUT",
          body: { window: policy.budget_window, budget_usd: Number(policy.budget_limit) }
        });
      }
      toast("Policy saved");
    } catch (e) { toast(e.message, "error"); } finally { setSaving(false); }
  };

  const PRESETS = [
    { label: "Conservative", max_concurrent: 5, rpm_limit: 60, tpm_limit: 50000, max_context_tokens: 8192, max_output_tokens: 1024 },
    { label: "Standard", max_concurrent: 20, rpm_limit: 300, tpm_limit: 200000, max_context_tokens: 32768, max_output_tokens: 4096 },
    { label: "Pro", max_concurrent: 50, rpm_limit: 1000, tpm_limit: 1000000, max_context_tokens: 131072, max_output_tokens: 8192 },
  ];

  const Field = ({ label, key_, unit }) => (
    <div>
      <div style={{ fontSize: 11, color: T.textTert, marginBottom: 4 }}>{label}{unit && <span style={{ marginLeft: 4, color: T.textTert }}>({unit})</span>}</div>
      <Input value={policy?.[key_] ?? ""} onChange={v => setPolicy(p => ({ ...p, [key_]: v }))} type="number" placeholder="—" />
    </div>
  );

  return (
    <Anim>
      <div style={{ display: "grid", gridTemplateColumns: "200px 1fr", gap: 16 }}>
        <Card>
          <SectionLabel>Tenant</SectionLabel>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {tenants.map(t => (
              <button key={t.id} onClick={() => setSelTenant(t.id)} style={{ textAlign: "left", padding: "8px 10px", borderRadius: 8, background: selTenant === t.id ? T.accentLight : "none", border: `1px solid ${selTenant === t.id ? T.accent + "40" : "transparent"}`, cursor: "pointer", fontSize: 13, fontWeight: 600, color: selTenant === t.id ? T.accent : T.text, fontFamily: "inherit" }}>{t.name}</button>
            ))}
          </div>
        </Card>
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {loading ? <div style={{ textAlign: "center", padding: 40 }}><Spinner size={18} /></div> : policy && (
            <>
              <Card>
                <SectionLabel>Presets</SectionLabel>
                <div style={{ display: "flex", gap: 8 }}>
                  {PRESETS.map(p => (
                    <Btn key={p.label} variant="secondary" small onClick={() => setPolicy(prev => ({ ...prev, max_concurrent: p.max_concurrent, rpm_limit: p.rpm_limit, tpm_limit: p.tpm_limit, max_context_tokens: p.max_context_tokens, max_output_tokens: p.max_output_tokens }))}>{p.label}</Btn>
                  ))}
                </div>
              </Card>
              <Card>
                <SectionLabel>Rate Limits</SectionLabel>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
                  <Field label="Max Concurrent" key_="max_concurrent" unit="reqs" />
                  <Field label="Requests/min" key_="rpm_limit" unit="rpm" />
                  <Field label="Tokens/min" key_="tpm_limit" unit="tpm" />
                </div>
              </Card>
              <Card>
                <SectionLabel>Token Limits</SectionLabel>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  <Field label="Max Context Tokens" key_="max_context_tokens" />
                  <Field label="Max Output Tokens" key_="max_output_tokens" />
                </div>
              </Card>
              <Card>
                <SectionLabel>Spend Policy</SectionLabel>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
                  <div>
                    <div style={{ fontSize: 11, color: T.textTert, marginBottom: 4 }}>Budget Cap ($)</div>
                    <Input value={policy.budget_limit} onChange={v => setPolicy(p => ({ ...p, budget_limit: v }))} type="number" placeholder="No limit" />
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: T.textTert, marginBottom: 4 }}>Window</div>
                    <Select value={policy.budget_window} onChange={v => setPolicy(p => ({ ...p, budget_window: v }))}>
                      <option value="day">Daily</option>
                      <option value="week">Weekly</option>
                      <option value="month">Monthly</option>
                    </Select>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: T.textTert, marginBottom: 4 }}>Alert Threshold (%)</div>
                    <Input value={policy.budget_alert_threshold} onChange={v => setPolicy(p => ({ ...p, budget_alert_threshold: v }))} type="number" placeholder="80" />
                  </div>
                </div>
                {policy.budget_limit && (
                  <div style={{ marginTop: 12 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: T.textTert, marginBottom: 4 }}>
                      <span>Budget limit set</span><span>${policy.budget_limit} / {policy.budget_window}</span>
                    </div>
                    <div style={{ background: T.bg, borderRadius: 6, height: 6, overflow: "hidden", border: `1px solid ${T.border}` }}>
                      <div style={{ height: "100%", width: "0%", background: T.success, borderRadius: 6 }} />
                    </div>
                    <div style={{ fontSize: 11, color: T.textTert, marginTop: 4 }}>$0 spent this {policy.budget_window}</div>
                  </div>
                )}
              </Card>
              <div style={{ display: "flex", justifyContent: "flex-end" }}>
                <Btn onClick={save} disabled={saving}>{saving ? <><Spinner /> Saving…</> : "Save Policy"}</Btn>
              </div>
            </>
          )}
        </div>
      </div>
    </Anim>
  );
}

/* ═══════════════════ AUDIT TAB [FIX 3: single /admin/audit endpoint] ═══════════════════ */
function AuditTab({ baseUrl, token }) {
  const [subTab, setSubTab] = useState("runs");
  const [logs, setLogs] = useState([]); const [loading, setLoading] = useState(true);
  const [tenants, setTenants] = useState([]); const [filterTenant, setFilterTenant] = useState("");
  const [filterWindow, setFilterWindow] = useState("24h");

  useEffect(() => { apiFetch(baseUrl, "/admin/tenants", { token }).then(r => setTenants(normalizeTenants(r))).catch(() => {}); }, []);
  useEffect(() => {
    setLoading(true);
    /* FIX 3: Use single /admin/audit endpoint — no /runs or /admin sub-paths */
    const path = `/admin/audit?window=${filterWindow}${filterTenant ? `&tenant_id=${filterTenant}` : ""}`;
    apiFetch(baseUrl, path, { token }).then(r => {
      const arr = Array.isArray(r) ? r : (r.data || r.runs || r.logs || r.events || r.items || []);
      setLogs(arr);
    }).catch(() => setLogs([])).finally(() => setLoading(false));
  }, [subTab, filterTenant, filterWindow]);

  const actionColors = {
    tenant_create: "success", seat_create: "success", key_create: "success", budget_set: "success",
    tenant_update_limits: "info", seat_update: "info", budget_update: "info",
    seat_activated: "success", seat_deactivated: "warning",
    key_revoke: "danger",
    profile_viewed: "accent", profile_generated: "accent",
  };
  const getVariant = (action = "") => actionColors[action] || "default";

  return (
    <Anim>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div style={{ display: "flex", gap: 0, border: `1px solid ${T.border}`, borderRadius: 9, overflow: "hidden" }}>
          {[
            { id: "runs", label: "All Events", icon: "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" },
          ].map(s => (
            <button key={s.id} onClick={() => setSubTab(s.id)} style={{ display: "flex", alignItems: "center", gap: 6, padding: "7px 14px", fontSize: 13, fontWeight: 600, border: "none", background: subTab === s.id ? T.accent : T.card, color: subTab === s.id ? "#fff" : T.textSec, cursor: "pointer", fontFamily: "inherit" }}>
              <SVG d={s.icon} size={13} color={subTab === s.id ? "#fff" : T.textSec} /> {s.label}
            </button>
          ))}
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <Select value={filterTenant} onChange={setFilterTenant}>
            <option value="">All tenants</option>
            {tenants.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
          </Select>
          <Select value={filterWindow} onChange={setFilterWindow}>
            {["1h", "24h", "7d"].map(w => <option key={w} value={w}>{w}</option>)}
          </Select>
        </div>
      </div>

      <div style={{ marginBottom: 12, background: T.infoBg, border: `1px solid ${T.infoBorder}`, borderRadius: 8, padding: "8px 14px", fontSize: 12, color: T.info }}>
        Admin audit log — tenant changes, key revocations, policy updates, profile views. Maps directly to SOC 2 change management requirements.
      </div>

      <Card style={{ padding: 0 }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr>
              {["Time", "Admin", "Action", "Resource", "Resource ID", "Request ID"].map(h =>
                <th key={h} style={{ textAlign: "left", padding: "9px 14px", fontSize: 11, fontWeight: 700, color: T.textTert, textTransform: "uppercase", letterSpacing: "0.04em", background: T.bg, borderBottom: `1px solid ${T.border}` }}>{h}</th>
              )}
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={6} style={{ textAlign: "center", padding: 32, color: T.textTert }}><Spinner size={18} /></td></tr>
            ) : logs.length === 0 ? (
              <tr><td colSpan={6} style={{ textAlign: "center", padding: 32, color: T.textTert, fontSize: 13 }}>No audit events in this window</td></tr>
            ) : logs.map((log, i) => (
              <tr key={log.event_id || i} style={{ borderBottom: `1px solid ${T.border}` }}>
                <td style={{ padding: "9px 14px", color: T.textSec, fontSize: 11 }}>{fmtTime(log.ts)}</td>
                <td style={{ padding: "9px 14px", fontWeight: 600, color: T.text }}>{log.admin_identity || "—"}</td>
                <td style={{ padding: "9px 14px" }}><Badge variant={getVariant(log.action)}>{log.action}</Badge></td>
                <td style={{ padding: "9px 14px", color: T.textSec }}>{log.resource_type || "—"}</td>
                <td style={{ padding: "9px 14px", color: T.textTert, fontFamily: "monospace", fontSize: 11 }}>{(log.resource_id || "—").slice(0, 16)}</td>
                <td style={{ padding: "9px 14px", color: T.textTert, fontFamily: "monospace", fontSize: 11 }}>{(log.request_id || "—").slice(0, 12)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </Anim>
  );
}

/* ═══════════════════ USAGE TAB ═══════════════════ */
function UsageTab({ baseUrl, token }) {
  const [rows, setRows] = useState([]); const [loading, setLoading] = useState(true);
  const [window_, setWindow_] = useState("24h");
  const [costEnabled, setCostEnabled] = useState(false);

  useEffect(() => {
    setLoading(true);
    apiFetch(baseUrl, `/admin/usage/tenants?window=${window_}`, { token })
      .then(r => { setRows(r.data || []); setCostEnabled(r.cost_estimation_enabled || false); })
      .catch(() => setRows([]))
      .finally(() => setLoading(false));
  }, [window_]);

  const totals = rows.reduce((acc, u) => ({
    requests: acc.requests + (u.requests || 0),
    errors: acc.errors + (u.errors || 0),
    tokens: acc.tokens + (u.total_tokens || 0),
    cost: acc.cost + (u.cost_est_sum || 0),
  }), { requests: 0, errors: 0, tokens: 0, cost: 0 });

  const errRate = totals.requests > 0 ? ((totals.errors / totals.requests) * 100).toFixed(1) : "0.0";

  const kpis = [
    { label: "Total Requests", value: totals.requests.toLocaleString(), icon: "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z", color: T.accent },
    { label: "Total Tokens", value: totals.tokens >= 1e6 ? `${(totals.tokens / 1e6).toFixed(2)}M` : totals.tokens.toLocaleString(), icon: "M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4", color: T.accent2 },
    { label: "Error Rate", value: `${errRate}%`, icon: "M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z", color: totals.errors > 0 ? T.danger : T.success },
    ...(costEnabled ? [{ label: "Est. Cost", value: `$${totals.cost.toFixed(4)}`, icon: "M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 10v-1m0-9a9 9 0 110 18A9 9 0 0112 3z", color: T.warning }] : []),
  ];

  return (
    <Anim>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 17, fontWeight: 700, color: T.text }}>Tenant Usage</h2>
        <Select value={window_} onChange={setWindow_}>
          {["1h", "24h", "7d", "30d"].map(w => <option key={w} value={w}>{w}</option>)}
        </Select>
      </div>

      {/* KPI summary cards */}
      {!loading && rows.length > 0 && (
        <div style={{ display: "grid", gridTemplateColumns: `repeat(${kpis.length}, 1fr)`, gap: 12, marginBottom: 16 }}>
          {kpis.map(k => (
            <Card key={k.label} style={{ display: "flex", alignItems: "center", gap: 12, padding: "14px 16px" }}>
              <div style={{ width: 36, height: 36, borderRadius: 10, background: k.color + "18", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                <SVG d={k.icon} size={18} color={k.color} />
              </div>
              <div>
                <div style={{ fontSize: 11, color: T.textTert, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.04em" }}>{k.label}</div>
                <div style={{ fontSize: 22, fontWeight: 700, color: k.color, lineHeight: 1.2 }}>{k.value}</div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {loading ? <div style={{ textAlign: "center", padding: 40 }}><Spinner size={20} /></div> : (
        <Card style={{ padding: 0 }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead><tr>{["Tenant", "Requests", "Errors", "Error Rate", "Prompt Tokens", "Completion Tokens", "Total Tokens", "p95 Latency", costEnabled ? "Cost Est" : null].filter(Boolean).map(h => <th key={h} style={{ textAlign: "left", padding: "10px 14px", fontSize: 11, fontWeight: 700, color: T.textTert, textTransform: "uppercase", letterSpacing: "0.04em", background: T.bg, borderBottom: `1px solid ${T.border}` }}>{h}</th>)}</tr></thead>
            <tbody>{rows.map(u => {
              const rate = u.requests > 0 ? ((u.errors / u.requests) * 100).toFixed(1) : "0.0";
              return (
                <tr key={u.tenant_id} style={{ borderBottom: `1px solid ${T.border}` }}>
                  <td style={{ padding: "10px 14px", fontWeight: 600, color: T.text }}>{u.tenant_name || u.tenant_id}</td>
                  <td style={{ padding: "10px 14px", color: T.textSec }}>{(u.requests || 0).toLocaleString()}</td>
                  <td style={{ padding: "10px 14px", color: u.errors > 0 ? T.danger : T.textSec }}>{u.errors || 0}</td>
                  <td style={{ padding: "10px 14px" }}>
                    <span style={{ color: parseFloat(rate) > 5 ? T.danger : parseFloat(rate) > 1 ? T.warning : T.success, fontWeight: 600 }}>{rate}%</span>
                  </td>
                  <td style={{ padding: "10px 14px", color: T.textSec }}>{(u.prompt_tokens || 0).toLocaleString()}</td>
                  <td style={{ padding: "10px 14px", color: T.textSec }}>{(u.completion_tokens || 0).toLocaleString()}</td>
                  <td style={{ padding: "10px 14px", color: T.textSec, fontWeight: 600 }}>{(u.total_tokens || 0).toLocaleString()}</td>
                  <td style={{ padding: "10px 14px", color: T.textSec }}>{u.p95_latency_ms ? `${u.p95_latency_ms}ms` : "—"}</td>
                  {costEnabled && <td style={{ padding: "10px 14px", color: T.text, fontWeight: 600 }}>${(u.cost_est_sum || 0).toFixed(4)}</td>}
                </tr>
              );
            })}</tbody>
          </table>
          {rows.length === 0 && <div style={{ textAlign: "center", padding: "28px 0", color: T.textTert, fontSize: 13 }}>No usage data in this window</div>}
        </Card>
      )}
    </Anim>
  );
}

/* ═══════════════════ TEST TAB ═══════════════════ */
function TestTab({ baseUrl, token, toast }) {
  const [tenants, setTenants] = useState([]); const [selTenant, setSelTenant] = useState("");
  const [tenantKeyIds, setTenantKeyIds] = useState(new Set());
  const [apiKey, setApiKey] = useState("");
  const [keyError, setKeyError] = useState("");
  const [prompt, setPrompt] = useState("Explain enterprise AI governance in 3 sentences."); const [result, setResult] = useState(null); const [loading, setLoading] = useState(false);

  useEffect(() => { apiFetch(baseUrl, "/admin/tenants", { token }).then(r => { const t = normalizeTenants(r); setTenants(t); if (t.length) setSelTenant(t[0].id); }).catch(() => {}); }, []);

  // When tenant changes, clear the API key field and load that tenant's key IDs for validation
  useEffect(() => {
    setApiKey("");
    setKeyError("");
    setResult(null);
    if (selTenant) {
      apiFetch(baseUrl, `/admin/keys?tenant_id=${selTenant}`, { token })
        .then(r => { const k = normalizeKeys(r); setTenantKeyIds(new Set(k.map(ki => ki.id))); })
        .catch(() => setTenantKeyIds(new Set()));
    }
  }, [selTenant]);

  const validateKey = (key) => {
    if (!key || !key.startsWith("cp_")) {
      setKeyError("API keys start with cp_ — paste the full key shown at creation time.");
    } else {
      setKeyError("");
    }
  };

  const run = async () => {
    if (!apiKey.trim()) { toast("Paste a cp_… API key first", "error"); return; }
    setLoading(true); setResult(null);
    try {
      const res = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${apiKey.trim()}`,
          "X-Expected-Tenant-Id": selTenant,
        },
        body: JSON.stringify({ model: "default", messages: [{ role: "user", content: prompt }], max_tokens: 256 })
      });
      const data = await res.json();
      if (!res.ok && data?.error?.message?.includes("tenant mismatch")) {
        setKeyError("This API key belongs to a different tenant. Select the correct tenant or use the right key.");
      }
      setResult(data);
    } catch (e) { toast(e.message, "error"); } finally { setLoading(false); }
  };

  const resultColor = result && (result.error || result.choices === undefined) ? T.danger : T.success;

  return (
    <Anim>
      <Card style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <SectionLabel>Test Inference</SectionLabel>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <div>
            <div style={{ fontSize: 11, color: T.textTert, marginBottom: 4 }}>Tenant</div>
            <Select value={selTenant} onChange={setSelTenant} style={{ width: "100%" }}>{tenants.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}</Select>
          </div>
          <div>
            <div style={{ fontSize: 11, color: T.textTert, marginBottom: 4 }}>API Key <span style={{ color: T.textTert }}>(paste the cp_… key — shown once at creation)</span></div>
            <Input value={apiKey} onChange={v => { setApiKey(v); validateKey(v); }} placeholder="cp_…" />
            {keyError && <div style={{ fontSize: 11, color: T.danger, marginTop: 4 }}>{keyError}</div>}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: T.textTert, marginBottom: 4 }}>Prompt</div>
          <textarea value={prompt} onChange={e => setPrompt(e.target.value)} rows={4} style={{ width: "100%", padding: "9px 11px", fontSize: 13, border: `1px solid ${T.border}`, borderRadius: 8, outline: "none", resize: "vertical", fontFamily: "inherit", color: T.text, background: T.bg, boxSizing: "border-box" }} />
        </div>
        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <Btn onClick={run} disabled={loading || !apiKey.trim()}>{loading ? <><Spinner /> Running…</> : "Send Request"}</Btn>
        </div>
        <div>
          <div style={{ fontSize: 11, color: T.textTert, marginBottom: 4 }}>Response</div>
          <pre style={{ background: T.bg, border: `1px solid ${result ? (result.error ? T.dangerBorder : T.successBorder) : T.border}`, borderRadius: 10, padding: 14, fontSize: 12, color: result ? resultColor : T.textTert, overflow: "auto", height: 280, whiteSpace: "pre-wrap", margin: 0, fontFamily: "'JetBrains Mono', monospace" }}>
            {result ? JSON.stringify(result, null, 2) : "Send a request to see results"}
          </pre>
        </div>
      </Card>
    </Anim>
  );
}

/* ═══════════════════ MAIN APP ═══════════════════ */
export default function App() {
  const [connected, setConnected] = useState(false); const [baseUrl, setBaseUrl] = useState(""); const [token, setToken] = useState(""); const [tab, setTab] = useState("overview"); const [toastMsg, setToastMsg] = useState(null);
  const toast = useCallback((msg, type = "info") => setToastMsg({ msg, type }), []);
  useEffect(() => { const s = loadSaved(); if (s.baseUrl && s.token) { apiFetch(s.baseUrl, "/admin/tenants", { token: s.token }).then(() => { setBaseUrl(s.baseUrl); setToken(s.token); setConnected(true); }).catch(() => {}); } }, []);
  if (!connected) return (
    <div style={{ fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" }}>
      <ConnectScreen onConnect={(u, t) => { setBaseUrl(u); setToken(t); setConnected(true); }} />
      <style>{`@keyframes spin{to{transform:rotate(360deg)}} @keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}} @keyframes slideUp{from{transform:translateY(12px);opacity:0}to{transform:translateY(0);opacity:1}}`}</style>
    </div>
  );
  const disconnect = () => { localStorage.removeItem(STORAGE_KEY); setConnected(false); };
  return (
    <div style={{ fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", background: T.bg, minHeight: "100vh" }}>
      <header style={{ background: T.card, borderBottom: `1px solid ${T.border}`, padding: "0 20px" }}>
        <div style={{ maxWidth: 1280, margin: "0 auto", display: "flex", justifyContent: "space-between", alignItems: "center", height: 50 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}><div style={{ width: 7, height: 7, borderRadius: "50%", background: T.accent }} /><span style={{ fontSize: 15, fontWeight: 700, color: T.text }}>mirae</span></div>
            <div style={{ width: 1, height: 18, background: T.border }} />
            <Badge variant="info">{(() => { try { return new URL(baseUrl).hostname.split(".")[0].slice(0, 18); } catch { return "local"; } })()}</Badge>
          </div>
          <button onClick={disconnect} style={{ fontSize: 12, color: T.textTert, background: "none", border: "none", cursor: "pointer", fontFamily: "inherit" }}>Disconnect</button>
        </div>
      </header>
      <nav style={{ background: T.card, borderBottom: `1px solid ${T.border}`, padding: "0 20px" }}>
        <div style={{ maxWidth: 1280, margin: "0 auto", display: "flex", gap: 0, overflowX: "auto" }}>
          {TABS.map(t => <button key={t.id} onClick={() => setTab(t.id)} style={{ display: "flex", alignItems: "center", gap: 5, padding: "9px 12px", fontSize: 13, fontWeight: 600, color: tab === t.id ? T.accent : T.textSec, background: "none", border: "none", borderBottom: `2px solid ${tab === t.id ? T.accent : "transparent"}`, cursor: "pointer", whiteSpace: "nowrap", fontFamily: "inherit" }}><SVG d={t.icon} size={14} />{t.label}</button>)}
        </div>
      </nav>
      <main style={{ maxWidth: 1280, margin: "0 auto", padding: "22px 20px" }}>
        {tab === "overview"   && <OverviewTab baseUrl={baseUrl} token={token} />}
        {tab === "live"       && <LiveTab baseUrl={baseUrl} token={token} />}
        {tab === "dynamo"     && <DynamoTab baseUrl={baseUrl} token={token} />}
        {tab === "tenants"    && <TenantsTab baseUrl={baseUrl} token={token} toast={toast} />}
        {tab === "keys"       && <KeysTab baseUrl={baseUrl} token={token} toast={toast} />}
        {tab === "policy"     && <PolicyTab baseUrl={baseUrl} token={token} toast={toast} />}
        {tab === "audit"      && <AuditTab baseUrl={baseUrl} token={token} />}
        {tab === "usage"      && <UsageTab baseUrl={baseUrl} token={token} />}
        {tab === "test"       && <TestTab baseUrl={baseUrl} token={token} toast={toast} />}
      </main>
      {toastMsg && <Toast message={toastMsg.msg} type={toastMsg.type} onDone={() => setToastMsg(null)} />}
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes slideUp { from { transform: translateY(12px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
        * { box-sizing: border-box; }
        input[type=number]::-webkit-inner-spin-button { opacity: 0.4; }
        select { outline: none; }
        input:focus { border-color: ${T.accent} !important; outline: none; box-shadow: 0 0 0 3px ${T.accent}18; }
      `}</style>
    </div>
  );
}
