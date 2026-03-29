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
    try { const j = await res.json(); msg = j.detail || j.error || msg; } catch {}
    throw new Error(msg);
  }
  if (res.status === 204) return {};
  return res.json();
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
// Change 1: Persistent key reveal modal with copy button and explicit dismiss
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

/* ═══════════════════ OVERVIEW TAB ═══════════════════ */
function OverviewTab({ baseUrl, token }) {
  const [data, setData] = useState(null); const [loading, setLoading] = useState(true);
  useEffect(() => {
    Promise.all([
      apiFetch(baseUrl, "/admin/tenants", { token }).catch(() => []),
      apiFetch(baseUrl, "/admin/usage/summary", { token }).catch(() => ({})),
      apiFetch(baseUrl, "/version").catch(() => ({})),
    ]).then(([tenants, usage, version]) => { setData({ tenants, usage, version }); setLoading(false); });
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
          {[
            ["Gateway Version", version.version || "—"],
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

/* ═══════════════════ LIVE TAB ═══════════════════ */
function LiveTab({ baseUrl, token }) {
  const [metrics, setMetrics] = useState(null); const [history, setHistory] = useState([]); const histRef = useRef([]);
  useEffect(() => {
    const poll = async () => {
      try {
        const m = await apiFetch(baseUrl, "/admin/dynamo/metrics", { token });
        setMetrics(m);
        const point = { t: Date.now(), inflight: m.total_inflight || 0, queued: m.total_queued || 0, tps: m.tokens_per_second || 0 };
        histRef.current = [...histRef.current.slice(-59), point];
        setHistory([...histRef.current]);
      } catch {}
    };
    poll(); const id = setInterval(poll, 3000); return () => clearInterval(id);
  }, []);
  const Sparkline = ({ data, key_, color }) => {
    if (data.length < 2) return <div style={{ height: 36, background: T.bg, borderRadius: 6 }} />;
    const vals = data.map(d => d[key_]); const max = Math.max(...vals, 1);
    const pts = vals.map((v, i) => `${(i / (vals.length - 1)) * 200},${36 - (v / max) * 34}`).join(" ");
    return <svg viewBox={`0 0 200 36`} style={{ width: "100%", height: 36 }}><polyline points={pts} fill="none" stroke={color} strokeWidth={1.5} /><polyline points={`0,36 ${pts} 200,36`} fill={color + "20"} stroke="none" /></svg>;
  };
  return (
    <Anim>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 14, marginBottom: 20 }}>
        {[
          { label: "In-flight", key_: "total_inflight", color: T.accent, icon: "M13 10V3L4 14h7v7l9-11h-7z" },
          { label: "Queued", key_: "total_queued", color: T.accent2, icon: "M4 6h16M4 10h16M4 14h16M4 18h16" },
          { label: "Tokens/sec", key_: "tps", color: T.info, icon: "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" },
        ].map(kpi => (
          <Card key={kpi.label}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: T.textTert, textTransform: "uppercase", letterSpacing: "0.05em" }}>{kpi.label}</span>
              <div style={{ display: "flex", gap: 4, alignItems: "center" }}><div style={{ width: 6, height: 6, borderRadius: "50%", background: T.success, animation: "pulse 2s infinite" }} /><span style={{ fontSize: 10, color: T.textTert }}>LIVE</span></div>
            </div>
            <div style={{ fontSize: 28, fontWeight: 700, color: T.text, marginBottom: 8 }}>{metrics ? (metrics[kpi.key_] ?? 0) : "—"}</div>
            <Sparkline data={history} key_={kpi.key_} color={kpi.color} />
          </Card>
        ))}
      </div>
      {metrics?.tenant_breakdown && (
        <Card>
          <SectionLabel>Per-Tenant Activity</SectionLabel>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead><tr>{["Tenant","In-flight","Queued","Tokens/sec","Status"].map(h => <th key={h} style={{ textAlign: "left", padding: "6px 10px", fontSize: 11, fontWeight: 700, color: T.textTert, textTransform: "uppercase", letterSpacing: "0.04em", borderBottom: `1px solid ${T.border}` }}>{h}</th>)}</tr></thead>
            <tbody>{Object.entries(metrics.tenant_breakdown).map(([tid, td]) => (
              <tr key={tid}>{[tid, td.inflight ?? 0, td.queued ?? 0, td.tps ?? 0].map((v, i) => <td key={i} style={{ padding: "8px 10px", borderBottom: `1px solid ${T.border}`, color: i === 0 ? T.text : T.textSec, fontWeight: i === 0 ? 600 : 400 }}>{v}</td>)}<td style={{ padding: "8px 10px", borderBottom: `1px solid ${T.border}` }}><Badge variant="success">Active</Badge></td></tr>
            ))}</tbody>
          </table>
        </Card>
      )}
      {!metrics && <div style={{ textAlign: "center", padding: 40, color: T.textTert, fontSize: 13 }}>Connecting to live metrics…<br /><span style={{ fontSize: 11 }}>Polling /admin/dynamo/metrics every 3s</span></div>}
    </Anim>
  );
}

/* ═══════════════════ INFRASTRUCTURE TAB (Change 2) ═══════════════════ */
// Added p50/p95/p99 latency, queue depth, KV cache utilization, TTFT, separate input/output TPS
function DynamoTab({ baseUrl, token }) {
  const [frontend, setFrontend] = useState(null); const [worker, setWorker] = useState(null); const [loading, setLoading] = useState(true);
  useEffect(() => {
    const poll = async () => {
      try {
        const [fe, wk] = await Promise.all([
          apiFetch(baseUrl, "/admin/dynamo/frontend/stats", { token }).catch(() => null),
          apiFetch(baseUrl, "/admin/dynamo/worker/stats", { token }).catch(() => null),
        ]);
        setFrontend(fe); setWorker(wk); setLoading(false);
      } catch { setLoading(false); }
    };
    poll(); const id = setInterval(poll, 5000); return () => clearInterval(id);
  }, []);
  const StatRow = ({ label, value, unit, tooltip }) => (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "9px 0", borderBottom: `1px solid ${T.border}` }}>
      <div>
        <span style={{ fontSize: 13, color: T.textSec }}>{label}</span>
        {tooltip && <div style={{ fontSize: 11, color: T.textTert, marginTop: 1 }}>{tooltip}</div>}
      </div>
      <span style={{ fontSize: 13, fontWeight: 700, color: T.text }}>{value ?? "—"}{unit && <span style={{ fontSize: 11, fontWeight: 400, color: T.textTert, marginLeft: 3 }}>{unit}</span>}</span>
    </div>
  );
  const pct = (v) => v != null ? `${(v * 100).toFixed(1)}%` : null;
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
            <Badge variant={frontend ? "success" : "warning"}>{frontend ? "Online" : "Polling…"}</Badge>
          </div>
          <div style={{ background: T.accentLight, border: `1px solid #99f6e4`, borderRadius: 8, padding: "8px 12px", marginBottom: 14, fontSize: 12, color: T.accent }}>
            Routes incoming requests, manages KV store discovery, and handles request migration across workers. Think of it as the load balancer layer.
          </div>
          <StatRow label="Active Connections" value={frontend?.active_connections} tooltip="Current open HTTP connections" />
          <StatRow label="Requests In-flight" value={frontend?.inflight_requests} tooltip="Requests currently being routed" />
          <StatRow label="Request Migration" value={frontend?.migration_enabled ? "Enabled" : "Disabled"} tooltip="Whether requests can migrate between workers" />
          <StatRow label="KV Store Backend" value={frontend?.kv_backend || "File"} tooltip="Discovery backend for worker coordination" />
          <StatRow label="Network Mode" value={frontend?.network_mode || "TCP"} />
          <StatRow label="Uptime" value={frontend?.uptime_seconds ? `${Math.floor(frontend.uptime_seconds / 60)}m` : null} />
          <div style={{ marginTop: 14 }}>
            <SectionLabel>Request Latency</SectionLabel>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
              {[["p50", frontend?.latency_p50_ms], ["p95", frontend?.latency_p95_ms], ["p99", frontend?.latency_p99_ms]].map(([p, v]) => (
                <div key={p} style={{ background: T.bg, borderRadius: 8, padding: "10px", textAlign: "center", border: `1px solid ${T.border}` }}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: T.textTert, textTransform: "uppercase", marginBottom: 4 }}>{p}</div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: T.text }}>{v != null ? v.toFixed(0) : "—"}</div>
                  <div style={{ fontSize: 10, color: T.textTert }}>ms</div>
                </div>
              ))}
            </div>
          </div>
        </Card>

        {/* vLLM Worker */}
        <Card>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
            <div>
              <div style={{ fontWeight: 700, fontSize: 15, color: T.text, marginBottom: 3 }}>vLLM Worker</div>
              <div style={{ fontSize: 12, color: T.textTert }}>GPU inference process · port 8081</div>
            </div>
            <Badge variant={worker ? "success" : "warning"}>{worker ? "Online" : "Polling…"}</Badge>
          </div>
          <div style={{ background: T.accent2Light, border: `1px solid #c7d2fe`, borderRadius: 8, padding: "8px 12px", marginBottom: 14, fontSize: 12, color: T.accent2 }}>
            Loads and runs the model on GPU. Manages KV cache, tokenization, and generation. Exposes Prometheus metrics for throughput and utilization.
          </div>
          <StatRow label="GPU Utilization" value={worker?.gpu_utilization_pct != null ? `${worker.gpu_utilization_pct.toFixed(1)}%` : null} tooltip="Current GPU compute usage" />
          <StatRow label="KV Cache Utilization" value={pct(worker?.kv_cache_utilization)} tooltip="Fraction of KV cache in use — high values may cause evictions" />
          <StatRow label="Queue Depth" value={worker?.queue_depth} tooltip="Requests waiting for GPU time" />
          <StatRow label="Running Sequences" value={worker?.num_running_seqs} tooltip="Sequences currently being generated" />
          <div style={{ marginTop: 14 }}>
            <SectionLabel>Throughput</SectionLabel>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
              {[
                ["Input TPS", worker?.prompt_tokens_per_sec, "tok/s"],
                ["Output TPS", worker?.generation_tokens_per_sec, "tok/s"],
                ["TTFT", worker?.time_to_first_token_ms, "ms"],
              ].map(([label, v, unit]) => (
                <div key={label} style={{ background: T.bg, borderRadius: 8, padding: "10px", textAlign: "center", border: `1px solid ${T.border}` }}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: T.textTert, textTransform: "uppercase", marginBottom: 4 }}>{label}</div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: T.text }}>{v != null ? v.toFixed(0) : "—"}</div>
                  <div style={{ fontSize: 10, color: T.textTert }}>{unit}</div>
                </div>
              ))}
            </div>
          </div>
          <div style={{ marginTop: 14 }}>
            <SectionLabel>Request Duration</SectionLabel>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
              {[["p50", worker?.req_duration_p50_ms], ["p95", worker?.req_duration_p95_ms], ["p99", worker?.req_duration_p99_ms]].map(([p, v]) => (
                <div key={p} style={{ background: T.bg, borderRadius: 8, padding: "10px", textAlign: "center", border: `1px solid ${T.border}` }}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: T.textTert, textTransform: "uppercase", marginBottom: 4 }}>{p}</div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: T.text }}>{v != null ? v.toFixed(0) : "—"}</div>
                  <div style={{ fontSize: 10, color: T.textTert }}>ms</div>
                </div>
              ))}
            </div>
          </div>
        </Card>
      </div>
    </Anim>
  );
}

/* ═══════════════════ TENANTS TAB (Change 4) ═══════════════════ */
// Added configurable parameters: rate limits, budget, model access, concurrency, region
function TenantsTab({ baseUrl, token, toast }) {
  const [tenants, setTenants] = useState([]); const [loading, setLoading] = useState(true);
  const [showAdd, setShowAdd] = useState(false);
  const [form, setForm] = useState({
    name: "", tier: "standard",
    max_concurrent: "", requests_per_minute: "", tokens_per_minute: "",
    budget_limit: "", budget_window: "month",
    model_access: "all", region: "us-east-1",
    max_context_tokens: "", max_output_tokens: "",
  });
  const [saving, setSaving] = useState(false);

  const load = () => apiFetch(baseUrl, "/admin/tenants", { token }).then(setTenants).catch(() => {}).finally(() => setLoading(false));
  useEffect(() => { load(); }, []);

  const TIERS = ["starter", "standard", "pro", "enterprise"];
  const PRESETS = {
    starter:    { max_concurrent: 5,   requests_per_minute: 60,  tokens_per_minute: 50000,  max_context_tokens: 8192,  max_output_tokens: 1024 },
    standard:   { max_concurrent: 20,  requests_per_minute: 300, tokens_per_minute: 200000, max_context_tokens: 32768, max_output_tokens: 4096 },
    pro:        { max_concurrent: 50,  requests_per_minute: 1000, tokens_per_minute: 1000000, max_context_tokens: 131072, max_output_tokens: 8192 },
    enterprise: { max_concurrent: 200, requests_per_minute: 5000, tokens_per_minute: 10000000, max_context_tokens: 131072, max_output_tokens: 8192 },
  };

  const applyTier = (tier) => {
    const p = PRESETS[tier];
    setForm(f => ({ ...f, tier, ...Object.fromEntries(Object.entries(p).map(([k, v]) => [k, String(v)])) }));
  };

  const add = async () => {
    if (!form.name.trim()) { toast("Name is required", "error"); return; }
    setSaving(true);
    try {
      const body = { name: form.name.trim(), tier: form.tier, region: form.region, model_access: form.model_access };
      const nums = ["max_concurrent","requests_per_minute","tokens_per_minute","budget_limit","max_context_tokens","max_output_tokens"];
      nums.forEach(k => { if (form[k]) body[k] = Number(form[k]); });
      if (form.budget_limit) { body.budget_window = form.budget_window; }
      await apiFetch(baseUrl, "/admin/tenants", { token, method: "POST", body });
      toast("Tenant created"); setShowAdd(false);
      setForm({ name: "", tier: "standard", max_concurrent: "", requests_per_minute: "", tokens_per_minute: "", budget_limit: "", budget_window: "month", model_access: "all", region: "us-east-1", max_context_tokens: "", max_output_tokens: "" });
      load();
    } catch (e) { toast(e.message, "error"); } finally { setSaving(false); }
  };

  return (
    <Anim>
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
                {[
                  ["Max Concurrent", "max_concurrent", "reqs"],
                  ["Requests/min", "requests_per_minute", "rpm"],
                  ["Tokens/min", "tokens_per_minute", "tpm"],
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
                {[
                  ["Max Context Tokens", "max_context_tokens"],
                  ["Max Output Tokens", "max_output_tokens"],
                ].map(([label, key]) => (
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

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <div>
                <SectionLabel>Model Access</SectionLabel>
                <Select value={form.model_access} onChange={v => setForm(f => ({ ...f, model_access: v }))}>
                  <option value="all">All Models</option>
                  <option value="restricted">Restricted (whitelist)</option>
                </Select>
              </div>
              <div>
                <SectionLabel>Region Tag</SectionLabel>
                <Select value={form.region} onChange={v => setForm(f => ({ ...f, region: v }))}>
                  {["us-east-1","us-west-2","eu-west-1","ap-southeast-1"].map(r => <option key={r} value={r}>{r}</option>)}
                </Select>
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
            <thead><tr>{["Name","ID","Tier","Region","Concurrency","RPM","Status",""].map(h => <th key={h} style={{ textAlign: "left", padding: "10px 14px", fontSize: 11, fontWeight: 700, color: T.textTert, textTransform: "uppercase", letterSpacing: "0.04em", borderBottom: `1px solid ${T.border}`, background: T.bg }}>{h}</th>)}</tr></thead>
            <tbody>{tenants.map(t => (
              <tr key={t.id} style={{ borderBottom: `1px solid ${T.border}` }}>
                <td style={{ padding: "10px 14px", fontWeight: 600, color: T.text }}>{t.name}</td>
                <td style={{ padding: "10px 14px", color: T.textTert, fontFamily: "monospace", fontSize: 11 }}>{t.id?.slice(0, 12)}…</td>
                <td style={{ padding: "10px 14px" }}><Badge variant="accent">{t.tier || "standard"}</Badge></td>
                <td style={{ padding: "10px 14px", color: T.textSec }}>{t.region || "—"}</td>
                <td style={{ padding: "10px 14px", color: T.textSec }}>{t.max_concurrent ?? "—"}</td>
                <td style={{ padding: "10px 14px", color: T.textSec }}>{t.requests_per_minute ?? "—"}</td>
                <td style={{ padding: "10px 14px" }}><Badge variant="success">Active</Badge></td>
                <td style={{ padding: "10px 14px" }}></td>
              </tr>
            ))}</tbody>
          </table>
          {tenants.length === 0 && <div style={{ textAlign: "center", padding: "30px 0", color: T.textTert, fontSize: 13 }}>No tenants yet — add one to get started</div>}
        </Card>
      )}
    </Anim>
  );
}

/* ═══════════════════ KEYS TAB (Change 1) ═══════════════════ */
// API key copy modal — persistent, explicit dismiss, copy-first UX
function KeysTab({ baseUrl, token, toast }) {
  const [tenants, setTenants] = useState([]); const [keys, setKeys] = useState([]);
  const [selTenant, setSelTenant] = useState(""); const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false); const [keyName, setKeyName] = useState("");
  const [revealKey, setRevealKey] = useState(null); // Change 1: show modal instead of toast

  const loadTenants = () => apiFetch(baseUrl, "/admin/tenants", { token }).then(t => { setTenants(t); if (t.length && !selTenant) setSelTenant(t[0].id); }).catch(() => {});
  const loadKeys = (tid) => { if (!tid) return; apiFetch(baseUrl, `/admin/tenants/${tid}/keys`, { token }).then(setKeys).catch(() => setKeys([])).finally(() => setLoading(false)); };
  useEffect(() => { loadTenants(); }, []);
  useEffect(() => { if (selTenant) loadKeys(selTenant); }, [selTenant]);

  const create = async () => {
    if (!selTenant) return;
    setCreating(true);
    try {
      const res = await apiFetch(baseUrl, `/admin/tenants/${selTenant}/keys`, { token, method: "POST", body: { name: keyName || `key-${Date.now()}` } });
      setRevealKey(res.key || res.api_key || res.token || JSON.stringify(res)); // Change 1: open modal
      setKeyName(""); loadKeys(selTenant);
    } catch (e) { toast(e.message, "error"); } finally { setCreating(false); }
  };

  const revoke = async (kid) => {
    if (!confirm("Revoke this key? This cannot be undone.")) return;
    try { await apiFetch(baseUrl, `/admin/tenants/${selTenant}/keys/${kid}`, { token, method: "DELETE" }); toast("Key revoked"); loadKeys(selTenant); }
    catch (e) { toast(e.message, "error"); }
  };

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
          <Card>
            <SectionLabel>New API Key</SectionLabel>
            <div style={{ display: "flex", gap: 8 }}>
              <Input value={keyName} onChange={setKeyName} placeholder="Key name (optional)" style={{ flex: 1 }} />
              <Btn onClick={create} disabled={creating || !selTenant}>{creating ? <><Spinner /> Creating…</> : <><SVG d="M12 4v16m8-8H4" size={13} /> Create Key</>}</Btn>
            </div>
          </Card>
          <Card style={{ padding: 0 }}>
            <div style={{ padding: "12px 16px", borderBottom: `1px solid ${T.border}` }}>
              <span style={{ fontSize: 13, fontWeight: 600, color: T.text }}>Active Keys</span>
              <span style={{ fontSize: 12, color: T.textTert, marginLeft: 8 }}>({keys.length})</span>
            </div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead><tr>{["Name","Key (masked)","Created","Last Used",""].map(h => <th key={h} style={{ textAlign: "left", padding: "8px 16px", fontSize: 11, fontWeight: 700, color: T.textTert, textTransform: "uppercase", letterSpacing: "0.04em", background: T.bg, borderBottom: `1px solid ${T.border}` }}>{h}</th>)}</tr></thead>
              <tbody>{keys.map(k => (
                <tr key={k.id} style={{ borderBottom: `1px solid ${T.border}` }}>
                  <td style={{ padding: "10px 16px", fontWeight: 600, color: T.text }}>{k.name || "—"}</td>
                  <td style={{ padding: "10px 16px", fontFamily: "monospace", color: T.textTert, fontSize: 12 }}>{k.key_prefix || k.masked || "sk-…"}</td>
                  <td style={{ padding: "10px 16px", color: T.textSec }}>{k.created_at ? fmtTime(k.created_at) : "—"}</td>
                  <td style={{ padding: "10px 16px", color: T.textSec }}>{k.last_used_at ? fmtTime(k.last_used_at) : "Never"}</td>
                  <td style={{ padding: "10px 16px" }}><Btn variant="danger" small onClick={() => revoke(k.id)}>Revoke</Btn></td>
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

/* ═══════════════════ POLICY TAB (Changes 5 & 6) ═══════════════════ */
// Change 5: Dynamo inference hints REMOVED — moved to infra config
// Change 6: Budget is now a Spend Policy section within Policies
function PolicyTab({ baseUrl, token, toast }) {
  const [tenants, setTenants] = useState([]); const [selTenant, setSelTenant] = useState("");
  const [policy, setPolicy] = useState(null); const [saving, setSaving] = useState(false); const [loading, setLoading] = useState(false);
  useEffect(() => { apiFetch(baseUrl, "/admin/tenants", { token }).then(t => { setTenants(t); if (t.length) setSelTenant(t[0].id); }).catch(() => {}); }, []);
  useEffect(() => {
    if (!selTenant) return; setLoading(true);
    apiFetch(baseUrl, `/admin/tenants/${selTenant}`, { token }).then(t => {
      setPolicy({
        max_concurrent: t.max_concurrent ?? "", requests_per_minute: t.requests_per_minute ?? "",
        tokens_per_minute: t.tokens_per_minute ?? "", max_context_tokens: t.max_context_tokens ?? "",
        max_output_tokens: t.max_output_tokens ?? "",
        budget_limit: t.budget_limit ?? "", budget_window: t.budget_window ?? "month",
        budget_alert_threshold: t.budget_alert_threshold ?? "80",
      });
    }).catch(() => {}).finally(() => setLoading(false));
  }, [selTenant]);

  const save = async () => {
    setSaving(true);
    try {
      const body = {};
      Object.entries(policy).forEach(([k, v]) => { if (v !== "") body[k] = isNaN(v) ? v : Number(v); });
      await apiFetch(baseUrl, `/admin/tenants/${selTenant}`, { token, method: "PATCH", body });
      toast("Policy saved");
    } catch (e) { toast(e.message, "error"); } finally { setSaving(false); }
  };

  const PRESETS = [
    { label: "Conservative", max_concurrent: 5, requests_per_minute: 60, tokens_per_minute: 50000, max_context_tokens: 8192, max_output_tokens: 1024 },
    { label: "Standard", max_concurrent: 20, requests_per_minute: 300, tokens_per_minute: 200000, max_context_tokens: 32768, max_output_tokens: 4096 },
    { label: "Pro", max_concurrent: 50, requests_per_minute: 1000, tokens_per_minute: 1000000, max_context_tokens: 131072, max_output_tokens: 8192 },
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
              {/* Presets */}
              <Card>
                <SectionLabel>Presets</SectionLabel>
                <div style={{ display: "flex", gap: 8 }}>
                  {PRESETS.map(p => (
                    <Btn key={p.label} variant="secondary" small onClick={() => setPolicy(prev => ({ ...prev, max_concurrent: p.max_concurrent, requests_per_minute: p.requests_per_minute, tokens_per_minute: p.tokens_per_minute, max_context_tokens: p.max_context_tokens, max_output_tokens: p.max_output_tokens }))}>{p.label}</Btn>
                  ))}
                </div>
              </Card>

              {/* Rate Limits */}
              <Card>
                <SectionLabel>Rate Limits</SectionLabel>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
                  <Field label="Max Concurrent" key_="max_concurrent" unit="reqs" />
                  <Field label="Requests/min" key_="requests_per_minute" unit="rpm" />
                  <Field label="Tokens/min" key_="tokens_per_minute" unit="tpm" />
                </div>
              </Card>

              {/* Token Limits */}
              <Card>
                <SectionLabel>Token Limits</SectionLabel>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  <Field label="Max Context Tokens" key_="max_context_tokens" />
                  <Field label="Max Output Tokens" key_="max_output_tokens" />
                </div>
              </Card>

              {/* Change 6: Spend Policy merged into Policies tab */}
              <Card>
                <div style={{ display: "flex", justify: "space-between", alignItems: "center", marginBottom: 12 }}>
                  <SectionLabel>Spend Policy</SectionLabel>
                </div>
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

/* ═══════════════════ AUDIT TAB (Change 7) ═══════════════════ */
// Change 7: Split into "Inference Runs" and "Admin Log" sub-tabs
function AuditTab({ baseUrl, token }) {
  const [subTab, setSubTab] = useState("runs");
  const [logs, setLogs] = useState([]); const [loading, setLoading] = useState(true);
  const [tenants, setTenants] = useState([]); const [filterTenant, setFilterTenant] = useState("");
  const [filterWindow, setFilterWindow] = useState("24h");

  useEffect(() => { apiFetch(baseUrl, "/admin/tenants", { token }).then(setTenants).catch(() => {}); }, []);
  useEffect(() => {
    setLoading(true);
    const path = subTab === "runs"
      ? `/admin/audit/runs?window=${filterWindow}${filterTenant ? `&tenant_id=${filterTenant}` : ""}`
      : `/admin/audit/admin?window=${filterWindow}${filterTenant ? `&tenant_id=${filterTenant}` : ""}`;
    apiFetch(baseUrl, path, { token }).then(setLogs).catch(() => setLogs([])).finally(() => setLoading(false));
  }, [subTab, filterTenant, filterWindow]);

  const actionColors = {
    create: "success", created: "success", update: "info", updated: "info", patch: "info",
    delete: "danger", deleted: "danger", revoke: "danger", revoked: "danger",
    budget: "warning", budget_exceeded: "warning", budget_alert: "warning",
    login: "accent", connect: "accent",
  };
  const getVariant = (action = "") => actionColors[action.toLowerCase()] || "default";

  return (
    <Anim>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div style={{ display: "flex", gap: 0, border: `1px solid ${T.border}`, borderRadius: 9, overflow: "hidden" }}>
          {[
            { id: "runs", label: "Inference Runs", icon: "M13 10V3L4 14h7v7l9-11h-7z" },
            { id: "admin", label: "Admin Changes", icon: "M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" },
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
            {["1h","6h","24h","7d","30d"].map(w => <option key={w} value={w}>{w}</option>)}
          </Select>
        </div>
      </div>

      {subTab === "runs" && (
        <div style={{ marginBottom: 12, background: T.infoBg, border: `1px solid ${T.infoBorder}`, borderRadius: 8, padding: "8px 14px", fontSize: 12, color: T.info }}>
          High-volume per-request logs. Use for debugging latency, tracing individual requests, and per-tenant usage attribution.
        </div>
      )}
      {subTab === "admin" && (
        <div style={{ marginBottom: 12, background: T.warningBg, border: `1px solid ${T.warningBorder}`, borderRadius: 8, padding: "8px 14px", fontSize: 12, color: T.warning }}>
          Low-volume, high-importance log of all admin actions — tenant changes, key revocations, policy updates. Maps directly to SOC 2 change management requirements.
        </div>
      )}

      <Card style={{ padding: 0 }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr>
              {(subTab === "runs"
                ? ["Time","Tenant","Request ID","Model","Tokens","Duration","Status"]
                : ["Time","Actor","Action","Target","Details",""]
              ).map(h => <th key={h} style={{ textAlign: "left", padding: "9px 14px", fontSize: 11, fontWeight: 700, color: T.textTert, textTransform: "uppercase", letterSpacing: "0.04em", background: T.bg, borderBottom: `1px solid ${T.border}` }}>{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={7} style={{ textAlign: "center", padding: 32, color: T.textTert }}><Spinner size={18} /></td></tr>
            ) : logs.length === 0 ? (
              <tr><td colSpan={7} style={{ textAlign: "center", padding: 32, color: T.textTert, fontSize: 13 }}>No {subTab === "runs" ? "inference runs" : "admin changes"} in this window</td></tr>
            ) : logs.map((log, i) => (
              subTab === "runs" ? (
                <tr key={i} style={{ borderBottom: `1px solid ${T.border}` }}>
                  <td style={{ padding: "9px 14px", color: T.textSec, fontSize: 11 }}>{fmtTime(log.ts || log.timestamp)}</td>
                  <td style={{ padding: "9px 14px", fontWeight: 600, color: T.text }}>{log.tenant_name || log.tenant_id || "—"}</td>
                  <td style={{ padding: "9px 14px", fontFamily: "monospace", color: T.textTert, fontSize: 11 }}>{(log.request_id || "—").slice(0, 12)}</td>
                  <td style={{ padding: "9px 14px", color: T.textSec }}>{log.model || "—"}</td>
                  <td style={{ padding: "9px 14px", color: T.textSec }}>{log.total_tokens ?? "—"}</td>
                  <td style={{ padding: "9px 14px", color: T.textSec }}>{log.duration_ms != null ? `${log.duration_ms}ms` : "—"}</td>
                  <td style={{ padding: "9px 14px" }}><Badge variant={log.error ? "danger" : "success"}>{log.error ? "Error" : "OK"}</Badge></td>
                </tr>
              ) : (
                <tr key={i} style={{ borderBottom: `1px solid ${T.border}` }}>
                  <td style={{ padding: "9px 14px", color: T.textSec, fontSize: 11 }}>{fmtTime(log.ts || log.timestamp)}</td>
                  <td style={{ padding: "9px 14px", fontWeight: 600, color: T.text }}>{log.actor || "admin"}</td>
                  <td style={{ padding: "9px 14px" }}><Badge variant={getVariant(log.action)}>{log.action}</Badge></td>
                  <td style={{ padding: "9px 14px", color: T.textSec }}>{log.target_type || "—"}</td>
                  <td style={{ padding: "9px 14px", color: T.textTert, fontSize: 12, maxWidth: 240, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{log.details || log.description || "—"}</td>
                  <td style={{ padding: "9px 14px" }}></td>
                </tr>
              )
            ))}
          </tbody>
        </table>
      </Card>
    </Anim>
  );
}

/* ═══════════════════ USAGE TAB ═══════════════════ */
function UsageTab({ baseUrl, token }) {
  const [tenants, setTenants] = useState([]); const [usage, setUsage] = useState({}); const [loading, setLoading] = useState(true);
  useEffect(() => {
    apiFetch(baseUrl, "/admin/tenants", { token }).then(async ts => {
      setTenants(ts);
      const entries = await Promise.all(ts.map(t => apiFetch(baseUrl, `/admin/tenants/${t.id}/usage`, { token }).then(u => [t.id, u]).catch(() => [t.id, {}])));
      setUsage(Object.fromEntries(entries)); setLoading(false);
    }).catch(() => setLoading(false));
  }, []);
  return (
    <Anim>
      {loading ? <div style={{ textAlign: "center", padding: 40 }}><Spinner size={20} /></div> : (
        <Card style={{ padding: 0 }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead><tr>{["Tenant","Total Requests","Prompt Tokens","Completion Tokens","Total Cost","Last Activity"].map(h => <th key={h} style={{ textAlign: "left", padding: "10px 14px", fontSize: 11, fontWeight: 700, color: T.textTert, textTransform: "uppercase", letterSpacing: "0.04em", background: T.bg, borderBottom: `1px solid ${T.border}` }}>{h}</th>)}</tr></thead>
            <tbody>{tenants.map(t => { const u = usage[t.id] || {}; return (
              <tr key={t.id} style={{ borderBottom: `1px solid ${T.border}` }}>
                <td style={{ padding: "10px 14px", fontWeight: 600, color: T.text }}>{t.name}</td>
                <td style={{ padding: "10px 14px", color: T.textSec }}>{(u.total_requests || 0).toLocaleString()}</td>
                <td style={{ padding: "10px 14px", color: T.textSec }}>{(u.prompt_tokens || 0).toLocaleString()}</td>
                <td style={{ padding: "10px 14px", color: T.textSec }}>{(u.completion_tokens || 0).toLocaleString()}</td>
                <td style={{ padding: "10px 14px", color: T.text, fontWeight: 600 }}>${(u.total_cost || 0).toFixed(4)}</td>
                <td style={{ padding: "10px 14px", color: T.textTert, fontSize: 11 }}>{u.last_request_at ? fmtTime(u.last_request_at) : "Never"}</td>
              </tr>
            ); })}</tbody>
          </table>
          {tenants.length === 0 && <div style={{ textAlign: "center", padding: "28px 0", color: T.textTert, fontSize: 13 }}>No usage data yet</div>}
        </Card>
      )}
    </Anim>
  );
}

/* ═══════════════════ TEST TAB ═══════════════════ */
function TestTab({ baseUrl, toast }) {
  const [tenants, setTenants] = useState([]); const [selTenant, setSelTenant] = useState("");
  const [keys, setKeys] = useState([]); const [selKey, setSelKey] = useState("");
  const [prompt, setPrompt] = useState("Explain enterprise AI governance in 3 sentences."); const [result, setResult] = useState(null); const [loading, setLoading] = useState(false);
  useEffect(() => { apiFetch(baseUrl, "/admin/tenants", {}).then(t => { setTenants(t); if (t.length) setSelTenant(t[0].id); }).catch(() => {}); }, []);
  useEffect(() => { if (selTenant) apiFetch(baseUrl, `/admin/tenants/${selTenant}/keys`, {}).then(k => { setKeys(k); if (k.length) setSelKey(k[0].id); }).catch(() => setKeys([])); }, [selTenant]);
  const run = async () => {
    setLoading(true); setResult(null);
    try {
      const apiKey = keys.find(k => k.id === selKey)?.raw_key || selKey;
      const res = await fetch(`${baseUrl}/v1/chat/completions`, { method: "POST", headers: { "Content-Type": "application/json", "Authorization": `Bearer ${apiKey}` }, body: JSON.stringify({ model: "default", messages: [{ role: "user", content: prompt }], max_tokens: 256 }) });
      const data = await res.json(); setResult(data);
    } catch (e) { toast(e.message, "error"); } finally { setLoading(false); }
  };
  return (
    <Anim>
      <Card style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <SectionLabel>Test Inference</SectionLabel>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <div><SectionLabel>Tenant</SectionLabel><Select value={selTenant} onChange={setSelTenant} style={{ width: "100%" }}>{tenants.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}</Select></div>
          <div><SectionLabel>API Key</SectionLabel><Select value={selKey} onChange={setSelKey} style={{ width: "100%" }}>{keys.map(k => <option key={k.id} value={k.id}>{k.name || k.id}</option>)}</Select></div>
        </div>
        <div><SectionLabel>Prompt</SectionLabel><textarea value={prompt} onChange={e => setPrompt(e.target.value)} rows={4} style={{ width: "100%", padding: "9px 11px", fontSize: 13, border: `1px solid ${T.border}`, borderRadius: 8, outline: "none", resize: "vertical", fontFamily: "inherit", color: T.text, background: T.bg, boxSizing: "border-box" }} /></div>
        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <Btn onClick={run} disabled={loading}>{loading ? <><Spinner /> Running…</> : "Send Request"}</Btn>
        </div>
        <div><SectionLabel>Response</SectionLabel><pre style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: 10, padding: 14, fontSize: 12, color: T.text, overflow: "auto", height: 280, whiteSpace: "pre-wrap", margin: 0, fontFamily: "'JetBrains Mono', monospace" }}>{result ? JSON.stringify(result, null, 2) : "Send a request to see results"}</pre></div>
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
        {tab === "test"       && <TestTab baseUrl={baseUrl} toast={toast} />}
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
