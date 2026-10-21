/* roboVLA console — shared helpers (design §7). Plain, readable JS:
   every page fetches the §9 JSON API and renders; SSE for live views. */
"use strict";

var $ = (sel, el = document) => el.querySelector(sel);
var $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

/* ---- API ----------------------------------------------------------------- */

async function api(path, opts = {}) {
  const isForm = opts.body instanceof FormData;
  const r = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(opts.headers ?? {}) },
    method: opts.method ?? (opts.body != null ? "POST" : "GET"),
    body: isForm ? opts.body : opts.body != null ? JSON.stringify(opts.body) : undefined,
  });
  if (r.status === 401) { location.href = "/ui/login.html"; throw new Error("auth"); }
  let data = null;
  try { data = await r.json(); } catch { /* non-JSON (file download) */ }
  if (!r.ok) {
    const msg = data && data.error ? data.error.message : `HTTP ${r.status}`;
    const err = new Error(msg);
    err.code = data && data.error && data.error.code;
    err.status = r.status;
    throw err;
  }
  return data;
}

/* identity: every page loads /api/auth/me and redirects to login if none */
async function me() {
  try { return await api("/api/auth/me"); }
  catch { return null; }
}

function badge(state) {
  const cls = ["running","ready","queued","done","failed","killed","uploading",
               "preparing","finishing","rejected","online","offline"]
              .includes(state) ? state : "";
  return `<span class="badge ${cls}">${state ?? "?"}</span>`;
}

function toast(msg, isErr = false) {
  let t = $("#toast");
  if (!t) { t = document.createElement("div"); t.id = "toast"; document.body.append(t); }
  t.textContent = msg;
  t.className = "show" + (isErr ? " err" : "");
  clearTimeout(t._h);
  t._h = setTimeout(() => (t.className = ""), 3500);
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g,
    c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmtEta(s) {
  if (s == null) return "—";
  if (s === Infinity) return "no ETA";
  s = Math.round(s);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}

function fmtBytes(n) {
  if (n == null) return "—";
  for (const [u, k] of [["GB", 1e9], ["MB", 1e6], ["KB", 1e3]])
    if (n >= k) return `${(n / k).toFixed(n >= 1e9 ? 0 : 1)}${u}`;
  return `${n}B`;
}

function fmtTs(rfc) {
  if (!rfc) return "—";
  const d = new Date(rfc);
  return Number.isNaN(d.getTime()) ? rfc : d.toISOString().replace("T", " ").slice(0, 16) + "Z";
}

/* ---- SSE ----------------------------------------------------------------- */
/* topic: job id or "class". onEvent(etype, data, seq). Returns close().
   Reconnects with Last-Event-ID semantics (EventSource is native). */
function sse(topic, onEvent) {
  const es = new EventSource(`/stream/${topic}`);
  const types = ["state", "metrics", "val", "log", "artifacts", "predict",
                 "queue", "slots", "node", "config", "dataset"];
  for (const t of types)
    es.addEventListener(t, e => {
      try { onEvent(t, JSON.parse(e.data), e.lastEventId); }
      catch { /* keep the stream alive on a bad frame */ }
    });
  es.onerror = () => { /* EventSource auto-reconnects; nothing to do */ };
  return () => es.close();
}

/* ---- sparkline (canvas, no deps) ------------------------------------------ */
function spark(canvas, series) {
  /* series: [{color, points: [[x,y],...]}] */
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 400, h = canvas.clientHeight || 90;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const s of series) for (const [x, y] of s.points) {
    xmin = Math.min(xmin, x); xmax = Math.max(xmax, x);
    ymin = Math.min(ymin, y); ymax = Math.max(ymax, y);
  }
  if (!isFinite(xmin)) return;
  if (xmin === xmax) { xmin -= 1; xmax += 1; }
  if (ymin === ymax) { ymin -= 1; ymax += 1; }
  const X = x => 4 + (x - xmin) / (xmax - xmin) * (w - 8);
  const Y = y => h - 4 - (y - ymin) / (ymax - ymin) * (h - 8);
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = 1.5; ctx.beginPath();
    s.points.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
    ctx.stroke();
  }
}

/* ---- nav ----------------------------------------------------------------- */
const PAGES = [
  ["index.html", "Dashboard"],
  ["datasets.html", "Datasets"],
  ["new-run.html", "New run"],
  ["class-board.html", "Class board"],
  ["slots.html", "Slots & robot"],
];

async function nav(active, user) {
  const el = $("#nav");
  if (!el) return;
  const links = PAGES.map(([p, t]) =>
    `<a href="/ui/${p}" class="${p === active ? "active" : ""}">${t}</a>`).join("");
  const teacher = user && user.role === "teacher"
    ? `<a href="/ui/teacher.html">Teacher</a><a href="/ui/accounts.html">Accounts</a>` : "";
  el.innerHTML =
    `<a class="brand" href="/ui/index.html">roboVLA</a>${links}${teacher}
     <span class="spacer"></span>
     <span class="who">${esc(user ? user.display_name : "")}</span>
     <a href="/ui/logout.html">log out</a>`;
}
