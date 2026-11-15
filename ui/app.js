/* roboVLA console — shared helpers (design §7). Plain, readable JS:
   every page fetches the §9 JSON API and renders; SSE for live views.
   No build step, no framework: the class reads this code. */
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

/* ---- formatting ----------------------------------------------------------- */

function badge(state) {
  const cls = ["running","ready","queued","done","failed","killed","uploading",
               "preparing","finishing","rejected","online","offline","disabled",
               "selected","estop","paused"]
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

/* Ask a question without window.prompt(): a modal <dialog> so the answer is
   styled with the rest of the console. Resolves null when dismissed. */
function ask({ title, body = "", value = "", okLabel = "OK", required = false,
              type = "text", trim = true }) {
  return new Promise((resolve) => {
    const d = document.createElement("dialog");
    d.innerHTML = `<form method="dialog">
      <h3>${esc(title)}</h3>
      ${body ? `<p class="small muted">${esc(body)}</p>` : ""}
      <input class="answer" type="${esc(type)}" value="${esc(value)}"
             autocomplete="off" autofocus>
      <div class="btn-row end mt3">
        <button class="ghost" value="cancel" formnovalidate>Cancel</button>
        <button value="ok">${esc(okLabel)}</button>
      </div></form>`;
    document.body.append(d);
    const input = d.querySelector(".answer");
    d.addEventListener("close", () => {
      // trim=false: a password keeps its edge characters, or "pw " and "pw"
      // would be the same secret and the operator would chase a ghost
      const v = trim ? input.value.trim() : input.value;
      const ok = d.returnValue === "ok" && (!required || v);
      d.remove();
      resolve(ok ? v : null);
    });
    d.querySelector('button[value="ok"]').addEventListener("click", (e) => {
      if (required && !input.value.trim()) e.preventDefault();
    });
    d.showModal();
    input.select();
  });
}

/* Destructive-action confirm (kills, purges, releases). */
function confirmAction({ title, body = "", okLabel = "Confirm", danger = true }) {
  return new Promise((resolve) => {
    const d = document.createElement("dialog");
    d.innerHTML = `<form method="dialog">
      <h3>${esc(title)}</h3>
      ${body ? `<p class="small muted">${esc(body)}</p>` : ""}
      <div class="btn-row end mt3">
        <button class="ghost" value="cancel" formnovalidate>Cancel</button>
        <button class="${danger ? "danger" : ""}" value="ok">${esc(okLabel)}</button>
      </div></form>`;
    document.body.append(d);
    d.addEventListener("close", () => { const ok = d.returnValue === "ok"; d.remove(); resolve(ok); });
    d.showModal();
  });
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

function fmtAgo(rfc, now = Date.now()) {
  if (!rfc) return "";
  const t = new Date(rfc).getTime();
  if (Number.isNaN(t)) return "";
  const s = Math.max(0, Math.round((now - t) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  return `${(s / 3600).toFixed(1)}h ago`;
}

const shortModel = (m) => String(m ?? "").split("/").pop();

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

/* Refresh on a timer while the tab is visible (a hidden tab costs the LAN
   nothing). Returns stop(). */
function poll(fn, ms = 15000) {
  let timer = null;
  const run = () => { if (!document.hidden) Promise.resolve(fn()).catch(() => {}); };
  const start = () => { stop(); timer = setInterval(run, ms); run(); };
  const stop = () => { if (timer) clearInterval(timer); timer = null; };
  document.addEventListener("visibilitychange", () => (document.hidden ? stop() : start()));
  window.addEventListener("pagehide", stop);
  start();
  return stop;
}

/* ---- charts (canvas, no dependencies) -------------------------------------- */
/* Every canvas here sizes its backing store to the device pixel ratio and
   redraws when its box changes, so a chart stays sharp on a HiDPI display and
   correct after a window resize — the old fixed-pixel canvases did neither. */
function mountChart(canvas, draw) {
  const paint = () => {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, canvas.clientWidth), h = Math.max(1, canvas.clientHeight);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    draw(ctx, w, h);
  };
  canvas._repaint = paint;
  if (!canvas._ro) {
    canvas._ro = new ResizeObserver(() => paint());
    canvas._ro.observe(canvas);
  }
  paint();
  return paint;
}

/* spark(canvas, series) — line chart over [[x, y], ...] per series.
   Redraws on demand (call the returned function) and on resize. */
function spark(canvas, series) {
  const paint = (ctx, w, h) => {
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const s of series) for (const [x, y] of s.points) {
      if (!isFinite(y)) continue;
      xmin = Math.min(xmin, x); xmax = Math.max(xmax, x);
      ymin = Math.min(ymin, y); ymax = Math.max(ymax, y);
    }
    if (!isFinite(xmin)) {
      ctx.fillStyle = "#8b98a9"; ctx.font = "12px ui-monospace, monospace";
      ctx.fillText("no data yet", 8, h / 2); return;
    }
    if (xmin === xmax) { xmin -= 1; xmax += 1; }
    if (ymin === ymax) { ymin -= 1; ymax += 1; }
    const pad = { l: 34, r: 6, t: 8, b: 14 };
    const X = (x) => pad.l + (x - xmin) / (xmax - xmin) * (w - pad.l - pad.r);
    const Y = (y) => h - pad.b - (y - ymin) / (ymax - ymin) * (h - pad.t - pad.b);
    ctx.strokeStyle = "#2d3646"; ctx.lineWidth = 1;
    ctx.fillStyle = "#8b98a9"; ctx.font = "10px ui-monospace, monospace";
    for (const gy of [ymin, (ymin + ymax) / 2, ymax]) {
      ctx.beginPath(); ctx.moveTo(pad.l, Y(gy)); ctx.lineTo(w - pad.r, Y(gy)); ctx.stroke();
      ctx.fillText(gy.toFixed(Math.abs(gy) < 10 ? 2 : 0), 3, Y(gy) + 3);
    }
    for (const s of series) {
      ctx.strokeStyle = s.color; ctx.lineWidth = 1.6; ctx.beginPath();
      let started = false;
      for (const [x, y] of s.points) {
        if (!isFinite(y)) continue;
        started ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)); started = true;
      }
      ctx.stroke();
    }
  };
  return mountChart(canvas, paint);
}

/* The last value of a series, for the caption next to a chart. */
function lastPoint(points) {
  for (let i = points.length - 1; i >= 0; i--)
    if (isFinite(points[i][1])) return points[i];
  return null;
}

/* Velocity arrow: top-down robot, +x forward (up the screen), +y left.
   `action` is the house 7-dim vector (vx, vy, wz, 0, 0, 0, 0). */
function arrowChart(canvas, action, { label = true } = {}) {
  const vx = Number(action?.[0] ?? 0), vy = Number(action?.[1] ?? 0);
  const wz = Number(action?.[2] ?? 0);
  return mountChart(canvas, (ctx, w, h) => {
    const cx = w / 2, cy = h / 2, r = Math.min(w, h) * 0.42;
    ctx.strokeStyle = "#2d3646"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();   // chassis
    ctx.beginPath();                                                      // forward tick
    ctx.moveTo(cx, cy - r); ctx.lineTo(cx, cy - r - 6); ctx.stroke();
    const mag = Math.min(1, Math.hypot(vx, vy) / (0.3 || 0.5));
    if (mag > 0.02) {
      const ang = Math.atan2(-vx, vy);
      const ex = cx + Math.cos(ang) * r * mag, ey = cy - Math.sin(ang) * r * mag;
      ctx.strokeStyle = "#e8590c"; ctx.lineWidth = 4; ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ex, ey); ctx.stroke();
      ctx.fillStyle = "#e8590c"; ctx.beginPath(); ctx.arc(ex, ey, 5, 0, Math.PI * 2); ctx.fill();
    }
    if (Math.abs(wz) > 0.01) {                       // yaw arc, clockwise = +wz
      ctx.strokeStyle = "#4cc2ff"; ctx.lineWidth = 2; ctx.beginPath();
      const a0 = -Math.PI / 2, a1 = a0 + Math.max(-1.6, Math.min(1.6, wz)) * 1.2;
      ctx.arc(cx, cy, r * 0.62, a0, a1, wz < 0); ctx.stroke();
    }
    if (label) {
      ctx.fillStyle = "#8b98a9"; ctx.font = "11px ui-monospace, monospace";
      ctx.fillText(`vx ${vx.toFixed(2)}  vy ${vy.toFixed(2)}  wz ${wz.toFixed(2)}`,
                   6, h - 6);
    }
  });
}

/* ---- nav ------------------------------------------------------------------ */
/* One nav, grouped by workflow, so the pages read as one product:
   work (queue/runs) · data · the robot · (teacher) administration. */
const PAGES = [
  { group: [["index.html", "Dashboard"], ["class-board.html", "Class board"]] },
  { group: [["datasets.html", "Datasets"], ["new-run.html", "New run"],
            ["endpoints.html", "Endpoints"]] },
  { group: [["slots.html", "Serving"], ["robot.html", "Robot"],
            ["drive.html", "Drive"]] },
];
const TEACHER_PAGES = [["teacher.html", "Teacher"], ["accounts.html", "Accounts"]];

async function nav(active, user) {
  const el = $("#nav");
  if (!el) return;
  const links = (pair) =>
    `<a href="/ui/${pair[0]}" class="${pair[0] === active ? "active" : ""}">${pair[1]}</a>`;
  const groups = PAGES.map(g => `<span class="navgroup">${g.group.map(links).join("")}</span>`);
  if (user && user.role === "teacher")
    groups.push(`<span class="navgroup">${TEACHER_PAGES.map(links).join("")}</span>`);
  el.innerHTML =
    `<a class="brand" href="/ui/index.html">roboVLA</a>${groups.join("")}
     <span class="spacer"></span>
     <span class="who">${esc(user ? user.display_name : "")}
       <span class="mono">${esc(user ? user.user_id : "")}</span></span>
     <a href="/ui/logout.html">log out</a>`;
}
