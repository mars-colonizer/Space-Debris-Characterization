/** Phase 2 Control Dashboard — client logic */

const STEP_ICONS = {
  WAITING: { icon: "○", cls: "step-waiting" },
  RUNNING: { icon: "◉", cls: "step-running" },
  COMPLETE: { icon: "✓", cls: "step-complete" },
  WARNING: { icon: "⚠", cls: "step-warning" },
  FAILED: { icon: "✕", cls: "step-failed" },
  STOPPED: { icon: "■", cls: "step-warning" },
};

const STATUS_BADGE = {
  IDLE: "status-idle",
  RUNNING: "status-running",
  COMPLETED: "status-completed",
  FAILED: "status-failed",
  STOPPED: "status-stopped",
};

let ws = null;
let wsReconnectTimer = null;
let elapsedTimer = null;
let pipelineStatus = "IDLE";
let startTime = null;
let stepsMeta = [];
let dataMode = "SYNTHETIC";

const $ = (sel) => document.querySelector(sel);
const terminal = () => $("#terminal-log");

function logClass(line) {
  if (/\[ERROR\]/i.test(line)) return "log-error";
  if (/\[WARNING\]/i.test(line)) return "log-warning";
  if (/\[OK\]/i.test(line)) return "log-ok";
  return "log-info";
}

function appendLog(line) {
  const el = terminal();
  if (!el) return;
  const span = document.createElement("span");
  span.className = logClass(line);
  span.textContent = line + "\n";
  el.appendChild(span);
  el.scrollTop = el.scrollHeight;
}

function clearTerminalView() {
  const el = terminal();
  if (el) el.innerHTML = "";
}

function formatElapsed(sec) {
  if (sec == null) return "—";
  if (sec < 60) return `${sec.toFixed(1)}s`;
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(0);
  return `${m}m ${s}s`;
}

function updateStatusUI(data) {
  pipelineStatus = data.pipeline_status || "IDLE";
  startTime = data.start_time || null;

  const badge = $("#status-badge");
  if (badge) {
    badge.textContent = pipelineStatus;
    badge.className = `${STATUS_BADGE[pipelineStatus] || "status-idle"} text-white text-xs font-bold px-3 py-1 rounded-full uppercase`;
  }

  $("#started-at").textContent = data.start_time_iso || "—";
  $("#elapsed").textContent = formatElapsed(data.elapsed_sec);
  $("#current-stage").textContent = data.current_exec_key || data.current_step_key || "—";

  const running = pipelineStatus === "RUNNING";
  $("#btn-run-full").disabled = running;
  $("#btn-stop").disabled = !running;
  $("#btn-reset-all").disabled = running;
  document.querySelectorAll(".stage-btn").forEach((b) => (b.disabled = running));
  document.querySelectorAll(".mode-btn").forEach((b) => (b.disabled = running));

  if (data.stage_status) renderStepper(data.stage_status);

  const errBanner = $("#error-banner");
  if (pipelineStatus === "FAILED" && data.failed_error) {
    errBanner.classList.remove("hidden");
    errBanner.textContent = `Failed at ${data.failed_stage || "unknown"}: ${data.failed_error}`;
  } else {
    errBanner.classList.add("hidden");
  }

  if (running && startTime) {
    if (!elapsedTimer) {
      elapsedTimer = setInterval(() => {
        const sec = (Date.now() / 1000) - startTime;
        $("#elapsed").textContent = formatElapsed(sec);
      }, 500);
    }
  } else if (elapsedTimer) {
    clearInterval(elapsedTimer);
    elapsedTimer = null;
  }
}

function renderStepper(stageStatus) {
  const ol = $("#stepper");
  if (!ol || !stepsMeta.length) return;
  ol.innerHTML = stepsMeta
    .map((step) => {
      const st = stageStatus[step.key] || "WAITING";
      const meta = STEP_ICONS[st] || STEP_ICONS.WAITING;
      return `<li class="flex items-start gap-2">
        <span class="${meta.cls} font-bold w-4">${meta.icon}</span>
        <span><span class="text-gray-400">${step.number}.</span> ${step.label}
        <span class="text-xs ${meta.cls} ml-1">${st}</span></span>
      </li>`;
    })
    .join("");
}

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

async function runPipeline() {
  try {
    clearTerminalView();
    await api("POST", "/api/run-pipeline");
  } catch (e) {
    alert(e.message);
  }
}

async function runStage(name) {
  try {
    await api("POST", `/api/run-stage/${name}`);
  } catch (e) {
    alert(e.message);
  }
}

async function stopPipeline() {
  try {
    await api("POST", "/api/stop-pipeline");
  } catch (e) {
    alert(e.message);
  }
}

function fmtNum(n) {
  if (n == null || n === "—") return "—";
  return Number(n).toLocaleString();
}

function renderMetrics(m) {
  const ing = m.ingestion || {};
  let ingHtml = "";
  if (ing.tle_ok) {
    ingHtml += `<div>Space-Track GP records: <strong>${fmtNum(ing.gp_records)}</strong></div>`;
    ingHtml += `<div>NORAD IDs: <strong>${fmtNum(ing.norad_ids)}</strong></div>`;
    ingHtml += `<div class="text-green-600">Space-Track: ✓</div>`;
  } else ingHtml += `<div>Space-Track: ○ (no data)</div>`;
  if (ing.discos_ok) {
    ingHtml += `<div>DISCOS objects: <strong>${fmtNum(ing.discos_objects)}</strong></div>`;
    ingHtml += `<div class="text-green-600">DISCOS: ✓</div>`;
  } else ingHtml += `<div>DISCOS: ○ (no data)</div>`;
  $("#ingestion-content").innerHTML = ingHtml || "No data yet.";

  const merge = m.merge || {};
  const meta = m.dataset_meta || {};
  let qHtml = "";
  if (ing.gp_records) qHtml += `<div>TLE records: <strong>${fmtNum(ing.gp_records)}</strong></div>`;
  if (ing.tle_objects) qHtml += `<div>Unique TLE objects: <strong>${fmtNum(ing.tle_objects)}</strong></div>`;
  if (merge.matched_objects != null) qHtml += `<div>Matched objects: <strong>${fmtNum(merge.matched_objects)}</strong></div>`;
  if (merge.unmatched_tle_objects != null) qHtml += `<div>Unmatched TLE: <strong>${fmtNum(merge.unmatched_tle_objects)}</strong></div>`;
  if (meta.train_objects != null) qHtml += `<div>Train objects: <strong>${fmtNum(meta.train_objects)}</strong></div>`;
  if (meta.test_objects != null) qHtml += `<div>Test objects: <strong>${fmtNum(meta.test_objects)}</strong></div>`;
  if (m.class_count) qHtml += `<div>Classes: <strong>${m.class_count}</strong></div>`;
  $("#quality-content").innerHTML = qHtml || "Run preparation to populate.";

  const photo = m.photometry || {};
  if (photo.photo_ok) {
    const src = photo.source ? ` · <span class="text-gray-400">${photo.source}</span>` : "";
    $("#photometry-content").innerHTML = `
      <div>Observations: <strong>${fmtNum(photo.obs_count)}</strong></div>
      <div>Objects with photometry: <strong>${fmtNum(photo.object_count)}</strong></div>
      <div>Avg amplitude (Δm): <strong>${photo.avg_delta_mag ?? "—"}</strong></div>
      <div>Max amplitude (Δm): <strong>${photo.max_delta_mag ?? "—"}</strong></div>
      <div>Tumbling fraction: <strong>${photo.tumbling_fraction != null ? (photo.tumbling_fraction * 100).toFixed(1) + "%" : "—"}</strong></div>
      <div class="text-xs mt-1">Source: <strong>${photo.source || "—"}</strong></div>`;
  } else {
    $("#photometry-content").textContent = "No photometric data yet.";
  }

  const leak = m.leakage || {};
  const removed = leak.removed_columns || [];
  if (removed.length) {
    const safe = leak.cospar_overlap === 0;
    $("#leakage-content").innerHTML = `
      <div class="mb-2">Removed columns:</div>
      <ul class="list-disc ml-5 mb-2">${removed.map((c) => `<li><code>${c}</code> ✓</li>`).join("")}</ul>
      <div>Features: <strong>${leak.feature_count ?? "—"}</strong></div>
      <div>COSPAR overlap: <strong>${leak.cospar_overlap ?? 0}</strong></div>
      <span class="inline-block mt-2 px-2 py-0.5 rounded text-xs font-bold ${safe ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}">${safe ? "STATUS: ✓ SAFE" : "STATUS: ✕ LEAKAGE"}</span>`;
  } else {
    $("#leakage-content").textContent = "Available after data preparation.";
  }

  const s1 = m.stage1 || [];
  if (s1.length) {
    const cols = ["model", "accuracy", "precision", "recall", "f1"];
    $("#stage1-content").innerHTML = `<table class="min-w-full text-xs"><thead><tr>${cols.map((c) => `<th class="text-left pr-3 py-1">${c}</th>`).join("")}</tr></thead><tbody>${s1.map((row) => `<tr>${cols.map((c) => `<td class="pr-3 py-1">${row[c] != null ? (typeof row[c] === "number" ? row[c].toFixed(4) : row[c]) : "—"}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  } else {
    $("#stage1-content").textContent = "No Stage 1 metrics yet.";
  }

  const s2classes = m.stage2_classes || [];
  const s2metrics = m.stage2 || [];
  let s2Html = "";
  if (s2classes.length) {
    const trained = s2classes.filter((x) => x.status === "OK").length;
    const skipped = s2classes.length - trained;
    s2Html += `<div class="mb-2">Models trained: <strong>${trained}</strong> | Skipped: <strong>${skipped}</strong></div>`;
    s2Html += `<table class="min-w-full text-xs mb-2"><thead><tr><th class="text-left pr-3">Class</th><th class="text-left pr-3">Samples</th><th>Status</th></tr></thead><tbody>${s2classes.map((r) => `<tr><td class="pr-3">${r.class}</td><td class="pr-3">${r.samples}</td><td>${r.status === "OK" ? "✓" : r.status === "insufficient" ? "⚠ insufficient" : "○"}</td></tr>`).join("")}</tbody></table>`;
  }
  if (s2metrics.length) {
    const cols = ["object_class", "target", "mae", "rmse", "r2"];
    s2Html += `<table class="min-w-full text-xs"><thead><tr>${cols.map((c) => `<th class="text-left pr-3 py-1">${c}</th>`).join("")}</tr></thead><tbody>${s2metrics.map((row) => `<tr>${cols.map((c) => `<td class="pr-3 py-1">${row[c] != null ? (typeof row[c] === "number" ? row[c].toFixed(4) : row[c]) : "—"}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  }
  $("#stage2-content").innerHTML = s2Html || "No Stage 2 metrics yet.";

  const inf = m.inference || {};
  if (Object.keys(inf).length) {
    const L = inf.length || "—", W = inf.width || "—", H = inf.height || "—";
    $("#inference-content").innerHTML = `
      <div class="grid grid-cols-2 gap-2">
        <div>COSPAR ID: <strong>${inf.cospar_id || "—"}</strong></div>
        <div>Confidence: <strong>${inf.confidence || "—"}%</strong></div>
        <div>True class: <strong>${inf.true_class || "—"}</strong></div>
        <div>Predicted: <strong>${inf.predicted_class || "—"}</strong></div>
      </div>
      <div class="mt-2">Sizing (L × W × H): <strong>${L} × ${W} × ${H}</strong> m</div>
      <div class="mt-2">Shape: <strong>${inf.shape || "—"}</strong></div>
      <div>Spin period: <strong>${inf.spin_period || "—"}</strong> s · Tumbling: <strong>${inf.tumbling || "—"}</strong></div>
      <div class="text-xs text-gray-500 mt-1">Latency: ${inf.latency_seconds || "—"} s</div>`;
  } else {
    $("#inference-content").textContent = "Run inference to see results.";
  }
}

async function loadMetrics() {
  try {
    const m = await api("GET", "/api/metrics");
    renderMetrics(m);
  } catch (_) { /* ignore */ }
}

function renderRuns(runs) {
  const tbody = $("#runs-tbody");
  if (!runs.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="py-4 text-gray-400">No saved runs yet.</td></tr>`;
    return;
  }
  tbody.innerHTML = runs
    .map(
      (r) => `<tr class="border-b hover:bg-gray-50" data-run-id="${r.run_id}">
        <td class="py-2 pr-4 font-mono text-xs">${r.run_id}</td>
        <td class="py-2 pr-4">${r.start_time || "—"}</td>
        <td class="py-2 pr-4">${r.status || "—"}</td>
        <td class="py-2 pr-4">${r.duration_sec != null ? r.duration_sec + "s" : "—"}</td>
        <td class="py-2 pr-4 text-xs">${(r.stages || []).join(", ") || "—"}</td>
        <td class="py-2"><button class="btn-delete-run text-red-600 hover:text-red-800 text-sm" data-run-id="${r.run_id}" title="Delete run">🗑️ Delete</button></td>
      </tr>`
    )
    .join("");
}

async function loadPreviousRuns() {
  try {
    const runs = await api("GET", "/api/previous-runs");
    renderRuns(runs);
  } catch (_) {
    $("#runs-tbody").innerHTML = `<tr><td colspan="6" class="py-4 text-red-400">Failed to load runs.</td></tr>`;
  }
}

async function deleteRun(runId) {
  if (!confirm(`Delete run ${runId}?`)) return;
  try {
    await api("DELETE", `/api/runs/${encodeURIComponent(runId)}`);
    const row = document.querySelector(`tr[data-run-id="${runId}"]`);
    if (row) row.remove();
    if (!$("#runs-tbody").querySelector("tr")) {
      $("#runs-tbody").innerHTML = `<tr><td colspan="6" class="py-4 text-gray-400">No saved runs yet.</td></tr>`;
    }
  } catch (e) {
    alert(e.message);
  }
}

async function clearAllRuns() {
  if (!confirm("Are you sure you want to delete all historical run data?")) return;
  try {
    const res = await api("DELETE", "/api/runs");
    $("#runs-tbody").innerHTML = `<tr><td colspan="6" class="py-4 text-gray-400">No saved runs yet. (${res.count} deleted)</td></tr>`;
  } catch (e) {
    alert(e.message);
  }
}

async function resetAllData() {
  if (!confirm("Are you sure you want to clear all processed datasets, models, logs, and metric cards?")) return;
  try {
    await api("POST", "/api/reset-all");
    const el = terminal();
    if (el) {
      el.innerHTML = '<div class="text-gray-500">[SYSTEM] All cached data cleared. Ready for fresh run.</div>';
    }
    const status = await api("GET", "/api/status");
    updateStatusUI(status);
    await loadMetrics();
  } catch (e) {
    alert(e.message);
  }
}

function updateModeUI(mode) {
  dataMode = mode;
  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.classList.toggle("mode-active", btn.dataset.mode === mode);
  });
}

async function loadCurrentMode() {
  try {
    const res = await api("GET", "/api/current-mode");
    updateModeUI(res.mode || "SYNTHETIC");
  } catch (_) {
    updateModeUI("SYNTHETIC");
  }
}

async function setDataMode(mode) {
  if (mode === dataMode) return;
  try {
    const res = await api("POST", "/api/set-mode", { mode });
    updateModeUI(res.mode);
    appendLog(`[INFO] Data mode switched to ${res.mode}`);
  } catch (e) {
    alert(e.message);
  }
}

function connectWebSocket() {
  if (ws && ws.readyState <= 1) return;
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws/logs`);

  ws.onmessage = (ev) => {
    let msg;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      return;
    }
    if (msg.type === "log" && msg.line) appendLog(msg.line);
    if (msg.type === "status" && msg.data) updateStatusUI(msg.data);
    if (msg.type === "metrics_refresh") loadMetrics();
  };

  ws.onclose = () => {
    ws = null;
    clearTimeout(wsReconnectTimer);
    wsReconnectTimer = setTimeout(connectWebSocket, 2000);
  };

  ws.onerror = () => ws.close();
}

async function init() {
  try {
    stepsMeta = await api("GET", "/api/steps");
  } catch (_) {
    stepsMeta = [];
  }

  const status = await api("GET", "/api/status");
  updateStatusUI(status);

  $("#btn-run-full").addEventListener("click", runPipeline);
  $("#btn-stop").addEventListener("click", stopPipeline);
  $("#btn-reset-all").addEventListener("click", resetAllData);
  $("#btn-clear-log").addEventListener("click", clearTerminalView);
  $("#btn-clear-runs").addEventListener("click", clearAllRuns);

  document.querySelectorAll(".stage-btn").forEach((btn) => {
    btn.addEventListener("click", () => runStage(btn.dataset.stage));
  });

  $("#runs-tbody").addEventListener("click", (e) => {
    const btn = e.target.closest(".btn-delete-run");
    if (btn) deleteRun(btn.dataset.runId);
  });

  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => setDataMode(btn.dataset.mode));
  });

  await loadCurrentMode();
  await loadMetrics();
  await loadPreviousRuns();
  connectWebSocket();
}

document.addEventListener("DOMContentLoaded", init);
