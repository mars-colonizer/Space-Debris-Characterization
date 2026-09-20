/** AI-Enabled Space Situational Awareness — dashboard client */

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

function fmtNum(n) {
  if (n == null || n === "—") return "—";
  return Number(n).toLocaleString();
}

function fmtPeriod(sec) {
  if (sec == null || sec === "—") return "—";
  const s = Number(sec);
  if (s >= 3600) return `${(s / 3600).toFixed(2)} h`;
  if (s >= 60) return `${(s / 60).toFixed(2)} min`;
  return `${s.toFixed(2)} s`;
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
  const phase3Btn = $("#btn-run-phase3");
  if (phase3Btn) phase3Btn.disabled = running;
  document.querySelectorAll(".stage-btn").forEach((b) => (b.disabled = running));
  document.querySelectorAll(".mode-btn").forEach((b) => (b.disabled = running));

  if (data.stage_status) renderStepper(data.stage_status);

  const errBanner = $("#error-banner");
  if ((pipelineStatus === "FAILED" || pipelineStatus === "STOPPED") && data.failed_error) {
    errBanner.classList.remove("hidden");
    const prefix = pipelineStatus === "STOPPED" ? "Stopped" : "Failed";
    errBanner.textContent = `${prefix} at ${data.failed_stage || "unknown"}: ${data.failed_error}`;
  } else {
    errBanner.classList.add("hidden");
  }

  if (running && startTime) {
    if (!elapsedTimer) {
      elapsedTimer = setInterval(() => {
        const sec = Date.now() / 1000 - startTime;
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

async function runPhase3() {
  try {
    clearTerminalView();
    await api("POST", "/api/run-phase3");
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

function renderIngestion(ing) {
  let html = "";
  if (ing.source) html += `<div class="text-xs text-gray-400 mb-1">Mode: <strong>${ing.source}</strong></div>`;
  if (ing.mmt_ok) {
    html += `<div>Observations: <strong>${fmtNum(ing.mmt_points)}</strong></div>`;
    html += `<div>Objects with photometry: <strong>${fmtNum(ing.mmt_objects)}</strong></div>`;
    html += `<div class="text-green-600">MMT-9: ✓</div>`;
  } else {
    html += `<div>MMT-9: ○ (no data)</div>`;
  }
  if (ing.source === "ACTUAL") {
    html += `<div class="text-xs text-gray-400 mt-2 pt-2 border-t">TLE/DISCOS are fetched in step 3 only after KeepTrack COSPAR resolution.</div>`;
  }
  $("#ingestion-content").innerHTML = html || "No data yet.";
}

function renderIdentifiers(ids) {
  if (!ids.identifiers_ok && !ids.lc_objects) {
    $("#identifiers-content").innerHTML = "Run ingestion first.";
    return;
  }
  let html = "";
  if (ids.lc_objects != null) {
    html += `<div>MMT objects (light curves): <strong>${fmtNum(ids.lc_objects)}</strong></div>`;
    html += `<div>Observations: <strong>${fmtNum(ids.lc_points)}</strong></div>`;
  }
  if (ids.total_objects != null) {
    html += `<div class="mt-1 pt-1 border-t">KeepTrack COSPAR resolved: <strong>${fmtNum(ids.cospar_resolved)}</strong></div>`;
    html += `<div>No COSPAR (metadata skipped): <strong>${fmtNum(ids.cospar_missing)}</strong></div>`;
    html += `<div>Resolution rate: <strong>${ids.coverage_pct ?? "—"}%</strong></div>`;
  }
  if (ids.lc_objects_with_cospar != null) {
    html += `<div class="text-xs text-gray-400 mt-1">Eligible for TLE/DISCOS: <strong>${fmtNum(ids.lc_objects_with_cospar)}</strong> objects</div>`;
  }
  $("#identifiers-content").innerHTML = html || "Run ingestion first.";
}

function renderMetadata(meta) {
  if (!meta.mmt_objects && !meta.metadata_eligible) {
    $("#metadata-content").innerHTML = "Awaiting KeepTrack resolution.";
    return;
  }
  let html = "";
  if (meta.gated) {
    html += `<div class="text-xs text-amber-700 bg-amber-50 border border-amber-100 rounded px-2 py-1 mb-2">KeepTrack-gated: TLE + DISCOS only for resolved MMT objects</div>`;
    if (!meta.keeptrack_enabled) {
      html += `<div class="text-amber-700">KeepTrack API key not set — TLE/DISCOS skipped</div>`;
    }
  }
  html += `<div>MMT objects: <strong>${fmtNum(meta.mmt_objects)}</strong></div>`;
  html += `<div>Eligible (KeepTrack COSPAR): <strong>${fmtNum(meta.metadata_eligible)}</strong></div>`;
  if (meta.metadata_skipped > 0) {
    html += `<div class="text-amber-700">Skipped (no COSPAR): <strong>${fmtNum(meta.metadata_skipped)}</strong></div>`;
  }
  html += `<div class="mt-1 pt-1 border-t">Space-Track objects fetched: <strong>${fmtNum(meta.tle_objects)}</strong></div>`;
  if (meta.tle_ok) {
    html += `<div>GP records: <strong>${fmtNum(meta.gp_records)}</strong></div>`;
    html += `<div class="text-green-600">Space-Track: ✓</div>`;
  } else {
    html += `<div class="text-gray-400">Space-Track: ○ (not fetched or no eligible objects)</div>`;
  }
  if (meta.discos_ok) {
    html += `<div>DISCOS objects: <strong>${fmtNum(meta.discos_objects)}</strong></div>`;
    html += `<div class="text-green-600">DISCOS: ✓</div>`;
  } else {
    html += `<div class="text-gray-400">DISCOS: ○ (not fetched or no eligible objects)</div>`;
  }
  $("#metadata-content").innerHTML = html;
}

function renderDatabase(db) {
  if (!db.database_ok) {
    $("#database-content").innerHTML = `<div class="text-gray-400">Database empty.</div><div class="text-xs mt-1">${db.db_path || "data/database/rso_poc.db"}</div>`;
    return;
  }
  $("#database-content").innerHTML = `
    <div class="text-xs text-gray-400 mb-1">${db.db_path || "rso_poc.db"}</div>
    <div>Objects: <strong>${fmtNum(db.objects_count)}</strong></div>
    <div>Light curves: <strong>${fmtNum(db.light_curves_count)}</strong></div>
    <div>Periodograms: <strong>${fmtNum(db.periodograms_count)}</strong></div>
    <div>Folded curves: <strong>${fmtNum(db.folded_light_curves_count)}</strong></div>
    <div>Photometric summaries: <strong>${fmtNum(db.photometric_observations_count)}</strong></div>`;
}

function renderPeriodAnalysis(pa) {
  if (!pa.period_ok) {
    $("#period-content").textContent = "Run photometric processing to populate.";
    return;
  }
  let html = `
    <div class="grid grid-cols-2 md:grid-cols-4 gap-2 mb-3">
      <div>In source: <strong>${fmtNum(pa.objects_in_source)}</strong></div>
      <div>Analyzed: <strong>${fmtNum(pa.objects_analyzed)}</strong></div>
      <div>Filtered out: <strong>${fmtNum(pa.objects_filtered_out)}</strong></div>
      <div>Mean PDM θ: <strong>${pa.mean_pdm_theta != null ? pa.mean_pdm_theta.toFixed(4) : "—"}</strong></div>
      <div>Stable rotators: <strong>${fmtNum(pa.stable_rotators)}</strong></div>
      <div>Tumbling: <strong>${fmtNum(pa.tumbling)}</strong></div>
    </div>`;

  const objects = pa.objects || [];
  if (objects.length) {
    html += `<div class="overflow-x-auto"><table class="min-w-full text-xs">
      <thead><tr>
        <th class="text-left pr-3 py-1">NORAD</th>
        <th class="text-left pr-3 py-1">Name</th>
        <th class="text-left pr-3 py-1">LSP P</th>
        <th class="text-left pr-3 py-1">PDM P</th>
        <th class="text-left pr-3 py-1">Selected P</th>
        <th class="text-left pr-3 py-1">PDM θ</th>
        <th class="text-left pr-3 py-1">Tumbling</th>
      </tr></thead><tbody>`;
    html += objects
      .slice(0, 20)
      .map(
        (o) => `<tr>
          <td class="pr-3 py-1 font-mono">${o.object_id ?? "—"}</td>
          <td class="pr-3 py-1">${o.object_name ?? "—"}</td>
          <td class="pr-3 py-1">${fmtPeriod(o.lsp_period_sec)}</td>
          <td class="pr-3 py-1">${fmtPeriod(o.pdm_period_sec)}</td>
          <td class="pr-3 py-1">${fmtPeriod(o.extracted_period_sec)}</td>
          <td class="pr-3 py-1">${o.pdm_theta != null ? Number(o.pdm_theta).toFixed(4) : "—"}</td>
          <td class="pr-3 py-1">${o.is_tumbling ? "yes" : "no"}</td>
        </tr>`
      )
      .join("");
    if (objects.length > 20) {
      html += `<tr><td colspan="7" class="py-1 text-gray-400">… and ${objects.length - 20} more</td></tr>`;
    }
    html += `</tbody></table></div>`;
  }
  $("#period-content").innerHTML = html;
}

function renderArtifacts(poc) {
  if (!poc.plot_count && !poc.periodogram_count) {
    $("#artifacts-summary").textContent = "No artifacts yet.";
    $("#artifact-gallery").innerHTML = `<p class="text-sm text-gray-400 col-span-full">Plots appear after photometric processing completes.</p>`;
    return;
  }
  $("#artifacts-summary").innerHTML = `
    <div>Periodogram &amp; diagnostic plots: <strong>${fmtNum(poc.plot_count)}</strong></div>
    <div>Periodograms: <strong>${fmtNum(poc.periodogram_count)}</strong></div>
    <div>Folded curves: <strong>${fmtNum(poc.folded_count)}</strong></div>`;

  const plots = poc.plots || [];
  if (!plots.length) {
    $("#artifact-gallery").innerHTML = `<p class="text-sm text-gray-400 col-span-full">No PNG plots on disk yet.</p>`;
    return;
  }
  $("#artifact-gallery").innerHTML = plots
    .map(
      (name) => `<a href="/api/artifacts/poc_plots/${encodeURIComponent(name)}" target="_blank" class="artifact-thumb" title="${name}">
        <img src="/api/artifacts/poc_plots/${encodeURIComponent(name)}" alt="${name}" loading="lazy" />
        <span class="artifact-label">${name.replace(/^norad_/, "").replace(/\.png$/, "")}</span>
      </a>`
    )
    .join("");
}

function fmtF1(v) {
  return v == null || Number.isNaN(Number(v)) ? "—" : Number(v).toFixed(4);
}

function renderPhase3Characterization(data) {
  const ablationEl = $("#phase3-ablation-content");
  const stage2El = $("#phase3-stage2-content");
  if (!ablationEl || !stage2El) return;

  const ablation = data.ablation || [];
  if (ablation.length) {
    ablationEl.innerHTML = `
      <table class="min-w-full text-sm bg-white rounded border border-slate-200 overflow-hidden">
        <thead class="bg-slate-100 text-slate-600 text-xs uppercase tracking-wide">
          <tr>
            <th class="text-left px-3 py-2">Model</th>
            <th class="text-right px-3 py-2">Orbital F1</th>
            <th class="text-right px-3 py-2">Fused F1</th>
            <th class="text-right px-3 py-2">Δ (Fused−Orb)</th>
          </tr>
        </thead>
        <tbody>
          ${ablation
            .map((row) => {
              const delta = Number(row.fused_f1) - Number(row.orbital_f1);
              const deltaCls = delta >= 0 ? "text-emerald-700" : "text-rose-700";
              return `<tr class="border-t border-slate-100">
                <td class="px-3 py-2 font-medium text-slate-800">${row.model}</td>
                <td class="px-3 py-2 text-right tabular-nums">${fmtF1(row.orbital_f1)}</td>
                <td class="px-3 py-2 text-right tabular-nums font-semibold">${fmtF1(row.fused_f1)}</td>
                <td class="px-3 py-2 text-right tabular-nums ${deltaCls}">${delta >= 0 ? "+" : ""}${delta.toFixed(4)}</td>
              </tr>`;
            })
            .join("")}
        </tbody>
      </table>`;
  } else {
    ablationEl.textContent = "No ablation metrics available.";
  }

  const stage2 = data.stage2 || [];
  if (stage2.length) {
    stage2El.innerHTML = `
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
        ${stage2
          .map(
            (row) => `
          <div class="bg-white border border-slate-200 rounded-lg p-3 shadow-sm">
            <div class="text-xs uppercase tracking-wide text-slate-500">${row.class}</div>
            <div class="text-sm text-slate-700 mt-1">Target: <strong>${row.target || "mass"}</strong> · n=${fmtNum(row.n_samples)}</div>
            <div class="mt-3 grid grid-cols-2 gap-2">
              <div class="rounded bg-slate-50 px-2 py-2">
                <div class="text-[10px] uppercase text-slate-400">LOOCV MAE</div>
                <div class="text-base font-semibold tabular-nums text-slate-800">${Number(row.mae).toFixed(2)}</div>
              </div>
              <div class="rounded bg-slate-50 px-2 py-2">
                <div class="text-[10px] uppercase text-slate-400">LOOCV R²</div>
                <div class="text-base font-semibold tabular-nums text-slate-800">${Number(row.r2).toFixed(4)}</div>
              </div>
            </div>
          </div>`
          )
          .join("")}
      </div>`;
  } else {
    stage2El.textContent = "No Stage 2 mass-regression metrics available.";
  }

  const bust = `?t=${Date.now()}`;
  const images = data.images || {};
  const cm = images.confusion_matrix || "/phase3_images/confusion_matrix.png";
  const shap = images.shap_feature_importance || "/phase3_images/shap_feature_importance.png";
  const cmImg = $("#phase3-cm-img");
  const shapImg = $("#phase3-shap-img");
  const cmLink = $("#phase3-cm-link");
  const shapLink = $("#phase3-shap-link");
  if (cmImg) {
    cmImg.classList.remove("hidden");
    cmImg.onerror = () => showXaiFallback("cm");
    cmImg.onload = () => {
      const fb = $("#phase3-cm-fallback");
      if (fb) fb.classList.add("hidden");
      cmImg.classList.remove("hidden");
    };
    cmImg.src = cm + bust;
  }
  if (shapImg) {
    shapImg.classList.remove("hidden");
    shapImg.onerror = () => showXaiFallback("shap");
    shapImg.onload = () => {
      const fb = $("#phase3-shap-fallback");
      if (fb) fb.classList.add("hidden");
      shapImg.classList.remove("hidden");
    };
    shapImg.src = shap + bust;
  }
  if (cmLink) cmLink.href = cm;
  if (shapLink) shapLink.href = shap;
}

function showXaiFallback(kind) {
  const img = kind === "cm" ? $("#phase3-cm-img") : $("#phase3-shap-img");
  const fb = kind === "cm" ? $("#phase3-cm-fallback") : $("#phase3-shap-fallback");
  if (img) img.classList.add("hidden");
  if (fb) fb.classList.remove("hidden");
}

async function loadPhase3Metrics() {
  try {
    const data = await api("GET", "/api/phase3/metrics");
    renderPhase3Characterization(data);
  } catch (e) {
    console.warn("phase3 metrics load failed:", e.message);
    const ablationEl = $("#phase3-ablation-content");
    const stage2El = $("#phase3-stage2-content");
    if (ablationEl) ablationEl.textContent = "Failed to load Phase 3 metrics.";
    if (stage2El) stage2El.textContent = "Failed to load Phase 3 metrics.";
  }
}

function renderPhase3(p3, inference) {
  const infEl = $("#inference-content");
  if (!infEl) return;

  const inf = inference || p3.inference || {};
  if (Object.keys(inf).length) {
    const massNum = inf.mass != null && inf.mass !== "" && !Number.isNaN(Number(inf.mass));
    const massLabel = massNum
      ? `${Number(inf.mass).toFixed(2)} kg`
      : (inf.mass_status || "n/a (no Stage 2 model)");
    const dims = (inf.length || inf.width || inf.height)
      ? `<div>Sizing (L × W × H): <strong>${inf.length || "—"} × ${inf.width || "—"} × ${inf.height || "—"}</strong> m</div>`
      : "";
    infEl.innerHTML = `
      <div class="grid grid-cols-2 gap-2">
        <div>COSPAR ID: <strong>${inf.cospar_id || "—"}</strong></div>
        <div>Confidence: <strong>${inf.confidence || "—"}%</strong></div>
        <div>True class: <strong>${inf.true_class || "—"}</strong></div>
        <div>Predicted: <strong>${inf.predicted_class || "—"}</strong></div>
      </div>
      <div class="mt-2">Estimated mass: <strong>${massLabel}</strong></div>
      ${dims}
      <div>Spin period: <strong>${inf.spin_period || "—"}</strong> s · Tumbling: <strong>${inf.tumbling || "—"}</strong></div>`;
  } else {
    infEl.textContent = "Run inference to see results.";
  }
}

function renderMetrics(m) {
  renderIngestion(m.ingestion || {});
  renderIdentifiers(m.identifiers || {});
  renderMetadata(m.metadata || {});
  renderDatabase(m.database || {});
  renderPeriodAnalysis(m.period_analysis || {});
  renderArtifacts(m.poc_artifacts || {});
  renderPhase3(m.phase3 || {}, m.inference_result);
}

let metricsTimer = null;

async function loadMetrics() {
  try {
    const m = await api("GET", "/api/metrics");
    renderMetrics(m);
  } catch (e) {
    console.warn("metrics load failed:", e.message);
  }
}

function scheduleMetricsRefresh() {
  if (metricsTimer) return;
  metricsTimer = setTimeout(async () => {
    metricsTimer = null;
    await loadMetrics();
    await loadPhase3Metrics();
  }, 250);
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
        <td class="py-2 pr-4 text-xs">${(r.exec_keys || r.stages || []).join(", ") || "—"}</td>
        <td class="py-2"><button class="btn-delete-run text-red-600 hover:text-red-800 text-sm" data-run-id="${r.run_id}" title="Delete run">Delete</button></td>
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
  if (!confirm("Delete all historical run logs?")) return;
  try {
    const res = await api("DELETE", "/api/runs");
    $("#runs-tbody").innerHTML = `<tr><td colspan="6" class="py-4 text-gray-400">No saved runs yet. (${res.count} deleted)</td></tr>`;
  } catch (e) {
    alert(e.message);
  }
}

async function resetAllData() {
  if (!confirm("Clear all processed datasets, models, logs, and metric cards?")) return;
  try {
    await api("POST", "/api/reset-all");
    const el = terminal();
    if (el) {
      el.innerHTML = '<div class="text-gray-500">[SYSTEM] Cached data cleared — ready for a fresh SSA pipeline run.</div>';
    }
    const status = await api("GET", "/api/status");
    updateStatusUI(status);
    await loadMetrics();
    await loadPhase3Metrics();
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
    if (msg.type === "status" && msg.data) {
      updateStatusUI(msg.data);
      const ps = msg.data.pipeline_status;
      if (ps === "COMPLETED" || ps === "FAILED" || ps === "STOPPED") {
        scheduleMetricsRefresh();
      }
    }
    if (msg.type === "metrics_refresh") scheduleMetricsRefresh();
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
  const p3Btn = $("#btn-run-phase3");
  if (p3Btn) p3Btn.addEventListener("click", runPhase3);
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
  await loadPhase3Metrics();
  await loadPreviousRuns();
  connectWebSocket();
}

document.addEventListener("DOMContentLoaded", init);
