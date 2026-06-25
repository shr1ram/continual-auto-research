// pipeline-controller.js — the CAR app UI.
//
// Thin over the API: launch runs (POST /api/runs), list them (GET /api/runs),
// stream a selected run (WS /ws/runs/{id}), resume/stop, and render the live
// chart + candidate table from the event stream. All data shown comes from the
// library's events — the API just routes them.

(function () {
  const $ = (id) => document.getElementById(id);
  const api = (p, opts) => fetch(p, opts).then((r) => (r.ok ? r.json() : Promise.reject(r)));

  let currentRunId = null;
  let ws = null;
  let points = []; // {iter, score, best}
  let traces = {}; // iteration -> {proposer, runner} trace event
  let direction = "min";

  // ── run list ────────────────────────────────────────────────────────────
  async function refreshRunList() {
    const { runs } = await api("/api/runs");
    const el = $("runList");
    el.innerHTML = "";
    runs.forEach((r) => {
      const div = document.createElement("div");
      div.className = "run-item" + (r.id === currentRunId ? " active" : "");
      div.innerHTML =
        `<div class="rid">${r.id} <span class="badge ${r.status}">${r.status}</span></div>` +
        `<div class="meta">${r.direction || ""} · best ${fmt(r.best_score)} · ${r.iterations} it</div>`;
      div.onclick = () => selectRun(r.id);
      el.appendChild(div);
    });
  }

  function fmt(v) { return v === null || v === undefined ? "—" : Number(v).toFixed(4); }

  // ── select / stream a run ──────────────────────────────────────────────
  function selectRun(id) {
    if (ws) { ws.close(); ws = null; }
    currentRunId = id;
    points = [];
    traces = {};
    $("rows").innerHTML = "";
    $("best").textContent = "—";
    $("status").textContent = "loading…";
    drawChart();
    refreshRunList();
    // load config (for direction) then stream
    api(`/api/runs/${id}`).then((rec) => {
      direction = (rec.config && rec.config.direction) || "min";
      openStream(id);
    });
  }

  function openStream(id) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/runs/${id}`);
    ws.onmessage = (m) => onEvent(JSON.parse(m.data));
    ws.onclose = () => { /* stream ended */ };
  }

  function onEvent(ev) {
    switch (ev.type) {
      case "scored": {
        addRow(ev);
        if (ev.best !== null && ev.best !== undefined) {
          points.push({ iter: ev.iteration, score: ev.score, best: ev.best });
          $("best").textContent = fmt(ev.best);
          drawChart();
        }
        if (ev.stale_rounds !== undefined) updatePlateau(ev.stale_rounds);
        $("status").textContent = "running";
        $("stop").disabled = false; $("resume").disabled = true;
        break;
      }
      case "trace": {
        // stash the full prompt/response/command/output keyed by iteration; the
        // candidate row's expander renders it when trace mode is on.
        traces[ev.iteration] = ev;
        break;
      }
      case "done": {
        $("status").textContent = `done — ${ev.stop_reason} (${ev.iterations} it)`;
        $("stop").disabled = true; $("resume").disabled = false;
        refreshRunList();
        break;
      }
      case "error": {
        $("status").textContent = "error: " + (ev.error || "");
        $("stop").disabled = true; $("resume").disabled = false;
        break;
      }
    }
  }

  function addRow(ev) {
    const tr = document.createElement("tr");
    tr.className = "cand" + (ev.improved ? " accepted" : "") + (ev.score === null ? " failed" : "");
    const pill = ev.improved ? '<span class="pill acc">best</span>' : '<span class="pill rej">·</span>';
    tr.dataset.iter = ev.iteration;
    tr.innerHTML =
      `<td>${ev.iteration}</td><td>${fmt(ev.score)}</td><td>${fmt(ev.best)} ${pill}</td>` +
      `<td>🔍</td>` +
      `<td title="${esc(ev.proposal || "")}">${esc((ev.proposal || "").slice(0, 60))}</td>`;
    // clicking a row opens the trace (prompt/response/command/output) for that iter
    tr.onclick = () => toggleTrace(tr, ev);
    $("rows").appendChild(tr);
  }

  // Full trace panel: the LLM prompt + response, the run command, and the full
  // GPU output for this iteration. Falls back to just the run output if no trace
  // was captured (e.g. an older run).
  function toggleTrace(tr, ev) {
    const next = tr.nextElementSibling;
    if (next && next.classList.contains("rawrow")) { next.remove(); return; }
    const t = traces[ev.iteration];
    const p = (t && t.proposer) || {};
    const r = (t && t.runner) || {};
    const sect = (label, body, cls) =>
      body ? `<div class="tsec"><div class="tlabel ${cls||''}">${label}</div><div class="raw">${esc(body)}</div></div>` : "";
    const html =
      sect("SYSTEM PROMPT", p.system, "sys") +
      sect("PROMPT → LLM" + (p.model ? ` (${esc(p.model)})` : ""), p.prompt, "prompt") +
      sect("LLM RESPONSE", p.response != null ? p.response : ev.proposal, "resp") +
      sect("RUN COMMAND", r.command, "cmd") +
      sect("RUN OUTPUT (full)", r.output != null ? r.output : ev.raw_result, "out");
    const row = document.createElement("tr");
    row.className = "rawrow";
    row.innerHTML = `<td colspan="5">${html || '<div class="raw">(no trace captured)</div>'}</td>`;
    tr.after(row);
  }

  function toggleRaw(tr, raw) {
    const next = tr.nextElementSibling;
    if (next && next.classList.contains("rawrow")) { next.remove(); return; }
    const r = document.createElement("tr");
    r.className = "rawrow";
    r.innerHTML = `<td colspan="5"><div class="raw">${esc(raw)}</div></td>`;
    tr.after(r);
  }

  function esc(s) { return String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

  function updatePlateau(stale) {
    const patience = parseInt($("patience").value, 10) || 4;
    const pct = Math.min(100, (stale / patience) * 100);
    $("plateau").style.width = pct + "%";
  }

  // ── live chart (best line + per-candidate dots), no deps ──────────────────
  function drawChart() {
    const svg = $("chart");
    const W = 600, H = 180, pad = 24;
    if (!points.length) { svg.innerHTML = ""; return; }
    const xs = points.map((p) => p.iter);
    const all = points.flatMap((p) => [p.score, p.best]).filter((v) => v !== null && v !== undefined);
    const xmin = Math.min(...xs), xmax = Math.max(...xs, xmin + 1);
    const ymin = Math.min(...all), ymax = Math.max(...all, ymin + 1e-9);
    const sx = (x) => pad + ((x - xmin) / (xmax - xmin || 1)) * (W - 2 * pad);
    const sy = (y) => H - pad - ((y - ymin) / (ymax - ymin || 1)) * (H - 2 * pad);
    let svgInner = "";
    // best line
    const bestPts = points.filter((p) => p.best !== null && p.best !== undefined);
    if (bestPts.length) {
      const d = bestPts.map((p, i) => `${i ? "L" : "M"}${sx(p.iter)},${sy(p.best)}`).join(" ");
      svgInner += `<path d="${d}" fill="none" stroke="#245A40" stroke-width="2"/>`;
    }
    // candidate dots
    points.forEach((p) => {
      if (p.score === null || p.score === undefined) return;
      svgInner += `<circle cx="${sx(p.iter)}" cy="${sy(p.score)}" r="3" fill="#C5A55A"/>`;
    });
    svg.innerHTML = svgInner;
  }

  // ── launch / resume / stop ────────────────────────────────────────────────
  function buildConfig() {
    const cfg = {
      direction: $("direction").value,
      max_iter: parseInt($("maxIter").value, 10),
      patience: parseInt($("patience").value, 10),
    };
    const obj = $("objective").value.trim();
    if (obj !== "") cfg.objective = obj;
    const t = $("target").value;
    if (t !== "") cfg.target_score = parseFloat(t);

    const rk = $("runner").value;
    if (rk === "broker") {
      cfg.runner = {
        kind: "broker",
        project_id: $("projectId").value,
        workspace_dir: $("workspaceDir").value,
        run_command: $("runCommand").value,
      };
    } else cfg.runner = { kind: "demo" };

    const pk = $("proposer").value;
    if (pk !== "demo") {
      cfg.proposer = { kind: pk };
      if ($("model").value) cfg.proposer.model = $("model").value;
    } else cfg.proposer = { kind: "demo" };
    return cfg;
  }

  async function launch() {
    $("launch").disabled = true;
    try {
      const rec = await api("/api/runs", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildConfig()),
      });
      await refreshRunList();
      selectRun(rec.id);
    } finally { $("launch").disabled = false; }
  }

  async function resume() {
    if (!currentRunId) return;
    await api(`/api/runs/${currentRunId}/resume`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ max_iter: (points.length || 0) + 10 }),
    });
    selectRun(currentRunId);
  }

  async function stop() {
    if (!currentRunId) return;
    await api(`/api/runs/${currentRunId}/stop`, { method: "POST" }).catch(() => {});
    $("status").textContent = "cancelling…";
  }

  // ── status lights ─────────────────────────────────────────────────────────
  async function loadLights() {
    try {
      const { status } = await api("/api/proposers");
      const el = $("lights");
      el.innerHTML = Object.entries(status).map(([k, v]) =>
        `<span class="light"><span class="dot ${v.ready ? "on" : "off"}"></span>${k}</span>`
      ).join("");
    } catch (_) { /* ignore */ }
  }

  // ── wire up ───────────────────────────────────────────────────────────────
  $("launch").addEventListener("click", launch);
  $("resume").addEventListener("click", resume);
  $("stop").addEventListener("click", stop);
  $("runner").addEventListener("change", () => {
    $("brokerFields").style.display = $("runner").value === "broker" ? "flex" : "none";
  });
  refreshRunList();
  loadLights();
  setInterval(refreshRunList, 4000); // keep the run list fresh
})();
