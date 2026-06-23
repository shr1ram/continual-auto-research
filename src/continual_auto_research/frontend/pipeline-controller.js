// pipeline-controller.js — the entire UI logic.
//
// Deliberately tiny: open a websocket, send a run config, render each event the
// HillClimber library emits (proposed / scored / accepted / done). The library
// is the product; this is a thin shell. If something can't be rendered from an
// event here, the fix belongs in the library's event stream, not in this file.

(function () {
  const $ = (id) => document.getElementById(id);
  const startBtn = $("start");
  const bestEl = $("best");
  const statusEl = $("status");
  const rows = $("rows");

  function reset() {
    rows.innerHTML = "";
    bestEl.textContent = "—";
    statusEl.textContent = "running…";
  }

  function fmt(v) {
    return v === null || v === undefined ? "—" : Number(v).toFixed(4);
  }

  function onEvent(ev) {
    switch (ev.type) {
      case "scored": {
        const tr = document.createElement("tr");
        if (ev.improved) tr.className = "accepted";
        if (ev.score === null) tr.className = "failed";
        const pill = ev.improved
          ? '<span class="pill acc">improved</span>'
          : '<span class="pill rej">no</span>';
        tr.innerHTML =
          `<td>${ev.iteration}</td><td>${fmt(ev.score)}</td>` +
          `<td>${pill}</td><td>${fmt(ev.best)}</td>`;
        rows.appendChild(tr);
        if (ev.best !== null && ev.best !== undefined) bestEl.textContent = fmt(ev.best);
        break;
      }
      case "done": {
        statusEl.textContent =
          `done — ${ev.stop_reason} after ${ev.iterations} iters`;
        startBtn.disabled = false;
        break;
      }
      // "proposed" / "accepted" carry no extra row state in this minimal UI.
    }
  }

  function start() {
    startBtn.disabled = true;
    reset();
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/run`);
    ws.onopen = () =>
      ws.send(JSON.stringify({ runner: "demo", direction: "max", max_iter: 20, patience: 4 }));
    ws.onmessage = (m) => onEvent(JSON.parse(m.data));
    ws.onerror = () => { statusEl.textContent = "connection error"; startBtn.disabled = false; };
    ws.onclose = () => { if (startBtn.disabled) { startBtn.disabled = false; } };
  }

  startBtn.addEventListener("click", start);
})();
