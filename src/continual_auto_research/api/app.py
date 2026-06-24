"""Web layer over the HillClimber library — a thin shell, not the product.

Exposes the library as a small app: create/list/get/resume/stop/delete runs
(REST), and a per-run websocket that forwards the climber's event stream. All
data the UI shows comes from the library's events/state — the API just persists
and routes them.

The library stays usable headless; none of this is needed to optimise in-process.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import store
from .runs import manager

_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="continual-auto-research")


# ── pages / health ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    idx = _FRONTEND / "index.html"
    return idx.read_text(encoding="utf-8") if idx.is_file() else "<h1>continual-auto-research</h1>"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/proposers")
def proposers() -> dict:
    """The proposer backends the UI can offer + their live readiness, so the
    launch form can show status lights and disable unavailable ones."""
    from ..core.proposers import backend_status, OLLAMA_PROXY_URL
    return {
        "backends": [
            {"kind": "claude", "label": "Claude (claude -p)", "needs": ["claude CLI"]},
            {"kind": "api", "label": "API (hosted proxy)", "needs": ["DEFAULT_API_BASE_URL", "key"]},
            {"kind": "ollama", "label": "Local (Ollama on UCL GPU)", "needs": ["ollama proxy"]},
        ],
        "status": backend_status(),
        "ollama_proxy": OLLAMA_PROXY_URL,
    }


@app.get("/api/health/backends")
def health_backends() -> dict:
    from ..core.proposers import backend_status
    return backend_status()


# ── runs REST ────────────────────────────────────────────────────────────────

@app.get("/api/runs")
def list_runs() -> dict:
    return {"runs": store.list_runs()}


@app.post("/api/runs")
async def create_run(cfg: dict) -> dict:
    """Create + start a run. Body is the full run config (direction, max_iter,
    patience, target_score, runner{...}, proposer{...}). Returns the run record."""
    loop = asyncio.get_running_loop()
    rec, _ = manager.create(cfg, loop)
    return rec


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    rec = store.load(run_id)
    if not rec:
        raise HTTPException(status_code=404, detail="run not found")
    return rec


@app.post("/api/runs/{run_id}/resume")
async def resume_run(run_id: str, body: dict) -> dict:
    """Resume a stored run with ``{max_iter}`` more budget."""
    max_iter = int(body.get("max_iter", 0))
    if max_iter <= 0:
        raise HTTPException(status_code=400, detail="max_iter must be > 0")
    loop = asyncio.get_running_loop()
    rec = manager.resume(run_id, max_iter, loop)
    if rec is None:
        raise HTTPException(status_code=404, detail="run not found")
    return rec


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str) -> dict:
    """Request cancellation; the run ends after its current iteration."""
    if not manager.cancel(run_id):
        raise HTTPException(status_code=404, detail="run not live (already finished?)")
    return {"status": "cancelling", "id": run_id}


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str) -> dict:
    if not store.delete(run_id):
        raise HTTPException(status_code=404, detail="run not found")
    return {"status": "deleted", "id": run_id}


# ── per-run live event stream ────────────────────────────────────────────────

@app.websocket("/ws/runs/{run_id}")
async def ws_run_events(ws: WebSocket, run_id: str) -> None:
    """Stream a live run's events. If the run already finished, replay its stored
    history as ``scored`` events then close — so the UI renders past runs too."""
    await ws.accept()
    live = manager.get_live(run_id)
    # Treat an already-finished run as not-live: replay from the store rather than
    # subscribing to a worker that has already emitted its _eof to nobody (which
    # would hang). Check both the asyncio done flag AND the persisted status, since
    # the store flips to a terminal status a beat before the done event is set.
    if live is not None:
        if live.done.is_set():
            live = None
        else:
            rec0 = store.load(run_id) or {}
            if rec0.get("status") in ("done", "failed"):
                live = None
    if live is None:
        # finished/unknown run: replay from the store, then done.
        rec = store.load(run_id)
        if rec:
            for c in rec.get("history", []):
                await ws.send_text(json.dumps({
                    "type": "scored", "iteration": c.get("iteration"),
                    "proposal": c.get("proposal"), "score": c.get("score"),
                    "improved": c.get("accepted"), "best": (rec.get("best") or {}).get("score"),
                    "raw_result": c.get("raw_result", ""), "replayed": True,
                }))
            await ws.send_text(json.dumps({
                "type": "done", "stop_reason": rec.get("stop_reason", ""),
                "iterations": rec.get("iterations", 0), "best": rec.get("best"),
            }))
        await ws.close()
        return

    q = live.subscribe()
    try:
        while True:
            event = await q.get()
            if event.get("type") == "_eof":
                break
            await ws.send_text(json.dumps(event))
    except WebSocketDisconnect:
        pass
    finally:
        live.unsubscribe(q)


# ── back-compat: the original one-shot ws/run (demo) ─────────────────────────

@app.websocket("/ws/run")
async def ws_run_legacy(ws: WebSocket) -> None:
    """Legacy single-run socket: client sends one config, server creates a run and
    forwards its events. Kept so the original minimal UI still works."""
    await ws.accept()
    try:
        cfg = json.loads(await ws.receive_text())
    except Exception:
        await ws.close(code=1003)
        return
    loop = asyncio.get_running_loop()
    # subscribe BEFORE starting so the first `proposed` event isn't missed.
    rec, live = manager.create(cfg, loop, autostart=False)
    q = live.subscribe()
    manager.start(rec["id"])
    try:
        while True:
            event = await q.get()
            if event.get("type") == "_eof":
                break
            await ws.send_text(json.dumps(event))
    except WebSocketDisconnect:
        pass
    finally:
        live.unsubscribe(q)


if _FRONTEND.is_dir():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")


def run() -> None:
    """Entry point: ``car-serve`` console script / ``python -m ...api.app``."""
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    run()
