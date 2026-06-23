"""Thin web layer over the HillClimber library.

This file is deliberately small: it constructs a :class:`HillClimber`, runs its
:meth:`stream` in a background thread, and forwards each event to a websocket.
Everything the UI renders comes from the library's event stream — if the UI ever
needs data not in an event, that's a gap to close in the library, not here.

The library is the product; this is a shell. A headless sweep needs none of this.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..core.climber import HillClimber
from ..core.runner import BrokerRunner, CallableRunner

_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="continual-auto-research")


def _build_climber(cfg: dict) -> HillClimber:
    """Construct a HillClimber from a UI/JSON config. The proposer here is a
    placeholder echo for the demo UI; a real deployment injects an LLM proposer.
    Swap ``runner`` to BrokerRunner(...) to run on the UCL GPU."""
    direction = cfg.get("direction", "min")

    def demo_proposer(context: str) -> str:
        return f"candidate based on:\n{context[:400]}"

    runner_kind = cfg.get("runner", "demo")
    if runner_kind == "broker":
        runner = BrokerRunner(
            project_id=cfg["project_id"],
            workspace_dir=cfg["workspace_dir"],
            run_command=cfg["run_command"],
            config_path=cfg.get("config_path", ""),
        )
    else:
        # Demo runner: a trivial in-process objective so the UI works with no GPU.
        import random
        _rng = random.Random(0)
        runner = CallableRunner(lambda proposal: (_rng.random(), "SCORE=%.4f" % _rng.random()))

    return HillClimber(propose=demo_proposer, runner=runner, direction=direction)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    idx = _FRONTEND / "index.html"
    return idx.read_text(encoding="utf-8") if idx.is_file() else "<h1>continual-auto-research</h1>"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.websocket("/ws/run")
async def ws_run(ws: WebSocket) -> None:
    """Start a climb and stream its events. The client sends one JSON config
    message; the server forwards every HillClimber event until ``done``."""
    await ws.accept()
    try:
        cfg = json.loads(await ws.receive_text())
    except Exception:
        await ws.close(code=1003)
        return

    climber = _build_climber(cfg)
    max_iter = int(cfg.get("max_iter", 20))
    patience = int(cfg.get("patience", 4))
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _produce() -> None:
        # stream() is blocking (the runner may sleep on a GPU poll), so it runs in
        # a thread; events are handed back to the event loop via the queue.
        for event in climber.stream(max_iter=max_iter, patience=patience):
            loop.call_soon_threadsafe(queue.put_nowait, event)
        loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    task = loop.run_in_executor(None, _produce)
    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            await ws.send_text(json.dumps(event))
    except WebSocketDisconnect:
        pass
    finally:
        await task


if _FRONTEND.is_dir():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")


def run() -> None:
    """Entry point: ``python -m continual_auto_research.api.app`` or the
    ``car-serve`` console script."""
    import os
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    run()
