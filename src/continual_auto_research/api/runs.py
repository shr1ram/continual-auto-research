"""Run manager — owns the lifecycle of climbs started via the API.

Bridges the synchronous, blocking :class:`HillClimber` (the runner may sleep on a
GPU poll) to the async web server. Each run executes in a thread; its events are
(a) persisted into the store as history accumulates and (b) fanned out to any
websocket clients subscribed to that run.

Keeps the library pure: the manager is app glue, the climber knows nothing about
HTTP, persistence, or other runs.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Optional

from loguru import logger

from ..core.climber import HillClimber
from ..core.hill_climb import HillClimbConfig, HillClimbState
from . import store
from .builders import build_climber


class _LiveRun:
    """A run executing in a background thread, with its event subscribers."""

    def __init__(self, run_id: str, climber: HillClimber, loop: asyncio.AbstractEventLoop):
        self.run_id = run_id
        self.climber = climber
        self._loop = loop
        self._subscribers: set[asyncio.Queue] = set()
        self._thread: Optional[threading.Thread] = None
        self.done = asyncio.Event()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def _emit(self, event: dict) -> None:
        # called from the worker thread → marshal to the loop thread. Skip the
        # hop entirely when nobody is listening (a REST-created run with no WS
        # subscriber), and tolerate a closed loop (the server/test loop may be
        # gone before a background run finishes) — persistence does not depend on
        # the loop, only live fan-out does.
        if not self._subscribers:
            return
        def _push():
            for q in list(self._subscribers):
                q.put_nowait(event)
        try:
            self._loop.call_soon_threadsafe(_push)
        except RuntimeError:
            pass  # event loop closed; live fan-out is best-effort

    def start(self, max_iter: int, patience: int) -> None:
        def _work():
            try:
                for event in self.climber.stream(max_iter=max_iter, patience=patience):
                    self._emit(event)
                    # persist incrementally so a crash/restart keeps progress + the
                    # list endpoint reflects live status without holding the run.
                    self._persist("running")
                self._persist("done")
            except Exception as exc:  # noqa: BLE001 — surface as a failed run, never crash the server
                logger.exception("run {} crashed: {}", self.run_id, exc)
                self._persist("failed", error=str(exc))
                self._emit({"type": "error", "error": str(exc)})
            finally:
                self._emit({"type": "_eof"})
                try:
                    self._loop.call_soon_threadsafe(self.done.set)
                except RuntimeError:
                    pass  # loop closed; done is only awaited by live subscribers

        self._thread = threading.Thread(target=_work, name=f"climb-{self.run_id}", daemon=True)
        self._thread.start()

    def _persist(self, status: str, error: str = "") -> None:
        rec = store.load(self.run_id) or {"id": self.run_id}
        st = self.climber.controller.state
        rec.update({
            "status": status,
            "state": st.to_dict(),
            "best": st.best,
            "history": st.history,
            "iterations": st.iteration,
            "stop_reason": st.stop_reason,
            "traces": self.climber.traces,   # per-iteration prompt/response/command/output
        })
        if error:
            rec["error"] = error
        store.save(rec)


class RunManager:
    """Process-wide registry of live + stored runs."""

    def __init__(self):
        self._live: dict[str, _LiveRun] = {}

    def create(self, cfg: dict, loop: asyncio.AbstractEventLoop,
               autostart: bool = True) -> tuple[dict, "_LiveRun"]:
        """Persist a new run, build its climber, register it. With
        ``autostart`` (the REST default) the worker starts immediately; pass
        ``autostart=False`` to subscribe to the live run BEFORE starting it, so a
        websocket client doesn't miss the first ``proposed`` event (a
        subscribe-after-start race). Returns ``(record, live_run)``."""
        seq = store.next_seq()
        run_id = store.new_run_id(seq)
        rec = {
            "id": run_id, "created_seq": seq, "status": "starting",
            "config": cfg, "state": None, "best": None, "history": [],
            "iterations": 0, "stop_reason": "",
        }
        store.save(rec)

        try:
            climber = build_climber(cfg, run_id=run_id)
        except Exception as exc:  # noqa: BLE001 — fail loud: a bad config must not start a run
            rec.update({"status": "failed", "error": str(exc)})
            store.save(rec)
            raise
        live = _LiveRun(run_id, climber, loop)
        self._live[run_id] = live
        if autostart:
            self._start(live, cfg)
            rec["status"] = "running"
            store.save(rec)
        return rec, live

    def _start(self, live: "_LiveRun", cfg: dict) -> None:
        live.start(int(cfg.get("max_iter", 20)), int(cfg.get("patience", 4)))

    def start(self, run_id: str) -> None:
        """Start a run created with ``autostart=False`` (after a WS subscribed)."""
        live = self._live.get(run_id)
        rec = store.load(run_id) or {}
        if live and live._thread is None:
            self._start(live, rec.get("config") or {})
            rec["status"] = "running"
            store.save(rec)

    def resume(self, run_id: str, max_iter: int, loop: asyncio.AbstractEventLoop) -> Optional[dict]:
        """Resume a stored run with a larger budget. Returns the record, or None
        if the run doesn't exist."""
        rec = store.load(run_id)
        if not rec:
            return None
        cfg = dict(rec.get("config") or {})
        cfg["max_iter"] = max_iter
        state = HillClimbState.from_dict(rec.get("state") or {})
        climber = build_climber(cfg, state=state, run_id=run_id)
        live = _LiveRun(run_id, climber, loop)
        self._live[run_id] = live
        live.start(max_iter, int(cfg.get("patience", 4)))
        rec["status"] = "running"
        store.save(rec)
        return rec

    def cancel(self, run_id: str) -> bool:
        live = self._live.get(run_id)
        if not live:
            return False
        live.climber.cancel()
        return True

    def get_live(self, run_id: str) -> Optional[_LiveRun]:
        return self._live.get(run_id)


# Process-wide singleton.
manager = RunManager()
