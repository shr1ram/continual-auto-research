"""Run store — JSON-file persistence for climbs.

The HillClimber library is ephemeral by design (a run lives in-process). The web
app needs runs to survive a restart and to list/compare them, so this adds a
small file-backed registry — one JSON file per run under a runs directory.

Deliberately minimal: no DB, no locking beyond an atomic write. A run record is
the config it was launched with + its live state (status, history, best) so the
controller state round-trips for resume. Concurrency is single-writer-per-run in
practice (one background task owns a run); the atomic replace guards against a
torn read by a concurrent lister.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from loguru import logger


def _runs_dir() -> Path:
    d = Path(os.environ.get("CAR_RUNS_DIR", str(Path.home() / ".continual-auto-research" / "runs")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(run_id: str) -> Path:
    # run_id is server-generated (see new_run_id) so it's filesystem-safe; guard
    # anyway against traversal if a caller ever passes an arbitrary id.
    safe = "".join(c for c in run_id if c.isalnum() or c in "-_")
    return _runs_dir() / f"{safe}.json"


def new_run_id(seq: int) -> str:
    """A sortable, filesystem-safe run id. ``seq`` makes it unique + ordered
    without needing a clock (Date.now is unavailable in some contexts); the
    caller passes a monotonic counter."""
    return f"run-{seq:06d}"


def save(record: dict) -> dict:
    """Atomically write a run record. Returns it. Requires record['id']."""
    rid = record["id"]
    p = _path(rid)
    # atomic: write to a temp file in the same dir, then replace.
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{rid}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return record


def load(run_id: str) -> Optional[dict]:
    """Read one run record, or None if absent/unreadable."""
    p = _path(run_id)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("run record {} unreadable: {}", run_id, exc)
        return None


def list_runs() -> list[dict]:
    """Summaries of all runs, newest id first. Each summary is a trimmed view
    (no full history) so the list endpoint stays light."""
    out = []
    for p in sorted(_runs_dir().glob("run-*.json"), reverse=True):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — skip a torn/corrupt file, don't fail the list
            continue
        out.append({
            "id": r.get("id"),
            "status": r.get("status"),
            "direction": (r.get("config") or {}).get("direction"),
            "best_score": (r.get("best") or {}).get("score") if r.get("best") else None,
            "iterations": r.get("iterations", 0),
            "stop_reason": r.get("stop_reason", ""),
            "created_seq": r.get("created_seq"),
        })
    return out


def delete(run_id: str) -> bool:
    """Remove a run record. True if it existed."""
    p = _path(run_id)
    if p.is_file():
        p.unlink()
        return True
    return False


def next_seq() -> int:
    """A monotonic counter for run ids, derived from existing files so it
    survives restarts without a stored cursor."""
    existing = list(_runs_dir().glob("run-*.json"))
    if not existing:
        return 1
    nums = []
    for p in existing:
        try:
            nums.append(int(p.stem.split("-")[1]))
        except (IndexError, ValueError):
            continue
    return (max(nums) + 1) if nums else 1
