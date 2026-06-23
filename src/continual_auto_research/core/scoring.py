"""Score extraction — the proposer/run output contract.

Two ways a run reports its objective, both consolidated here as pure functions:

1. ``result.json`` (authoritative): ``{"score": <n>, "direction": "min"|"max",
   "status": ...}`` written into the run workspace. A non-executed status
   (draft / proposal / skipped / error) DROPS the score — a placeholder that was
   never run must not be trusted as a real measurement. This is the #73 fix from
   the fork: a fabricated ``{"score": 0.0, "status": "draft"}`` once became the
   incumbent under a min direction and poisoned the whole climb.

2. ``SCORE=<n>`` sentinel in stdout (the engine-run contract): the run prints its
   objective as a line-anchored ``SCORE=<number>`` as its final stdout line. The
   LAST occurrence wins. Strict, case-sensitive, whole-line — so incidental
   stdout like ``best_score=0.9`` can't be mistaken for the sentinel.

``read_result_json`` wins over ``extract_score`` when both are present.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Optional

from loguru import logger

# Statuses that mean the run actually executed (so its score is real). An empty
# status is the legacy clean-run shape and is also trusted.
_EXECUTED_STATUSES = {
    "", "done", "ok", "success", "succeeded", "complete", "completed", "finished",
}

# SCORE=<number> sentinel: line-anchored, case-sensitive, whole-line.
_SCORE_SENTINEL = re.compile(
    r"(?m)^[ \t]*SCORE[ \t]*=[ \t]*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)[ \t]*$",
)


def _num(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_result_json(workspace_dir: str) -> Optional[dict]:
    """Read ``<workspace_dir>/result.json`` and return ``{"score"?, "direction"?}``.

    The authoritative score source. Returns ``None`` if the file is
    absent/unreadable/empty. A non-executed ``status`` drops the score (but keeps
    a declared direction — that's an intent, not a fabricated metric). A
    non-finite score is passed through as NaN (a failed run, not a missing one).
    """
    try:
        p = Path(workspace_dir) / "result.json"
        if not p.is_file():
            return None
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — best-effort; fall back to prose
        logger.warning("result.json unreadable ({}); falling back to prose parse", exc)
        return None
    if not isinstance(obj, dict):
        return None

    out: dict = {}
    raw_status = obj.get("status")
    status = raw_status.strip().lower() if isinstance(raw_status, str) else ""
    score_is_trusted = status in _EXECUTED_STATUSES
    if not score_is_trusted:
        logger.warning(
            "result.json status={!r} is not an executed run — ignoring its score "
            "(a draft/placeholder must not score as real)", raw_status,
        )
    for key in ("score", "metric"):
        if key in obj and score_is_trusted:
            f = _num(obj[key])
            if f is None:
                continue
            out["score"] = f if math.isfinite(f) else float("nan")
            break

    raw_direction = obj.get("direction")
    d = raw_direction.strip().lower() if isinstance(raw_direction, str) else ""
    if d in ("min", "max"):
        out["direction"] = d
    return out or None


def extract_score(result: str) -> Optional[float]:
    """Extract a scalar score from a run's stdout/prose, in order:

      0. a ``SCORE=<number>`` sentinel line (primary contract; LAST one wins)
      1. a fenced ```json block with a "score"/"metric" key
      2. a bare JSON object anywhere with "score"/"metric"
      3. a ``score: <n>`` / ``metric: <n>`` line

    Returns the float, or ``None`` if nothing parseable is present.
    """
    if not result:
        return None

    # 0) SCORE=<n> sentinel — the run prints it as its final line; last wins.
    hits = _SCORE_SENTINEL.findall(result)
    if hits:
        n = _num(hits[-1])
        if n is not None:
            return n

    # 1) fenced ```json / ```RESULT_JSON blocks
    for m in re.finditer(r"```(?:json|RESULT_JSON)?\s*(\{.*?\})\s*```", result, re.DOTALL | re.IGNORECASE):
        try:
            obj = json.loads(m.group(1))
        except Exception:  # noqa: BLE001
            continue
        for key in ("score", "metric"):
            if key in obj and _num(obj[key]) is not None:
                return _num(obj[key])

    # 2) any bare {...} containing score/metric
    for m in re.finditer(r"\{[^{}]*?(?:\"score\"|\"metric\")[^{}]*?\}", result):
        try:
            obj = json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            continue
        for key in ("score", "metric"):
            if key in obj and _num(obj[key]) is not None:
                return _num(obj[key])

    # 3) "score: 0.83" / "metric = 12.4" prose
    m = re.search(r"\b(?:score|metric)\b\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", result, re.IGNORECASE)
    if m:
        return _num(m.group(1))
    return None


def resolve_score(workspace_dir: str, result_text: str) -> Optional[float]:
    """The authoritative result.json score if present and trusted, else the
    score parsed from the run's prose. ``None`` if neither yields a number
    (the caller scores that as a failed/non-finite run)."""
    rj = read_result_json(workspace_dir) or {}
    if "score" in rj:
        return rj["score"]
    return extract_score(result_text)
