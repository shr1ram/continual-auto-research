"""Build a HillClimber from a JSON run config.

Centralises runner + proposer selection so both the REST create path and the
resume path construct runs identically. Task #33 extends ``_build_proposer`` with
the real claude/api/ollama backends; for now it ships the demo proposer plus a
hook for the broker runner.
"""
from __future__ import annotations

import random
from typing import Callable, Optional

from ..core.climber import HillClimber
from ..core.hill_climb import HillClimbConfig, HillClimbState
from ..core.runner import BrokerRunner, CallableRunner, Runner


def _build_runner(cfg: dict) -> Runner:
    """Pick the runner. ``demo`` is an in-process objective so the UI works with
    no GPU; ``broker`` runs real experiments on the UCL GPU via ucl-gpu-infra."""
    r = cfg.get("runner") or {}
    kind = r.get("kind", "demo") if isinstance(r, dict) else str(r)
    if kind == "broker":
        return BrokerRunner(
            project_id=r["project_id"],
            workspace_dir=r["workspace_dir"],
            run_command=r["run_command"],
            config_path=r.get("config_path", ""),
            poll_interval_s=float(r.get("poll_interval_s", 15.0)),
            timeout_s=float(r.get("timeout_s", 3600.0)),
        )
    # demo: a deterministic-ish toy objective; seed varies per process so repeated
    # demo runs differ but a single run is reproducible.
    rng = random.Random(cfg.get("seed", 0))
    return CallableRunner(lambda proposal: (rng.random(), "SCORE=%.4f" % rng.random()))


def _build_proposer(cfg: dict) -> Callable[[str], str]:
    """Pick the proposer. Task #33 wires claude/api/ollama here; until then a
    placeholder that echoes the context (works with the demo runner)."""
    p = cfg.get("proposer") or {}
    kind = p.get("kind", "demo") if isinstance(p, dict) else str(p)
    try:
        from ..core.proposers import build_proposer  # available once #33 lands
        if kind != "demo":
            return build_proposer(p)
    except Exception:  # noqa: BLE001 — proposers module not present yet / misconfig
        pass

    def demo_proposer(context: str) -> str:
        return f"candidate based on:\n{context[:400]}"

    return demo_proposer


def build_climber(cfg: dict, state: Optional[HillClimbState] = None) -> HillClimber:
    """Construct a HillClimber from a run config (the REST/WS body)."""
    hc_cfg = HillClimbConfig()
    if cfg.get("direction"):
        hc_cfg.direction = cfg["direction"]
    if cfg.get("patience") is not None:
        hc_cfg.patience = int(cfg["patience"])
    if cfg.get("max_iter") is not None:
        hc_cfg.max_iterations = int(cfg["max_iter"])
    ts = cfg.get("target_score")
    hc_cfg.target_score = float(ts) if ts not in (None, "") else None

    return HillClimber(
        propose=_build_proposer(cfg),
        runner=_build_runner(cfg),
        config=hc_cfg,
        state=state,
    )
