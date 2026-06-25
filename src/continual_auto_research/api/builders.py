"""Build a HillClimber from a JSON run config.

Centralises runner + proposer selection so both the REST create path and the
resume path construct runs identically. Task #33 extends ``_build_proposer`` with
the real claude/api/ollama backends; for now it ships the demo proposer plus a
hook for the broker runner.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Optional

from ..core.climber import HillClimber
from ..core.hill_climb import HillClimbConfig, HillClimbState
from ..core.runner import BrokerRunner, CallableRunner, Runner


def _auto_workspace_dir(run_id: str) -> str:
    """The experiment workspace for a GPU run, created automatically so the user
    never has to type a path. Mirrors the fork's per-run ``experiment/`` dir:
    ``<runs_dir>/<run_id>/experiment``. The proposer writes the candidate's code
    here and the run leaves ``result.json`` here."""
    base = Path(os.environ.get(
        "CAR_RUNS_DIR", str(Path.home() / ".continual-auto-research" / "runs")))
    d = base / (run_id or "run") / "experiment"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001 — best effort; submit will surface a real error
        pass
    return str(d)


def _build_runner(cfg: dict, run_id: str = "") -> Runner:
    """Pick the runner.

    * ``demo`` — an in-process toy objective so the UI works with no GPU.
    * ``broker`` — real experiments on the UCL lab GPU (leased via the broker).
    * ``h100`` — real experiments submitted directly to the H100 box (no broker).

    For ``broker``/``h100`` the three plumbing fields are AUTOMATED so a UI launch
    needs none of them: ``project_id`` defaults to the run id, ``workspace_dir`` is
    auto-created under the runs dir, and ``run_command`` (when empty) is parsed from
    each proposal. Any explicitly-provided value still wins (advanced/manual use)."""
    r = cfg.get("runner") or {}
    kind = (r.get("kind", "demo") if isinstance(r, dict) else str(r)).lower()
    if kind in ("broker", "h100"):
        # project_id: the broker lease holder + omc/<id>/<iter> workdir marker.
        # Default to the run id so every run is uniquely keyed without user input.
        project_id = (r.get("project_id") or run_id or "car-run").strip()
        # workspace_dir: auto-create one unless the caller pinned a path.
        workspace_dir = (r.get("workspace_dir") or "").strip() or _auto_workspace_dir(run_id)
        # run_command: empty → BrokerRunner parses it from the proposal each
        # iteration (the fork contract). A {proposal} placeholder is honoured by
        # BrokerRunner._command_for when a fixed command is given.
        run_command = (r.get("run_command") or "").strip()
        return BrokerRunner(
            project_id=project_id,
            workspace_dir=workspace_dir,
            run_command=run_command,
            config_path=r.get("config_path", ""),
            poll_interval_s=float(r.get("poll_interval_s", 15.0)),
            timeout_s=float(r.get("timeout_s", 3600.0)),
            direct=(kind == "h100"),
        )
    # demo: a deterministic-ish toy objective; seed varies per process so repeated
    # demo runs differ but a single run is reproducible.
    rng = random.Random(cfg.get("seed", 0))
    return CallableRunner(lambda proposal: (rng.random(), "SCORE=%.4f" % rng.random()))


def _build_proposer(cfg: dict, objective: str = "") -> Callable[[str], str]:
    """Pick the proposer. Task #33 wires claude/api/ollama here; until then a
    placeholder that echoes the context (works with the demo runner).

    ``objective`` (the user's task description) is folded into the proposer's
    system prompt when the proposer config doesn't already set one — so the LLM's
    standing instruction names the task, not just the per-iteration context."""
    p = dict(cfg.get("proposer") or {})  # copy: we may inject `system`
    kind = p.get("kind", "demo") if isinstance(p, dict) else str(p)
    # Fold the objective into the proposer's system prompt (unless one was given
    # explicitly). proposer_context() also carries it per-iteration; this makes
    # it the standing instruction too.
    if objective and not p.get("system"):
        p["system"] = (
            "You are an optimisation proposer in a hill-climbing loop.\n"
            f"OBJECTIVE: {objective}\n"
            "You are given the best candidate so far and recent attempts with their "
            "measured scores. Propose exactly ONE new candidate that should beat the "
            "best, on this objective. Be concrete and concise. If the task specifies a "
            "run command or a SCORE= contract, follow it exactly. Output ONLY the "
            "candidate — no preamble, no explanation."
        )
    try:
        from ..core.proposers import build_proposer  # available once #33 lands
        if kind != "demo":
            return build_proposer(p)
    except Exception:  # noqa: BLE001 — proposers module not present yet / misconfig
        pass

    def demo_proposer(context: str) -> str:
        return f"candidate based on:\n{context[:400]}"

    return demo_proposer


def build_climber(cfg: dict, state: Optional[HillClimbState] = None,
                  run_id: str = "") -> HillClimber:
    """Construct a HillClimber from a run config (the REST/WS body).

    ``run_id`` is used to auto-derive the broker/h100 runner's project_id and
    workspace_dir so a UI launch needs no manual plumbing fields."""
    hc_cfg = HillClimbConfig()
    if cfg.get("direction"):
        hc_cfg.direction = cfg["direction"]
    if cfg.get("patience") is not None:
        hc_cfg.patience = int(cfg["patience"])
    if cfg.get("max_iter") is not None:
        hc_cfg.max_iterations = int(cfg["max_iter"])
    ts = cfg.get("target_score")
    hc_cfg.target_score = float(ts) if ts not in (None, "") else None
    # The objective/prompt: what the run is optimising. Threaded into both the
    # per-iteration proposer context (via the config) and the proposer's system
    # prompt (below). Accept a couple of aliases the UI/clients might send.
    objective = (cfg.get("objective") or cfg.get("prompt") or cfg.get("task") or "").strip()
    hc_cfg.objective = objective

    return HillClimber(
        propose=_build_proposer(cfg, objective=objective),
        runner=_build_runner(cfg, run_id=run_id),
        config=hc_cfg,
        state=state,
    )
