"""Runners — execute a proposed candidate and return its score.

A runner is the *how* the controller leaves abstract: given a candidate's
proposal text, run it and return ``(score_or_None, raw_output)``. The library
ships two:

* :class:`CallableRunner` — wraps any ``def run(proposal) -> (score, output)``.
  Zero infra; ideal for tests, local experiments, or any in-process objective.
* :class:`BrokerRunner` — the real UCL GPU path: claim a GPU via the broker,
  submit the candidate's run command through ``ucl_gpu_infra.stage6_infra``, poll
  the per-run shim to terminal via ``run_poller``, then score from the run
  workspace's ``result.json`` / ``SCORE=`` sentinel. This is the ~50-line app
  glue that used to live tangled inside the fork's pipeline_engine.

Both honour the same protocol, so the :class:`HillClimber` facade is agnostic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple

from loguru import logger

from . import scoring


class Runner(Protocol):
    """Run a candidate; return ``(score_or_None, raw_output)``. A ``None`` score
    means the run failed / produced no measurable objective — the controller
    records it but does not let it count toward the plateau."""

    def run(self, proposal: str, iteration: int) -> Tuple[Optional[float], str]:
        ...


class CallableRunner:
    """Adapt a plain callable into a :class:`Runner`. The callable receives the
    proposal text and returns either a bare score or ``(score, output)``."""

    def __init__(self, fn: Callable[[str], object]):
        self._fn = fn

    def run(self, proposal: str, iteration: int) -> Tuple[Optional[float], str]:
        out = self._fn(proposal)
        if isinstance(out, tuple) and len(out) == 2:
            score, text = out
            return (None if score is None else float(score)), str(text)
        return (None if out is None else float(out)), ""


@dataclass
class BrokerRunner:
    """Run candidates on the UCL GPU broker.

    ``workspace_dir`` is where the proposer has written the candidate's code and
    where the run leaves ``result.json``. ``run_command`` is how to execute the
    candidate (the proposer should also print ``SCORE=<n>`` as its final line).
    ``project_id`` is the broker lease holder + the ``omc/<id>/<iter>`` workdir
    marker the poller matches on.
    """

    project_id: str
    workspace_dir: str
    run_command: str
    config_path: str = ""
    poll_interval_s: float = 15.0
    timeout_s: float = 3600.0

    def run(self, proposal: str, iteration: int) -> Tuple[Optional[float], str]:
        # Imported here so the library imports without the infra package present
        # (e.g. for CallableRunner-only / test use).
        from ucl_gpu_infra import gpu_broker, stage6_infra, run_poller

        iter_id = f"iter_{iteration:03d}"
        remote_dest = f"omc/{self.project_id}/{iter_id}"
        holder = self.project_id

        # Clear any stale result.json so a run that fails to write a fresh one
        # scores as None, not as last iteration's number.
        try:
            from pathlib import Path
            stale = Path(self.workspace_dir) / "result.json"
            if stale.exists():
                stale.unlink()
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not clear stale result.json: {}", exc)

        lease, status = gpu_broker.claim(holder, holder="hc_run")
        if status == "unavailable":
            logger.warning("iter {} — no free GPU; scoring as failed run", iteration)
            return None, "engine run unavailable: no free GPU"
        env = gpu_broker.env_for(lease)
        try:
            scripts = stage6_infra.find_infra_scripts()
            receipt = stage6_infra.Receipt(
                smoke_cmd=self.run_command, code_dir=self.workspace_dir, remote_dest=remote_dest,
            )
            res = stage6_infra.submit(receipt, scripts, self.config_path, kind="smoke", env=env)
            if not res.ok:
                logger.warning("iter {} submit failed: {}", iteration, res.error)
                return None, f"submit failed: {res.error}"

            poller = run_poller.RunPoller(find_marker=lambda rid: f"omc/{rid}/{iter_id}")
            deadline = time.monotonic() + self.timeout_s
            while time.monotonic() < deadline:
                if poller.all_terminal(self.project_id, [res.run_id]):
                    break
                time.sleep(self.poll_interval_s)
            else:
                logger.warning("iter {} timed out after {}s", iteration, self.timeout_s)
                return None, "run timed out"

            info = stage6_infra.query_status(res.run_id, scripts, env=env) or {}
            output = info.get("log_tail", "") or ""
            score = scoring.resolve_score(self.workspace_dir, output)
            return score, output
        finally:
            gpu_broker.release(holder)
