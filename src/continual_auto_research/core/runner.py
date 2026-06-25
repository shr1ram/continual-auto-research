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
        self.last_trace: Optional[dict] = None

    def run(self, proposal: str, iteration: int) -> Tuple[Optional[float], str]:
        out = self._fn(proposal)
        if isinstance(out, tuple) and len(out) == 2:
            score, text = out
            score, text = (None if score is None else float(score)), str(text)
        else:
            score, text = (None if out is None else float(out)), ""
        self.last_trace = {"runner": "callable", "command": "(in-process callable)", "output": text}
        return score, text


@dataclass
class BrokerRunner:
    """Run candidates on the UCL GPU broker.

    ``workspace_dir`` is where the proposer has written the candidate's code and
    where the run leaves ``result.json``. ``run_command`` is how to execute the
    candidate (the proposer should also print ``SCORE=<n>`` as its final line);
    it may be a fixed string, a callable ``(proposal) -> command``, OR empty/None
    — in which case the command is PARSED FROM THE PROPOSAL each iteration (the
    proposer emits a ``cd … && python …`` run command, exactly as the fork's
    hill-climbing mode did). ``project_id`` is the broker lease holder + the
    ``omc/<id>/<iter>`` workdir marker the poller matches on.

    ``direct`` selects the infra target: ``False`` (default) leases a GPU via the
    broker (UCL lab); ``True`` skips the broker claim/release and submits straight
    to whatever ``INFRA_SERVER_URL``/``INFRA_SESSION_KEY`` point at (the H100 box).
    """

    project_id: str
    workspace_dir: str
    run_command: object = ""     # str | Callable[[str], str] | "" (parse from proposal)
    config_path: str = ""
    poll_interval_s: float = 15.0
    timeout_s: float = 3600.0
    direct: bool = False         # True → H100 direct-submit (no broker lease)

    def _command_for(self, proposal: str) -> str:
        rc = self.run_command
        if callable(rc):
            return rc(proposal)
        if rc:
            # a fixed command; still honour a {proposal} placeholder for params
            return rc.replace("{proposal}", str(proposal).strip()) if "{proposal}" in rc else rc
        # No command configured — parse one out of the proposal itself (the
        # proposer's `cd … && python …` line). This is what makes a UI-launched
        # broker run work with NO manually-typed run_command (the fork contract).
        try:
            from ucl_gpu_infra import stage6_infra
            receipt = stage6_infra.parse_receipt(
                proposal or "", project_id=self.project_id,
            )
            return receipt.smoke_cmd or receipt.full_cmd or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not parse run command from proposal: {}", exc)
            return ""

    def run(self, proposal: str, iteration: int) -> Tuple[Optional[float], str]:
        # Imported here so the library imports without the infra package present
        # (e.g. for CallableRunner-only / test use).
        from ucl_gpu_infra import gpu_broker, stage6_infra, run_poller

        iter_id = f"iter_{iteration:03d}"
        remote_dest = f"omc/{self.project_id}/{iter_id}"
        holder = self.project_id
        command = self._command_for(proposal)

        target = "h100" if self.direct else "ucl-broker"

        def _trace(output: str, run_id: str = "") -> None:
            # Record the run's command + FULL output (per the trace-window design)
            # so the UI can show exactly what executed on the GPU.
            self.last_trace = {
                "runner": "broker", "target": target, "command": command,
                "output": output, "run_id": run_id, "workspace_dir": self.workspace_dir,
            }

        # No runnable command in the proposal (and none configured): the candidate
        # didn't emit a `cd … && python …` line / SCORE= contract. Score as a
        # failed run with a clear reason rather than the opaque "no smoke command".
        if not command:
            logger.warning("iter {} — proposal has no runnable command; scoring as failed", iteration)
            _trace("no runnable command found in proposal (expected a `cd … && python …` line)")
            return None, "no runnable command found in proposal"

        # Clear any stale result.json so a run that fails to write a fresh one
        # scores as None, not as last iteration's number.
        try:
            from pathlib import Path
            stale = Path(self.workspace_dir) / "result.json"
            if stale.exists():
                stale.unlink()
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not clear stale result.json: {}", exc)

        # Infra target. Default: lease a GPU from the UCL broker. H100 (direct):
        # skip the broker entirely and submit straight to the team H100 server.
        # The H100 endpoint lives in its OWN env vars (H100_INFRA_SERVER_URL /
        # H100_INFRA_SESSION_KEY) so it doesn't collide with the UCL static-shim
        # creds (INFRA_SERVER_URL/INFRA_SESSION_KEY) the broker path uses. We do
        # NOT fall back to the plain INFRA_* here: on the box those are set to the
        # UCL shim, so a fallback would silently submit an H100 run to the UCL
        # backend (the wrong place). Missing H100 config must fail fast instead.
        if self.direct:
            import os as _os
            h_url = _os.environ.get("H100_INFRA_SERVER_URL", "")
            h_key = _os.environ.get("H100_INFRA_SESSION_KEY", "")
            if not (h_url and h_key):
                logger.warning("iter {} — H100 infra not configured "
                               "(set H100_INFRA_SERVER_URL/H100_INFRA_SESSION_KEY)", iteration)
                _trace("H100 infra not configured: set H100_INFRA_SERVER_URL / H100_INFRA_SESSION_KEY")
                return None, "H100 infra not configured"
            env = dict(_os.environ, INFRA_SERVER_URL=h_url, INFRA_SESSION_KEY=h_key)
            lease = None
        else:
            lease, status = gpu_broker.claim(holder, holder="hc_run")
            if status == "unavailable":
                logger.warning("iter {} — no free GPU; scoring as failed run", iteration)
                _trace("engine run unavailable: no free GPU")
                return None, "engine run unavailable: no free GPU"
            env = gpu_broker.env_for(lease)
        try:
            scripts = stage6_infra.find_infra_scripts()
            receipt = stage6_infra.Receipt(
                smoke_cmd=command, code_dir=self.workspace_dir, remote_dest=remote_dest,
            )
            res = stage6_infra.submit(receipt, scripts, self.config_path, kind="smoke", env=env)
            if not res.ok:
                logger.warning("iter {} submit failed: {}", iteration, res.error)
                _trace(f"submit failed: {res.error}")
                return None, f"submit failed: {res.error}"

            poller = run_poller.RunPoller(find_marker=lambda rid: f"omc/{rid}/{iter_id}")
            deadline = time.monotonic() + self.timeout_s
            while time.monotonic() < deadline:
                if poller.all_terminal(self.project_id, [res.run_id]):
                    break
                time.sleep(self.poll_interval_s)
            else:
                logger.warning("iter {} timed out after {}s", iteration, self.timeout_s)
                _trace("run timed out", res.run_id)
                return None, "run timed out"

            info = stage6_infra.query_status(res.run_id, scripts, env=env) or {}
            output = info.get("log_tail", "") or ""
            _trace(output, res.run_id)
            # Score precedence (see scoring.resolve_score): a result.json in the
            # workspace wins, else the SCORE= sentinel in the run's stdout. NOTE:
            # the workspace result.json is only visible here if ``workspace_dir``
            # is shared with the run (the UCL NFS case). For a non-shared remote
            # run, the result.json lands on the remote box and is NOT readable
            # locally — there the SCORE=<n> sentinel in ``log_tail`` is the
            # authoritative score, which is why the proposer contract REQUIRES it.
            score = scoring.resolve_score(self.workspace_dir, output)
            return score, output
        finally:
            # Only release a broker lease we actually claimed. The direct/H100
            # path never claimed one — calling release there would be a no-op at
            # best and could disturb an unrelated lease at worst.
            if not self.direct:
                gpu_broker.release(holder)
