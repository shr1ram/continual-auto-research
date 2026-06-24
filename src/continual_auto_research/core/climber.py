"""HillClimber — the programmatic entry point.

This is the library face of continual-auto-research. Everything the UI shows is
obtainable from here; the web app is a thin shell that constructs a HillClimber,
calls :meth:`stream`, and forwards events. No FastAPI, no websocket, no DB is
needed to optimise — a batch sweep is just a loop over :meth:`run`.

    from continual_auto_research import HillClimber, CallableRunner

    hc = HillClimber(
        propose=my_proposer,           # (context: str) -> proposal text
        runner=CallableRunner(my_fn),  # or BrokerRunner(...) for the UCL GPU
        direction="min",
    )
    result = hc.run(max_iter=20, patience=4)     # blocking; returns RunResult
    # or, for a live feed:
    for event in hc.stream(max_iter=20):
        ...   # {"type": "scored"|"proposed"|"accepted"|"done", ...}

``propose`` is any callable taking the controller's proposer context (incumbent
best + recent attempts + plateau guidance) and returning the next candidate's
proposal text. ``runner`` executes that candidate and returns its score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from .hill_climb import HillClimbConfig, HillClimbController, HillClimbState
from .runner import CallableRunner, Runner


@dataclass
class RunResult:
    """The outcome of a climb: the winning candidate, why it stopped, and the
    full history (every candidate's iteration/proposal/score/accepted)."""

    best: Optional[dict]
    stop_reason: str
    iterations: int
    history: list

    @property
    def best_score(self) -> Optional[float]:
        return (self.best or {}).get("score") if self.best else None


class HillClimber:
    """Drive the propose → run → score → climb loop programmatically.

    Thin orchestration over :class:`HillClimbController` (policy) and a
    :class:`Runner` (execution). Construct once, then :meth:`run` (blocking) or
    :meth:`stream` (event iterator). Stateless between calls except for the
    controller it owns — pass an existing ``state`` to resume.
    """

    def __init__(
        self,
        propose: Callable[[str], str],
        runner: Runner,
        *,
        direction: Optional[str] = None,
        config: Optional[HillClimbConfig] = None,
        state: Optional[HillClimbState] = None,
    ):
        if not callable(propose):
            raise TypeError("propose must be callable: (context: str) -> proposal text")
        if callable(runner) and not hasattr(runner, "run"):
            runner = CallableRunner(runner)  # accept a bare callable for convenience
        self._propose = propose
        self._runner = runner
        cfg = config or HillClimbConfig()
        if direction:
            cfg.direction = direction
        self.controller = HillClimbController(config=cfg, state=state)
        self._cancelled = False

    def cancel(self) -> None:
        """Request the loop stop after the current iteration. Thread-safe (a plain
        flag check); the run ends with ``stop_reason="cancelled"``. Set from
        another thread (e.g. the API's stop route) while the climb runs."""
        self._cancelled = True

    # -- the event-producing core; run() and stream() are thin wrappers --------
    def _iterate(self, max_iter: Optional[int], patience: Optional[int]) -> Iterator[dict]:
        ctrl = self.controller
        if max_iter is not None:
            ctrl.config.max_iterations = max_iter
        if patience is not None:
            ctrl.config.patience = patience

        # Resume: a state snapshotted after a prior run stopped on BUDGET is
        # reactivated when this call raises the budget above the iterations
        # already done — the budget was the only thing that stopped it. A
        # plateau/target stop is a genuine convergence and is NOT auto-resumed
        # (re-running it would silently spin); the caller must reset stop_reason
        # explicitly to override that.
        if (ctrl.state.phase in ("done", "failed")
                and ctrl.state.stop_reason == "budget"
                and ctrl.state.iteration < ctrl.config.max_iterations):
            ctrl.state.phase = "propose"
            ctrl.state.stop_reason = ""

        while ctrl.should_continue():
            if self._cancelled:
                ctrl.state.stop_reason = "cancelled"
                break
            ctrl.begin_iteration()
            it = ctrl.state.iteration

            proposal = self._propose(ctrl.proposer_context())
            cand = ctrl.on_proposed(proposal)
            yield {"type": "proposed", "iteration": it, "proposal": proposal}

            raw_full = ""
            score, raw = self._runner.run(proposal, it)
            raw_full = (raw or "")[:4000]
            improved = ctrl.on_scored(cand, score if score is not None else float("nan"),
                                      raw_result=raw_full)
            yield {
                "type": "scored",
                "iteration": it,
                "proposal": proposal,           # echo so the UI table needs no join
                "score": cand.score,            # None if the run failed/non-finite
                "improved": improved,
                "best": ctrl.best_score,
                "stale_rounds": ctrl.state.stale_rounds,
                "raw_result": raw_full,         # measured run output (objective breakdown)
            }
            if improved:
                yield {"type": "accepted", "iteration": it, "best": ctrl.best_score}

        ctrl.finish()
        yield {
            "type": "done",
            "stop_reason": ctrl.state.stop_reason,
            "iterations": ctrl.state.iteration,
            "best": ctrl.state.best,
        }

    def stream(self, *, max_iter: Optional[int] = None,
               patience: Optional[int] = None) -> Iterator[dict]:
        """Yield an event dict per step: ``proposed`` → ``scored`` →
        (``accepted``) … → ``done``. The UI consumes this; so can a script."""
        yield from self._iterate(max_iter, patience)

    def run(self, *, max_iter: Optional[int] = None,
            patience: Optional[int] = None) -> RunResult:
        """Run the climb to termination (blocking) and return the result.
        Equivalent to draining :meth:`stream`."""
        for _ in self._iterate(max_iter, patience):
            pass
        s = self.controller.state
        return RunResult(best=s.best, stop_reason=s.stop_reason,
                         iterations=s.iteration, history=s.history)
