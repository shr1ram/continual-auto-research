"""Hill-climbing controller — a standard metric-optimising loop.

The core of continual-auto-research. The objective is a single scalar metric;
there is no ideation, methodology, paper, or LLM critic. The loop is:

    propose a candidate → run & score it on the GPU → compare the score to the
    incumbent best → keep the better one → repeat until the metric plateaus or
    the iteration budget runs out.

This module is the OPTIMISATION POLICY only — engine-agnostic. It decides *what
should happen next* (propose / accept-or-reject / terminate) and leaves the *how*
(calling a proposer, running on infra) to the caller (:mod:`runner` /
:mod:`climber`). The metric-gate is a pure comparison (``score > best_score``),
NOT an LLM critic — no rubric, no feasibility gate, none of the critic-
degeneration failure modes.

(Internal symbols keep the accurate ``HillClimb*`` names — this *is* hill
climbing. Only the project's user-facing surface is branded "auto-research".)

State machine (``phase``):

    propose  → a proposer is generating the next candidate
    run      → the candidate is executing on infra
    compare  → deterministic metric-gate: did the score beat the incumbent?
    done     → terminated (plateau / budget / target); ``best`` holds the winner
    failed   → a candidate run errored irrecoverably
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── Stage list for hill-climbing mode ───────────────────────────────────────
# Distinct from pipeline_engine.STAGES (the 7 research stages). Looked up by id,
# like the auto-scientist stages. "Run & Score" REUSES the stage-6
# ``experimentalist`` skill — the one piece both modes share.
HC_STAGES = [
    {"id": 1, "skill": "candidate_proposer", "name": "Propose Solution"},
    {"id": 2, "skill": "experimentalist",    "name": "Run & Score"},
    {"id": 3, "skill": "improvement_analyst", "name": "Analyze & Improve"},
]

# Optimisation direction. "max" keeps the highest score (e.g. accuracy);
# "min" keeps the lowest (e.g. loss, or the commaai controls cost where lower
# is better). Read from config so a task can flip it without code changes.
DIRECTION_MAX = "max"
DIRECTION_MIN = "min"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _measured_output(raw_result: str, max_chars: int = 600) -> str:
    """Pull the informative lines out of a candidate's stored run output so they
    can be threaded back to the proposer. We prefer the lines that report the
    objective and its components (so the model can see the breakdown the scalar
    score hides), then fall back to the tail of the output. Returns '' if there
    is nothing useful. Kept short — this is added per recent attempt."""
    import re as _re
    if not raw_result:
        return ""
    text = str(raw_result)
    # Lines that look like a metric report: "<name>_cost: <n>", "score: <n>",
    # "<name>: <number>", "SCORE=<n>", "total ... <number>".
    metric_re = _re.compile(
        r"(cost|score|metric|error|loss|reward|total|accuracy|rmse|mae|f1|time)\b",
        _re.IGNORECASE,
    )
    hits = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        # must contain a number AND a metric-ish word to count as a report line
        if metric_re.search(ln) and _re.search(r"[-+]?\d", ln):
            hits.append(ln)
    picked = "\n     ".join(hits[-6:]) if hits else text.strip()[-max_chars:]
    return picked[:max_chars]


def _normalize_direction(raw):
    """Coerce HC_DIRECTION to "min"/"max". Case- and whitespace-insensitive, with
    a few friendly aliases. An UNRECOGNISED value would otherwise fall through to
    max logic silently and invert the optimisation (e.g. a minimise-cost task
    would maximise) - so we log and default to max explicitly instead."""
    v = (raw or "").strip().lower()
    if v in ("min", "minimize", "minimise", "minimum", "lower", "lowest"):
        return DIRECTION_MIN
    if v in ("max", "maximize", "maximise", "maximum", "higher", "highest"):
        return DIRECTION_MAX
    if v:
        import logging
        logging.getLogger(__name__).warning(
            "Unrecognised HC_DIRECTION=%r; defaulting to %r", raw, DIRECTION_MAX,
        )
    return DIRECTION_MAX


def _direction_env_is_recognised():
    """True only when HC_DIRECTION is set AND parses to a real min/max value.
    A typo (e.g. "minimze") is NOT recognised, so it does not count as an
    explicit override - it falls through to proposer inference instead of being
    treated as authoritative and silently optimising the wrong way."""
    raw = os.environ.get("HC_DIRECTION", "").strip()
    if not raw:
        return False
    v = raw.lower()
    return v in (
        "min", "minimize", "minimise", "minimum", "lower", "lowest",
        "max", "maximize", "maximise", "maximum", "higher", "highest",
    )


def direction_is_explicit():
    """True if a human pinned a VALID HC_DIRECTION in the environment. When set,
    it wins over any agent-inferred direction. An unrecognised/typo value is
    treated as NOT explicit so inference still runs (see _direction_env_is_recognised)."""
    return _direction_env_is_recognised()



def extract_direction(text):
    """Pull an optimisation direction the proposer declared, e.g.
    ``DIRECTION: min`` / ``optimize: minimize`` / a ``"direction": "min"`` JSON
    field. Returns "min"/"max", or None if nothing parseable is present.

    This is how the system learns whether LOWER or HIGHER is better WITHOUT a
    human setting HC_DIRECTION - critical because the default is max, which
    silently optimises a cost-minimisation task (e.g. the commaai controls cost)
    backwards."""
    import re as _re
    if not text:
        return None
    # 1) explicit "direction"/"optimize"/"goal": min|max|minimi*|maximi* (prose or JSON)
    m = _re.search(
        r'(?:direction|optimi[sz]e|optimization|goal|objective)"?\s*[:=]\s*"?'
        r'(min|max|minimi[sz]e?|maximi[sz]e?|lower|higher)',
        text, _re.IGNORECASE,
    )
    if m:
        return _normalize_direction(m.group(1))
    # 2) "lower is better" / "higher is better" / minimise / maximise phrasing
    if _re.search(r'lower\b[^.\n]{0,20}\bbetter|\bminimi[sz]e?\b', text, _re.IGNORECASE):
        return DIRECTION_MIN
    if _re.search(r'higher\b[^.\n]{0,20}\bbetter|\bmaximi[sz]e?\b', text, _re.IGNORECASE):
        return DIRECTION_MAX
    return None


@dataclass
class HillClimbConfig:
    """Termination + direction knobs, sourced from env/profile so they switch
    the same way as ``llm``/``infra`` do."""

    direction: str = field(default_factory=lambda: _normalize_direction(_env("HC_DIRECTION", DIRECTION_MAX)))
    # What is being optimised — the task description the user types in the launch
    # form (or HC_OBJECTIVE). Threaded into proposer_context() on EVERY iteration
    # (and especially the first, where there is no history yet) so the proposer
    # knows what problem it is solving. Empty = no objective given (the proposer
    # falls back to its generic system prompt and whatever the run output reveals).
    objective: str = field(default_factory=lambda: _env("HC_OBJECTIVE", ""))
    # Hard cap on iterations regardless of progress.
    max_iterations: int = field(default_factory=lambda: int(_env("HC_MAX_ITERATIONS", "20")))
    # Plateau: stop after this many consecutive non-improving iterations.
    patience: int = field(default_factory=lambda: int(_env("HC_PATIENCE", "4")))
    # Optional absolute target — stop early once the metric crosses it. Empty = none.
    target_score: Optional[float] = field(
        default_factory=lambda: (
            float(_env("HC_TARGET_SCORE", "")) if _env("HC_TARGET_SCORE", "") else None
        )
    )

    def is_better(self, candidate: float, incumbent: Optional[float]) -> bool:
        """The metric-gate. Pure comparison — this REPLACES the LLM critic."""
        if incumbent is None:
            return True
        if self.direction == DIRECTION_MIN:
            return candidate < incumbent
        return candidate > incumbent

    def target_reached(self, score: float) -> bool:
        if self.target_score is None:
            return False
        if self.direction == DIRECTION_MIN:
            return score <= self.target_score
        return score >= self.target_score


@dataclass
class Candidate:
    """One proposed solution and the score it achieved."""

    iteration: int
    proposal: str            # the candidate description / params / code reference
    score: Optional[float] = None
    raw_result: str = ""     # full execution output, for the next proposer's context
    accepted: bool = False   # became the new incumbent best?


@dataclass
class HillClimbState:
    """Persisted optimisation state. Serialised alongside the pipeline state so a
    run survives restarts (the same way ``stage_results`` does for auto-scientist)."""

    phase: str = "propose"   # propose | run | compare | done | failed
    iteration: int = 0
    best: Optional[dict] = None          # asdict(Candidate) of the incumbent
    history: list = field(default_factory=list)  # [asdict(Candidate), ...]
    stale_rounds: int = 0    # consecutive non-improving iterations (plateau counter)
    stop_reason: str = ""    # "plateau" | "budget" | "target" | "" (still running)
    # Optimisation direction RESOLVED for this run ("min"/"max"/""). Set once,
    # early — from an explicit HC_DIRECTION env if a human pinned one, else
    # inferred from the proposer's declaration (see _extract_direction). Persisted
    # so it stays stable across iterations and survives restarts. Empty until
    # resolved, at which point the controller uses it instead of the env default.
    resolved_direction: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HillClimbState":
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


class HillClimbController:
    """Drives the propose → run → compare loop.

    This is intentionally engine-agnostic: it decides *what should happen next*
    (propose another candidate / accept-or-reject a scored one / terminate) and
    leaves the *how* (dispatching a proposer agent, running on infra) to the
    caller, which owns the agent/infra plumbing. That keeps all the LLM and
    GPU-execution wiring in one place (the engine) and all the optimisation
    policy here.
    """

    def __init__(self, config: Optional[HillClimbConfig] = None,
                 state: Optional[HillClimbState] = None):
        self.config = config or HillClimbConfig()
        self.state = state or HillClimbState()

    # -- queries --------------------------------------------------------------
    @property
    def best_score(self) -> Optional[float]:
        return (self.state.best or {}).get("score") if self.state.best else None

    def should_continue(self) -> bool:
        """Termination check, run before proposing the next candidate."""
        # Already terminated — e.g. on_scored hit target_score and set phase=done.
        # Without this, a target hit was recorded but the loop dispatched another
        # iteration anyway, wasting a GPU run.
        if self.state.phase in ("done", "failed") or self.state.stop_reason == "target":
            return False
        if self.state.iteration >= self.config.max_iterations:
            self.state.stop_reason = "budget"
            return False
        if self.state.stale_rounds >= self.config.patience:
            self.state.stop_reason = "plateau"
            return False
        return True

    # -- transitions ----------------------------------------------------------
    def begin_iteration(self) -> int:
        """Open a new iteration. Returns the iteration number."""
        self.state.iteration += 1
        self.state.phase = "propose"
        return self.state.iteration

    def on_proposed(self, proposal: str) -> Candidate:
        """Record a freshly proposed candidate; move to the run phase."""
        cand = Candidate(iteration=self.state.iteration, proposal=proposal)
        self.state.phase = "run"
        return cand

    def on_scored(self, cand: Candidate, score: float, raw_result: str = "") -> bool:
        """The metric-gate. Compare to incumbent, accept-or-reject, update the
        plateau counter, append to history. Returns True if this became the new
        best. After this, call :meth:`should_continue` to decide whether to loop.
        """
        # A non-finite score means the run crashed/errored. Store it as None
        # (JSON null) rather than NaN: NaN is not JSON-compliant and crashed the
        # /status endpoint's json.dumps (500 on every poll of a failed HC run).
        # None round-trips cleanly and reads as "no score" everywhere.
        finite = math.isfinite(score)
        cand.score = score if finite else None
        cand.raw_result = raw_result
        self.state.phase = "compare"

        # A non-finite score means the run crashed/errored (the engine scores a
        # failed candidate as NaN). Record it in history for the proposer's
        # context, but it is NOT an improvement and must NOT count toward the
        # plateau — otherwise a streak of flaky/failed runs would be mistaken for
        # convergence and stop the optimisation early. (Matches the engine's
        # _hc_on_task_failed contract: "recorded but not allowed to plateau".)
        improved = finite and self.config.is_better(score, self.best_score)
        cand.accepted = improved
        if improved:
            self.state.best = asdict(cand)
            self.state.stale_rounds = 0
        elif finite:
            self.state.stale_rounds += 1
        # non-finite: leave stale_rounds unchanged (neither progress nor plateau)

        self.state.history.append(asdict(cand))

        if finite and self.config.target_reached(score):
            self.state.stop_reason = "target"
            self.state.phase = "done"
        return improved

    def finish(self) -> dict:
        """Mark the loop done and return the winning candidate (or None)."""
        if self.state.phase != "done":
            self.state.phase = "done"
        return self.state.best

    def fail(self, reason: str = "") -> None:
        self.state.phase = "failed"
        self.state.stop_reason = reason or "error"

    # -- proposer context -----------------------------------------------------
    def proposer_context(self) -> str:
        """Feedback for the next proposer: the incumbent best and recent tries,
        so the LLM proposes an *improvement* rather than starting blind. This is
        hill-climbing's analogue of threading critic feedback into the producer.

        Outcome-aware and plateau-aware: the closing instruction adapts to whether
        the search is still climbing (exploit) or stuck (explore a new approach).
        The incumbent and recent proposals are shown in FULL (not truncated) so the
        model can build on a non-trivial solution rather than a clipped fragment.
        """
        objective = (self.config.objective or "").strip()
        # First iteration: no history yet. Still give the proposer the objective
        # so the INITIAL candidate is on-task rather than blind. Without this the
        # first proposal is generated with no idea what is being optimised.
        if not self.state.history:
            if objective:
                return (
                    f"OBJECTIVE: {objective}\n"
                    f"Optimisation direction: {self.config.direction} "
                    f"({'lower' if self.config.direction == DIRECTION_MIN else 'higher'} is better).\n"
                    "No candidates yet. Propose an initial solution for this objective."
                )
            return "No candidates yet. Propose an initial solution."
        # The objective heads the context on EVERY iteration — it is the standing
        # task the recent attempts below are all trying to improve.
        lines = []
        if objective:
            lines.append(f"OBJECTIVE: {objective}")
        lines.append(f"Optimisation direction: {self.config.direction}.")
        if self.state.best:
            b = self.state.best
            lines.append(f"Best so far (iter {b['iteration']}): score={b['score']}.")
            lines.append(f"  Its approach: {b['proposal']}")
            best_out = _measured_output(b.get("raw_result", ""))
            if best_out:
                lines.append(f"  Its MEASURED run output: {best_out}")
        recent = self.state.history[-5:]
        lines.append("Recent attempts (most recent last):")
        for c in recent:
            tag = "ACCEPTED" if c["accepted"] else "rejected"
            lines.append(f"  iter {c['iteration']} [{tag}] score={c['score']}: {c['proposal']}")
            # Surface the ACTUAL measured run output, not just the model's own
            # description. The run output carries the objective breakdown (e.g. a
            # cost split into components) that the scalar score hides — without it
            # the proposer optimises blind and can misread which component is the
            # real problem.
            meas = _measured_output(c.get("raw_result", ""))
            if meas:
                lines.append(f"     measured: {meas}")
        lines.append(
            "IMPORTANT: read the MEASURED run output above (not just the scalar "
            "score). If the objective is a sum/weighted combination of components, "
            "work out from the measurements which component actually dominates the "
            "score and target THAT — do not assume; the numbers are right there."
        )

        # Plateau-aware closing instruction. stale_rounds counts consecutive
        # non-improving (finite) rounds; patience is the give-up threshold. When we
        # are within one round of patience (or already there), refining parameters
        # is exhausted — push the proposer to change the CLASS of approach.
        stale = int(self.state.stale_rounds)
        patience = int(self.config.patience)
        plateauing = patience > 0 and stale >= max(1, patience - 1)
        if plateauing:
            lines.append(
                f"PLATEAU: the score has not improved for {stale} round(s) "
                f"(patience={patience}). Parameter tuning is exhausted — do NOT "
                "propose another small tweak to the same approach. Propose a "
                "DIFFERENT CLASS of solution than anything in the attempts above."
            )
        else:
            last = self.state.history[-1]
            if last.get("accepted"):
                lines.append(
                    "The last attempt improved the best — EXPLOIT: refine in that "
                    "direction with a focused change."
                )
            else:
                lines.append(
                    "The last attempt did not beat the best. Either back off and try "
                    "a different lever, or switch approach. Do not re-try something "
                    "already above."
                )
            lines.append("Propose a NEW candidate that beats the best score above.")
        return "\n".join(lines)
