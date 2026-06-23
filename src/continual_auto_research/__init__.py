"""continual-auto-research — a programmatic hill-climbing optimiser for
experiment candidates, with a thin web UI on top.

Library-first: optimise with no server.

    from continual_auto_research import HillClimber, CallableRunner, BrokerRunner

    hc = HillClimber(propose=my_proposer, runner=CallableRunner(my_fn), direction="min")
    result = hc.run(max_iter=20, patience=4)        # blocking
    # or: for event in hc.stream(max_iter=20): ...   # live feed (what the UI uses)

The web UI (continual_auto_research.api) is a small shell around this — anything
it displays comes from the library's public return values / event stream.
"""
from __future__ import annotations

from .core.climber import HillClimber, RunResult
from .core.hill_climb import HillClimbConfig, HillClimbController, HillClimbState
from .core.runner import BrokerRunner, CallableRunner, Runner

__all__ = [
    "HillClimber",
    "RunResult",
    "CallableRunner",
    "BrokerRunner",
    "Runner",
    "HillClimbConfig",
    "HillClimbController",
    "HillClimbState",
]
__version__ = "0.1.0"
