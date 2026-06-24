"""End-to-end library tests: the HillClimber runs programmatically with a
CallableRunner, no server and no GPU. This is the contract the user asked for —
'runs programmatically, the UI is just a layer on top'."""
import pytest

from continual_auto_research import HillClimber, CallableRunner


def test_run_blocking_returns_best_and_history():
    # Objective: maximise. The proposer count grows each call; the runner scores
    # the iteration number, so the best should be the last iteration's score.
    scores = {}

    def proposer(context: str) -> str:
        return f"cand-{len(scores)}"

    def runner_fn(proposal: str):
        n = len(scores) + 1
        scores[proposal] = n
        return float(n), f"SCORE={n}"

    hc = HillClimber(propose=proposer, runner=CallableRunner(runner_fn), direction="max")
    result = hc.run(max_iter=5)
    assert result.iterations == 5
    assert result.best_score == 5.0          # monotone increasing → last wins
    assert result.stop_reason == "budget"
    assert len(result.history) == 5


def test_min_direction_keeps_lowest():
    seq = iter([10.0, 3.0, 7.0, 1.0, 9.0])

    hc = HillClimber(
        propose=lambda ctx: "c",
        runner=CallableRunner(lambda p: (next(seq), "")),
        direction="min",
    )
    result = hc.run(max_iter=5)
    assert result.best_score == 1.0


def test_plateau_stops_early():
    # First score is best; nothing beats it → plateau after `patience` rounds.
    seq = iter([1.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    hc = HillClimber(
        propose=lambda ctx: "c",
        runner=CallableRunner(lambda p: (next(seq), "")),
        direction="max",
    )
    result = hc.run(max_iter=20, patience=3)
    assert result.stop_reason == "plateau"
    assert result.iterations == 4          # iter1 best + 3 stale rounds


def test_failed_run_scores_none_and_does_not_plateau():
    # A None score is a failed run: recorded, but must NOT count toward patience.
    calls = {"n": 0}

    def runner_fn(proposal: str):
        calls["n"] += 1
        if calls["n"] <= 3:
            return None, "crashed"          # 3 failed runs
        return 5.0, "SCORE=5"               # then a real improvement

    hc = HillClimber(propose=lambda ctx: "c", runner=CallableRunner(runner_fn), direction="max")
    result = hc.run(max_iter=6, patience=2)
    # If failures had ticked the plateau, it would have stopped at iter 2. It must
    # not have — the real score at iter 4 must be reached and become best.
    assert result.best_score == 5.0


def test_stream_emits_expected_event_sequence():
    hc = HillClimber(
        propose=lambda ctx: "c",
        runner=CallableRunner(lambda p: (1.0, "SCORE=1")),
        direction="max",
    )
    events = list(hc.stream(max_iter=1))
    types = [e["type"] for e in events]
    # a `trace` event is emitted between proposed and scored (the trace window).
    assert types == ["proposed", "trace", "scored", "accepted", "done"]
    assert events[-1]["stop_reason"] == "budget"
    scored = next(e for e in events if e["type"] == "scored")
    assert scored["score"] == 1.0


def test_bare_callable_runner_accepted():
    # convenience: a plain callable is wrapped automatically
    hc = HillClimber(propose=lambda ctx: "c", runner=lambda p: (2.0, ""), direction="max")
    assert hc.run(max_iter=1).best_score == 2.0
