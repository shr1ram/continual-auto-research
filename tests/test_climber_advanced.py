"""Advanced/realistic programmatic tests: convergence, resume-from-state,
target-score early stop, determinism, and proposer-context plumbing."""
import pytest

from continual_auto_research import HillClimber, CallableRunner, HillClimbConfig, HillClimbState


def test_realistic_convergence_on_a_quadratic():
    """A proposer that nudges a scalar toward the optimum should converge. The
    objective is -(x-3)^2 (maximised at x=3). The proposer reads the incumbent
    from the context and steps toward better x; the runner scores f(x)."""
    state = {"x": 0.0, "step": 1.0}

    def proposer(context: str) -> str:
        # naive hill-climb: try a step up from the current best x
        return str(state["x"] + state["step"])

    def runner_fn(proposal: str):
        x = float(proposal)
        score = -((x - 3.0) ** 2)
        # if this x is better than the running best, adopt it; else shrink+flip step
        if score > -((state["x"] - 3.0) ** 2):
            state["x"] = x
        else:
            state["step"] *= -0.5
        return score, f"SCORE={score}"

    hc = HillClimber(propose=proposer, runner=CallableRunner(runner_fn), direction="max")
    result = hc.run(max_iter=40, patience=40)
    # should get close to the optimum score of 0 at x=3
    assert result.best_score is not None
    assert result.best_score > -0.5, f"expected near-optimal, got {result.best_score}"


def test_target_score_stops_early():
    seq = iter([1.0, 5.0, 9.0, 11.0, 12.0])
    cfg = HillClimbConfig()
    cfg.direction = "max"
    cfg.target_score = 8.0           # stop once we cross 8
    hc = HillClimber(propose=lambda c: "x",
                     runner=CallableRunner(lambda p: (next(seq), "")),
                     config=cfg)
    result = hc.run(max_iter=20, patience=20)
    assert result.stop_reason == "target"
    assert result.best_score == 9.0   # first to cross 8
    assert result.iterations == 3


def test_resume_from_persisted_state():
    # Run 3 iters, snapshot state, then resume into a NEW climber and continue.
    seq = iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    runner = CallableRunner(lambda p: (next(seq), ""))
    hc1 = HillClimber(propose=lambda c: "x", runner=runner, direction="max")
    r1 = hc1.run(max_iter=3, patience=10)
    assert r1.best_score == 3.0 and r1.iterations == 3

    snap = HillClimbState.from_dict(hc1.controller.state.to_dict())
    hc2 = HillClimber(propose=lambda c: "x", runner=runner, direction="max", state=snap)
    r2 = hc2.run(max_iter=6, patience=10)   # continues from iter 3
    assert r2.iterations == 6
    assert r2.best_score == 6.0
    assert len(r2.history) == 6, "history accumulates across the resume"


def test_resume_does_not_reactivate_a_converged_plateau():
    # A plateau stop is genuine convergence — resuming with a bigger budget must
    # NOT silently re-run it (only a budget stop auto-resumes).
    seq = iter([5.0, 1.0, 1.0, 1.0])
    runner = CallableRunner(lambda p: (next(seq), ""))
    hc1 = HillClimber(propose=lambda c: "x", runner=runner, direction="max")
    r1 = hc1.run(max_iter=20, patience=2)
    assert r1.stop_reason == "plateau"
    done_iters = r1.iterations

    snap = HillClimbState.from_dict(hc1.controller.state.to_dict())
    hc2 = HillClimber(propose=lambda c: "x", runner=runner, direction="max", state=snap)
    r2 = hc2.run(max_iter=50, patience=2)        # bigger budget
    assert r2.iterations == done_iters, "a converged plateau must not auto-resume"


def test_state_roundtrips_through_dict():
    hc = HillClimber(propose=lambda c: "x",
                     runner=CallableRunner(lambda p: (1.0, "")), direction="max")
    hc.run(max_iter=2)
    d = hc.controller.state.to_dict()
    back = HillClimbState.from_dict(d)
    assert back.iteration == 2
    assert back.to_dict() == d, "state must round-trip losslessly (for persistence)"


def test_determinism_same_inputs_same_result():
    def make():
        seq = iter([3.0, 1.0, 4.0, 1.0, 5.0])
        return HillClimber(propose=lambda c: "x",
                           runner=CallableRunner(lambda p: (next(seq), "")),
                           direction="max").run(max_iter=5, patience=10)
    a, b = make(), make()
    assert a.best_score == b.best_score == 5.0
    assert [h["score"] for h in a.history] == [h["score"] for h in b.history]


def test_proposer_receives_incumbent_in_context():
    seen = []

    def proposer(context: str) -> str:
        seen.append(context)
        return "x"

    seq = iter([5.0, 1.0])
    HillClimber(propose=proposer, runner=CallableRunner(lambda p: (next(seq), "")),
                direction="max").run(max_iter=2, patience=10)
    # iter 1 context: no candidates yet; iter 2 context: must mention the best=5.0
    assert "No candidates yet" in seen[0]
    assert "5.0" in seen[1] and "Best so far" in seen[1]


def test_plateau_pushes_proposer_to_change_class():
    # after enough non-improving rounds, the context should tell the proposer to
    # switch approach (the anti-plateau guidance).
    seen = []
    seq = iter([10.0, 1.0, 1.0, 1.0, 1.0])
    HillClimber(propose=lambda c: seen.append(c) or "x",
                runner=CallableRunner(lambda p: (next(seq), "")),
                direction="max").run(max_iter=5, patience=3)
    assert any("PLATEAU" in c or "DIFFERENT CLASS" in c for c in seen), \
        "near-patience context must steer toward a different class of solution"


def test_empty_proposal_and_zero_iterations():
    # max_iter=0 → should_continue is False immediately; clean done, no crash.
    hc = HillClimber(propose=lambda c: "", runner=CallableRunner(lambda p: (1.0, "")),
                     direction="max")
    result = hc.run(max_iter=0)
    assert result.iterations == 0
    assert result.best is None
    assert result.stop_reason == "budget"
