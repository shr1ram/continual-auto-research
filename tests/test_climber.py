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


def test_all_failures_trips_circuit_breaker():
    # Every iteration fails. Failures don't plateau (above), so without the
    # breaker this would burn all 20 iterations; it must stop at
    # max_consecutive_failures with its own terminal reason — distinct from
    # plateau/budget — and a failed phase.
    from continual_auto_research.core.hill_climb import HillClimbConfig

    cfg = HillClimbConfig()
    cfg.max_consecutive_failures = 3
    hc = HillClimber(propose=lambda ctx: "c",
                     runner=CallableRunner(lambda p: (None, "crashed")),
                     config=cfg, direction="max")
    result = hc.run(max_iter=20, patience=4)
    assert result.stop_reason == "failing"
    assert result.iterations == 3
    assert result.best is None
    assert hc.controller.state.phase == "failed"


def test_scored_iteration_resets_failure_streak():
    # 2 failures, a success, 2 failures, a success: the streak never reaches the
    # threshold of 3, so the run must use its whole budget and keep the best.
    from continual_auto_research.core.hill_climb import HillClimbConfig

    seq = iter([None, None, 5.0, None, None, 6.0])
    cfg = HillClimbConfig()
    cfg.max_consecutive_failures = 3
    hc = HillClimber(propose=lambda ctx: "c",
                     runner=CallableRunner(lambda p: (next(seq), "")),
                     config=cfg, direction="max")
    result = hc.run(max_iter=6, patience=10)
    assert result.stop_reason == "budget"
    assert result.iterations == 6
    assert result.best_score == 6.0
    assert hc.controller.state.consecutive_failures == 0  # the final success reset it


def test_circuit_breaker_leaves_plateau_logic_unchanged():
    # Finite non-improving scores still plateau exactly as before, and failures
    # still don't count toward patience — the breaker is a separate counter.
    from continual_auto_research.core.hill_climb import HillClimbConfig

    seq = iter([1.0, 0.5, None, 0.5, 0.5])
    cfg = HillClimbConfig()
    cfg.max_consecutive_failures = 4
    hc = HillClimber(propose=lambda ctx: "c",
                     runner=CallableRunner(lambda p: (next(seq), "")),
                     config=cfg, direction="max")
    result = hc.run(max_iter=20, patience=3)
    assert result.stop_reason == "plateau"
    # iter1 best + 3 stale (finite) rounds + 1 failed round that doesn't count
    assert result.iterations == 5


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


def test_objective_is_in_first_proposer_context():
    # The objective must reach the proposer on the FIRST iteration (no history yet)
    # — that is the whole point of the launch-form prompt field. Without it the
    # initial candidate is generated blind.
    from continual_auto_research.core.hill_climb import HillClimbConfig

    seen = []

    def proposer(context: str) -> str:
        seen.append(context)
        return "c"

    cfg = HillClimbConfig()
    cfg.objective = "Minimise the comma.ai controls cost; lower is better."
    hc = HillClimber(propose=proposer, runner=CallableRunner(lambda p: (1.0, "")),
                     config=cfg, direction="min")
    hc.run(max_iter=1)
    assert "comma.ai controls cost" in seen[0]
    assert "OBJECTIVE" in seen[0]


def test_objective_threads_through_builder_and_config():
    # The API builder lifts cfg["objective"] into the config so proposer_context
    # carries it, and aliases (prompt/task) are accepted.
    from continual_auto_research.api.builders import build_climber

    hc = build_climber({"objective": "maximise accuracy", "direction": "max",
                        "runner": {"kind": "demo"}})
    assert hc.controller.config.objective == "maximise accuracy"
    hc2 = build_climber({"prompt": "via the prompt alias", "runner": {"kind": "demo"}})
    assert hc2.controller.config.objective == "via the prompt alias"


def test_bad_proposer_config_raises_no_demo_fallback(monkeypatch):
    # A misconfigured proposer must FAIL LOUD at build time, not silently fall
    # back to the demo (echo) proposer — the fallback let a broken config burn
    # the full GPU budget on a garbage run.
    from continual_auto_research.api.builders import build_climber

    with pytest.raises(ValueError, match="unknown proposer kind"):
        build_climber({"proposer": {"kind": "telepathy"}, "runner": {"kind": "demo"}})
    # missing required model for the api backend also surfaces
    monkeypatch.delenv("DEFAULT_LLM_MODEL", raising=False)
    with pytest.raises(ValueError, match="model"):
        build_climber({"proposer": {"kind": "api"}, "runner": {"kind": "demo"}})


def test_demo_proposer_still_available_when_requested():
    # kind="demo" (and the implicit default) keeps the echo proposer working.
    from continual_auto_research.api.builders import build_climber

    for cfg in ({"runner": {"kind": "demo"}},
                {"proposer": {"kind": "demo"}, "runner": {"kind": "demo"}}):
        hc = build_climber(cfg)
        assert "candidate based on" in hc._propose("some context")


def test_broker_runner_auto_plumbing_from_run_id(tmp_path, monkeypatch):
    # A UI broker launch sends only {kind: broker} — project_id/workspace_dir must
    # be auto-derived from the run id, and run_command left empty (parsed later).
    monkeypatch.setenv("CAR_RUNS_DIR", str(tmp_path))
    from continual_auto_research.api.builders import build_climber

    hc = build_climber({"runner": {"kind": "broker"}}, run_id="run-000042")
    runner = hc._runner
    assert runner.project_id == "run-000042"           # auto from run id
    assert runner.workspace_dir.endswith("run-000042/experiment")
    assert runner.run_command == ""                     # empty → parsed from proposal
    assert runner.direct is False                       # broker, not h100
    import os
    assert os.path.isdir(runner.workspace_dir)          # auto-created


def test_h100_runner_kind_sets_direct(tmp_path, monkeypatch):
    monkeypatch.setenv("CAR_RUNS_DIR", str(tmp_path))
    from continual_auto_research.api.builders import build_climber

    hc = build_climber({"runner": {"kind": "h100"}}, run_id="run-000007")
    assert hc._runner.direct is True
    assert hc._runner.project_id == "run-000007"


def test_broker_explicit_overrides_still_win(tmp_path, monkeypatch):
    monkeypatch.setenv("CAR_RUNS_DIR", str(tmp_path))
    from continual_auto_research.api.builders import build_climber

    hc = build_climber({"runner": {
        "kind": "broker", "project_id": "myproj",
        "workspace_dir": "/custom/ws", "run_command": "python go.py",
    }}, run_id="run-000099")
    assert hc._runner.project_id == "myproj"
    assert hc._runner.workspace_dir == "/custom/ws"
    assert hc._runner.run_command == "python go.py"
