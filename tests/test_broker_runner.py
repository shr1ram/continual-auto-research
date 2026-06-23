"""Integration tests for BrokerRunner — the real UCL-GPU glue path.

The GPU broker / infra is mocked at the ``ucl_gpu_infra`` boundary so the FULL
flow (claim → submit → poll-to-terminal → score → release) is exercised without a
live GPU. These cover the production runner the unit tests skipped.
"""
import sys
import types
import json

import pytest

from continual_auto_research.core.runner import BrokerRunner


# ── a fake ucl_gpu_infra installed into sys.modules for the duration of a test ─

class _FakeBroker:
    def __init__(self):
        self.claimed = []
        self.released = []
        self.claim_status = "ok"
        self.lease = {"INFRA_SERVER_URL": "http://shim", "INFRA_SESSION_KEY": "k"}

    def claim(self, run_id, holder="stage6"):
        self.claimed.append((run_id, holder))
        if self.claim_status != "ok":
            return None, self.claim_status
        return self.lease, "ok"

    def env_for(self, lease, base=None):
        return {"INFRA_SERVER_URL": "http://shim", "INFRA_SESSION_KEY": "k"}

    def release(self, run_id):
        self.released.append(run_id)


class _FakeSubmitResult:
    def __init__(self, ok, run_id="", error=""):
        self.ok, self.run_id, self.error = ok, run_id, error


class _FakeStage6:
    def __init__(self):
        self.submit_ok = True
        self.status = {"status": "succeeded", "log_tail": "SCORE=42"}

    class Receipt:
        def __init__(self, smoke_cmd="", code_dir="", remote_dest=""):
            self.smoke_cmd, self.code_dir, self.remote_dest = smoke_cmd, code_dir, remote_dest

    def find_infra_scripts(self):
        return {"fast_submit.sh": "/x", "fast_push_code.sh": "/y", "fast_query_exp_status.sh": "/z"}

    def submit(self, receipt, scripts, config_path, kind="smoke", env=None):
        return _FakeSubmitResult(self.submit_ok, run_id="run_abc123", error="" if self.submit_ok else "boom")

    def query_status(self, run_id, scripts, env=None):
        return self.status


class _FakeRunPoller:
    """all_terminal returns True after `terminal_after` polls (default immediately)."""
    terminal_after = 0

    def __init__(self, find_marker=None):
        self._n = 0

    def all_terminal(self, owner_id, run_ids, limit=100):
        self._n += 1
        return self._n > _FakeRunPoller.terminal_after


@pytest.fixture
def fake_infra(monkeypatch):
    broker = _FakeBroker()
    s6 = _FakeStage6()
    pkg = types.ModuleType("ucl_gpu_infra")
    pkg.gpu_broker = broker
    pkg.stage6_infra = s6
    rp_mod = types.ModuleType("ucl_gpu_infra.run_poller")
    rp_mod.RunPoller = _FakeRunPoller
    pkg.run_poller = rp_mod
    monkeypatch.setitem(sys.modules, "ucl_gpu_infra", pkg)
    monkeypatch.setitem(sys.modules, "ucl_gpu_infra.run_poller", rp_mod)
    _FakeRunPoller.terminal_after = 0
    return types.SimpleNamespace(broker=broker, s6=s6, poller=_FakeRunPoller)


def _runner(tmp_path, **kw):
    kw.setdefault("poll_interval_s", 0.0)
    kw.setdefault("timeout_s", 5.0)
    return BrokerRunner(
        project_id="proj1",
        workspace_dir=str(tmp_path),
        run_command="cd exp && python run.py",
        **kw,
    )


def test_happy_path_scores_from_sentinel_and_releases(fake_infra, tmp_path):
    score, out = _runner(tmp_path).run("a candidate", 1)
    assert score == 42.0                       # from log_tail "SCORE=42"
    assert "SCORE=42" in out
    assert fake_infra.broker.claimed == [("proj1", "hc_run")]
    assert fake_infra.broker.released == ["proj1"], "lease must be released"


def test_result_json_wins_over_sentinel(fake_infra, tmp_path):
    # The run writes result.json DURING execution (simulating a shared/NFS
    # workspace). The pre-run stale-clear has already happened, so a file present
    # at score time is this run's — and it must win over the SCORE= sentinel.
    def submit_then_write(receipt, scripts, config_path, kind="smoke", env=None):
        (tmp_path / "result.json").write_text(json.dumps({"score": 7.0, "status": "done"}))
        return _FakeSubmitResult(True, run_id="run_abc123")
    fake_infra.s6.submit = submit_then_write
    score, _ = _runner(tmp_path).run("c", 1)
    assert score == 7.0                        # result.json authoritative over SCORE=42


def test_draft_result_json_scored_none(fake_infra, tmp_path):
    # the run writes a fabricated DRAFT result.json; it must NOT score, and with
    # the sentinel present it falls through to the sentinel (42).
    def submit_then_draft(receipt, scripts, config_path, kind="smoke", env=None):
        (tmp_path / "result.json").write_text(json.dumps({"score": 0.0, "status": "draft"}))
        return _FakeSubmitResult(True, run_id="run_abc123")
    fake_infra.s6.submit = submit_then_draft
    score, _ = _runner(tmp_path).run("c", 1)
    assert score == 42.0, "draft result.json ignored → falls back to SCORE= sentinel"


def test_no_free_gpu_scores_none_and_no_submit(fake_infra, tmp_path):
    fake_infra.broker.claim_status = "unavailable"
    score, out = _runner(tmp_path).run("c", 1)
    assert score is None and "no free GPU" in out
    # nothing was submitted; no lease to release (claim returned no lease)
    assert fake_infra.broker.released == []


def test_submit_failure_scores_none_and_releases(fake_infra, tmp_path):
    fake_infra.s6.submit_ok = False
    score, out = _runner(tmp_path).run("c", 1)
    assert score is None and "submit failed" in out
    assert fake_infra.broker.released == ["proj1"], "lease released even on submit failure"


def test_timeout_scores_none_and_releases(fake_infra, tmp_path):
    _FakeRunPoller.terminal_after = 10_000     # never terminal within the timeout
    score, out = _runner(tmp_path, timeout_s=0.0).run("c", 1)
    assert score is None and "timed out" in out
    assert fake_infra.broker.released == ["proj1"]


def test_stale_result_json_cleared_before_run(fake_infra, tmp_path):
    # a result.json from a prior iteration must be removed before the run, so a
    # run that writes none doesn't re-score the old number.
    (tmp_path / "result.json").write_text(json.dumps({"score": 999.0, "status": "done"}))
    fake_infra.s6.status = {"status": "succeeded", "log_tail": "no score here"}

    # patch submit to confirm the file is already gone by submit time
    seen = {}
    orig_submit = fake_infra.s6.submit
    def spy(receipt, scripts, config_path, kind="smoke", env=None):
        seen["existed_at_submit"] = (tmp_path / "result.json").exists()
        return orig_submit(receipt, scripts, config_path, kind, env)
    fake_infra.s6.submit = spy

    score, _ = _runner(tmp_path).run("c", 1)
    assert seen["existed_at_submit"] is False, "stale result.json must be cleared before submit"
    assert score is None, "no fresh score written and no sentinel → None (not the stale 999)"


def test_run_command_callable_per_candidate(fake_infra, tmp_path):
    # run_command may be a callable so each candidate bakes its own hyperparams in.
    seen = {}
    orig = fake_infra.s6.submit
    def spy(receipt, scripts, config_path, kind="smoke", env=None):
        seen["cmd"] = receipt.smoke_cmd
        return orig(receipt, scripts, config_path, kind, env)
    fake_infra.s6.submit = spy

    r = BrokerRunner(project_id="p", workspace_dir=str(tmp_path),
                     run_command=lambda proposal: f"LR={proposal} python run.py",
                     poll_interval_s=0.0, timeout_s=5.0)
    r.run("0.05", 1)
    assert seen["cmd"] == "LR=0.05 python run.py", "callable run_command receives the proposal"


def test_lease_released_even_if_query_raises(fake_infra, tmp_path):
    def boom(run_id, scripts, env=None):
        raise RuntimeError("query exploded")
    fake_infra.s6.query_status = boom
    with pytest.raises(RuntimeError):
        _runner(tmp_path).run("c", 1)
    # the finally: must still have released the lease
    assert fake_infra.broker.released == ["proj1"]
