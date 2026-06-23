"""TRUE end-to-end: HillClimber → BrokerRunner → the REAL ucl_gpu_infra package
(gpu_broker / stage6_infra / run_poller), with only the OS boundary faked.

Unlike test_broker_runner.py (which mocks ucl_gpu_infra out), this exercises the
actual package code: real claim parsing, real submit/run_id extraction, real HTTP
polling (httpx.MockTransport), real scoring. It proves the two repos work together
through a full multi-iteration climb — the integration nothing else covered.

The only fakes are: the broker/infra SHELL SCRIPTS (emit the real JSON shapes) and
the shim's HTTP /api/list_runs (always reports the submitted run terminal).
"""
import stat
import json

import httpx
import pytest

from continual_auto_research import HillClimber, BrokerRunner


def _script(path, body):
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def real_infra(tmp_path, monkeypatch):
    """Fake scripts + mocked shim HTTP so the REAL package code runs end to end."""
    # --- broker scripts (real gpu_broker parses these) ---
    bdir = tmp_path / "ucl-infra"
    bdir.mkdir()
    _script(bdir / "claim-gpu.sh",
            'echo \'{"box":"gpu01","INFRA_SERVER_URL":"http://shim","INFRA_SESSION_KEY":"k"}\'\n')
    _script(bdir / "release-gpu.sh", 'exit 0\n')
    _script(bdir / "gpu-leases.sh", 'echo \'{}\'\n')   # no extra lease shims
    monkeypatch.setenv("GPU_BROKER", "1")
    monkeypatch.setenv("UCL_INFRA_DIR", str(bdir))

    # --- infra scripts (real stage6_infra parses these) ---
    sdir = tmp_path / "infra-scripts"
    sdir.mkdir()
    _script(sdir / "fast_push_code.sh", 'exit 0\n')
    # a NEW run_id per submit so each iteration is distinct
    _script(sdir / "fast_submit.sh",
            'echo "{\\"run_id\\":\\"run_$RANDOM\\",\\"status\\":\\"queued\\"}"\n')
    _script(sdir / "fast_query_exp_status.sh",
            'echo \'{"run_id":"x","status":"succeeded","log_tail":"final cost computed\\nSCORE=7.5"}\'\n')
    monkeypatch.setenv("EXPERIMENT_INFRA_SCRIPTS", str(sdir))
    monkeypatch.setenv("INFRA_SERVER_URL", "http://shim")
    monkeypatch.setenv("INFRA_SESSION_KEY", "k")

    # --- mock the shim's /api/list_runs so polling sees the run as terminal ---
    # run_poller.httpx.post must report ALL runs succeeded under the iter marker.
    from ucl_gpu_infra import run_poller

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"runs": [
            # broad: claims every plausible run_id this poll asks about as terminal,
            # under both iter markers used in the test.
            {"run_id": rid, "status": "succeeded", "workdir": wd}
            for wd in ("omc/ctrl/iter_001", "omc/ctrl/iter_002", "omc/ctrl/iter_003")
            for rid in (_SUBMITTED.get(wd, "none"),)
        ]})

    def fake_post(url, **kwargs):
        t = httpx.MockTransport(handler)
        with httpx.Client(transport=t) as c:
            return c.post(url, **kwargs)

    monkeypatch.setattr(run_poller.httpx, "post", fake_post)
    return tmp_path


# Track which run_id each iteration's submit produced, so the mocked shim can mark
# exactly that run terminal under its workdir marker.
_SUBMITTED: dict = {}


def test_full_climb_through_real_package(real_infra, monkeypatch):
    _SUBMITTED.clear()

    # Wrap stage6_infra.submit so we learn the run_id it assigned per iteration and
    # can have the mocked shim report THAT id terminal. (We patch at the call site
    # the runner imports, exercising the real submit underneath.)
    from ucl_gpu_infra import stage6_infra as real_s6
    orig_submit = real_s6.submit

    def submit_spy(receipt, scripts, config_path, kind="smoke", env=None):
        res = orig_submit(receipt, scripts, config_path, kind=kind, env=env)
        if res.ok:
            _SUBMITTED[receipt.remote_dest] = res.run_id
        return res

    monkeypatch.setattr(real_s6, "submit", submit_spy)

    runner = BrokerRunner(
        project_id="ctrl",
        workspace_dir=str(real_infra),
        run_command="cd exp && python run.py",
        poll_interval_s=0.0,
        timeout_s=5.0,
    )
    hc = HillClimber(propose=lambda ctx: "a PID controller", runner=runner, direction="min")
    result = hc.run(max_iter=2, patience=10)

    # Every iteration ran through claim → submit → poll → score(SCORE=7.5) → release.
    assert result.iterations == 2
    assert result.best_score == 7.5, "scored from the real SCORE= sentinel via the real package"
    assert len(_SUBMITTED) >= 1, "real stage6_infra.submit actually produced run_ids"
    assert all(h["score"] == 7.5 for h in result.history)


def test_no_free_gpu_propagates_through_real_broker(real_infra, monkeypatch):
    # Make the REAL broker script report no free GPU; the climb records it as a
    # failed (non-finite) iteration and continues — no fabricated score.
    bdir = real_infra / "ucl-infra"
    _script(bdir / "claim-gpu.sh", 'echo \'{"error":"no free GPU available"}\'\n')

    runner = BrokerRunner(project_id="ctrl", workspace_dir=str(real_infra),
                          run_command="python run.py", poll_interval_s=0.0, timeout_s=5.0)
    hc = HillClimber(propose=lambda ctx: "x", runner=runner, direction="min")
    result = hc.run(max_iter=2, patience=10)
    assert result.best is None, "no GPU → no score → nothing became the incumbent"
    assert all(h["score"] is None for h in result.history)
