"""API tests — the run store + REST + per-run websocket, against the demo runner
(no GPU). Uses a temp runs dir so tests don't touch real state."""
import json
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CAR_RUNS_DIR", str(tmp_path / "runs"))
    # import after env is set so the store picks up the temp dir
    import importlib
    from continual_auto_research.api import store as store_mod
    importlib.reload(store_mod)
    from continual_auto_research.api import app as app_mod
    importlib.reload(app_mod)
    return TestClient(app_mod.app)


def _wait_done(client, run_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        rec = client.get(f"/api/runs/{run_id}").json()
        if rec["status"] in ("done", "failed"):
            return rec
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish in {timeout}s")


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_create_run_lifecycle(client):
    r = client.post("/api/runs", json={"direction": "max", "max_iter": 5, "patience": 10})
    assert r.status_code == 200
    rec = r.json()
    rid = rec["id"]
    assert rec["status"] in ("running", "starting")

    done = _wait_done(client, rid)
    assert done["status"] == "done"
    assert done["iterations"] == 5
    assert len(done["history"]) == 5
    assert done["stop_reason"] == "budget"
    assert done["best"] is not None


def test_list_runs(client):
    client.post("/api/runs", json={"direction": "max", "max_iter": 2, "patience": 5})
    client.post("/api/runs", json={"direction": "min", "max_iter": 2, "patience": 5})
    runs = client.get("/api/runs").json()["runs"]
    assert len(runs) == 2
    # newest id first
    assert runs[0]["id"] > runs[1]["id"]
    assert {r["direction"] for r in runs} == {"max", "min"}


def test_get_missing_run_404(client):
    assert client.get("/api/runs/run-999999").status_code == 404


def test_create_run_bad_proposer_is_400_not_demo_fallback(client):
    # A misconfigured proposer must reject the run with the builder's message —
    # not silently start a demo (echo) run that burns the GPU budget.
    r = client.post("/api/runs", json={"max_iter": 2, "patience": 5,
                                       "proposer": {"kind": "telepathy"}})
    assert r.status_code == 400
    assert "telepathy" in r.json()["detail"]
    # the aborted record is persisted as failed, not left dangling as "starting"
    runs = client.get("/api/runs").json()["runs"]
    assert runs and runs[0]["status"] == "failed"


def test_delete_run(client):
    rid = client.post("/api/runs", json={"max_iter": 2, "patience": 5}).json()["id"]
    _wait_done(client, rid)
    assert client.delete(f"/api/runs/{rid}").json()["status"] == "deleted"
    assert client.get(f"/api/runs/{rid}").status_code == 404
    assert client.delete(f"/api/runs/{rid}").status_code == 404


def test_resume_extends_budget(client):
    rid = client.post("/api/runs", json={"direction": "max", "max_iter": 3, "patience": 20}).json()["id"]
    first = _wait_done(client, rid)
    assert first["iterations"] == 3

    r = client.post(f"/api/runs/{rid}/resume", json={"max_iter": 6})
    assert r.status_code == 200
    resumed = _wait_done(client, rid)
    assert resumed["iterations"] == 6, "resume continues from where it stopped"


def test_resume_requires_positive_budget(client):
    rid = client.post("/api/runs", json={"max_iter": 2, "patience": 5}).json()["id"]
    _wait_done(client, rid)
    assert client.post(f"/api/runs/{rid}/resume", json={"max_iter": 0}).status_code == 400


def test_ws_replays_finished_run(client):
    rid = client.post("/api/runs", json={"direction": "max", "max_iter": 3, "patience": 20}).json()["id"]
    _wait_done(client, rid)
    # connect AFTER it finished → should replay history then done
    with client.websocket_connect(f"/ws/runs/{rid}") as ws:
        types = []
        while True:
            ev = json.loads(ws.receive_text())
            types.append(ev["type"])
            if ev["type"] == "done":
                break
    assert types.count("scored") == 3
    assert types[-1] == "done"


def test_ws_live_run_streams_events(client):
    # start via the legacy ws/run (creates + streams in one socket)
    with client.websocket_connect("/ws/run") as ws:
        ws.send_text(json.dumps({"direction": "max", "max_iter": 2, "patience": 10}))
        types = []
        while True:
            ev = json.loads(ws.receive_text())
            types.append(ev["type"])
            if ev["type"] == "done":
                break
    assert types[0] == "proposed"
    assert "scored" in types and types[-1] == "done"


def test_startup_loads_shared_secrets(tmp_path, monkeypatch):
    # a populated shared secret file → api backend reads ready after startup.
    sec = tmp_path / "secrets.env"
    sec.write_text("DEFAULT_API_BASE_URL=https://proxy/v1\nCUSTOM_API_KEY=k\nDEFAULT_LLM_MODEL=m\n")
    monkeypatch.setenv("UCL_GPU_INFRA_SECRETS", str(sec))
    monkeypatch.setenv("CAR_RUNS_DIR", str(tmp_path / "runs"))
    import importlib
    from continual_auto_research.api import app as app_mod
    importlib.reload(app_mod)
    # context manager fires startup events (where load_secrets runs)
    with TestClient(app_mod.app) as c:
        st = c.get("/api/proposers").json()["status"]
    assert st["api"]["ready"] is True
    import os
    assert os.environ["DEFAULT_API_BASE_URL"] == "https://proxy/v1"


def test_broker_run_command_proposal_placeholder():
    # {proposal} in run_command must be substituted with the candidate so a
    # UI-driven broker run can inject the LLM's hyperparameter into the command.
    from continual_auto_research.api.builders import _build_runner
    runner = _build_runner({"runner": {
        "kind": "broker", "project_id": "p", "workspace_dir": "/w",
        "run_command": "LR={proposal} python3 train.py"}})
    cmd = runner._command_for("0.008")
    assert cmd == "LR=0.008 python3 train.py"

    # no placeholder → verbatim
    r2 = _build_runner({"runner": {
        "kind": "broker", "project_id": "p", "workspace_dir": "/w",
        "run_command": "python3 train.py"}})
    assert r2._command_for("anything") == "python3 train.py"
