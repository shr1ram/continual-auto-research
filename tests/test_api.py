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
