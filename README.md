# continual-auto-research

A programmatic **hill-climbing optimiser** for experiment candidates, with a thin
web UI on top. Propose a candidate → run it on a GPU → score it → keep the best →
repeat until the metric plateaus or the budget runs out.

Extracted from the Memento-Research fork into a clean, standalone, library-first
project. The optimisation policy is engine-agnostic; the GPU execution lives in
the shared [`ucl-gpu-infra`](../ucl-gpu-infra) package.

## Library-first

You can optimise with **no server**:

```python
from continual_auto_research import HillClimber, CallableRunner

hc = HillClimber(
    propose=my_proposer,            # (context: str) -> proposal text
    runner=CallableRunner(my_fn),   # (proposal) -> (score, output)
    direction="min",                # or "max"
)
result = hc.run(max_iter=20, patience=4)     # blocking
print(result.best_score, result.stop_reason, len(result.history))
```

Or consume a live event feed (this is exactly what the UI uses):

```python
for event in hc.stream(max_iter=20):
    # {"type": "proposed"|"scored"|"accepted"|"done", ...}
    ...
```

A batch sweep is just a loop over `run()`. No FastAPI, no websocket, no DB.

## Running on the UCL GPU

Swap the runner for `BrokerRunner`, which claims a GPU, submits the candidate via
`ucl-gpu-infra`, polls to terminal, and scores from the run's `result.json` /
`SCORE=` sentinel:

```python
from continual_auto_research import HillClimber, BrokerRunner

hc = HillClimber(
    propose=my_llm_proposer,
    runner=BrokerRunner(
        project_id="ctrl-challenge",
        workspace_dir="/path/to/candidate/workspace",
        run_command="cd exp && python run.py",
    ),
    direction="min",
)
result = hc.run(max_iter=20, patience=4)
```

The proposer must print `SCORE=<n>` as the run's final stdout line, or write
`{"score": <n>, "direction": "min"|"max"}` to `result.json` in the workspace. A
`result.json` marked `status: "draft"` (or any non-executed status) is ignored —
a placeholder that never ran must not be scored as real.

## The web app

A full app over the library: launch/configure runs, watch them live, and manage
past runs. Still a thin shell — everything it shows comes from the library's event
stream and persisted state.

```bash
pip install -e ".[web,llm]"
car-serve            # → http://127.0.0.1:8000
```

**API:**

| Method | Route | |
|--------|-------|--|
| `GET/POST` | `/api/runs` | list / create+start a run |
| `GET/DELETE` | `/api/runs/{id}` | fetch / delete a run |
| `POST` | `/api/runs/{id}/resume` | resume with `{max_iter}` more budget |
| `POST` | `/api/runs/{id}/stop` | cancel after the current iteration |
| `WS` | `/ws/runs/{id}` | live event stream (or replay if finished) |
| `GET` | `/api/proposers` · `/api/health/backends` | backend readiness (status lights) |

**Proposer backends** (the `proposer.kind` in a run config) — mirroring
auto-scientist's `llm` switch:
- `claude` — the `claude -p` subscription CLI (no API key)
- `api` — a hosted OpenAI-compatible proxy (`DEFAULT_API_BASE_URL` + key)
- `ollama`/`local` — Ollama on a UCL GPU via the on-box wake-proxy (`:11435`)

`api` and `ollama` share one OpenAI-compatible client (differ only in `base_url`);
`claude` is a subprocess. Runs are persisted under `$CAR_RUNS_DIR`
(default `~/.continual-auto-research/runs`).

## Layout

```
src/continual_auto_research/
  core/
    hill_climb.py    # the controller — optimisation policy (engine-agnostic)
    scoring.py       # SCORE= sentinel + draft-guarded result.json reader
    runner.py        # CallableRunner (tests/local) + BrokerRunner (UCL GPU)
    climber.py       # HillClimber facade: run() / stream()
  api/app.py         # thin FastAPI: /ws/run forwards stream() events
  frontend/          # index.html + pipeline-controller.js (the whole UI)
```

> Internal symbols keep the accurate `HillClimb*` names — it *is* hill climbing.
> Only the project's user-facing surface is branded "auto-research".

## Tests

```bash
pip install -e ../ucl-gpu-infra      # the shared infra dep, editable locally
pip install -e ".[test,web]"
pytest                                # 24 tests; no GPU required
```
