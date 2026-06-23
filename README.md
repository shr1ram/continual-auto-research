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

## The web UI

A small shell over the library — it opens a websocket, sends a run config, and
renders each `HillClimber` event. Anything the UI shows comes from the event
stream; if it needs data an event doesn't carry, that's a library API gap.

```bash
pip install -e ".[web]"
car-serve            # → http://127.0.0.1:8000
```

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
