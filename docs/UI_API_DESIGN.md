# CAR full-app: API + proposer design

Turns the minimal demo UI into a real app: **configure + launch runs**, **live
visualization**, and **multi-run management** — with three proposer backends
(`claude -p`, LLM-via-API, Ollama-on-UCL-GPU), mirroring auto-scientist's switch.

The guiding rule stays: **the library is the product; the API/UI are a thin shell
over it.** Anything the UI shows must come from the library's public surface — if
it can't, that's a library gap to close first.

## What already exists (no new library code needed to surface)

The `HillClimber` library already emits/holds everything below; today's UI just
drops most of it.

- **Config knobs** — `HillClimbConfig`: `direction` (min/max), `max_iterations`,
  `patience`, `target_score`. (`/ws/run` accepts the first three; `target_score`
  is a real knob not yet wired to the UI.)
- **Per-candidate history** — `state.history` = list of `{iteration, proposal,
  score, accepted, raw_result}`. The `raw_result` holds the **measured run
  output** (the objective breakdown), already captured, never surfaced.
- **Live signals** — every `scored` event carries `improved`, `best`,
  `stale_rounds` (the plateau counter); `done` carries `stop_reason`.
- **Resume** — `HillClimber(state=…)` resumes; budget-resume already works.

## New API (REST + the existing WS)

Persistence is the one genuinely-new backend piece: the library is ephemeral, so
add a small JSON-file run store under a runs dir.

| Method | Route | Purpose |
|--------|-------|---------|
| `GET`  | `/api/runs` | list runs (id, status, best, direction, started) |
| `POST` | `/api/runs` | create + start a run; body = full config (below) |
| `GET`  | `/api/runs/{id}` | one run: config + full history + best + stop_reason |
| `POST` | `/api/runs/{id}/resume` | resume with `{max_iter}` more budget |
| `POST` | `/api/runs/{id}/stop` | request cancel (needs a stop flag in the loop) |
| `DELETE` | `/api/runs/{id}` | delete a stored run |
| `WS`   | `/ws/runs/{id}` | live event stream for THAT run (replaces the single `/ws/run`) |
| `GET`  | `/api/proposers` | available proposer backends + their ready state |
| `GET`  | `/api/health/backends` | claude CLI / ollama proxy / API key readiness (status lights) |

**Run config (POST body):**
```json
{
  "direction": "min",
  "max_iter": 20, "patience": 4, "target_score": null,
  "runner": {"kind": "broker", "project_id": "...", "workspace_dir": "...",
             "run_command": "cd exp && python run.py"},
  "proposer": {"kind": "claude|api|ollama", "model": "...", "system": "..."}
}
```

## The proposer backends

CAR's equivalent of auto-scientist's `llm=local|api|claude` switch. The fork
routes at *executor* level (vessel/employee machinery CAR doesn't have); CAR only
needs a **proposer factory** returning the `(context) -> proposal` callable the
`HillClimber` already takes.

**Key simplification — verified from the fork's box-local env profiles:** `api`
and `local`/`ollama` are the SAME code path. The fork's `llm.api.env` and
`llm.local.env` are structurally identical — both `DEFAULT_API_PROVIDER=custom`,
`CUSTOM_CHAT_CLASS=openai`, differing only in `base_url` + model:

```
# llm.api.env   → base_url = https://litellm.yangtzeailab.com/v1  (team proxy)
# llm.local.env → base_url = http://127.0.0.1:11435/v1           (on-box Ollama proxy)
```

Ollama exposes an OpenAI-compatible endpoint, so there is **no Ollama-specific
code** — both go through one `ChatOpenAI(base_url, model, api_key)` client. (My
earlier "Ollama is a stub" note was wrong: it works via this base_url swap; the
config just lives in the box-local, git-excluded `env-profiles/`, not the repo.)

So CAR needs only **two** proposer implementations:

```python
# core/proposers.py
def build_proposer(cfg: dict) -> Callable[[str], str]:
    kind = cfg["kind"]
    if kind == "claude":
        return ClaudeCliProposer(model=cfg.get("model"), system=cfg.get("system"))
    # "api" and "ollama"/"local" are the same client — only base_url + model differ
    return OpenAICompatProposer(model=cfg["model"], base_url=cfg["base_url"],
                                api_key=cfg.get("api_key", "x"), system=cfg.get("system"))
```

### `OpenAICompatProposer` — covers BOTH api and local/ollama
`ChatOpenAI(model, base_url, api_key).invoke(messages).content`. Switched by
`base_url`:
- **api**: the team LiteLLM proxy (`https://.../v1`, model e.g. `deepseek-v4-flash`)
- **local/ollama**: the on-box wake-proxy `http://127.0.0.1:11435/v1`
  (`ucl-infra/start-ollama-proxy.sh`; idle-kills ollama after 900s, wakes on
  demand; model e.g. `qwen3.6-64k:27b-q4_K_M`). For local, ping `/_proxy/health`
  first to wake the GPU and tolerate cold-start latency.

This matches the fork's env profiles exactly — the UI just presents api/local as
two presets that fill in `base_url` + `model`.

### `ClaudeCliProposer` — the `claude -p` subscription CLI
The one genuinely-different backend (subprocess, not an HTTP client). Modeled on
the fork's `claude_session.py` (verified real). A subprocess call:
```
claude -p --output-format json --model <model> <<< "<system+context prompt>"
```
Capture stdout, extract the candidate. Uses the CLI's own OAuth (no API key in
env). Box has `claude` v2.1.112 at `~/.local/bin/claude`.

## Frontend (the viz)

All data already flows over the per-run WS; this is rendering:
- **Score-over-time chart** (iteration → best & per-candidate score), min/max aware
- **Candidate table**: iteration · proposal · score · accept/reject · expand →
  `raw_result` measured output
- **Live status**: current phase, `stale_rounds` plateau meter, stop reason
- **Run list** sidebar (from `/api/runs`): switch between runs, resume, delete
- **Launch form**: direction, budget, patience, target, runner kind + params,
  proposer kind + model — with backend status lights from `/api/health/backends`

## Build sequence (PRs)

1. **Library gaps** — add a cancel/stop flag to `stream()`; ensure `raw_result`
   and `stale_rounds` are in every event. (small)
2. **Run store + REST** — JSON-file persistence, the `/api/runs*` routes,
   per-run WS. UI run-list + launch form against it. (the app skeleton)
3. **Proposers** — `core/proposers.py` with the 3 backends + `/api/proposers` +
   `/api/health/backends` status lights. (the headline feature)
4. **Viz** — chart + candidate table + drill-down. (polish)

Each PR keeps the app runnable. The library stays usable headless throughout.
```
