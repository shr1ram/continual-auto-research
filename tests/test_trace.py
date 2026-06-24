"""Trace capture tests: proposer/runner expose .last_trace; climber emits trace
events + accumulates climber.traces; the run store persists them."""
import types
from continual_auto_research import HillClimber, CallableRunner
from continual_auto_research.core import proposers as P


def test_climber_emits_trace_events_and_accumulates():
    # a proposer exposing last_trace (mimics the real OpenAICompatProposer)
    class TracingProposer:
        def __init__(self): self.last_trace = None
        def __call__(self, context):
            self.last_trace = {"backend": "test", "system": "sys", "prompt": context, "response": "cand"}
            return "cand"
    prop = TracingProposer()
    hc = HillClimber(propose=prop, runner=CallableRunner(lambda p: (1.0, "SCORE=1\nlog output")),
                     direction="max")
    events = list(hc.stream(max_iter=2))
    traces = [e for e in events if e["type"] == "trace"]
    assert len(traces) == 2
    t = traces[0]
    assert t["proposer"]["prompt"]  # the context sent to the LLM
    assert t["proposer"]["response"] == "cand"
    assert t["runner"]["output"] == "SCORE=1\nlog output"   # full runner output
    # climber accumulates them for persistence
    assert len(hc.traces) == 2


def test_trace_falls_back_when_proposer_has_no_last_trace():
    # a bare callable proposer (no .last_trace) → trace still emitted with the context
    hc = HillClimber(propose=lambda ctx: "x", runner=CallableRunner(lambda p: (1.0, "out")),
                     direction="max")
    traces = [e for e in hc.stream(max_iter=1) if e["type"] == "trace"]
    assert len(traces) == 1
    assert "prompt" in traces[0]["proposer"]     # the context we sent
    assert traces[0]["runner"]["output"] == "out"


def test_openai_proposer_records_last_trace(monkeypatch):
    p = P.OpenAICompatProposer(model="m", base_url="http://x", api_key="k")
    fake = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
        create=lambda **kw: types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="resp"))]))))
    monkeypatch.setattr(p, "_ensure_client", lambda: fake)
    p("the prompt context")
    assert p.last_trace["prompt"] == "the prompt context"
    assert p.last_trace["response"] == "resp"
    assert p.last_trace["system"]   # the system prompt is captured
    assert p.last_trace["model"] == "m"


def test_claude_proposer_records_last_trace(monkeypatch):
    import json
    monkeypatch.setattr(P.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=0, stdout=json.dumps({"result": "claude-cand"}), stderr=""))
    p = P.ClaudeCliProposer()
    p("ctx-in")
    assert p.last_trace["prompt"] == "ctx-in"
    assert p.last_trace["response"] == "claude-cand"
    assert p.last_trace["backend"] == "claude_cli"
