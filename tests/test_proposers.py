"""Proposer tests — the factory, the OpenAI-compat client (mocked), and the
claude CLI proposer (subprocess mocked). No real LLM is called."""
import json
import types

import pytest

from continual_auto_research.core import proposers as P


# ── factory ──────────────────────────────────────────────────────────────────

def test_factory_routes_kinds():
    assert isinstance(P.build_proposer({"kind": "claude"}), P.ClaudeCliProposer)
    assert isinstance(P.build_proposer({"kind": "api", "model": "m", "base_url": "http://x"}),
                      P.OpenAICompatProposer)
    assert isinstance(P.build_proposer({"kind": "ollama", "model": "m"}),
                      P.OpenAICompatProposer)
    assert isinstance(P.build_proposer({"kind": "local", "model": "m"}),
                      P.OpenAICompatProposer)


def test_factory_unknown_kind_raises():
    with pytest.raises(ValueError):
        P.build_proposer({"kind": "telepathy"})


def test_ollama_defaults_to_proxy_url_and_wake():
    p = P.build_proposer({"kind": "ollama", "model": "qwen"})
    assert "11435" in p.base_url
    assert p.wake_url and "_proxy/health" in p.wake_url


def test_api_reads_env_base_url(monkeypatch):
    monkeypatch.setenv("DEFAULT_API_BASE_URL", "https://proxy/v1")
    monkeypatch.setenv("CUSTOM_API_KEY", "k123")
    p = P.build_proposer({"kind": "api", "model": "deepseek"})
    assert p.base_url == "https://proxy/v1"
    assert p.api_key == "k123"


# ── OpenAICompatProposer (mock the openai client) ────────────────────────────

class _FakeChoice:
    def __init__(self, content):
        self.message = types.SimpleNamespace(content=content)


class _FakeCompletions:
    def __init__(self, content):
        self._content = content
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return types.SimpleNamespace(choices=[_FakeChoice(self._content)])


def test_openai_compat_calls_and_returns_text(monkeypatch):
    p = P.OpenAICompatProposer(model="m", base_url="http://x", api_key="k")
    fake = types.SimpleNamespace(chat=types.SimpleNamespace(completions=_FakeCompletions("  cand-X  ")))
    monkeypatch.setattr(p, "_ensure_client", lambda: fake)
    out = p("best so far: 5\npropose better")
    assert out == "cand-X"  # stripped
    call = fake.chat.completions.calls[0]
    assert call["model"] == "m"
    assert call["messages"][0]["role"] == "system"
    assert "propose better" in call["messages"][1]["content"]


def test_openai_compat_wake_is_best_effort(monkeypatch):
    # a failing wake ping must not break the proposal
    p = P.OpenAICompatProposer(model="m", base_url="http://x", wake_url="http://dead/health")
    fake = types.SimpleNamespace(chat=types.SimpleNamespace(completions=_FakeCompletions("ok")))
    monkeypatch.setattr(p, "_ensure_client", lambda: fake)
    assert p("ctx") == "ok"


# ── ClaudeCliProposer (mock subprocess) ──────────────────────────────────────

def test_claude_cli_parses_json_result(monkeypatch):
    def fake_run(cmd, input=None, capture_output=None, text=None, timeout=None):
        assert "claude" in cmd[0] and "-p" in cmd
        return types.SimpleNamespace(returncode=0, stdout=json.dumps({"result": "claude-cand"}), stderr="")
    monkeypatch.setattr(P.subprocess, "run", fake_run)
    assert P.ClaudeCliProposer()("ctx") == "claude-cand"


def test_claude_cli_falls_back_to_raw_stdout(monkeypatch):
    def fake_run(cmd, **kw):
        return types.SimpleNamespace(returncode=0, stdout="plain text candidate", stderr="")
    monkeypatch.setattr(P.subprocess, "run", fake_run)
    assert P.ClaudeCliProposer()("ctx") == "plain text candidate"


def test_claude_cli_nonzero_raises(monkeypatch):
    def fake_run(cmd, **kw):
        return types.SimpleNamespace(returncode=1, stdout="", stderr="boom")
    monkeypatch.setattr(P.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="claude CLI failed"):
        P.ClaudeCliProposer()("ctx")


def test_claude_cli_missing_binary_raises(monkeypatch):
    def fake_run(cmd, **kw):
        raise FileNotFoundError("no claude")
    monkeypatch.setattr(P.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="not found"):
        P.ClaudeCliProposer()("ctx")


# ── backend status ───────────────────────────────────────────────────────────

def test_backend_status_shape(monkeypatch):
    # force claude probe to "found" and api env present; ollama unreachable
    def fake_run(cmd, **kw):
        return types.SimpleNamespace(returncode=0, stdout="2.1.0", stderr="")
    monkeypatch.setattr(P.subprocess, "run", fake_run)
    monkeypatch.setenv("DEFAULT_API_BASE_URL", "https://x/v1")
    monkeypatch.setenv("CUSTOM_API_KEY", "k")
    st = P.backend_status()
    assert set(st) == {"claude", "api", "ollama"}
    assert st["claude"]["ready"] is True
    assert st["api"]["ready"] is True
    assert "ready" in st["ollama"]
